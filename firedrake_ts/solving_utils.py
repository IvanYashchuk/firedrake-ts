from firedrake import assemble
from firedrake.bcs import DirichletBC
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import MeshGeometry
from firedrake.ufl_expr import TestFunction
import numpy

import itertools
from firedrake import homogenize, derivative, adjoint
from numpy.core.shape_base import block
from pyadjoint import block_variable

from pyop2 import op2
from firedrake import function, dmhooks, Constant, TrialFunction
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import PETSc
from firedrake import vector
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.utils import cached_property

from pyadjoint import no_annotations

print = lambda x: PETSc.Sys.Print(x)


def _make_reasons(reasons):
    return dict(
        [(getattr(reasons, r), r) for r in dir(reasons) if not r.startswith("_")]
    )


TSReasons = _make_reasons(PETSc.TS.ConvergedReason())


def check_ts_convergence(ts):
    r = ts.getConvergedReason()
    # TODO: submit PR to petsc4py to add the following reasons
    # TSFORWARD_DIVERGED_LINEAR_SOLVE = -3,
    # TSADJOINT_DIVERGED_LINEAR_SOLVE = -4
    if r == -3:
        raise ConvergenceError(
            f"TS solve failed to converge. Reason: TSFORWARD_DIVERGED_LINEAR_SOLVE"
        )
    if r == -4:
        raise ConvergenceError(
            f"TS solve failed to converge. Reason: TSADJOINT_DIVERGED_LINEAR_SOLVE"
        )
    reason = TSReasons[r]
    if r < 0:
        raise ConvergenceError(
            f"TS solve failed to converge after {ts.getStepNumber()} iterations. Reason: {reason}"
        )


class _TSContext(object):
    r"""
    Context holding information for TS callbacks.

    :arg problem: a :class:`DAEProblem`.
    :arg mat_type: Indicates whether the Jacobian is assembled
        monolithically ('aij'), as a block sparse matrix ('nest') or
        matrix-free (as :class:`~.ImplicitMatrix`, 'matfree').
    :arg pmat_type: Indicates whether the preconditioner (if present) is assembled
        monolithically ('aij'), as a block sparse matrix ('nest') or
        matrix-free (as :class:`~.ImplicitMatrix`, 'matfree').
    :arg appctx: Any extra information used in the assembler.  For the
        matrix-free case this will contain the Newton state in
        ``"state"``.
    :arg pre_jacobian_callback: User-defined function called immediately
        before Jacobian assembly
    :arg pre_function_callback: User-defined function called immediately
        before residual assembly
    :arg post_jacobian_callback: User-defined function called immediately
        after Jacobian assembly
    :arg post_function_callback: User-defined function called immediately
        after residual assembly
    :arg options_prefix: The options prefix of the TS.
    :arg transfer_manager: Object that can transfer functions between
        levels, typically a :class:`~.TransferManager`

    The idea here is that the TS holds a shell DM which contains
    this object as "user context".  When the TS calls back to the
    user form_function code, we pull the DM out of the TS and then
    get the context (which is one of these objects) to find the
    Firedrake level information.
    """

    def __init__(
        self,
        problem,
        mat_type,
        pmat_type,
        appctx=None,
        pre_jacobian_callback=None,
        pre_function_callback=None,
        post_jacobian_callback=None,
        post_function_callback=None,
        options_prefix=None,
        transfer_manager=None,
    ):
        from firedrake.bcs import DirichletBC
        from firedrake import ufl_expr

        if pmat_type is None:
            pmat_type = mat_type
        self.mat_type = mat_type
        self.pmat_type = pmat_type
        self.options_prefix = options_prefix

        matfree = mat_type == "matfree"
        pmatfree = pmat_type == "matfree"

        self._problem = problem
        self._pre_jacobian_callback = pre_jacobian_callback
        self._pre_function_callback = pre_function_callback
        self._post_jacobian_callback = post_jacobian_callback
        self._post_function_callback = post_function_callback

        self.fcp = problem.form_compiler_parameters

        # timeshift value provided by the solver
        self.shift = Constant(1.0)

        F = self._problem.F
        udot = self._problem.udot
        u = self._problem.u

        self.J = self._problem.J or self.shift * ufl_expr.derivative(
            F, udot
        ) + ufl_expr.derivative(F, u)

        if appctx is None:
            appctx = {}
        # A split context will already get the full state.
        # TODO, a better way of doing this.
        # Now we don't have a temporary state inside the snes
        # context we could just require the user to pass in the
        # full state on the outside.
        appctx.setdefault("state", self._problem.u)
        appctx.setdefault("form_compiler_parameters", self.fcp)

        self.appctx = appctx
        self.matfree = matfree
        self.pmatfree = pmatfree

        # For Jp to equal J, bc.Jp must equal bc.J for all EquationBC objects.
        Jp_eq_J = problem.Jp is None and all(bc.Jp_eq_J for bc in problem.bcs)

        if mat_type != pmat_type or not Jp_eq_J:
            # Need separate pmat if either Jp is different or we want
            # a different pmat type to the mat type.
            if problem.Jp is None:
                self.Jp = self.J
            else:
                self.Jp = problem.Jp
        else:
            # pmat_type == mat_type and Jp_eq_J
            self.Jp = None

        self.bcs_F = [
            bc if isinstance(bc, DirichletBC) else bc._F for bc in problem.bcs
        ]
        self.bcs_J = [
            bc if isinstance(bc, DirichletBC) else bc._J for bc in problem.bcs
        ]
        self.bcs_Jp = [
            bc if isinstance(bc, DirichletBC) else bc._Jp for bc in problem.bcs
        ]

        self.create_assemble_residual()

        self._jacobian_assembled = False
        self._splits = {}
        self._coarse = None
        self._fine = None

        self._nullspace = None
        self._nullspace_T = None
        self._near_nullspace = None
        self._transfer_manager = transfer_manager

        # Keep a map of the local indices in the stacked vector where each block variable
        # is placed
        self.bv_indices_map = {}

    def create_assemble_residual(self):
        from firedrake.assemble import create_assembly_callable

        self._assemble_residual = create_assembly_callable(
            self._problem.F,
            tensor=self._F,
            bcs=self.bcs_F,
            form_compiler_parameters=self.fcp,
        )

    @property
    def transfer_manager(self):
        """This allows the transfer manager to be set from options, e.g.

        solver_parameters = {"ksp_type": "cg",
                             "pc_type": "mg",
                             "mg_transfer_manager": __name__ + ".manager"}

        The value for "mg_transfer_manager" can either be a specific instantiated
        object, or a function or class name. In the latter case it will be invoked
        with no arguments to instantiate the object.

        If "snes_type": "fas" is used, the relevant option is "fas_transfer_manager",
        with the same semantics.
        """
        if self._transfer_manager is None:
            opts = PETSc.Options()
            prefix = self.options_prefix or ""
            if opts.hasName(prefix + "mg_transfer_manager"):
                managername = opts[prefix + "mg_transfer_manager"]
            elif opts.hasName(prefix + "fas_transfer_manager"):
                managername = opts[prefix + "fas_transfer_manager"]
            else:
                managername = None

            if managername is None:
                from firedrake import TransferManager

                transfer = TransferManager(use_averaging=True)
            else:
                (modname, objname) = managername.rsplit(".", 1)
                mod = __import__(modname)
                obj = getattr(mod, objname)
                if isinstance(obj, type):
                    transfer = obj()
                else:
                    transfer = obj

            self._transfer_manager = transfer
        return self._transfer_manager

    @transfer_manager.setter
    def transfer_manager(self, manager):
        if self._transfer_manager is not None:
            raise ValueError("Must set transfer manager before first use.")
        self._transfer_manager = manager

    class RHSJacPShell:
        def __init__(self, ctx):
            self.ctx = ctx

        def scale(self, A, alpha):
            raise RuntimeError("Not implemented")

        def mult(self, A, x, y):
            "y <- A * x"
            raise RuntimeError("Not implemented")

        def multTranspose(self, A, x, y):
            "y <- A' * x"
            y_ownership = y.getOwnershipRange()
            block_outputs = [dep.output for dep in self.ctx.block.get_outputs()]
            bv_indices_map = self.ctx.bv_indices_map
            for block_variable in self.ctx.block.get_dependencies():
                coeff = block_variable.output
                c_rep = block_variable.saved_output
                if coeff in block_outputs:
                    continue
                if c_rep not in self.ctx._problem.F.coefficients():
                    continue
                if isinstance(c_rep, DirichletBC):
                    RuntimeWarning(
                        "DirichletBC control not supported, ignoring this dependency"
                    )
                    continue
                elif isinstance(c_rep, MeshGeometry):
                    RuntimeWarning(
                        " MeshGeometry control not supported, ignoring this dependency"
                    )
                    continue
                elif isinstance(c_rep, Constant):
                    mesh = self.ctx._problem.F.ufl_domain()
                    tmp = function.Function(c_rep._ad_function_space(mesh))
                    dFdf = derivative(self.ctx._problem.F, c_rep)
                    local_indices = bv_indices_map[coeff]
                    with tmp.dat.vec as y_vec:
                        local_range = y_vec.getOwnershipRange()
                        assemble(adjoint(dFdf)).petscmat.mult(x, y_vec)
                    y[
                        (y_ownership[0] + local_indices[0]) : (
                            y_ownership[0] + local_indices[1]
                        )
                    ] = tmp.dat.data
                else:
                    tmp = function.Function(c_rep.function_space())
                    dFdf = derivative(self.ctx._problem.F, c_rep)
                    local_indices = bv_indices_map[coeff]
                    with tmp.dat.vec as y_vec:
                        local_range = y_vec.getOwnershipRange()
                        assemble(adjoint(dFdf)).petscmat.mult(x, y_vec)
                        y[
                            (y_ownership[0] + local_indices[0]) : (
                                y_ownership[0] + local_indices[1]
                            )
                        ] = y_vec[local_range[0] : (local_range[1])]

            y.assemble()

    def set_ifunction(self, ts):
        r"""Set the function to compute F(t,U,U_t) where F() = 0 is the DAE to be solved."""
        with self._F.dat.vec_wo as v:
            ts.setIFunction(self.form_function, f=v)

    def set_ijacobian(self, ts):
        r"""Set the function to compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t) is the function provided with set_ifunction()"""
        ts.setIJacobian(self.form_jacobian, J=self._jac.petscmat, P=self._pjac.petscmat)

    def set_quad_rhsfunction(self, ts):
        r"""Set the function to compute the cost function integrand"""
        ts.setRHSFunction(self.form_cost_integrand)

    def set_quad_rhsjacobian(self, ts, zero=False):
        r"""Set the function to compute the cost function integrand jacobian w.r.t U"""
        if zero:
            ts.setRHSJacobian(self.form_cost_jacobian_zero, J=self._Mjac_x)
        else:
            ts.setRHSJacobian(self.form_cost_jacobian, J=self._Mjac_x)

    def set_quad_rhsjacobianP(self, ts, zero=False):
        r"""Set the function to compute the cost function integrand jacobian w.r.t p"""
        if zero:
            ts.setRHSJacobianP(self.form_cost_jacobianP_zero, A=self._Mjac_p)
        else:
            ts.setRHSJacobianP(self.form_cost_jacobianP, A=self._Mjac_p)

    def set_rhsjacobianP(self, ts):
        r"""Set the function to compute the residual RHS jacobian w.r.t p"""
        ts.setIJacobianP(self.form_jacobianP, self._dFdp)

    def set_cost_gradients(self, ts, adj_input=None):

        self._dMdp.zeroEntries()
        with self._dMdx.dat.vec as dMdx_vec:
            dMdx_vec.zeroEntries()
        if isinstance(adj_input, vector.Vector):
            if self.bcs_J:
                bcs = (homogenize(self.bcs_J),)
                bcs[0][0].apply(adj_input)  # TODO what if Dirichlet is a control
            with adj_input.dat.vec_ro as adj_vec, self._dMdx.dat.vec as dmdx_vec:
                adj_vec.copy(dmdx_vec)

        with self._dMdx.dat.vec as dMdu_vec:
            ts.setCostGradients(dMdu_vec, self._dMdp)

    def set_nullspace(self, nullspace, ises=None, transpose=False, near=False):
        if nullspace is None:
            return
        nullspace._apply(self._jac, transpose=transpose, near=near)
        if self.Jp is not None:
            nullspace._apply(self._pjac, transpose=transpose, near=near)
        if ises is not None:
            nullspace._apply(ises, transpose=transpose, near=near)

    def split(self, fields):
        from firedrake import replace, as_vector, split
        from firedrake_ts.ts_solver import DAEProblem
        from firedrake.bcs import DirichletBC, EquationBC

        fields = tuple(tuple(f) for f in fields)
        splits = self._splits.get(tuple(fields))
        if splits is not None:
            return splits

        splits = []
        problem = self._problem
        splitter = ExtractSubBlock()
        for field in fields:
            F = splitter.split(problem.F, argument_indices=(field,))
            J = splitter.split(self.J, argument_indices=(field, field))
            us = problem.u.split()
            V = F.arguments()[0].function_space()
            # Exposition:
            # We are going to make a new solution Function on the sub
            # mixed space defined by the relevant fields.
            # But the form may refer to the rest of the solution
            # anyway.
            # So we pull it apart and will make a new function on the
            # subspace that shares data.
            pieces = [us[i].dat for i in field]
            if len(pieces) == 1:
                (val,) = pieces
                subu = function.Function(V, val=val)
                subsplit = (subu,)
            else:
                val = op2.MixedDat(pieces)
                subu = function.Function(V, val=val)
                # Split it apart to shove in the form.
                subsplit = split(subu)
            # Permutation from field indexing to indexing of pieces
            field_renumbering = dict([f, i] for i, f in enumerate(field))
            vec = []
            for i, u in enumerate(us):
                if i in field:
                    # If this is a field we're keeping, get it from
                    # the new function. Otherwise just point to the
                    # old data.
                    u = subsplit[field_renumbering[i]]
                if u.ufl_shape == ():
                    vec.append(u)
                else:
                    for idx in numpy.ndindex(u.ufl_shape):
                        vec.append(u[idx])

            # So now we have a new representation for the solution
            # vector in the old problem. For the fields we're going
            # to solve for, it points to a new Function (which wraps
            # the original pieces). For the rest, it points to the
            # pieces from the original Function.
            # IOW, we've reinterpreted our original mixed solution
            # function as being made up of some spaces we're still
            # solving for, and some spaces that have just become
            # coefficients in the new form.
            u = as_vector(vec)
            F = replace(F, {problem.u: u})
            J = replace(J, {problem.u: u})
            if problem.Jp is not None:
                Jp = splitter.split(problem.Jp, argument_indices=(field, field))
                Jp = replace(Jp, {problem.u: u})
            else:
                Jp = None
            bcs = []
            for bc in problem.bcs:
                if isinstance(bc, DirichletBC):
                    bc_temp = bc.reconstruct(
                        field=field,
                        V=V,
                        g=bc.function_arg,
                        sub_domain=bc.sub_domain,
                        method=bc.method,
                    )
                elif isinstance(bc, EquationBC):
                    bc_temp = bc.reconstruct(field, V, subu, u)
                if bc_temp is not None:
                    bcs.append(bc_temp)
            new_problem = DAEProblem(
                F,
                subu,
                problem.udot,
                problem.tspan,
                bcs=bcs,
                J=J,
                Jp=Jp,
                form_compiler_parameters=problem.form_compiler_parameters,
            )
            new_problem._constant_jacobian = problem._constant_jacobian
            splits.append(
                type(self)(
                    new_problem,
                    mat_type=self.mat_type,
                    pmat_type=self.pmat_type,
                    appctx=self.appctx,
                    transfer_manager=self.transfer_manager,
                )
            )
        return self._splits.setdefault(tuple(fields), splits)

    @staticmethod
    def form_function(ts, t, X, Xdot, F):
        r"""Form the residual for this problem

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg Xdot: time derivative of state vector
        :arg F: function vector
        """
        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        # X may not be the same vector as the vec behind self._problem.u, so
        # copy guess in from X.
        with ctx._problem.u.dat.vec_wo as v:
            X.copy(v)
        with ctx._problem.udot.dat.vec_wo as v:
            Xdot.copy(v)
        ctx._problem.time.assign(t)

        if ctx._pre_function_callback is not None:
            ctx._pre_function_callback(X, Xdot, t)

        from firedrake import assemble

        ctx._assemble_residual()

        if ctx._post_function_callback is not None:
            with ctx._F.dat.vec as F_:
                ctx._post_function_callback(X, Xdot, F_, t)

        # F may not be the same vector as self._F, so copy
        # residual out to F.
        with ctx._F.dat.vec_ro as v:
            v.copy(F)

    @staticmethod
    def form_jacobian(ts, t, X, Xdot, shift, J, P):
        r"""Form the Jacobian for this problem

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg Xdot: time derivative of state vector
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake import assemble

        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        problem = ctx._problem

        assert J.handle == ctx._jac.petscmat.handle
        # TODO: Check how to use constant jacobian properly for TS
        if problem._constant_jacobian and ctx._jacobian_assembled:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        ctx._jacobian_assembled = True

        # X may not be the same vector as the vec behind self._problem.u, so
        # copy guess in from X.
        with ctx._problem.u.dat.vec_wo as v:
            X.copy(v)
        with ctx._problem.udot.dat.vec_wo as v:
            Xdot.copy(v)
        ctx._problem.time.assign(t)

        if ctx._pre_jacobian_callback is not None:
            ctx._pre_jacobian_callback(X, Xdot, t)

        ctx.shift.assign(shift)

        from firedrake import ufl_expr

        F = ctx._problem.F
        udot = ctx._problem.udot
        u = ctx._problem.u

        # Update jacobian
        ctx.J = ctx._problem.J or ctx.shift * ufl_expr.derivative(
            F, udot
        ) + ufl_expr.derivative(F, u)
        assemble(
            ctx.J,
            tensor=ctx._jac,
            bcs=ctx.bcs_J,
            form_compiler_parameters=ctx.fcp,
            mat_type=ctx.mat_type,
        )

        if ctx._post_jacobian_callback is not None:
            ctx._post_jacobian_callback(X, Xdot, J)

        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac()

        ises = ctx.J.arguments()[0].function_space()._ises
        ctx.set_nullspace(ctx._nullspace, ises, transpose=False, near=False)
        ctx.set_nullspace(ctx._nullspace_T, ises, transpose=True, near=False)
        ctx.set_nullspace(ctx._near_nullspace, ises, transpose=False, near=True)

    @staticmethod
    @no_annotations
    def form_cost_integrand(ts, t, X, R):
        r"""Form the integrand of the nost function

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg R: function vector
        """
        from firedrake.assemble import assemble

        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        # X may not be the same vector as the vec behind self._problem.u, so
        # copy guess in from X.
        with ctx._problem.u.dat.vec_wo as v:
            X.copy(v)
        ctx._problem.time.assign(t)

        j_value = assemble(ctx._problem.M)
        R.set(j_value)

    @staticmethod
    @no_annotations
    def form_cost_jacobian(ts, t, X, J, P):
        r"""Form the jacobian of the nost function

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake.assemble import assemble
        from firedrake import derivative

        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        # X may not be the same vector as the vec behind self._problem.u, so
        # copy guess in from X.
        with ctx._problem.u.dat.vec_wo as v:
            X.copy(v)
        ctx._problem.time.assign(t)

        assemble(
            derivative(ctx._problem.M, ctx._problem.u),
            tensor=ctx._Mjac_x_vec,
            bcs=homogenize(ctx.bcs_J),
        )
        Jmat_array = J.getDenseArray()
        # Here we're getting only the local copy with getDenseArray
        with ctx._Mjac_x_vec.dat.vec_ro as v:
            Jmat_array[:, 0] = v.array[:]
        J.assemble()

    @staticmethod
    @no_annotations
    def form_cost_jacobian_zero(ts, t, X, J, P):
        r"""Form the jacobian of the nost function

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        pass

    @staticmethod
    @no_annotations
    def form_cost_jacobianP(ts, t, X, J):
        r"""Form the jacobian of the cost function w.r.t. p

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake.assemble import assemble
        from firedrake import derivative

        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        # X may not be the same vector as the vec behind self._problem.u, so
        # copy guess in from X.
        with ctx._problem.u.dat.vec_wo as v:
            X.copy(v)
        ctx._problem.time.assign(t)

        if ctx.block.get_dependencies():
            bv_indices_map = ctx.bv_indices_map
            for i, block_variable in enumerate(ctx.block.get_dependencies()):
                c_rep = block_variable.saved_output
                coeff = block_variable.output
                if c_rep not in ctx._problem.M.coefficients():
                    continue
                if isinstance(c_rep, function.Function):
                    trial_function = TestFunction(c_rep.function_space())
                elif isinstance(c_rep, Constant):
                    mesh = ctx._problem.M.ufl_domain()
                    trial_function = TestFunction(c_rep._ad_function_space(mesh))
                elif isinstance(c_rep, DirichletBC):
                    RuntimeWarning(
                        "DirichletBC control not supported, ignoring this dependency"
                    )
                    continue
                elif isinstance(c_rep, MeshGeometry):
                    RuntimeWarning(
                        " MeshGeometry control not supported, ignoring this dependency"
                    )
                    continue

                assemble(
                    derivative(ctx._problem.M, c_rep, trial_function),
                    tensor=ctx._Mjac_p_vecs[coeff],
                )

                J_ownership = J.getOwnershipRange()
                local_indices = bv_indices_map[coeff]
                if isinstance(c_rep, function.Function):
                    with ctx._Mjac_p_vecs[coeff].dat.vec_ro as v:
                        local_range = v.getOwnershipRange()
                        J[
                            (J_ownership[0] + local_indices[0]) : (
                                J_ownership[0] + local_indices[1]
                            ),
                            0,
                        ] = v[local_range[0] : local_range[1]]
                elif isinstance(c_rep, Constant):
                    J[
                        (J_ownership[0] + local_indices[0]) : (
                            J_ownership[0] + local_indices[1]
                        ),
                        0,
                    ] = ctx._Mjac_p_vecs[coeff].dat.data
            J.assemble()

    @staticmethod
    @no_annotations
    def form_cost_jacobianP_zero(ts, t, X, J):
        r"""Form the jacobian of the cost function w.r.t. p

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        pass

    @staticmethod
    @no_annotations
    def form_jacobianP(ts, t, X, Xdot, a, J):
        r"""Form the jacobian of the RHS function w.r.t. p (not P the preconditioner)
        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake.assemble import assemble
        from firedrake import derivative

        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        ctx._problem.time.assign(t)
        with ctx._problem.u.dat.vec_wo as v:
            X.copy(v)
        with ctx._problem.udot.dat.vec_wo as v:
            Xdot.copy(v)

    @staticmethod
    def compute_operators(ksp, J, P):
        r"""Form the Jacobian for this problem

        :arg ksp: a PETSc KSP object
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake.bcs import DirichletBC

        dm = ksp.getDM()
        ctx = dmhooks.get_appctx(dm)
        problem = ctx._problem
        assert J.handle == ctx._jac.petscmat.handle
        # TODO: Check how to use constant jacobian properly for TS
        if problem._constant_jacobian and ctx._jacobian_assembled:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        ctx._jacobian_assembled = True

        fine = ctx._fine
        if fine is not None:
            manager = dmhooks.get_transfer_operators(
                fine._problem.u.function_space().dm
            )
            manager.inject(fine._problem.u, ctx._problem.u)

            for bc in itertools.chain(*ctx._problem.bcs):
                if isinstance(bc, DirichletBC):
                    bc.apply(ctx._problem.u)

        ctx._assemble_jac()

        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac()

    @cached_property
    def _jac(self):
        from firedrake.assemble import allocate_matrix
        from firedrake import ufl_expr

        return allocate_matrix(
            self.J,
            bcs=self.bcs_J,
            form_compiler_parameters=self.fcp,
            mat_type=self.mat_type,
            appctx=self.appctx,
            options_prefix=self.options_prefix,
        )

    @cached_property
    def _assemble_jac(self):
        from firedrake.assemble import create_assembly_callable

        return create_assembly_callable(
            self.J,
            tensor=self._jac,
            bcs=self.bcs_J,
            form_compiler_parameters=self.fcp,
            mat_type=self.mat_type,
        )

    # The jacobian of the cost function has to be a dense matrix per PETSc requirements,
    # but Firedrake is not able to assemble the liner form resulting from derivative(M, u)
    # into a dense matrix, only into a vector (not even augmenting the 0-form with Real space)
    # We decide to have two copies of the same information. One Firedrake vector to assemble
    # the linear form into (_Mjac_x_vec) and a dense matrix to pass the information to PETSc
    # We do the same for the jacobian w.r.t. the control
    @cached_property
    def _Mjac_x_vec(self):
        return function.Function(self._problem.F.arguments()[0].function_space())

    @cached_property
    def _Mjac_x(self):
        from firedrake import derivative
        from firedrake.assemble import assemble

        local_dofs = self._Mjac_x_vec.dat.data.size
        djdu_transposed_mat = PETSc.Mat().createDense(
            [[local_dofs, self._Mjac_x_vec.ufl_function_space().dim()], [1, 1]]
        )
        djdu_transposed_mat.setUp()
        # djdu_transposed_mat.assemble()

        return djdu_transposed_mat

    @cached_property
    def _Mjac_p(self):
        # TODO _Mjac_p and the form_cost_jacobianP share the same loop over the dependencies
        # You should refactor them together or something
        # This vector is allocated each time the adjoint is evaluated. Refactor?
        local_m_size = 0
        m = 0
        block_outputs = [dep.output for dep in self.block.get_outputs()]
        for block_variable in self.block.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in block_outputs or (c_rep not in self._problem.M.coefficients()):
                continue
            # if self.u0:
            #    if coeff == self.u0:
            #        continue
            if isinstance(coeff, DirichletBC):
                RuntimeWarning(
                    "DirichletBC control not supported, ignoring this dependency"
                )
                continue
            elif isinstance(coeff, MeshGeometry):
                RuntimeWarning(
                    " MeshGeometry control not supported, ignoring this dependency"
                )
                continue
            elif isinstance(coeff, Constant):
                local_m_size += coeff.dat.data.size
                m += local_m_size
            else:
                local_m_size += coeff.dat.data.size
                m += coeff.function_space().dim()

        djdp_transposed_mat = PETSc.Mat().createDense([[local_m_size, m], [1, 1]])
        djdp_transposed_mat.setUp()
        # djdp_transposed_mat.assemble()

        return djdp_transposed_mat

    @cached_property
    def _dFdp(self):
        # TODO another loop that it is the same than _Mjac_p and _Mjac_p_vecs... REFACTOR!!
        from firedrake import derivative
        from firedrake.assemble import assemble

        u = self._problem.u
        local_n_size = u.dat.data.size
        n = u.function_space().dim()

        local_m_size = 0
        m = 0
        block_outputs = [dep.output for dep in self.block.get_outputs()]
        for block_variable in self.block.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in block_outputs or (c_rep not in self._problem.F.coefficients()):
                continue
            if isinstance(coeff, DirichletBC):
                RuntimeWarning(
                    "DirichletBC control not supported, ignoring this dependency"
                )
                continue
            elif isinstance(coeff, MeshGeometry):
                RuntimeWarning(
                    " MeshGeometry control not supported, ignoring this dependency"
                )
                continue
            elif isinstance(coeff, Constant):
                local_m_size += coeff.dat.data.size
                m += local_m_size
            else:
                local_m_size += coeff.dat.data.size
                m += coeff.function_space().dim()
            self.bv_indices_map[coeff] = (
                local_m_size - coeff.dat.data.size,
                local_m_size,
            )
        J = PETSc.Mat().createAIJ([[local_n_size, n], [local_m_size, m]])
        J.setType("python")
        shell = self.RHSJacPShell(self)
        J.setPythonContext(shell)
        J.setUp()
        J.assemble()
        return J

    @cached_property
    def _dMdx(self):
        return function.Function(self._problem.F.arguments()[0].function_space())

    @cached_property
    def _dMdp(self):
        return self._dFdp.createVecRight()

    @cached_property
    def is_mixed(self):
        return self._jac.block_shape != (1, 1)

    @cached_property
    def _pjac(self):
        if self.mat_type != self.pmat_type or self._problem.Jp is not None:
            from firedrake.assemble import allocate_matrix

            return allocate_matrix(
                self.Jp,
                bcs=self.bcs_Jp,
                form_compiler_parameters=self.fcp,
                mat_type=self.pmat_type,
                appctx=self.appctx,
                options_prefix=self.options_prefix,
            )
        else:
            return self._jac

    @cached_property
    def _assemble_pjac(self):
        from firedrake.assemble import create_assembly_callable

        return create_assembly_callable(
            self.Jp,
            tensor=self._pjac,
            bcs=self.bcs_Jp,
            form_compiler_parameters=self.fcp,
            mat_type=self.pmat_type,
        )

    @cached_property
    def _F(self):
        return function.Function(self._problem.F.arguments()[0].function_space())
