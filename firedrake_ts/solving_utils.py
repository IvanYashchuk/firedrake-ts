from itertools import chain

import numpy

from pyop2 import op2
from firedrake_configuration import get_config
from firedrake import function, cofunction, dmhooks
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import PETSc
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.utils import cached_property
from firedrake.logging import warning


from firedrake.solving_utils import _make_reasons


TSReasons = _make_reasons(PETSc.TS.ConvergedReason())



def check_ts_convergence(ts):
    r = ts.getConvergedReason()
    # TODO: submit PR to petsc4py to add the following reasons
    # TSFORWARD_DIVERGED_LINEAR_SOLVE = -3,
    # TSADJOINT_DIVERGED_LINEAR_SOLVE = -4
    if r == -3:
        raise ConvergenceError(
            "TS solve failed to converge. Reason: TSFORWARD_DIVERGED_LINEAR_SOLVE"
        )
    if r == -4:
        raise ConvergenceError(
            "TS solve failed to converge. Reason: TSADJOINT_DIVERGED_LINEAR_SOLVE"
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
    :arg post_jacobian_callback: User-defined function called immediately
        after Jacobian assembly
    :arg pre_function_callback: User-defined function called immediately
        before residual assembly
    :arg post_function_callback: User-defined function called immediately
        after residual assembly
    :arg options_prefix: The options prefix of the TS.
    :arg project_rhs: If True the right-hand-side term is projected using a mass matrix solver.
    :arg rhs_projection_parameters: Solver parameters of the right-hand-side projection solver.
    :arg transfer_manager: Object that can transfer functions between
        levels, typically a :class:`~.TransferManager`

    The idea here is that the TS holds a shell DM which contains
    this object as "user context".  When the TS calls back to the
    user form_function code, we pull the DM out of the TS and then
    get the context (which is one of these objects) to find the
    Firedrake level information.
    """
    @PETSc.Log.EventDecorator()
    def __init__(self, problem, mat_type, pmat_type, appctx=None,
                 pre_jacobian_callback=None, pre_function_callback=None,
                 post_jacobian_callback=None, post_function_callback=None,
                 options_prefix=None,
                 transfer_manager=None,
                 project_rhs=True,
                 rhs_projection_parameters=None):
        from firedrake.assemble import get_assembler

        if pmat_type is None:
            pmat_type = mat_type
        self.mat_type = mat_type
        self.pmat_type = pmat_type
        self.options_prefix = options_prefix

        matfree = mat_type == 'matfree'
        pmatfree = pmat_type == 'matfree'

        self._problem = problem
        self._pre_jacobian_callback = pre_jacobian_callback
        self._pre_function_callback = pre_function_callback
        self._post_jacobian_callback = post_jacobian_callback
        self._post_function_callback = post_function_callback

        self.fcp = problem.form_compiler_parameters
        # Function to hold current guess
        self._x = problem.u
        # Function to hold time derivative
        self._xdot = problem.udot
        # Constant to hold time shift value
        self._shift = problem.shift
        # Constant to hold current time value
        self._time = problem.time

        if appctx is None:
            appctx = {}
        # A split context will already get the full state.
        # TODO, a better way of doing this.
        # Now we don't have a temporary state inside the snes
        # context we could just require the user to pass in the
        # full state on the outside.
        appctx.setdefault("state", self._x)
        appctx.setdefault("form_compiler_parameters", self.fcp)

        self.appctx = appctx
        self.matfree = matfree
        self.pmatfree = pmatfree
        self.F = problem.F
        self.J = problem.J
        self.G = problem.G
        self.dGdu = problem.dGdu

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

        self.bcs_F = tuple(bc.extract_form('F') for bc in problem.bcs)
        self.bcs_J = tuple(bc.extract_form('J') for bc in problem.bcs)
        self.bcs_Jp = tuple(bc.extract_form('Jp') for bc in problem.bcs)
        self.bcs_G = tuple(bc.extract_form('F') for bc in problem.bcs)
        self.bcs_dGdu = tuple(bc.extract_form('J') for bc in problem.bcs)

        self._assemble_residual = get_assembler(self.F, bcs=self.bcs_F,
                                                form_compiler_parameters=self.fcp,
                                                zero_bc_nodes=True).assemble
        
        
        if self.G is not None:
            self._assemble_rhs_residual = get_assembler(self.G, bcs=self.bcs_G,
                                                        form_compiler_parameters=self.fcp,
                                                        zero_bc_nodes=True).assemble
            self.rhs_projection_parameters = rhs_projection_parameters
            self.project_rhs = project_rhs
            self._G_or_projected_G = self._projected_G if self.project_rhs else self._G
            self._rhs_jacobian_assembled = False

        self._jacobian_assembled = False
        self._splits = {}
        self._coarse = None
        self._fine = None

        self._nullspace = None
        self._nullspace_T = None
        self._near_nullspace = None
        self._transfer_manager = transfer_manager

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
                (modname, objname) = managername.rsplit('.', 1)
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

    def set_ifunction(self, ts):
        r"""Set the function to compute F(t,U,U_t) where F() = 0 is the DAE to be solved."""
        with self._F.dat.vec_wo as v:
            ts.setIFunction(self.form_function, v)

    def set_ijacobian(self, ts):
        r"""Set the function to compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t) is the function provided with set_ifunction()"""
        ts.setIJacobian(self.form_jacobian, J=self._jac.petscmat, P=self._pjac.petscmat)

    def set_rhs_function(self, ts):
        r"""Set the function to compute G(t,u) where F() = G() is the equation to be solved."""
        if self.G is not None:
            with self._G_or_projected_G.dat.vec_wo as v:
                ts.setRHSFunction(self.form_rhs_function, v)

    def set_rhs_jacobian(self, ts):
        r"""Set the function to compute the Jacobian of G, where U_t = G(U,t), as well as the location to store the matrix."""
        if self.G is not None:
            ts.setRHSJacobian(self.form_rhs_jacobian, J=self._rhs_jac.petscmat, P=self._rhs_pjac.petscmat)

    def set_nullspace(self, nullspace, ises=None, transpose=False, near=False):
        if nullspace is None:
            return
        nullspace._apply(self._jac, transpose=transpose, near=near)
        if self.Jp is not None:
            nullspace._apply(self._pjac, transpose=transpose, near=near)
        if ises is not None:
            nullspace._apply(ises, transpose=transpose, near=near)

    @PETSc.Log.EventDecorator()
    def split(self, fields):
        from firedrake import replace, as_vector, split
        from firedrake_ts.ts_solver import DAEProblem as DAEP
        from firedrake.bcs import DirichletBC, EquationBC
        fields = tuple(tuple(f) for f in fields)
        splits = self._splits.get(tuple(fields))
        if splits is not None:
            return splits

        splits = []
        problem = self._problem
        splitter = ExtractSubBlock()
        for field in fields:
            F = splitter.split(problem.F, argument_indices=(field, ))
            J = splitter.split(problem.J, argument_indices=(field, field))
            us = problem.u.subfunctions
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
                val, = pieces
                subu = function.Function(V, val=val)
                subsplit = (subu, )
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
            if problem.G is not None:
                G = splitter.split(problem.G, argument_indices=(field,))
                G = replace(G, {problem.u: u})
            else:
                G = None
            bcs = []
            for bc in problem.bcs:
                if isinstance(bc, DirichletBC):
                    bc_temp = bc.reconstruct(field=field, V=V, g=bc.function_arg, sub_domain=bc.sub_domain)
                elif isinstance(bc, EquationBC):
                    bc_temp = bc.reconstruct(field, V, subu, u)
                if bc_temp is not None:
                    bcs.append(bc_temp)
            new_problem = DAEP(F, subu,
            		       problem.udot, problem.tspan,
                	       bcs=bcs, J=J, Jp=Jp,
                               form_compiler_parameters=problem.form_compiler_parameters,
                               G=G)
            new_problem._constant_jacobian = problem._constant_jacobian
            splits.append(type(self)(new_problem, mat_type=self.mat_type, pmat_type=self.pmat_type,
                                     appctx=self.appctx,
                                     transfer_manager=self.transfer_manager))
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
        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec_wo as v:
            X.copy(v)
        with ctx._xdot.dat.vec_wo as v:
            Xdot.copy(v)
        ctx._time.assign(t)

        if ctx._pre_function_callback is not None:
            ctx._pre_function_callback(X, Xdot)

        ctx._assemble_residual(tensor=ctx._F)

        if ctx._post_function_callback is not None:
            with ctx._F.dat.vec as F_:
                ctx._post_function_callback(X, Xdot, F_)

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

        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec_wo as v:
            X.copy(v)
        with ctx._xdot.dat.vec_wo as v:
            Xdot.copy(v)
        ctx._time.assign(t)

        if ctx._pre_jacobian_callback is not None:
            ctx._pre_jacobian_callback(X, Xdot)

        ctx._shift.assign(shift)
        ctx._assemble_jac(ctx._jac)

        if ctx._post_jacobian_callback is not None:
            ctx._post_jacobian_callback(X, Xdot, J)

        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac(ctx._pjac)

        ises = problem.J.arguments()[0].function_space()._ises
        ctx.set_nullspace(ctx._nullspace, ises, transpose=False, near=False)
        ctx.set_nullspace(ctx._nullspace_T, ises, transpose=True, near=False)
        ctx.set_nullspace(ctx._near_nullspace, ises, transpose=False, near=True)

    @staticmethod
    def form_rhs_function(ts, t, X, G):
        r"""Form the residual for this problem

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg G: function vector
        """
        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec_wo as v:
            X.copy(v)
        ctx._time.assign(t)

        # TODO: Add pre_rhs_function_callback

        ctx._assemble_projected_rhs_residual()

        # TODO: Add post_rhs_function_callback

        # G may not be the same vector as self._projected_G, so copy
        # residual out to G.
        with ctx._G_or_projected_G.dat.vec_ro as v:
            v.copy(G)

    @staticmethod
    def form_rhs_jacobian(ts, t, X, J, P):
        r"""Form the Jacobian for this problem

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        problem = ctx._problem

        assert J.handle == ctx._rhs_jac.petscmat.handle
        # TODO: Check how to use constant jacobian properly for TS
        if problem._constant_rhs_jacobian and ctx._rhs_jacobian_assembled:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        ctx._rhs_jacobian_assembled = True

        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec_wo as v:
            X.copy(v)
        ctx._time.assign(t)

        # TODO: Add pre_rhs_jacobian_callback

        ctx._assemble_rhs_jac(ctx._dGdu)

        # TODO: Add post_rhs_jacobian_callback

        ises = problem.dGdu.arguments()[0].function_space()._ises
        ctx.set_nullspace(ctx._nullspace, ises, transpose=False, near=False)
        ctx.set_nullspace(ctx._nullspace_T, ises, transpose=True, near=False)
        ctx.set_nullspace(ctx._near_nullspace, ises, transpose=False, near=True)

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
            manager = dmhooks.get_transfer_manager(fine._x.function_space().dm)
            manager.inject(fine._x, ctx._x)

            for bc in chain(*ctx._problem.bcs):
                if isinstance(bc, DirichletBC):
                    bc.apply(ctx._x)

        ctx._assemble_jac(ctx._jac)
        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac(ctx._pjac)

    @cached_property
    def _assembler_jac(self):
        from firedrake.assemble import get_assembler
        return get_assembler(self.J, bcs=self.bcs_J, form_compiler_parameters=self.fcp, mat_type=self.mat_type, options_prefix=self.options_prefix, appctx=self.appctx)

    @cached_property
    def _jac(self):
        return self._assembler_jac.allocate()

    @cached_property
    def _assemble_jac(self):
        return self._assembler_jac.assemble

    @cached_property
    def is_mixed(self):
        return self._jac.block_shape != (1, 1)

    @cached_property
    def _assembler_pjac(self):
        from firedrake.assemble import get_assembler
        if self.mat_type != self.pmat_type or self._problem.Jp is not None:
            return get_assembler(self.Jp, bcs=self.bcs_Jp, form_compiler_parameters=self.fcp, mat_type=self.pmat_type, options_prefix=self.options_prefix, appctx=self.appctx)
        else:
            return self._assembler_jac

    @cached_property
    def _pjac(self):
        if self.mat_type != self.pmat_type or self._problem.Jp is not None:
            return self._assembler_pjac.allocate()
        else:
            return self._jac

    @cached_property
    def _assemble_pjac(self):
        return self._assembler_pjac.assemble

    @cached_property
    def _F(self):
        return cofunction.Cofunction(self.F.arguments()[0].function_space().dual())

    @cached_property
    def _G(self):
        return cofunction.Cofunction(self.G.arguments()[0].function_space().dual())

    @cached_property
    def _projected_G(self):
        return function.Function(self.G.arguments()[0].function_space())

    @cached_property
    def _rhs_projection_solver(self):
        if self.G is not None:
            from firedrake import LinearSolver, assemble, ufl_expr

            mass_matrix = assemble(
                ufl_expr.derivative(self.F, self._xdot), bcs=self.bcs_F
            )

            prefix = self.options_prefix or ""
            prefix += "rhs_projection_solver"

            _rhs_projection_solver = LinearSolver(
                mass_matrix,
                solver_parameters=self.rhs_projection_parameters,
                options_prefix=prefix,
            )
            return _rhs_projection_solver
        else:
            return None

    def _assemble_projected_rhs_residual(self):
        """
        solve(mass_matrix_form == self.G, self._projected_G, self.bcs_G)
        """
        if self.G is not None:
            self._assemble_rhs_residual(self._G)
            if self.project_rhs:
                # TODO maybe the riesz_repre is the correct way?
                #assign(self._projected_G, self._G.riesz_representation())
                self._rhs_projection_solver.solve(self._projected_G, self._G)
            # else assembled rhs residual that is saved in self._G is used

    @cached_property
    def _rhs_jac(self):
        if self.G is not None:
	        return self._assembler_rhs_jac.allocate()
        else:
            return None

    @cached_property
    def _rhs_pjac(self):
        return self._rhs_jac

    @cached_property
    def _assemble_rhs_jac(self):
        if self.G is not None:
	        return self._assembler_rhs_jac.assemble
        else:
            return None

    @cached_property
    def _assemble_rhs_pjac(self):
        return self._assemble_rhs_jac

    @cached_property
    def _assembler_rhs_jac(self):
        from firedrake.assemble import get_assembler
        return get_assembler(self.dGdu, bcs=self.bcs_dGdu, form_compiler_parameters=self.fcp, mat_type=self.mat_type, options_prefix=self.options_prefix, appctx=self.appctx)
