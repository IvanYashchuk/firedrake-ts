import ufl
from itertools import chain
from contextlib import ExitStack

from firedrake import dmhooks, slate, solving, solving_utils, ufl_expr, utils
from firedrake import function
from firedrake.petsc import PETSc, OptionsManager, flatten_parameters
from firedrake.bcs import DirichletBC

from firedrake_ts.solving_utils import check_ts_convergence, _TSContext


def check_pde_args(F, G, J, Jp):
    if not isinstance(F, (ufl.BaseForm, slate.TensorBase)):
        raise TypeError("Provided residual is a '%s', not a BaseForm or Slate Tensor" % type(F).__name__)
    if len(F.arguments()) != 1:
        raise ValueError("Provided residual is not a linear form")
    if G is not None and not isinstance(G, (ufl.BaseForm, slate.TensorBase)):
        raise TypeError(f"Provided G residual is a '{type(G).__name__}', not a BaseForm or Slate Tensor")
    if G is not None and len(G.arguments()) != 1:
        raise ValueError("Provided G residual is not a linear form")
    if not isinstance(J, (ufl.BaseForm, slate.TensorBase)):
        raise TypeError("Provided Jacobian is a '%s', not a BaseForm or Slate Tensor" % type(J).__name__)
    if len(J.arguments()) != 2:
        raise ValueError("Provided Jacobian is not a bilinear form")
    if Jp is not None and not isinstance(Jp, (ufl.BaseForm, slate.TensorBase)):
        raise TypeError("Provided preconditioner is a '%s', not a BaseForm or Slate Tensor" % type(Jp).__name__)
    if Jp is not None and len(Jp.arguments()) != 2:
        raise ValueError("Provided preconditioner is not a bilinear form")


def is_form_consistent(is_linear, bcs):
    # Check form style consistency
    if not (is_linear == all(bc.is_linear for bc in bcs if not isinstance(bc, DirichletBC))
            or not is_linear == all(not bc.is_linear for bc in bcs if not isinstance(bc, DirichletBC))):
        raise TypeError("Form style mismatch: some forms are given in 'F == 0' style, but others are given in 'A == b' style.")


class DAEProblem(object):
    r"""Nonlinear variational problem in DAE form F(u̇, u, t) = G(u, t)."""

    def __init__(
        self,
        F,
        u,
        udot,
        tspan,
        time=None,
        bcs=None,
        J=None,
        Jp=None,
        form_compiler_parameters=None,
        is_linear=False,
        G=None,
    ):
        r"""
        :param F: the nonlinear form
        :param u: the :class:`.Function` to solve for
        :param udot: the :class:`.Function` for time derivative
        :param tspan: the tuple for start time and end time
        :param time: the :class:`.Constant` for time-dependent weak forms
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = sigma*dF/du̇ + dF/du (optional)
        :param Jp: a form used for preconditioning the linear system,
                 optional, if not supplied then the Jacobian itself
                 will be used.
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        :is_linear: internally used to check if all domain/bc forms
            are given either in 'A == b' style or in 'F == 0' style.
        :param G: G(t, u) term that will be treated explicitly
            when using an IMEX method for solving F(u̇, u, t) = G(u, t).
            If G is `None` the G(u, t) term in the equation is considered to be equal to zero.
        """
        from firedrake import solving
        from firedrake import function, Constant

        self.bcs = solving._extract_bcs(bcs)
        # Check form style consistency
        self.is_linear = is_linear
        is_form_consistent(self.is_linear, self.bcs)
        self.Jp_eq_J = Jp is None

        self.u = u
        self.udot = udot
        self.tspan = tspan
        self.F = F
        self.G = G
        self.Jp = Jp
        if not isinstance(self.u, function.Function):
            raise TypeError(
                "Provided solution is a '%s', not a Function" % type(self.u).__name__
            )
        if not isinstance(self.udot, function.Function):
            raise TypeError(
                "Provided time derivative is a '%s', not a Function"
                % type(self.udot).__name__
            )

        # current value of time that may be used in weak form
        self.time = time or Constant(0.0)
        # timeshift value provided by the solver
        self.shift = Constant(1.0)

        # Use the user-provided Jacobian. If none is provided, derive
        # the Jacobian from the residual.
        self.J = self.shift * ufl_expr.derivative(F, udot) + (J or ufl_expr.derivative(F, u))

        # Derive the Jacobian for the G residual
        self.dGdu = ufl_expr.derivative(G, u) if G is not None else None

        # Argument checking
        check_pde_args(self.F, self.G, self.J, self.Jp)

        # Store form compiler parameters
        self.form_compiler_parameters = form_compiler_parameters
        self._constant_jacobian = False
        self._constant_rhs_jacobian = False

    def dirichlet_bcs(self):
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    @utils.cached_property
    def dm(self):
        return self.u.function_space().dm


class DAESolver(OptionsManager):
    r"""Solves a :class:`DAEProblem`."""

    def __init__(self, problem, **kwargs):
        r"""
        :arg problem: A :class:`DAEProblem` to solve.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        :kwarg transpose_nullspace: as for the nullspace, but used to
               make the right hand side consistent.
        :kwarg near_nullspace: as for the nullspace, but used to
               specify the near nullspace (for multigrid solvers).
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
               This should be a dict mapping PETSc options to values.
        :kwarg appctx: A dictionary containing application context that
               is passed to the preconditioner if matrix-free.
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.
        :kwarg pre_jacobian_callback: A user-defined function that will
               be called immediately before Jacobian assembly. This can
               be used, for example, to update a coefficient function
               that has a complicated dependence on the unknown solution.
        :kwarg pre_function_callback: As above, but called immediately
               before residual assembly.
        :kwarg post_jacobian_callback: As above, but called after the
               Jacobian has been assembled.
        :kwarg post_function_callback: As above, but called immediately
               after residual assembly.
        :kwarg monitor_callback: A user-defined function that will
               be used at every timestep to display the iteration's progress.
        Example usage of the ``solver_parameters`` option: to set the
        nonlinear solver type to just use a linear solver, use
        .. code-block:: python
            {'snes_type': 'ksponly'}
        PETSc flag options (where the presence of the option means something) should
        be specified with ``None``.
        For example:
        .. code-block:: python
            {'snes_monitor': None}
        To use the ``pre_jacobian_callback`` or ``pre_function_callback``
        functionality, the user-defined function must accept the current
        solution and the current time derivative as a petsc4py Vec. Example usage is given below:
        .. code-block:: python
            def update_diffusivity(current_solution, current_time_derivative):
                with cursol.dat.vec_wo as v:
                    current_solution.copy(v)
                solve(trial*test*dx == dot(grad(cursol), grad(test))*dx, diffusivity)
            solver = DAESolver(problem, pre_jacobian_callback=update_diffusivity)
        """
        assert isinstance(problem, DAEProblem)

        parameters = kwargs.get("solver_parameters")
        if "parameters" in kwargs:
            raise TypeError("Use solver_parameters, not parameters")
        nullspace = kwargs.get("nullspace")
        nullspace_T = kwargs.get("transpose_nullspace")
        near_nullspace = kwargs.get("near_nullspace")
        options_prefix = kwargs.get("options_prefix")
        pre_j_callback = kwargs.get("pre_jacobian_callback")
        pre_f_callback = kwargs.get("pre_function_callback")
        post_j_callback = kwargs.get("post_jacobian_callback")
        post_f_callback = kwargs.get("post_function_callback")
        monitor_callback = kwargs.get("monitor_callback")

        self.nullspace = nullspace
        self.near_nullspace = near_nullspace
        self.nullspace_T = nullspace_T

        super(DAESolver, self).__init__(parameters, options_prefix)

        # Allow anything, interpret "matfree" as matrix_free.
        mat_type = self.parameters.get("mat_type")
        pmat_type = self.parameters.get("pmat_type")
        matfree = mat_type == "matfree"
        pmatfree = pmat_type == "matfree"

        appctx = kwargs.get("appctx")

        ctx = _TSContext(
            problem,
            mat_type=mat_type,
            pmat_type=pmat_type,
            appctx=appctx,
            pre_jacobian_callback=pre_j_callback,
            pre_function_callback=pre_f_callback,
            post_jacobian_callback=post_j_callback,
            post_function_callback=post_f_callback,
            options_prefix=self.options_prefix,
        )

        # No preconditioner by default for matrix-free
        if (problem.Jp is not None and pmatfree) or matfree:
            self.set_default_parameter("pc_type", "none")
        elif ctx.is_mixed:
            # Mixed problem, use jacobi pc if user has not supplied
            # one.
            self.set_default_parameter("pc_type", "jacobi")

        self.ts = PETSc.TS().create(comm=problem.dm.comm)
        self.snes = self.ts.getSNES()

        self._problem = problem

        self._ctx = ctx
        self._work = problem.u.dof_dset.layout_vec.duplicate()

        self.ts.setDM(problem.dm)
        self.ts.setMonitor(monitor_callback)

        self.ts.setTime(problem.tspan[0])
        self.ts.setMaxTime(problem.tspan[1])

        if problem.G is None:
            # If G is not provided set the equation type as implicit
            # leave a default type otherwise
            self.ts.setEquationType(PETSc.TS.EquationType.IMPLICIT)
        else:
            # If G is provided use the arkimex solver
            self.set_default_parameter("ts_type", "arkimex")

        self.set_default_parameter("ts_exact_final_time", "stepover")
        # allow a certain number of failures (step will be rejected and retried)
        #self.set_default_parameter("ts_max_snes_failures", 5)

        self._set_problem_eval_funcs(
            ctx, problem, nullspace, nullspace_T, near_nullspace
        )

        # Set from options now. We need the
        # DM with an app context in place so that if the DM is active
        # on a subKSP the context is available.
        dm = self.ts.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx, save=False):
            self.set_from_options(self.ts)

        # Used for custom grid transfer.
        self._transfer_operators = ()
        self._setup = False

    def _set_problem_eval_funcs(
        self, ctx, problem, nullspace, nullspace_T, near_nullspace
    ):
        r"""
        :arg ctx: A :class:`_TSContext` that contains the residual evaluations
        :arg problem: A :class:`DAEProblem` to solve.
        :arg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        :arg nullspace_T: as for the nullspace, but used to
               make the right hand side consistent.
        :arg near_nullspace: as for the nullspace, but used to
               specify the near nullspace (for multigrid solvers).
        """
        ctx.set_ifunction(self.ts)
        ctx.set_ijacobian(self.ts)
        ctx.set_rhs_function(self.ts)
        ctx.set_rhs_jacobian(self.ts)
        ctx.set_nullspace(
            nullspace,
            problem.J.arguments()[0].function_space()._ises,
            transpose=False,
            near=False,
        )
        ctx.set_nullspace(
            nullspace_T,
            problem.J.arguments()[1].function_space()._ises,
            transpose=True,
            near=False,
        )
        ctx.set_nullspace(
            near_nullspace,
            problem.J.arguments()[0].function_space()._ises,
            transpose=False,
            near=True,
        )
        ctx._nullspace = nullspace
        ctx._nullspace_T = nullspace_T
        ctx._near_nullspace = near_nullspace

    def set_transfer_manager(self, manager):
        r"""Set the object that manages transfer between grid levels.
        Typically a :class:`~.TransferManager` object.
        :arg manager: Transfer manager, should conform to the
            TransferManager interface.
        :raises ValueError: if called after the transfer manager is setup.
        """
        self._ctx.transfer_manager = manager

    def solve(self, bounds=None):
        r"""Solve the time-dependent variational problem.
        :arg bounds: Optional bounds on the solution (lower, upper).
            ``lower`` and ``upper`` must both be
            :class:`~.Function`\s. or :class:`~.Vector`\s.
        .. note::
           If bounds are provided the ``snes_type`` must be set to
           ``vinewtonssls`` or ``vinewtonrsls``.
        """

        self._set_problem_eval_funcs(
            self._ctx,
            self._problem,
            self.nullspace,
            self.nullspace_T,
            self.near_nullspace,
        )
        # Make sure appcontext is attached to the DM before we solve.
        dm = self.ts.getDM()
        for dbc in self._problem.dirichlet_bcs():
            dbc.apply(self._problem.u)

        if bounds is not None:
            lower, upper = bounds
            with lower.dat.vec_ro as lb, upper.dat.vec_ro as ub:
                self.snes.setVariableBounds(lb, ub)

        work = self._work
        with self._problem.u.dat.vec as u:
            u.copy(work)
            with ExitStack() as stack:
                # Ensure options database has full set of options (so monitors
                # work right)
                for ctx in chain(
                        (
                            self.inserted_options(),
                            dmhooks.add_hooks(dm, self, appctx=self._ctx),
                        ),
                        self._transfer_operators,
                ):
                    stack.enter_context(ctx)
                self.ts.solve(work)
            work.copy(u)
        self._setup = True
        check_ts_convergence(self.ts)

    def adjoint_solve(self):
        r"""Solve the adjoint problem.
        """
        self._set_problem_eval_funcs(
            self._ctx,
            self._problem,
            self.nullspace,
            self.nullspace_T,
            self.near_nullspace,
        )
        # Make sure appcontext is attached to the DM before the adjoint solve.
        dm = self.ts.getDM()

        with ExitStack() as stack:
            for ctx in chain(
                (
                    self.inserted_options(),
                    dmhooks.add_hooks(dm, self, appctx=self._ctx),
                ),
                self._transfer_operators,
            ):
                stack.enter_context(ctx)
            self.ts.adjointSolve()

        check_ts_convergence(self.ts)
