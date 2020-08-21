from firedrake.adjoint.blocks import GenericSolveBlock, solve_init_params
from pyadjoint.tape import stop_annotating
import ufl

import firedrake_ts


class DAESolverBlock(GenericSolveBlock):
    def __init__(
        self,
        F,
        u,
        udot,
        tspan,
        dt,
        bcs,
        M,
        p,
        problem_J,
        solver_params,
        solver_kwargs,
        **kwargs
    ):
        self.problem_J = problem_J
        self.solver_params = solver_params.copy()
        self.solver_kwargs = solver_kwargs

        super().__init__(F, 0, u, bcs, **{**solver_kwargs, **kwargs})

        self.udot = udot
        self.M = M

        if self.problem_J is not None:
            for coeff in self.problem_J.coefficients():
                self.add_dependency(coeff, no_duplicates=True)

        # Each block creates and stores its own DAESolver to reuse it with
        # the recompute() and evaluate_adj() operations
        self.problem = firedrake_ts.DAEProblem(F, u, udot, tspan, dt, bcs=bcs, M=M, p=p)
        self.solver = firedrake_ts.DAESolver(self.problem, **self.solver_kwargs)
        self.solver.parameters.update(self.solver_params)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        pass

    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        input = adj_inputs[0]
        prepared = self._replace_recompute_form()

        self.problem.F = prepared[0]
        self.problem.u = prepared[2]
        self.problem.udot = prepared[3]
        self.problem.bcs_F = prepared[4]
        self.problem.M = prepared[5]

        self.solver.adjoint_solve()
        dJdu, dJdf = self.solver.get_cost_gradients()

        return input * dJdf

    def _replace_form(self, form, func=None, velfunc=None):
        """Replace the form coefficients with checkpointed values

        func represents the initial guess if relevant.
        """
        replace_map = self._replace_map(form)
        if func is not None and self.func in replace_map:
            self.backend.Function.assign(func, replace_map[self.func])
            replace_map[self.func] = func
        if velfunc is not None and self.udot in replace_map:
            self.backend.Function.assign(velfunc, replace_map[self.udot])
            replace_map[self.udot] = velfunc
        return ufl.replace(form, replace_map)

    def _replace_recompute_form(self):
        func = self._create_initial_guess()
        velfunc = self._create_initial_guess()

        bcs = self._recover_bcs()
        lhs = self._replace_form(self.lhs, func=func, velfunc=velfunc)
        rhs = 0
        if self.linear:
            rhs = self._replace_form(self.rhs)

        M = self._replace_form(self.M, func=func, velfunc=velfunc)

        return lhs, rhs, func, velfunc, bcs, M

    def recompute_component(self, inputs, block_variable, idx, prepared):
        from firedrake import Function

        prepared = self._replace_recompute_form()

        self.problem.F = prepared[0]
        self.problem.u = prepared[2]
        self.problem.udot = prepared[3]
        self.problem.bcs_F = prepared[4]
        self.problem.M = prepared[5]

        # Necessary to reset the problem as solver.solve() starts from the last time step used.
        self.solver.ts.setTimeStep(self.solver.dt)
        self.solver.ts.setTime(self.solver.tspan[0])
        self.solver.ts.setStepNumber(0)
        if self.solver._problem.M:
            self.solver.ts.getCostIntegral().getArray()[0] = 0.0

        return self.solver.solve()

