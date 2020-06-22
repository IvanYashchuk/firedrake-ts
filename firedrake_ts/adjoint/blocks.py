from firedrake.adjoint.blocks import GenericSolveBlock, solve_init_params

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
        pass

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self._replace_form(self.problem.F)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        lhs = prepared[0]
        rhs = prepared[1]
        u = prepared[2]
        bcs = prepared[3]

        return self._forward_solve(lhs, rhs, u, bcs)

    def _forward_solve(self, lhs, rhs, u, bcs, **kwargs):
        J = self.problem_J
        if J is not None:
            J = self._replace_form(J, u)
        problem = DAEProblem(lhs, u, udot, tspan, dt, bcs, J=J)
        solver = DAESolver(problem, **self.solver_kwargs)
        solver.parameters.update(self.solver_params)
        solver.solve()
        return u

