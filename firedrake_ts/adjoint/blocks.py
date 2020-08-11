from firedrake.adjoint.blocks import GenericSolveBlock, solve_init_params
from pyadjoint.tape import stop_annotating

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
        return self._replace_recompute_form()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        from firedrake import Function

        self.problem.F = prepared[0]
        self.problem.u = prepared[2]
        self.problem.bcs = prepared[3]

        for dep in self.problem.F.coefficients():
            print(dep.dat.data)

        return self.solver.solve()

