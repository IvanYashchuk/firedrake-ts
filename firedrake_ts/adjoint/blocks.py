import copy
from firedrake.adjoint.blocks import GenericSolveBlock, solve_init_params
from firedrake import Constant
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

        if self.M is not None:
            for coeff in self.M.coefficients():
                self.add_dependency(coeff, no_duplicates=True)

        # Each block creates and stores its own DAESolver to reuse it with
        # the recompute() and evaluate_adj() operations
        self.problem = firedrake_ts.DAEProblem(F, u, udot, tspan, dt, bcs=bcs, M=M, p=p)
        self.pfunc = p
        if isinstance(p, Constant):
            self.pfunc_copy = copy.deepcopy(p)
        else:
            self.pfunc_copy = p.copy(deepcopy=True)

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
        self.problem.p = prepared[6]

        self.solver.adjoint_solve()
        dJdu, dJdf = self.solver.get_cost_gradients()

        self._ad_tsvs.adjoint_solve()
        self._ad_tsvs.get_cost_gradients()
        return input * dJdf

    def prepare_recompute_component(self, inputs, relevant_outputs):
        pass

    def _replace_map(self, form):
        replace_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in form.coefficients():
                replace_coeffs[coeff] = block_variable.saved_output
        return replace_coeffs

    def _replace_form(self, form, func=None, velfunc=None, pfunc=None):
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

        if pfunc is not None and self.pfunc in replace_map:
            self.backend.Function.assign(pfunc, replace_map[self.pfunc])
            replace_map[self.pfunc] = pfunc

        return ufl.replace(form, replace_map)

    def _replace_recompute_form(self):
        func = self._create_initial_guess()
        velfunc = self._create_initial_guess()
        pfunc = self.pfunc_copy

        bcs = self._recover_bcs()
        lhs = self._replace_form(self.lhs, func=func, velfunc=velfunc, pfunc=pfunc)
        rhs = 0
        if self.linear:
            rhs = self._replace_form(self.rhs)

        M = self._replace_form(self.M, func=func, velfunc=velfunc, pfunc=pfunc)

        return lhs, rhs, func, velfunc, bcs, M, pfunc

    def recompute_component(self, inputs, block_variable, idx, prepared):
        from firedrake import Function

        prepared = self._replace_recompute_form()

        self.problem.F = prepared[0]
        self.problem.u = prepared[2]
        self.problem.udot = prepared[3]
        self.problem.bcs_F = prepared[4]
        self.problem.M = prepared[5]
        self.problem.p = prepared[6]

        self._ad_tsvs_replace_forms()
        self._ad_tsvs.parameters.update(self.solver_params)
        self._ad_tsvs.solve(self._ad_tsvs._problem.u)

        return self.solver.solve(self.problem.u)

    def _ad_assign_map(self, form):
        count_map = self._ad_tsvs._problem._ad_count_map
        assign_map = {}
        form_ad_count_map = dict(
            (count_map[coeff], coeff) for coeff in form.coefficients()
        )
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if isinstance(coeff, (self.backend.Coefficient, self.backend.Constant)):
                coeff_count = coeff.count()
                if coeff_count in form_ad_count_map:
                    assign_map[
                        form_ad_count_map[coeff_count]
                    ] = block_variable.saved_output
        return assign_map

    def _ad_assign_coefficients(self, form, func=None, velfunc=None):
        assign_map = self._ad_assign_map(form)
        if func is not None and self._ad_tsvs._problem.u in assign_map:
            self.backend.Function.assign(func, assign_map[self._ad_tsvs._problem.u])
            assign_map[self._ad_tsvs._problem.u] = func

        if velfunc is not None and self._ad_tsvs._problem.udot in assign_map:
            self.backend.Function.assign(
                velfunc, assign_map[self._ad_tsvs._problem.udot]
            )
            assign_map[self._ad_tsvs._problem.udot] = velfunc

        for coeff, value in assign_map.items():
            coeff.assign(value)

    def _ad_tsvs_replace_forms(self):
        problem = self._ad_tsvs._problem
        func = self.backend.Function(problem.u.function_space())
        velfunc = self.backend.Function(problem.u.function_space())
        self._ad_assign_coefficients(problem.F, func, velfunc)
        self._ad_assign_coefficients(problem.M, func, velfunc)
        self._ad_assign_coefficients(problem.J)
