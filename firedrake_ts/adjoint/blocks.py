import copy
from firedrake.adjoint.blocks import GenericSolveBlock, solve_init_params
from firedrake import Constant, DirichletBC, function, vector
from firedrake.mesh import MeshGeometry
from numpy.lib.arraysetops import isin
from pyadjoint.tape import stop_annotating
import ufl

import firedrake_ts


class DAESolverBlock(GenericSolveBlock):
    def __init__(
        self, F, u, udot, tspan, dt, bcs, M, u0, solver_params, solver_kwargs, **kwargs
    ):
        self.solver_params = solver_params.copy()
        self.solver_kwargs = solver_kwargs

        super().__init__(F, 0, u, bcs, **{**solver_kwargs, **kwargs})

        self.udot = udot
        self.M = M

        if self.M is not None:
            for coeff in self.M.coefficients():
                self.add_dependency(coeff, no_duplicates=True)

        self.add_dependency(u0, no_duplicates=True)
        # Keep a reference to the initial condition to avoid
        # generating its jacobians and to assign it the proper
        # gradient
        self.u0 = u0

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)

    def evaluate_adj(self, markings):
        outputs = self.get_outputs()
        adj_inputs = []
        has_input = False
        for output in outputs:
            adj_inputs.append(output.adj_value)
            if output.adj_value is not None:
                has_input = True

        if not has_input:
            return

        deps = self.get_dependencies()
        inputs = [bv.saved_output for bv in deps]
        relevant_dependencies = [
            (i, bv) for i, bv in enumerate(deps) if bv.marked_in_path or not markings
        ]

        if len(relevant_dependencies) <= 0:
            return

        input = adj_inputs[0] or adj_inputs[1]

        problem = self._ad_tsvs._problem

        # TODO clean and refactor the next ~30 lines
        func = self.backend.Function(problem.u.function_space())
        velfunc = self.backend.Function(problem.u.function_space())
        assign_map_F = self._ad_create_assign_map(problem.F, func, velfunc)
        # assign_map_J = self._ad_create_assign_map(problem.J)

        problem.F = ufl.replace(problem.F, assign_map_F)
        # problem.J = ufl.replace(problem.J, assign_map_J)

        # TODO is there a better way to avoid checking for this property everywhere?
        if hasattr(problem, "M") and problem.M:
            assign_map_M = self._ad_create_assign_map(problem.M, func, velfunc)
            problem.M = ufl.replace(problem.M, assign_map_M)

        if problem.M and isinstance(input, float):
            self._ad_tsvs.set_adjoint_jacobians(self._ad_tsvs._ctx)
        elif isinstance(input, vector.Vector) and problem.M:
            self._ad_tsvs.set_adjoint_jacobians(self._ad_tsvs._ctx, zero=True)

        problem.u = assign_map_F[problem.u]
        problem.udot = assign_map_F[problem.udot]

        self._ad_tsvs.adjoint_solve(adj_input=input)

        revs_assign_map_F = {v: k for k, v in assign_map_F.items()}
        # revs_assign_map_J = {v: k for k, v in assign_map_J.items()}

        problem.F = ufl.replace(problem.F, revs_assign_map_F)
        # problem.J = ufl.replace(problem.J, revs_assign_map_J)

        # TODO is there a better way to avoid checking for this property everywhere?
        if hasattr(problem, "M") and problem.M:
            revs_assign_map_M = {v: k for k, v in assign_map_M.items()}
            problem.M = ufl.replace(problem.M, revs_assign_map_M)

        problem.u = revs_assign_map_F[problem.u]
        problem.udot = revs_assign_map_F[problem.udot]

        dJdu, dJdf = self._ad_tsvs.get_cost_gradients()
        y_ownership = dJdf.getOwnershipRange()

        local_shift = 0
        for idx, dep in relevant_dependencies:
            # TODO this is working for now because self.u0 is the
            # same than the state response
            if dep.output == self.u0:
                if isinstance(input, float):
                    dep.add_adj_output(input * dJdu)
                else:
                    dep.add_adj_output(dJdu)
            else:
                c_rep = dep.saved_output
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
                    mesh = self._ad_tsvs._ctx._problem.F.ufl_domain()
                    tmp = function.Function(c_rep._ad_function_space(mesh))
                else:
                    tmp = function.Function(c_rep.function_space())

                with tmp.dat.vec as y_vec:
                    local_range = y_vec.getOwnershipRange()
                    local_size = local_range[1] - local_range[0]
                    y_vec[:] = dJdf[
                        (y_ownership[0] + local_shift) : (
                            y_ownership[0] + local_shift + local_size
                        )
                    ]
                local_shift += local_size
                if tmp is not None:
                    # Whether the output is the solution or the integral
                    if isinstance(input, float):
                        dep.add_adj_output(input * tmp)
                    else:
                        dep.add_adj_output(tmp)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        pass

    def recompute_component(self, inputs, block_variable, idx, prepared):
        from firedrake import Function

        self._ad_tsvs_replace_forms()
        self._ad_tsvs.parameters.update(self.solver_params)
        u = self._ad_tsvs._problem.u
        # Pyadjoint uses the output as the new checkpoint.
        # We're providing a new one. TODO does it make sense?
        # It should not, since you're not recording
        u_func = u.copy(deepcopy=True)

        if self._ad_tsvs._problem.M:
            u_func, m = self._ad_tsvs.solve(u_func)
            if isinstance(block_variable.output, float):
                return m
            else:
                return u_func
        else:
            u_func = self._ad_tsvs.solve(u_func)
            return u_func

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

    def _ad_create_assign_map(self, form, func=None, velfunc=None):
        assign_map = self._ad_assign_map(form)
        if func is not None and self._ad_tsvs._problem.u in assign_map:
            self.backend.Function.assign(func, assign_map[self._ad_tsvs._problem.u])
            assign_map[self._ad_tsvs._problem.u] = func

        if velfunc is not None and self._ad_tsvs._problem.udot in assign_map:
            self.backend.Function.assign(
                velfunc, assign_map[self._ad_tsvs._problem.udot]
            )
            assign_map[self._ad_tsvs._problem.udot] = velfunc

        return assign_map

    def _ad_assign_coefficients(self, form, func=None, velfunc=None):
        assign_map = self._ad_create_assign_map(form, func, velfunc)
        for coeff, value in assign_map.items():
            coeff.assign(value)

    def _ad_tsvs_replace_forms(self):
        problem = self._ad_tsvs._problem
        func = self.backend.Function(problem.u.function_space())
        velfunc = self.backend.Function(problem.u.function_space())
        self._ad_assign_coefficients(problem.F, func, velfunc)
        # TODO is there a better way to avoid checking for this property everywhere?
        if hasattr(problem, "M") and problem.M:
            self._ad_assign_coefficients(problem.M, func, velfunc)
        # self._ad_assign_coefficients(problem.J)
