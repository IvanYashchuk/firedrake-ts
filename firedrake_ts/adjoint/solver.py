import copy
from pyadjoint.tape import (
    get_working_tape,
    stop_annotating,
    annotate_tape,
    no_annotations,
)
from functools import wraps
from pyadjoint.overloaded_type import create_overloaded_object
from ufl import replace

from firedrake_ts.adjoint.blocks import DAESolverBlock
from firedrake import DirichletBC


class DAEProblemMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self._ad_F = self.F
            self._ad_u = self.u
            self._ad_udot = self.udot
            self._ad_tspan = self.tspan
            self._ad_bcs = self.bcs
            self._ad_J = self.J
            self._ad_M = self.M or None
            self._ad_kwargs = {
                "Jp": self.Jp,
                "form_compiler_parameters": self.form_compiler_parameters,
                "is_linear": self.is_linear,
            }

        return wrapper

    def _ad_count_map_update(self, updated_ad_count_map):
        self._ad_count_map = updated_ad_count_map


class DAESolverMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, problem, *args, **kwargs):
            init(self, problem, *args, **kwargs)
            self._ad_problem = problem
            self._ad_args = args
            self._ad_kwargs = kwargs
            self._ad_tsvs = None

        return wrapper

    @staticmethod
    def _ad_annotate_solve(solve):
        @wraps(solve)
        def wrapper(self, *args, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
            for the purposes of the adjoint computation (such as projecting fields to other function spaces
            for the purposes of visualisation)."""

            annotate = annotate_tape(kwargs)
            problem = self._problem
            if annotate:
                u0 = args[0]
                tape = get_working_tape()
                problem = self._ad_problem
                sb_kwargs = DAESolverBlock.pop_kwargs(kwargs)
                sb_kwargs.update(kwargs)
                block = DAESolverBlock(
                    problem._ad_F,
                    problem._ad_u,
                    problem._ad_udot,
                    problem._ad_tspan,
                    problem._ad_bcs,
                    problem._ad_M,
                    u0=u0,
                    problem_J=problem._ad_J,
                    solver_params=self.parameters,
                    solver_kwargs=self._ad_kwargs,
                    **sb_kwargs,
                )

                if not self._ad_tsvs:
                    from firedrake_ts import DAESolver

                    self._ad_tsvs = DAESolver(
                        self._ad_problem_clone(
                            self._ad_problem, block.get_dependencies()
                        ),
                        **self._ad_kwargs,
                    )
                    # Attach block to context to access them from TSAdjoint
                    self._ad_tsvs._ctx.block = block

                block._ad_tsvs = self._ad_tsvs
                block.u0 = u0
                tape.add_block(block)

            with stop_annotating():
                if problem.M:
                    out_u, out_m = solve(self, *args, **kwargs)
                else:
                    out = solve(self, *args, **kwargs)

            if annotate:
                if problem.M:
                    out_m = create_overloaded_object(out_m)
                    block.add_output(out_m.block_variable)
                block.add_output(self._ad_problem._ad_u.create_block_variable())
            return (out_u, out_m) if problem.M else out

        return wrapper

    @no_annotations
    def _ad_problem_clone(self, problem, dependencies):
        """Replaces every coefficient in the residual and jacobian with a deepcopy to return
        a clone of the original NonlinearVariationalProblem instance. We'll be modifying the
        numerical values of the coefficients in the residual and jacobian, so in order not to
        affect the user-defined self._ad_problem.F, self._ad_problem.J and self._ad_problem.u
        expressions, we'll instead create clones of them.
        """
        from firedrake_ts import DAEProblem
        from firedrake import Constant

        F_replace_map = {}
        J_replace_map = {}
        M_replace_map = {}

        F_coefficients = problem.F.coefficients()
        if problem.J:
            J_coefficients = problem.J.coefficients()
        # TODO is there a better way to avoid checking for this property everywhere?
        if hasattr(problem, "M") and problem.M:
            M_coefficients = problem.M.coefficients()

        _ad_count_map = {}
        for block_variable in dependencies:
            coeff = block_variable.output
            if coeff in F_coefficients and coeff not in F_replace_map:
                if isinstance(coeff, Constant):
                    F_replace_map[coeff] = copy.deepcopy(coeff)
                else:
                    F_replace_map[coeff] = coeff.copy(deepcopy=True)
                _ad_count_map[F_replace_map[coeff]] = coeff.count()

            if problem.J and coeff in J_coefficients and coeff not in J_replace_map:
                if coeff in F_replace_map:
                    J_replace_map[coeff] = F_replace_map[coeff]
                elif isinstance(coeff, Constant):
                    J_replace_map[coeff] = copy.deepcopy(coeff)
                else:
                    J_replace_map[coeff] = coeff.copy()
                _ad_count_map[J_replace_map[coeff]] = coeff.count()

            # TODO is there a better way to avoid checking for this property everywhere?
            if (
                hasattr(problem, "M")
                and problem.M
                and coeff in M_coefficients
                and coeff not in M_replace_map
            ):
                if coeff in F_replace_map:
                    M_replace_map[coeff] = F_replace_map[coeff]
                elif isinstance(coeff, Constant):
                    M_replace_map[coeff] = copy.deepcopy(coeff)
                else:
                    M_replace_map[coeff] = coeff.copy()
                _ad_count_map[M_replace_map[coeff]] = coeff.count()

        # TODO is there a better way to avoid checking for this property everywhere?
        if hasattr(problem, "M") and problem.M:
            M_rep = replace(problem.M, M_replace_map)
        else:
            M_rep = None
        J_repl = replace(problem.J, J_replace_map) if problem.J else None
        tsvp = DAEProblem(
            replace(problem.F, F_replace_map),
            F_replace_map[problem.u],
            F_replace_map[problem.udot],
            problem._ad_tspan,
            bcs=problem.bcs,
            M=M_rep,
            J=J_repl,
        )
        tsvp._ad_count_map_update(_ad_count_map)

        return tsvp
