from pyadjoint.tape import (
    get_working_tape,
    stop_annotating,
    annotate_tape,
    no_annotations,
)
from pyadjoint.overloaded_type import create_overloaded_object

from firedrake_ts.adjoint.blocks import DAESolverBlock


class DAEProblemMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        def wrapper(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self._ad_F = self.F
            self._ad_u = self.u
            self._ad_udot = self.udot
            self._ad_tspan = self.tspan
            self._ad_dt = self.dt
            self._ad_bcs = self.bcs
            self._ad_J = self.J
            if self.M:
                self._ad_M = self.M
                self._ad_p = self.p
            else:
                self._ad_M = None
                self._ad_p = None

            self._ad_kwargs = {
                "Jp": self.Jp,
                "form_compiler_parameters": self.form_compiler_parameters,
                "is_linear": self.is_linear,
            }

        return wrapper


class DAESolverMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        def wrapper(self, problem, *args, **kwargs):
            init(self, problem, *args, **kwargs)
            self._ad_problem = problem
            self._ad_args = args
            self._ad_kwargs = kwargs

        return wrapper

    @staticmethod
    def _ad_annotate_solve(solve):
        def wrapper(self, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
            for the purposes of the adjoint computation (such as projecting fields to other function spaces
            for the purposes of visualisation)."""

            annotate = annotate_tape(kwargs)
            if annotate:
                tape = get_working_tape()
                problem = self._ad_problem
                sb_kwargs = DAESolverBlock.pop_kwargs(kwargs)
                sb_kwargs.update(kwargs)
                block = DAESolverBlock(
                    problem._ad_F,
                    problem._ad_u,
                    problem._ad_udot,
                    problem._ad_tspan,
                    problem._ad_dt,
                    problem._ad_bcs,
                    problem._ad_M,
                    problem._ad_p,
                    problem_J=problem._ad_J,
                    solver_params=self.parameters,
                    solver_kwargs=self._ad_kwargs,
                    **sb_kwargs
                )
                tape.add_block(block)

            with stop_annotating():
                out = solve(self, **kwargs)

            if annotate:
                if problem.M:
                    out = create_overloaded_object(out)
                    block.add_output(out.block_variable)
                else:
                    block.add_output(self._ad_problem._ad_u.create_block_variable())

            return out

        return wrapper
