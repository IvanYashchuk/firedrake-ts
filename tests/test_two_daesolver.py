import pytest
from firedrake import *
import firedrake_ts


def test_two_ts():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - 1.0 * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)


    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0), bcs=bc)
    solver = firedrake_ts.DAESolver(problem)

    V2 = FunctionSpace(mesh, "P", 1)
    u2 = Function(V2)
    u_t2 = Function(V2)
    v = TestFunction(V2)
    F2 = inner(u_t2, v) * dx + inner(grad(u2), grad(v)) * dx - 1.0 * v * dx
    bc2 = DirichletBC(V2, 0.0, "on_boundary")


    problem2 = firedrake_ts.DAEProblem(F2, u2, u_t2, (0.0, 1.0), bcs=bc2)
    solver2 = firedrake_ts.DAESolver(problem2)

    solver.solve()
