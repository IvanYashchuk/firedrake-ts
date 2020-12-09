import pytest
from firedrake import *
from firedrake_adjoint import *
import petsc4py
import firedrake_ts
import numpy as np

petsc4py.PETSc.Sys.popErrorHandler()


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_cost_function_recompute(control):

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    if control == "constant":
        f = Constant(5.0)
    elif control == "function":
        f = Function(V).interpolate(Constant(5.0))
    else:
        f = Function(V).interpolate(Constant(5.0))
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)
    M = u * u * dx

    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_type": "theta",
        "ts_theta_theta": 0.5,
        "ts_exact_final_time": "matchstep",
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u, m = solver.solve(u)

    c = Control(f)
    Jhat = ReducedFunctional(m, c)
    print(f"Second m: {Jhat(f)}")
    assert np.allclose(m, Jhat(f), 1e-8)


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_cost_function_adjoint(control):

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    if control == "constant":
        f = Constant(5.0)
        h = Constant(1.0)
    elif control == "function":
        f = Function(V).interpolate(Constant(5.0))
        h = Function(V).interpolate(Constant(1.0))
    else:
        f = Function(V).interpolate(Constant(5.0))
        h = Function(V).interpolate(Constant(1.0))

    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)
    M = u * u * dx

    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_type": "theta",
        "ts_theta_theta": 0.5,
        "ts_exact_final_time": "matchstep",
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u, m = solver.solve(u)
    print(f"First m: {m}")

    c = Control(f)
    Jhat = ReducedFunctional(m, c)
    print(f"Second m: {Jhat(f)}")
    djdf_adjoint = Jhat.derivative()
    print(f"derivative {djdf_adjoint.dat.data}")

    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_control_in_cost_function_adjoint(control):

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)

    if control == "constant":
        f = Constant(5.0)
        h = Constant(1.0)
    elif control == "function":
        f = Function(V).interpolate(Constant(5.0))
        h = Function(V).interpolate(Constant(1.0))
    else:
        f = Function(V).interpolate(Constant(5.0))
        h = Function(V).interpolate(Constant(1.0))

    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)
    M = u * u * f * f * dx

    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_type": "theta",
        "ts_theta_theta": 0.5,
        "ts_exact_final_time": "matchstep",
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u, m = solver.solve(u)
    print(f"First m: {m}")

    c = Control(f)
    Jhat = ReducedFunctional(m, c)
    print(f"Second m: {Jhat(f)}")
    djdf_adjoint = Jhat.derivative()
    print(f"derivative {djdf_adjoint.dat.data}")

    assert taylor_test(Jhat, f, h) > 1.9


def test_terminal_cost_function_adjoint():

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    f = Function(V).interpolate(Constant(5.0))
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)
    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_theta_theta": 0.5,
        "ts_theta_endpoint": None,
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u = solver.solve(u)

    J = assemble(u * u * dx)

    c = Control(f)
    Jhat = ReducedFunctional(J, c)
    print(f"Second m: {Jhat(f)}")
    djdf_adjoint = Jhat.derivative()
    print(f"derivative {djdf_adjoint.dat.data}")

    h = Function(V).interpolate(Constant(1.0e1))
    assert taylor_test(Jhat, f, h) > 1.9


def test_combined_cost_function_adjoint():

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    f = Function(V).interpolate(Constant(5.0))
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)
    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_theta_theta": 0.5,
        "ts_theta_endpoint": None,
    }

    M = u * u * dx
    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u, m = solver.solve(u)

    c = Control(f)

    J = assemble(u * u * dx)
    Jhat = ReducedFunctional(J, c)
    print(f"Second m: {Jhat(f)}")
    djdf_adjoint = Jhat.derivative()
    print(f"derivative J {djdf_adjoint.dat.data}")

    h = Function(V).interpolate(Constant(1.0e1))
    assert taylor_test(Jhat, f, h) > 1.9

    Ghat = ReducedFunctional(m, c)
    Ghat(f)
    dgdf_adjoint = Ghat.derivative()
    print(f"derivative G {dgdf_adjoint.dat.data}")

    h = Function(V).interpolate(Constant(1.0e1))
    assert taylor_test(Ghat, f, h) > 1.9


if __name__ == "__main__":
    test_integral_control_in_cost_function_adjoint()
    test_integral_cost_function_adjoint()
    test_integral_cost_function_recompute()
    test_terminal_cost_function_adjoint()
    test_combined_cost_function_adjoint()
