import pytest
from firedrake import *
from firedrake_adjoint import *
import petsc4py
import firedrake_ts
import numpy as np

petsc4py.PETSc.Sys.popErrorHandler()
print = lambda x: PETSc.Sys.Print(x, comm=COMM_SELF)


@pytest.mark.skip()
@pytest.mark.parametrize("control", ["constant", "function"])
def test_burgers(control):

    n = 30
    mesh = UnitSquareMesh(n, n)
    V = VectorFunctionSpace(mesh, "CG", 2)
    RHO = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    if control == "constant":
        a = Constant(5.0)
    elif control == "function":
        a = Function(RHO).interpolate(sin(x[0]))
    else:
        a = Function(RHO).interpolate(sin(x[0]))
    F = (
        inner(u_t, v) + inner(dot(u, nabla_grad(u)), v) + a * inner(grad(u), grad(v))
    ) * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    ic = project(as_vector([sin(pi * x[0]), 0]), V)
    u.interpolate(ic)
    M = inner(u, u) * dx

    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_theta_theta": 0.5,
        "ts_exact_final_time": "matchstep",
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u, m = solver.solve(u)

    c = Control(a)
    Jhat = ReducedFunctional(m, c)
    m2 = Jhat(a)
    assert np.allclose(m, m2, 1e-8)

    h = Function(RHO).interpolate(Constant(1.0))
    assert taylor_test(Jhat, a, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_cost_function_recompute(control):

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)
    x = SpatialCoordinate(mesh)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    a = Function(V).interpolate(sin(x[0]))
    if control == "constant":
        f = Constant(5.0)
    elif control == "function":
        f = Function(V).interpolate(Constant(5.0))
    else:
        f = Function(V).interpolate(Constant(5.0))
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - a * f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)
    M = u * u * dx

    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
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
    m2 = Jhat(f)
    assert np.allclose(m, m2, 1e-8)


def test_initial_condition_recompute():

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
    ic = Function(V).interpolate(bump)
    u.assign(ic)
    M = u * u * dx

    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_theta_theta": 0.5,
        "ts_exact_final_time": "matchstep",
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u, m = solver.solve(u)

    c = Control(ic)
    Jhat = ReducedFunctional(m, c)
    m2 = Jhat(ic)
    assert np.allclose(m, m2, 1e-8)


def test_initial_condition_adjoint():

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    f = Function(V).interpolate(Constant(5.0))
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    ic = Function(V)
    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    ic.interpolate(bump)
    u.assign(ic)

    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_theta_theta": 0.5,
        "ts_exact_final_time": "matchstep",
        # "ts_adjoint_monitor_sensi": None,
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u_fin = solver.solve(u)
    m = assemble(u_fin * u_fin * dx)

    c = Control(ic)
    Jhat = ReducedFunctional(m, c)
    Jhat(ic)
    djdf_adjoint = Jhat.derivative()

    h = Function(V).interpolate(Constant(1.0))
    assert taylor_test(Jhat, ic, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_cost_function_adjoint(control):

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)
    comm = mesh.comm

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
        "ts_theta_theta": 0.5,
        "ts_exact_final_time": "matchstep",
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u, m = solver.solve(u)

    c = Control(f)
    Jhat = ReducedFunctional(m, c)
    Jhat(f)
    djdf_adjoint = Jhat.derivative()

    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_control_in_cost_function_adjoint(control):

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)
    comm = mesh.comm

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
        "ts_theta_theta": 0.5,
        "ts_exact_final_time": "matchstep",
    }

    dt = 1e-1
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

    u, m = solver.solve(u)

    c = Control(f)
    Jhat = ReducedFunctional(m, c)

    Jhat(f)
    djdf_adjoint = Jhat.derivative()

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
    Jhat(f)
    djdf_adjoint = Jhat.derivative()

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
    Jhat(f)
    djdf_adjoint = Jhat.derivative()

    h = Function(V).interpolate(Constant(1.0e1))
    assert taylor_test(Jhat, f, h) > 1.9

    Ghat = ReducedFunctional(m, c)
    Ghat(f)
    dgdf_adjoint = Ghat.derivative()

    h = Function(V).interpolate(Constant(1.0e1))
    assert taylor_test(Ghat, f, h) > 1.9


if __name__ == "__main__":
    # test_integral_cost_function_adjoint("function")
    # test_integral_control_in_cost_function_adjoint("function")
    # test_integral_cost_function_recompute("function")
    # test_integral_cost_function_adjoint("constant")
    # test_integral_control_in_cost_function_adjoint("constant")
    # test_integral_cost_function_recompute("constant")
    # test_terminal_cost_function_adjoint()
    # test_combined_cost_function_adjoint()
    # test_initial_condition_recompute()
    test_initial_condition_adjoint()
    # test_burgers("function")
