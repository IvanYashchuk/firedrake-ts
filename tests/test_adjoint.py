import pytest
from firedrake import *
from firedrake_adjoint import *
import petsc4py
import firedrake_ts
from pyadjoint.tape import no_annotations, stop_annotating
import numpy as np

petsc4py.PETSc.Sys.popErrorHandler()
print = lambda x: PETSc.Sys.Print(x, comm=COMM_SELF)


@pytest.fixture
def solver_parameters():
    return {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_dt": 0.1,
        "ts_theta_theta": 0.5,
        "ts_theta_endpoint": None,
    }


@pytest.mark.skip()
def test_shape_derivative(solver_parameters):

    n = 10
    mesh = UnitSquareMesh(n, n)

    S = mesh.coordinates.function_space()
    s = Function(S)
    mesh.coordinates.assign(mesh.coordinates + s)

    V = FunctionSpace(mesh, "CG", 1)
    RHO = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)

    F = (inner(u_t, v) + u * u.dx(0) * v + a * u.dx(0) * v.dx(0)) * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    ic = interpolate(sin(2 * pi * x[0]), V)
    u.interpolate(ic)

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc)
    solver_parameters["ts_type"] = "beuler"
    solver_parameters.pop("ts_theta_theta")
    solver_parameters.pop("ts_theta_endpoint")
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u = solver.solve(u)

    m = assemble(inner(u, u) * dx)
    print(f"m: {m}")
    c = Control(ic)
    Jhat = ReducedFunctional(m, c)
    m2 = Jhat(ic)
    assert np.allclose(m, m2, 1e-6)

    h = Function(V).interpolate(Constant(1.0))
    assert taylor_test(Jhat, ic, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_time_dependent_bcs(control, solver_parameters):

    n = 10
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 1)
    RHO = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    if control == "constant":
        a = Constant(0.5)
    else:
        a = Function(RHO)
        with stop_annotating():
            a.interpolate(Constant(0.5))
    F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - a * v * dx

    rate = 0.1
    vd = Function(V)
    with stop_annotating():
        vd.interpolate(Constant(0.0))
    bc = DirichletBC(V, vd, "on_boundary")

    u_bc = Function(V)

    @no_annotations
    def apply_time_bcs(ts, steps, time, X):
        with u_bc.dat.vec as u_vec:
            X.copy(u_vec)
        bc.function_arg.assign(time * rate)
        bc.apply(u_bc)
        with u_bc.dat.vec as u_vec:
            u_vec.copy(X)

    ic = interpolate(sin(2 * pi * x[0]), V)
    u.assign(ic)

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc)
    # solver_parameters["ts_type"] = "beuler"
    # solver_parameters["snes_monitor"] = None
    # solver_parameters.pop("ts_theta_theta")
    # solver_parameters.pop("ts_theta_endpoint")
    solver = firedrake_ts.DAESolver(
        problem,
        solver_parameters=solver_parameters,
        monitor_callback=apply_time_bcs,
    )

    u = solver.solve(u)

    m = assemble(inner(u, u) * dx)
    print(f"m: {m}")
    c = Control(ic)
    Jhat = ReducedFunctional(m, c)
    m2 = Jhat(ic)
    assert np.allclose(m, m2, 1e-6)

    h = Function(V).interpolate(Constant(1.0))
    assert taylor_test(Jhat, ic, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_burgers(control, solver_parameters):

    n = 10
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 1)
    RHO = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    if control == "constant":
        a = Constant(0.5)
    elif control == "function":
        a = Function(RHO).interpolate(Constant(0.5))
    else:
        a = Function(RHO).interpolate(Constant(0.5))
    F = (inner(u_t, v) + u * u.dx(0) * v + a * u.dx(0) * v.dx(0)) * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    ic = interpolate(sin(2 * pi * x[0]), V)
    u.interpolate(ic)

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc)
    solver_parameters["ts_type"] = "beuler"
    solver_parameters.pop("ts_theta_theta")
    solver_parameters.pop("ts_theta_endpoint")
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u = solver.solve(u)

    m = assemble(inner(u, u) * dx)
    print(f"m: {m}")
    c = Control(ic)
    Jhat = ReducedFunctional(m, c)
    m2 = Jhat(ic)
    assert np.allclose(m, m2, 1e-6)

    h = Function(V).interpolate(Constant(1.0))
    assert taylor_test(Jhat, ic, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_cost_function_recompute(control, solver_parameters):

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

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u, m = solver.solve(u)

    c = Control(f)
    Jhat = ReducedFunctional(m, c)
    m2 = Jhat(f)
    assert np.allclose(m, m2, 1e-8)


def test_initial_condition_recompute(solver_parameters):

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

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u, m = solver.solve(u)

    c = Control(ic)
    Jhat = ReducedFunctional(m, c)
    m2 = Jhat(ic)
    assert np.allclose(m, m2, 1e-8)


def test_initial_condition_adjoint(solver_parameters):

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

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u_fin = solver.solve(u)
    m = assemble(u_fin * u_fin * dx)

    c = Control(ic)
    Jhat = ReducedFunctional(m, c)
    Jhat(ic)
    djdf_adjoint = Jhat.derivative()

    h = Function(V).interpolate(Constant(1.0))
    assert taylor_test(Jhat, ic, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_cost_function_adjoint(control, solver_parameters):

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

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u, m = solver.solve(u)

    c = Control(f)
    Jhat = ReducedFunctional(m, c)
    Jhat(f)
    djdf_adjoint = Jhat.derivative()

    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.parametrize("control", ["constant", "function"])
def test_integral_control_in_cost_function_adjoint(control, solver_parameters):

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

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u, m = solver.solve(u)

    c = Control(f)
    Jhat = ReducedFunctional(m, c)

    Jhat(f)
    djdf_adjoint = Jhat.derivative()

    assert taylor_test(Jhat, f, h) > 1.9


def test_terminal_cost_function_multiple_deps_in_form_adjoint(solver_parameters):

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    # a = Constant(2.0)
    a = Function(V)
    f = Function(V)
    rho = Function(V)
    with stop_annotating():
        a.interpolate(Constant(5.0))
        f.interpolate(Constant(0.5))
        rho.interpolate(Constant(0.2))
    F = inner(rho * u_t, v) * dx + inner(grad(u), grad(v)) * dx - f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u = solver.solve(u)

    J = assemble(u * u * dx)

    c = Control(rho)
    Jhat = ReducedFunctional(J, c)
    print(f"m: {Jhat(rho)}")
    djdf_adjoint = Jhat.derivative()
    print(djdf_adjoint.dat.data)

    h = Function(V).interpolate(Constant(1.0e0))
    assert taylor_test(Jhat, rho, h) > 1.9


def test_terminal_cost_function_adjoint(solver_parameters):

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

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

    u = solver.solve(u)

    J = assemble(u * u * dx)

    c = Control(f)
    Jhat = ReducedFunctional(J, c)
    Jhat(f)
    djdf_adjoint = Jhat.derivative()

    h = Function(V).interpolate(Constant(1.0e1))
    assert taylor_test(Jhat, f, h) > 1.9


def test_combined_cost_function_adjoint(solver_parameters):

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

    M = u * u * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

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


def test_multiple_coeffs(solver_parameters):

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "P", 1)

    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    f = Function(V).interpolate(Constant(5.0))
    a = Constant(2.0)
    b = Constant(2.0)
    F = inner(u_t, v) * dx + inner(b * grad(u), grad(v)) * dx - f * v * dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    x = SpatialCoordinate(mesh)
    bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)

    M = u * u * f * b * dx
    problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), bcs=bc, M=M)
    solver = firedrake_ts.DAESolver(problem, solver_parameters=solver_parameters)

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
    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ts_type": "theta",
        "ts_dt": 0.1,
        "ts_theta_theta": 0.5,
        "ts_theta_endpoint": None,
    }
    # test_integral_cost_function_adjoint("function", params)
    # test_integral_control_in_cost_function_adjoint("function", params)
    # test_integral_cost_function_recompute("function", params)
    # test_integral_cost_function_adjoint("constant", params)
    # test_integral_control_in_cost_function_adjoint("constant", params)
    # test_integral_cost_function_recompute("constant", params)
    # test_terminal_cost_function_adjoint(params)
    # test_combined_cost_function_adjoint(params)
    # test_initial_condition_recompute(params)
    # test_initial_condition_adjoint(params)
    # test_burgers("function", params)
    # test_time_dependent_bcs("function", params)
    # test_terminal_cost_function_multiple_deps_in_form_adjoint(params)
    test_multiple_coeffs(params)
