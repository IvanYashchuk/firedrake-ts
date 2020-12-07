import pytest
from firedrake import *
from firedrake_adjoint import *
from firedrake import PETSc
import firedrake_ts

PETSc.Sys.popErrorHandler()

print = lambda x: PETSc.Sys.Print(x)


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


params = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ts_type": "theta",
    "ts_type": "theta",
    # "ts_monitor_solution": None,
    "ts_adjoint_monitor": None,
    "ts_monitor": None,
    "ts_theta_theta": 0.5,  # implicit midpoint method | the Gauss–Legendre method of order two
    "ts_exact_final_time": "matchstep",
}


dt = 1e-1
problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.3), dt, bcs=bc, M=M)
solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

m = solver.solve(u)
print(f"First m: {m}")

# solver.adjoint_solve()
# dJdu, dJdf = solver.get_cost_gradients()
# print(f"Second dJdf: {dJdf.dat.data}")

c = Control(f)
Jhat = ReducedFunctional(m, c)
# Jhat.optimize_tape()
# print(f"Second m: {Jhat(f)}")
# djdf_adjoint = Jhat.derivative()
# print(f"derivative {djdf_adjoint.dat.data}")
# print(f"Third m: {Jhat(interpolate(Constant(5.0), V))}")
# djdf_adjoint = Jhat.derivative()
# print(f"derivative {djdf_adjoint.dat.data}")
# print(f"Fourth m: {Jhat(interpolate(Constant(5.0), V))}")

h = Function(V).interpolate(Constant(1.0e-6))
assert taylor_test(Jhat, f, h) > 1.9
