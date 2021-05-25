# This example is based on
# https://github.com/rckirby/Irksome/blob/master/demos/bbm/demo_bbm.py.rst

from firedrake import *
import firedrake_ts


def sech(x):
    return 2 / (exp(x) + exp(-x))


N = 1000
L = 100
h = L / N
msh = PeriodicIntervalMesh(N, L)

t = Constant(0)
dt = 10 * h

(x,) = SpatialCoordinate(msh)

V = FunctionSpace(msh, "CG", 1)
u = Function(V)
u_t = Function(V)
v = TestFunction(V)

c = Constant(0.5)
center = 30.0
delta = -c * center
uexact = (
    3 * c ** 2 / (1 - c ** 2) * sech(0.5 * (c * x - c * t / (1 - c ** 2) + delta)) ** 2
)
u.interpolate(uexact)

F = (
    inner(u_t, v) * dx
    + inner(u.dx(0), v) * dx
    + inner(u * u.dx(0), v) * dx
    + inner(u_t.dx(0), v.dx(0)) * dx
)

I1 = u * dx
I2 = (u ** 2 + (u.dx(0)) ** 2) * dx
I3 = ((u.dx(0)) ** 2 - u ** 3 / 3) * dx
I1s = []
I2s = []
I3s = []


def ts_monitor(ts, steps, time, X):
    I1s.append(assemble(I1))
    I2s.append(assemble(I2))
    I3s.append(assemble(I3))
    print(f"Time: {time} | I1: {I1s[-1]} | I2: {I2s[-1]} | I3: {I3s[-1]}")


params = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ts_type": "theta",
    "ts_theta_theta": 0.5,  # implicit midpoint method | the Gauss–Legendre method of order two
    "ts_exact_final_time": "matchstep",
}

problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 18.0))
solver = firedrake_ts.DAESolver(
    problem, solver_parameters=params, monitor_callback=ts_monitor
)

ts = solver.ts
ts.setTimeStep(dt)

solver.solve()

# Update t constant so that uexact is updated
t.assign(ts.getTime())

print(f"errornorm(uexact, u) / norm(uexact): {errornorm(uexact, u) / norm(uexact)}")
