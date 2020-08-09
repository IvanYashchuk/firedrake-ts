# This example is based on https://firedrakeproject.org/demos/burgers.py.html
from firedrake import *
import firedrake_ts

n = 30
mesh = UnitSquareMesh(n, n)

V = VectorFunctionSpace(mesh, "CG", 2)
V_out = VectorFunctionSpace(mesh, "CG", 1)

u = Function(V, name="Velocity")
u_t = Function(V, name="VelocityTimeDerivative")

v = TestFunction(V)

x = SpatialCoordinate(mesh)
ic = project(as_vector([sin(pi * x[0]), 0]), V)

u.assign(ic)

nu = 0.0001

F = (
    inner(u_t, v) + inner(dot(u, nabla_grad(u)), v) + nu * inner(grad(u), grad(v))
) * dx

outfile = File("burgers.pvd")
outfile.write(project(u, V_out, name="Velocity"), time=0.0)


def ts_monitor(ts, steps, time, X):
    outfile.write(project(u, V_out, name="Velocity"), time=time)


problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.5))
solver = firedrake_ts.DAESolver(problem, monitor_callback=ts_monitor)

timestep = 1.0 / n
ts = solver.ts
ts.setTimeStep(timestep)

solver.solve()
