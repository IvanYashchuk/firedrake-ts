from firedrake import *
import firedrake_ts
from firedrake.__future__ import interpolate

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "P", 1)

u = Function(V)
u_t = Function(V)
v = TestFunction(V)

# F(u_t, u, t) = G(u, t)
F = inner(u_t, v) * dx
G = -(inner(grad(u), grad(v)) * dx - 1.0 * v * dx)

bc1 = DirichletBC(V, 1.0, 1)
bc2 = DirichletBC(V, 0.0, 2)
bcs=[bc1, bc2]

x = SpatialCoordinate(mesh)
bump = conditional(lt(x[0], 0.5), 1.0, 0.0)
assemble(interpolate(bump, u), tensor=u)
print(f'{u.dat.data}=')


def monitor(ts, step, t, x):
    print(f'{solver.ts.getTime()=}')
    print(f'{u.dat.data}=')

problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 0.4), bcs=bcs, G=G)

solver = firedrake_ts.DAESolver(problem, options_prefix='', monitor_callback=monitor)

solver.solve()

