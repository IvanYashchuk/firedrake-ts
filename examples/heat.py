from firedrake import *
import firedrake_ts
from firedrake.__future__ import interpolate

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "P", 1)

u = Function(V)
u_t = Function(V)
v = TestFunction(V)
F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - 1.0 * v * dx

bc_val=Constant(0.0)
bc = DirichletBC(V, bc_val, "on_boundary")

x = SpatialCoordinate(mesh)
bump = conditional(lt(x[0], 0.5), 1.0, 0.0)
assemble(interpolate(bump, u), tensor=u)
print(f'{u.dat.data}=')


p0=NonlinearVariationalProblem(F, u, bcs=bc)
s0=NonlinearVariationalSolver(p0)
s0.solve()
print(f'{u.dat.data}=')

bc_val.assign(1.0)
s0.solve()
print(f'{u.dat.data}=')


def monitor(ts, step, t, x):
    print(f'{solver.ts.getTime()=} {problem.time=} {bc_val=}')
    print(f'{u.dat.data}=')

problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0), bcs=bc, time=bc_val)
solver = firedrake_ts.DAESolver(problem, options_prefix='', monitor_callback=monitor)

bc_val.assign(2.0)
solver.solve()
print(id(bc_val), id(problem.time))

bc_val.assign(4.0)
solver.solve()

print(solver.ts.getTime())
# solver.ts.view()
print(u.dat.data)
