from firedrake import *
import firedrake_ts
from firedrake.__future__ import interpolate

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "P", 1)

u = Function(V)
u_t = Function(V)
v = TestFunction(V)

# F(u_t, u, t)=0
F = inner(u_t, v) * dx
G = -inner(grad(u), grad(v)) * dx + 1.0 * v * dx
F = F - G

bc = DirichletBC(V, 1.0, "on_boundary")

x = SpatialCoordinate(mesh)
bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
assemble(interpolate(bump, u))

problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0), bcs=bc)
solver = firedrake_ts.DAESolver(problem, options_prefix='')

solver.solve()

print(solver.ts.getTime())
print(u.dat.data)
