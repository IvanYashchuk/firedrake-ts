from firedrake import *
import firedrake_ts

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "P", 1)

u = Function(V)
u_t = Function(V)
v = TestFunction(V)
F = inner(u_t, v) * dx + inner(grad(u), grad(v)) * dx - 1.0 * v * dx

bc = DirichletBC(V, 0.0, "on_boundary")

x = SpatialCoordinate(mesh)
# gaussian = exp(-30*(x[0]-0.5)**2)
bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
u.interpolate(bump)

problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0), bcs=bc)
solver = firedrake_ts.DAESolver(problem)

solver.solve()

print(solver.ts.getTime())
# solver.ts.view()
print(u.dat.data)
