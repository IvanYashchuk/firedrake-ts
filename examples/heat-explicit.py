from firedrake import *
import firedrake_ts

mesh = UnitIntervalMesh(11)
V = FunctionSpace(mesh, "P", 1)

u = Function(V)
u_t = Function(V)
v = TestFunction(V)

# F(u̇, u, t) = G(u, t)
F = inner(u_t, v) * dx
G = -(inner(grad(u), grad(v)) * dx - 1.0 * v * dx)

bc = DirichletBC(V, 0.0, "on_boundary")

x = SpatialCoordinate(mesh)
bump = conditional(lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
u.interpolate(bump)

problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 1.0), bcs=bc, G=G)
solver = firedrake_ts.DAESolver(problem)

solver.solve()

print(solver.ts.getTime())
# solver.ts.view()
print(u.dat.data)
