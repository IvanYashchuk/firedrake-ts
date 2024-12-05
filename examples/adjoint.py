# This example is based on
# http://www.dolfin-adjoint.org/en/latest/documentation/tutorial.html

import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks
import ufl
import firedrake_ts

equation = "burgers"  # 'burgers' or 'heat'
n = 50
mesh = fd.UnitSquareMesh(n, n)
V = fd.VectorFunctionSpace(mesh, "CG", 2)

x = ufl.SpatialCoordinate(mesh)
expr = ufl.as_vector([ufl.sin(2 * ufl.pi * x[0]), ufl.cos(2 * ufl.pi * x[1])])
u = fd.Function(V)
u.interpolate(expr)

u_dot = fd.Function(V)
v = fd.TestFunction(V)

nu = fd.Constant(0.0001)  # for burgers
if equation == "heat":
    nu = fd.Constant(0.1)  # for heat

M = fd.derivative(fd.inner(u, v) * fd.dx, u)
R = -(fd.inner(fd.grad(u) * u, v) + nu * fd.inner(fd.grad(u), fd.grad(v))) * fd.dx
if equation == "heat":
    R = -nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
F = fd.action(M, u_dot) - R

bc = fd.DirichletBC(V, (0.0, 0.0), "on_boundary")

t = 0.0
end = 0.1
tspan = (t, end)

state_out = fd.File("result/state.pvd")


def ts_monitor(ts, steps, time, X):
    state_out.write(u, time=time)


problem = firedrake_ts.DAEProblem(F, u, u_dot, tspan, bcs=bc)
solver = firedrake_ts.DAESolver(problem, monitor_callback=ts_monitor)

ts = solver.ts
ts.setTimeStep(0.01)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
ts.setSaveTrajectory()

ts.setType(PETSc.TS.Type.THETA)
ts.setTheta(0.99)  # adjoint for 1.0 (backward Euler) is currently broken in PETSc

solver.solve()

J = fd.inner(u, u) * fd.dx

# dJ/du at final time
dJdu = fd.assemble(fd.derivative(J, u))

with dJdu.dat.vec as vec:
    dJdu_vec = vec

fdJdu=dJdu.riesz_representation()
    
print(f"Norm of dJdu before the adjoint solve: {dJdu_vec.norm()=} {fd.norm(fdJdu)=}")

# setCostGradients accepts two PETSc Vecs
# J is the objective function
# u is the solution state
# m is the parameter
# then the input is dJdu (at final time), dJdm (at final time)
# After calling solver.adjoint_solve() these Vecs will be overwritten by results
ts.setCostGradients(dJdu_vec, None)

solver.adjoint_solve()

fdJdu=dJdu.riesz_representation()
print(f"Norm of dJdu after the adjoint solve: {dJdu_vec.norm()=} {fd.norm(fdJdu)=}")


adj_out = fd.File("result/adj.pvd")
adj_out.write(fdJdu, time=0)
