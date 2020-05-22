# This example is based on
# http://www.dolfin-adjoint.org/en/latest/documentation/tutorial.html

import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks
import ufl
import firedrake_ts

equation = "heat"  # 'burgers' or 'heat'
n = 50
mesh = fd.UnitSquareMesh(n, n)
V = fd.VectorFunctionSpace(mesh, "CG", 2)

x = ufl.SpatialCoordinate(mesh)
expr = ufl.as_vector([ufl.sin(2 * ufl.pi * x[0]), ufl.cos(2 * ufl.pi * x[1])])
u = fd.interpolate(expr, V)

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

solver.solve()

J = fd.inner(u, u) * fd.dx

# dJ/du at final time
dJdu = fd.assemble(fd.derivative(J, u))

with dJdu.dat.vec as vec:
    dJdu_vec = vec

print(f"Norm of dJdu before the adjoint solve: {fd.norm(dJdu)}")

# setCostGradients accepts two PETSc Vecs
# J is the objective function
# u is the solution state
# m is the parameter
# then the input is dJdu (at final time), dJdm (at final time)
# After calling ts.adjointSolve() these Vecs will be overwritten by results
ts.setCostGradients(dJdu_vec, None)

adj_out = fd.File("result/adj.pvd")


dm = ts.getDM()
with dmhooks.add_hooks(dm, solver, appctx=solver._ctx):
    # ts.adjointSolve() # "works" only if one timestep was done during the forward run
    ts.adjointStep()  # the first adjoint step seems to work correctly
    print(f"Norm of dJdu, adjoint step #1: {fd.norm(dJdu)}")
    adj_out.write(dJdu, time=0)
    ts.adjointStep()  # trying to call adjointStep second time breaks the calculation
    print(f"Norm of dJdu, adjoint step #2: {fd.norm(dJdu)}")
    adj_out.write(dJdu, time=1)
