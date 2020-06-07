import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks
import ufl
import numpy as np
import firedrake_ts

mesh = fd.UnitSquareMesh(100, 100, quadrilateral=True)
V = fd.FunctionSpace(mesh, "DG", 0)

x = ufl.SpatialCoordinate(mesh)
t = fd.Constant(0.0)
data = 100 * x[0] * (x[0] - 1) * x[1] * (x[1] - 1) * ufl.sin(ufl.pi * t)

T = 1.0

b = 2.0
u = fd.Function(V, name="solution")

u_t = fd.Function(V, name="time derivative of solution")
v = fd.TestFunction(V)

f = fd.Function(V, name="source")
f.interpolate(fd.Constant(b))

bump = ufl.conditional(ufl.lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
u.interpolate(bump)

inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
F = (u_t * v + inner(grad(u), grad(v)) - f * v) * dx
bc = fd.DirichletBC(V, 0, "on_boundary")
j = ((u - data) ** 2 + t * f) * dx

problem = firedrake_ts.DAEProblem(F, u, u_t, (0, T), bcs=bc, time=t, M=j, p=f)
solver = firedrake_ts.DAESolver(problem)

ts = solver.ts

ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
ts.setSaveTrajectory()

ts.setType(PETSc.TS.Type.THETA)
ts.setTheta(0.5)  # adjoint for 1.0 (backward Euler) is currently broken in PETSc
ts.setTimeStep(0.01)

solver.solve()

Jm = ts.getCostIntegral().getArray()
PETSc.Sys.Print(f"Cost integral numerical value is {Jm}")

solver.adjoint_solve()
