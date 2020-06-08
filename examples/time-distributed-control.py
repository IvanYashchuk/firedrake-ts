import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks
import ufl
import numpy as np
import firedrake_ts

mesh = fd.UnitSquareMesh(10, 10, quadrilateral=True)
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
j = ((u - data) ** 2 + t * f * f) * dx

dt = 0.001
problem = firedrake_ts.DAEProblem(F, u, u_t, (0, T), dt, bcs=bc, time=t, M=j, p=f)
solver = firedrake_ts.DAESolver(problem)

solver.solve()

Jm = solver.get_cost_function()
PETSc.Sys.Print(f"Cost integral numerical value is {Jm}")

solver.adjoint_solve()

dJdu, dJdf = solver.get_cost_gradients()


def taylor_test(Jm, f, dJdf, h):

    # Direction
    with h.dat.vec_ro as hv, dJdf.dat.vec_ro as dJdf_vec:
        dJdm = dJdf_vec.dot(hv)

    epsilons = [0.01 / 2 ** i for i in range(4)]
    residuals = []
    fh = f.copy(deepcopy=True)
    for eps in epsilons:
        with f.dat.vec_ro as fv, fh.dat.vec_ro as fhv, h.dat.vec_ro as hv:
            fv.waxpy(eps, hv, fhv)

        solver.solve()
        Jp = solver.get_cost_function()
        res = abs(Jp - Jm - eps * dJdm)
        residuals.append(res)

    print("Computed residuals: {}".format(residuals))

    from numpy import log

    r = []
    for i in range(1, len(epsilons)):
        r.append(
            log(residuals[i] / residuals[i - 1]) / log(epsilons[i] / epsilons[i - 1])
        )

    print("Computed convergence rates: {}".format(r))

    return False


h = fd.Function(V).interpolate(fd.Constant(0.1))
taylor_test(Jm, f, dJdf, h)
