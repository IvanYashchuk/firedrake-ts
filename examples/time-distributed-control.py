# This example is based on
# http://www.dolfin-adjoint.org/en/latest/documentation/time-distributed-control/time-distributed-control.html

import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks
from firedrake.assemble import create_assembly_callable
import ufl
import firedrake_ts

mesh = fd.UnitSquareMesh(8, 8)
V = fd.FunctionSpace(mesh, "CG", 1)

x = ufl.SpatialCoordinate(mesh)
t = fd.Constant(0.0)
# define the expressions for observational data 𝑑 and the viscosity 𝜈.
data = 100 * x[0] * (x[0] - 1) * x[1] * (x[1] - 1) * ufl.sin(ufl.pi * t)

dt = fd.Constant(0.1)
T = 1.0

u = fd.Function(V, name="solution")
u_t = fd.Function(V, name="time derivative of solution")
v = fd.TestFunction(V)

f = fd.Function(V, name="source")

inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
F = (u_t * v + inner(grad(u), grad(v)) - f * v) * dx
bc = fd.DirichletBC(V, 0, "on_boundary")

problem = firedrake_ts.DAEProblem(F, u, u_t, (0, T), bcs=bc, time=t)
solver = firedrake_ts.DAESolver(problem)

ts = solver.ts

ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
ts.setSaveTrajectory()

ts.setType(PETSc.TS.Type.THETA)
ts.setTheta(0.99)  # adjoint for 1.0 (backward Euler) is currently broken in PETSc
ts.setTimeStep(0.1)

quad_ts = ts.createQuadratureTS(forward=False)
quad_ts.setDM(problem.dm)

R = fd.FunctionSpace(mesh, "R", 0)
vr = fd.TestFunction(R)

# how to include time derivative of f?
j = ((u - data) ** 2 + t * f) * vr * dx

j_value = fd.Function(R)


def form_cost_integrand(ts, t, X, R):
    r"""Form the integrand of the cost function

    :arg ts: a PETSc TS object
    :arg t: the time at step/stage being solved
    :arg X: state vector
    :arg R: function vector
    """
    dm = ts.getDM()
    ctx = dmhooks.get_appctx(dm)
    # X may not be the same vector as the vec behind self._x, so
    # copy guess in from X.
    with ctx._x.dat.vec_wo as v:
        X.copy(v)
    ctx._time.assign(t)

    fd.assemble(j, tensor=j_value)
    with j_value.dat.vec_ro as v:
        v.copy(R)


_djdu = fd.assemble(fd.derivative(j, u))


def form_djdu(ts, t, X, Amat, Pmat):
    dm = ts.getDM()
    ctx = dmhooks.get_appctx(dm)
    with ctx._x.dat.vec_wo as v:
        X.copy(v)
    ctx._time.assign(t)

    fd.assemble(fd.derivative(j, u), tensor=_djdu)


_djdf = fd.assemble(fd.derivative(j, f))


def form_djdf(ts, t, X, Amat):
    dm = ts.getDM()
    ctx = dmhooks.get_appctx(dm)
    with ctx._x.dat.vec_wo as v:
        X.copy(v)
    ctx._time.assign(t)

    fd.assemble(fd.derivative(j, f), tensor=_djdf)


# integrand_vec = PETSc.Vec().createSeq(1)

# with j_value.dat.vec_wo as v:
#     quad_ts.setRHSFunction(form_cost_integrand, f=v)
quad_ts.setRHSFunction(form_cost_integrand)

# # djdu_mat = PETSc.Mat().createDense([1, u.ufl_function_space().dim()])
# quad_ts.setRHSJacobian(form_djdu, J=_djdu.petscmat)

# # djdf_mat = PETSc.Mat().createDense([1, f.ufl_function_space().dim()])
# quad_ts.setRHSJacobianP(form_djdf, A=_djdf.petscmat)

bump = ufl.conditional(ufl.lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
u.interpolate(bump)

# I get error at this point
# Error: error code 75
# [0] TSSolve() line 4127 in /Users/yashchi1/devdir/firedrake/src/petsc/src/ts/interface/ts.c
# [0] TSStep() line 3721 in /Users/yashchi1/devdir/firedrake/src/petsc/src/ts/interface/ts.c
# [0] TSStep_Theta() line 223 in /Users/yashchi1/devdir/firedrake/src/petsc/src/ts/impls/implicit/theta/theta.c
# [0] TSTheta_SNESSolve() line 185 in /Users/yashchi1/devdir/firedrake/src/petsc/src/ts/impls/implicit/theta/theta.c
# [0] SNESSolve() line 4516 in /Users/yashchi1/devdir/firedrake/src/petsc/src/snes/interface/snes.c
# [0] SNESSolve_NEWTONLS() line 175 in /Users/yashchi1/devdir/firedrake/src/petsc/src/snes/impls/ls/ls.c
# [0] SNESComputeFunction() line 2379 in /Users/yashchi1/devdir/firedrake/src/petsc/src/snes/interface/snes.c
# [0] SNESTSFormFunction() line 4983 in /Users/yashchi1/devdir/firedrake/src/petsc/src/ts/interface/ts.c
# [0] SNESTSFormFunction_Theta() line 973 in /Users/yashchi1/devdir/firedrake/src/petsc/src/ts/impls/implicit/theta/theta.c
# [0] VecAXPBYPCZ() line 684 in /Users/yashchi1/devdir/firedrake/src/petsc/src/vec/vec/interface/rvector.c
# [0] Arguments are incompatible
# [0] Incompatible vector global lengths parameter # 1 global size 1 != parameter # 5 global size 81

solver.solve()

dJdu = fd.Function(V)
dJdf = fd.Function(V)

with dJdu.dat.vec as dJdu_vec, dJdf.dat.vec as dJdf_vec:
    ts.setCostGradients(dJdu_vec, dJdf_vec)
