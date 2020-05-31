# This example is based on
# http://www.dolfin-adjoint.org/en/latest/documentation/time-distributed-control/time-distributed-control.html

import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks
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

# Now create QuadratureTS for integrating the cost integral
quad_ts = ts.createQuadratureTS(forward=True)

# We want to attach solver._ctx to DM of QuadratureTS
# to be able to modify the data attached to the solver
# from the RHSFunction, Jacobian, JacobianP
quad_dm = quad_ts.getDM()
dmhooks.push_appctx(quad_dm, solver._ctx)

# how to include time derivative of f?
j = ((u - data) ** 2 + t * f) * dx


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

    j_value = fd.assemble(j)
    R.set(j_value)


_djdu = fd.assemble(fd.derivative(j, u))
djdu_transposed_mat = PETSc.Mat().createDense([_djdu.ufl_function_space().dim(), 1])
djdu_transposed_mat.setUp()


def form_djdu(ts, t, X, Amat, Pmat):
    dm = ts.getDM()
    ctx = dmhooks.get_appctx(dm)
    with ctx._x.dat.vec_wo as v:
        X.copy(v)
    ctx._time.assign(t)

    fd.assemble(fd.derivative(j, u), tensor=_djdu)
    Amat_array = Amat.getDenseArray()
    Amat_array[:] = _djdu.dat.data.reshape(Amat_array.shape)

    # the following is wrong in parallel, since getValues can only get values on the same processor
    # with _djdu.dat.vec_ro as v:
    #     Amat_array[:] = v.getValues(range(*Amat.getOwnershipRange())).reshape(Amat_array.shape)
    # Amat.assemble()


_djdf = fd.assemble(fd.derivative(j, f))
djdf_transposed_mat = PETSc.Mat().createDense([_djdf.ufl_function_space().dim(), 1])
djdf_transposed_mat.setUp()


def form_djdf(ts, t, X, Amat):
    dm = ts.getDM()
    ctx = dmhooks.get_appctx(dm)
    with ctx._x.dat.vec_wo as v:
        X.copy(v)
    ctx._time.assign(t)

    fd.assemble(fd.derivative(j, f), tensor=_djdf)
    Amat_array = Amat.getDenseArray()
    Amat_array[:] = _djdf.dat.data.reshape(Amat_array.shape)

    # the following is wrong in parallel, since getValues can only get values on the same processor
    # with _djdf.dat.vec_ro as v:
    #     Amat_array[:] = v.getValues(getValues(range(*Amat.getOwnershipRange()))).reshape(Amat_array.shape)
    # Amat.assemble()


quad_ts.setRHSFunction(form_cost_integrand)
quad_ts.setRHSJacobian(form_djdu, J=djdu_transposed_mat)
quad_ts.setRHSJacobianP(form_djdf, A=djdf_transposed_mat)

bump = ufl.conditional(ufl.lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
u.interpolate(bump)

solver.solve()

PETSc.Sys.Print(f"Cost integral value is {ts.getCostIntegral().getArray()}")

dJdu = fd.Function(V)
dJdf = fd.Function(V)

# TODO: Add wrappers of TSSetIJacobianP to petsc4py
# AttributeError: 'petsc4py.PETSc.TS' object has no attribute 'setIJacobianP'

with dJdu.dat.vec as dJdu_vec, dJdf.dat.vec as dJdf_vec:
    ts.setCostGradients(dJdu_vec, None)

PETSc.Sys.Print(f"Norm before the adjoint solve: {fd.norm(dJdu)}")

solver.adjoint_solve()

# Something happens
PETSc.Sys.Print(f"Norm after the adjoint solve: {fd.norm(dJdu)}")
