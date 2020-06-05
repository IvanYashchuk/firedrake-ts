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

def evalfunc(f):
    bump = ufl.conditional(ufl.lt(abs(x[0] - 0.5), 0.1), 1.0, 0.0)
    u.interpolate(bump)

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    F = (u_t * v + inner(grad(u), grad(v)) - f * v) * dx
    bc = fd.DirichletBC(V, 0, "on_boundary")

    problem = firedrake_ts.DAEProblem(F, u, u_t, (0, T), bcs=bc, time=t)
    solver = firedrake_ts.DAESolver(problem)

    ts = solver.ts

    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
    ts.setSaveTrajectory()

    ts.setType(PETSc.TS.Type.THETA)
    ts.setTheta(0.5)  # adjoint for 1.0 (backward Euler) is currently broken in PETSc
    ts.setTimeStep(0.01)

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
        r"""Form the integrand of the nost function

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
    local_dofs = _djdu.dat.data.shape[0]
    djdu_transposed_mat = PETSc.Mat().createDense([[local_dofs, _djdu.ufl_function_space().dim()], [1, 1]])
    djdu_transposed_mat.setUp()


    def form_djdu(ts, t, X, Amat, Pmat):
        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        with ctx._x.dat.vec_wo as v:
            X.copy(v)
        ctx._time.assign(t)

        fd.assemble(fd.derivative(j, u), tensor=_djdu)
        Amat_array = Amat.getDenseArray()
        with _djdu.dat.vec_ro as v:
            Amat_array[:, 0] = v.array[:]
        Amat.assemble()


    _djdf = fd.assemble(fd.derivative(j, f))
    local_dofs = _djdf.dat.data.shape[0]
    djdf_transposed_mat = PETSc.Mat().createDense([[local_dofs, _djdf.ufl_function_space().dim()], [1, 1]])
    djdf_transposed_mat.setUp()


    def form_djdf(ts, t, X, Amat):
        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        with ctx._x.dat.vec_wo as v:
            X.copy(v)
        ctx._time.assign(t)

        fd.assemble(fd.derivative(j, f), tensor=_djdf)
        Amat_array = Amat.getDenseArray()
        with _djdf.dat.vec_ro as v:
            Amat_array[:, 0] = v.array[:]
        Amat.assemble()


    dFdf = fd.derivative(-F, f)
    _dFdf = fd.assemble(dFdf)
    def form_dFdf(ts, t, X, Jp):
        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        ctx._time.assign(t)
        with ctx._x.dat.vec_wo as v:
            X.copy(v)

        fd.assemble(dFdf, tensor=_dFdf)

    ts.setRHSJacobianP(form_dFdf, _dFdf.petscmat)

    quad_ts.setRHSFunction(form_cost_integrand)
    quad_ts.setRHSJacobian(form_djdu, J=djdu_transposed_mat)
    quad_ts.setRHSJacobianP(form_djdf, A=djdf_transposed_mat)

    solver.solve()

    Jm = ts.getCostIntegral().getArray()
    PETSc.Sys.Print(f"Cost integral numerical value is {Jm}")

    dJdu = fd.Function(V)
    dJdf = fd.Function(V)

    # TODO: Add wrappers of TSSetIJacobianP to petsc4py
    # AttributeError: 'petsc4py.PETSc.TS' object has no attribute 'setIJacobianP'

    with dJdu.dat.vec as dJdu_vec, dJdf.dat.vec as dJdf_vec:
        ts.setCostGradients(dJdu_vec, dJdf_vec)

    solver.adjoint_solve()

    # Reset time step (becomes negative after adjointSolve
    ts.setTimeStep(0.001)
    # reset step counter
    ts.setStepNumber(0)


    return Jm, dJdf

def taylor_test(Jm, f, dJdf, h):

    # Direction
    with h.dat.vec_ro as hv, dJdf.dat.vec_ro as dJdf_vec:
        dJdm = dJdf_vec.dot(hv)

    epsilons = [0.01 / 2 ** i for i in range(4)]
    residuals = []
    fh = f.copy(deepcopy=True)
    for eps in epsilons:
        with f.dat.vec_ro as fv, fh.dat.vec_ro as fhv, h.dat.vec_ro as hv:
            fhv.waxpy(eps, hv, fv)

        Jp, _ = evalfunc(fh)
        res = abs(Jp - Jm - eps * dJdm)
        residuals.append(res)

    print("Computed residuals: {}".format(residuals))

    from numpy import log
    r = []
    for i in range(1, len(epsilons)):
        r.append(log(residuals[i] / residuals[i - 1])
                 / log(epsilons[i] / epsilons[i - 1]))

    print("Computed convergence rates: {}".format(r))

    return False

h = fd.Function(V).interpolate(fd.Constant(0.001))

Jm, dJdf = evalfunc(f)
taylor_test(Jm, f, dJdf, h)
