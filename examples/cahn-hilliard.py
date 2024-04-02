# This example is based on
# https://fenicsproject.org/docs/dolfin/latest/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html
# and
# https://github.com/firedrakeproject/firedrake-bench/blob/master/cahn_hilliard/firedrake_cahn_hilliard.py

import firedrake_ts
from firedrake import *
import numpy as np

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06  # time step

mesh = UnitSquareMesh(96, 96, quadrilateral=True)
V = FunctionSpace(mesh, "Lagrange", 1)
ME = V * V

# Define test functions
q, v = TestFunctions(ME)

# Define functions
u = Function(ME)  # current solution
u_t = Function(ME)  # time derivative

# Split mixed functions
c, mu = split(u)
c_t, mu_t = split(u_t)


# Compute the chemical potential df/dc
c = variable(c)
f = 100 * c ** 2 * (1 - c) ** 2
dfdc = diff(f, c)

# Weak statement of the equations
F0 = c_t * q * dx + dot(grad(mu), grad(q)) * dx
F1 = mu * v * dx - dfdc * v * dx - lmbda * dot(grad(c), grad(v)) * dx
F = F0 + F1

rng = np.random.default_rng(11)
c , mu = u.subfunctions
with c.dat.vec as vv:
    vv[:]=0.63 + 0.2*(0.5-rng.random(vv.size))

    
pc = "fieldsplit"
ksp = "lgmres"
inner_ksp = "preonly"
maxit = 1
params = {
    "mat_type": "aij",
    "pc_type": pc,
    "ksp_type": ksp,
    "snes_rtol": 1e-9,
    "snes_atol": 1e-10,
    "snes_stol": 1e-16,
    "snes_linesearch_type": "basic",
    "snes_linesearch_max_it": 1,
    "ksp_rtol": 1e-6,
    "ksp_atol": 1e-15,
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "lower",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_0_ksp_type": inner_ksp,
    "fieldsplit_0_ksp_max_it": maxit,
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": inner_ksp,
    "fieldsplit_1_ksp_max_it": maxit,
    "fieldsplit_1_pc_type": "mat",
}

params["snes_monitor"] = None
params["ts_monitor"] = None
#params["ts_view"] = None

problem = firedrake_ts.DAEProblem(F, u, u_t, (0.0, 2 * dt))
solver = firedrake_ts.DAESolver(problem, solver_parameters=params)

if pc in ["fieldsplit", "ilu"]:
    sigma = 100
    # PC for the Schur complement solve
    trial = TrialFunction(V)
    test = TestFunction(V)
    mass = assemble(inner(trial, test) * dx).M.handle
    a = 1
    c = (dt * lmbda) / (1 + dt * sigma)
    hats = assemble(
        sqrt(a) * inner(trial, test) * dx
        + sqrt(c) * inner(grad(trial), grad(test)) * dx
    ).M.handle

    from firedrake.petsc import PETSc

    ksp_hats = PETSc.KSP()
    ksp_hats.create()
    ksp_hats.setOperators(hats)
    opts = PETSc.Options()

    opts["ksp_type"] = inner_ksp
    opts["ksp_max_it"] = maxit
    opts["pc_type"] = "hypre"
    ksp_hats.setFromOptions()

    class SchurInv(object):
        def mult(self, mat, x, y):
            tmp1 = y.duplicate()
            tmp2 = y.duplicate()
            ksp_hats.solve(x, tmp1)
            mass.mult(tmp1, tmp2)
            ksp_hats.solve(tmp2, y)

    pc_schur = PETSc.Mat()
    pc_schur.createPython(mass.getSizes(), SchurInv())
    pc_schur.setUp()
    pc = solver.snes.ksp.pc
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, pc_schur)

ts = solver.ts
ts.setTimeStep(dt)

solver.solve()
