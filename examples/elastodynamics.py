# Adaptive Alpha integration for a square beam
import firedrake_ts
from firedrake import *

n = 10
mesh = UnitSquareMesh(n, n)

V = VectorFunctionSpace(mesh, "CG", 1)
W = V*V

E = 1.0
nu = 0.3
mu = Constant(E/2/(1+nu))
lmbda = Constant(E*nu/(1+nu)/(1-2*nu))


def eps(v):
    return sym(grad(v))
def sigma(v):
    return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

t = Constant(0.0)
z = Function(W)
zt = Function(W) # Time Derivatives
q,v = TestFunctions(W)
u,ut = split(z)
w,wt = split(zt)

rho = Constant(1.0)
F1 = inner(w, q)*dx-inner(ut,q)*dx 
F2 = rho*inner(wt,v)*dx + inner(sigma(u),eps(v)) * dx - dot(as_vector([0, 1e-2*cos(t)]),v)*ds(2)
F = F1+F2
outfile = File("output/elastodyn.pvd")
outfile.write(project(u, V, name="Disp"), time=0.0)

params={"ts_type":"alpha"}
params["ts_adapt_type"]="basic"
params["ts_adapt_monitor"]=None

def ts_monitor(ts, steps, time, X):
    print(time)
    outfile.write(project(u, V, name="Disp"), time=time)

bc = DirichletBC(W.sub(0), Constant((0,0)), 1)
problem = firedrake_ts.DAEProblem(F, z, zt, (0.0, 5),time=t,bcs=bc)
solver = firedrake_ts.DAESolver(problem, monitor_callback=ts_monitor, solver_parameters=params)

timestep = 1.0 / (n)
ts = solver.ts
ts.setTimeStep(timestep)
solver.solve()
