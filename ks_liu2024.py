

import numpy as np
rng = np.random.default_rng(seed=3500)
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#parameters
Lx = 32 * np.pi
Nx = 256
epsilon = 20 #vary to check against multiple schemes
k_0 = 4
T_f = 20 * np.sqrt(2)
omega = 2 * np.pi / Lx
dealias = 3/2
# start_sim_time = 2.98e5
# stop_sim_time = sim_start_time + 2000
start_sim_time = 0
stop_sim_time = 500
timestepper = d3.SBDF2
timestep = 1e-2
dtype = np.float64

#bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.RealFourier(xcoord, size = Nx, bounds = (0, Lx), dealias = dealias)

#fields
u = dist.Field(name = 'u', bases = xbasis)

#substitutions
dx = lambda A: d3.Differentiate(A, xcoord)

# Create x as a Field for use in equations
x = dist.Field(bases=xbasis)
x['g'] = dist.local_grid(xbasis)
t = dist.Field(name="t")


# Make pi available for equations
pi = np.pi

#problem
problem = d3.IVP([u], time=t, namespace = locals())
# problem.add_equation("dt(u) + dx(dx(u)) + dx(dx(dx(dx(u)))) = -u * dx(u) + epsilon * sin(k_0 * omega * x)")
problem.add_equation("dt(u) + dx(dx(u)) + dx(dx(dx(dx(u)))) = -u * dx(u) + epsilon * sin(k_0 * omega * x) * sin((2 * pi/T_f) * t)")
#for now keep forcing constant in time...change after


#initial condition
alpha = 8.1e-3
beta = 0.74
g = 9.8
w_0 = g / 16
theta = rng.uniform()
k = np.fft.rfftfreq(Nx, Lx / (2 * np.pi))
S = alpha * g * g * k ** 5 * np.exp(-beta * (k * w_0) ** 4)
u_0_hat = np.exp(2 * np.pi * 1j * theta) * S
u['g'] = np.fft.irfft(u_0_hat)

#solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#main loop
u_list = [u['g', 1].copy()]
t_list = [solver.sim_time]
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 10000 == 0:
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    if solver.iteration % 25 == 0:
        u_list.append(u['g',1].copy())
        t_list.append(solver.sim_time)
# Plot
x_grid = np.linspace(0, Lx, Nx, endpoint=False)
plt.figure(figsize=(6,4))
plt.pcolormesh(np.array(t_list), x_grid, np.array(u_list).T, cmap='viridis', rasterized=True, )
plt.xlim(start_sim_time, stop_sim_time)
plt.ylim(0, Lx)
plt.xlabel('t')
plt.ylabel('x')
plt.title(f'fKSe, (epsilon,theta)=({epsilon},{theta})')
plt.tight_layout()
plt.show()
