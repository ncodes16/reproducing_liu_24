
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

seed = 1
rng = np.random.default_rng(seed=seed)

# Official Parula colormap RGB values
parula_colors = [
    (0.2081, 0.1663, 0.5292),
    (0.2116, 0.1898, 0.5777),
    (0.2123, 0.2138, 0.6270),
    (0.2081, 0.2386, 0.6771),
    (0.1959, 0.2645, 0.7279),
    (0.1707, 0.2919, 0.7792),
    (0.1253, 0.3242, 0.8303),
    (0.0591, 0.3598, 0.8683),
    (0.0117, 0.3875, 0.8819),
    (0.0060, 0.4086, 0.8828),
    (0.0165, 0.4266, 0.8786),
    (0.0329, 0.4430, 0.8720),
    (0.0498, 0.4586, 0.8641),
    (0.0629, 0.4737, 0.8554),
    (0.0723, 0.4887, 0.8467),
    (0.0779, 0.5040, 0.8384),
    (0.0793, 0.5200, 0.8312),
    (0.0749, 0.5375, 0.8263),
    (0.0641, 0.5570, 0.8240),
    (0.0488, 0.5772, 0.8228),
    (0.0343, 0.5966, 0.8199),
    (0.0265, 0.6137, 0.8135),
    (0.0239, 0.6287, 0.8038),
    (0.0231, 0.6418, 0.7913),
    (0.0228, 0.6535, 0.7768),
    (0.0267, 0.6642, 0.7607),
    (0.0384, 0.6743, 0.7436),
    (0.0590, 0.6838, 0.7254),
    (0.0843, 0.6928, 0.7062),
    (0.1133, 0.7015, 0.6859),
    (0.1453, 0.7098, 0.6646),
    (0.1801, 0.7177, 0.6424),
    (0.2178, 0.7250, 0.6193),
    (0.2586, 0.7317, 0.5954),
    (0.3022, 0.7376, 0.5712),
    (0.3482, 0.7424, 0.5473),
    (0.3953, 0.7459, 0.5244),
    (0.4420, 0.7481, 0.5033),
    (0.4871, 0.7491, 0.4840),
    (0.5300, 0.7491, 0.4661),
    (0.5709, 0.7485, 0.4494),
    (0.6099, 0.7473, 0.4337),
    (0.6473, 0.7456, 0.4188),
    (0.6834, 0.7435, 0.4044),
    (0.7184, 0.7411, 0.3905),
    (0.7525, 0.7384, 0.3768),
    (0.7858, 0.7356, 0.3633),
    (0.8185, 0.7327, 0.3498),
    (0.8507, 0.7299, 0.3360),
    (0.8824, 0.7274, 0.3217),
    (0.9139, 0.7258, 0.3063),
    (0.9450, 0.7261, 0.2886),
    (0.9739, 0.7314, 0.2666),
    (0.9938, 0.7455, 0.2403),
    (0.9990, 0.7653, 0.2164),
    (0.9955, 0.7861, 0.1967),
    (0.9880, 0.8066, 0.1794),
    (0.9789, 0.8271, 0.1633),
    (0.9697, 0.8481, 0.1475),
    (0.9626, 0.8705, 0.1309),
    (0.9589, 0.8949, 0.1132),
    (0.9598, 0.9218, 0.0948),
    (0.9661, 0.9514, 0.0755),
    (0.9763, 0.9831, 0.0538)
]

# Create the colormap
parula_cmap = LinearSegmentedColormap.from_list('parula', parula_colors, N=256)

#parameters
Lx = 32 * np.pi
Nx = 256
epsilon = 0.0426#vary to check against multiple schemes
k_0 = 4
T_f = 10
T_f_2 = np.sqrt(2) * 10
omega = 2 * np.pi / Lx
dealias = 3/2
# start_sim_time = 2.98e5
# stop_sim_time = sim_start_time + 2000
start_sim_time = 0
stop_sim_time =2000
timestepper = d3.SBDF2
timestep = 1e-2
renorm_int = 50
transient_steps = 99 // timestep
lyapunov_sum = 0
dtype = np.float64

#bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.RealFourier(xcoord, size = Nx, bounds = (0, Lx), dealias = dealias)

#fields
u = dist.Field(name = 'u', bases = xbasis)
du = dist.Field(name = 'du', bases = xbasis)

#substitutions
dx = lambda A: d3.Differentiate(A, xcoord)

# Create x as a Field for use in equations
x = dist.Field(bases=xbasis)
x['g'] = dist.local_grid(xbasis)
t = dist.Field(name="t")


# Make pi available for equations
pi = np.pi

#problem
problem = d3.IVP([u, du], time=t, namespace = locals())
problem.add_equation("dt(u) + dx(dx(u)) + dx(dx(dx(dx(u)))) = -u * dx(u) + epsilon * sin(k_0 * omega * x) * (sin((2 * pi/T_f) * t))")
# problem.add_equation("dt(u) + dx(dx(u)) + dx(dx(dx(dx(u)))) = -u * dx(u) + epsilon * sin(k_0 * omega * x) * (sin((2 * pi/T_f) * t) + sin((2 * pi/T_f_2) * t))")
#perturbation equation
problem.add_equation("dt(du) + dx(dx(dx(dx(du)))) + dx(dx(du)) = -dx(u)*du - dx(du) * u")



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
du['g'] = np.random.standard_normal(du['g'].shape)

dx_grid = Lx / Nx

def norm(vector):
    return np.sqrt(np.sum(vector['g']**2) * dx_grid)

d0 = 1.0
scale = d0 / norm(du)
du['g'][:] = du['g'] * scale

if __name__ == "__main__":
    try:
        data = np.load(f"runs/data_seed={seed}_epsilon={epsilon}_Tf={T_f}_simtime={stop_sim_time}_transInt={round(transient_steps * timestep)}.npz")
        if input("File for this data already exists. Are you sure you want to re-run simulation? [y/n] ") == 'n':
            sys.exit()
    except FileNotFoundError:
        pass
    #solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    #main loop
    u_list = [u['g'][:Nx].copy()]
    du_list = [du['g'][:Nx].copy()]
    t_list = [solver.sim_time]
    lyapunov_list = [0.0]
    lyapunov_inst_list = [0.0]
    last_lyapunov = 0.0
    inst_lambda = 0.0
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 10000 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
        if solver.iteration % 5 == 0:
            u_list.append(u['g'][:Nx].copy())
            du_list.append(du['g'][:Nx].copy())
            t_list.append(solver.sim_time)
            lyapunov_list.append(last_lyapunov)
            lyapunov_inst_list.append(inst_lambda)
        if solver.iteration % renorm_int == 0:
            growth_factor = norm(du) / d0
            inst_lambda = np.log(growth_factor) / (renorm_int * timestep)

            if solver.iteration > transient_steps:
                lyapunov_sum += inst_lambda
                active_steps = solver.iteration - transient_steps
                current_renorm_count = active_steps // renorm_int
                if current_renorm_count != 0:
                    last_lyapunov = (lyapunov_sum / current_renorm_count)
                else:
                    last_lyapunov = 0.0
                # if solver.iteration % (renorm_int * 100) == 0:
                #     print(f"Avg max lambda: {last_lyapunov:.5f}")
            du['g'] /= growth_factor
    # df = pd.read_csv("T_f_3.csv", header=None)
    # graph_data = df.to_numpy()

    np.savez_compressed(f"runs/data_seed={seed}_epsilon={epsilon}_Tf={T_f}_simtime={stop_sim_time}_transInt={round(transient_steps * timestep)}", u=np.array(u_list), du=np.array(du_list), t=np.array(t_list), lyapunov=np.array(lyapunov_list))
    print("Data generated.")
# # Plot
# x_grid = np.linspace(0, Lx, Nx, endpoint=False)
# plt.figure(figsize=(3.73*4,0.66*4))
# plt.pcolormesh(np.array(t_list), x_grid, np.array(u_list).T, cmap=parula_cmap, rasterized=True,)
# plt.xlim(start_sim_time, stop_sim_time)
# plt.ylim(0, Lx)
# plt.xlabel('t')
# plt.ylabel('x')
# plt.title(f'fKSe, (epsilon,theta)=({epsilon},{theta})')
# plt.tight_layout()
# # plt.figure(figsize=(4, 4))
# # u_last = np.asarray(u_list)[-1]
# # x_vals = graph_data[:, 0]
# # y_vals = graph_data[:, 1]

# # n = min(len(u_last), len(x_vals), len(y_vals))
# # u_last = u_last[:n]
# # x_vals = x_vals[:n]
# # y_vals = y_vals[:n]

# # plt.plot(x_grid, np.array(u_list)[-1] - y_vals)
# # plt.plot(t_list, lyapunov_list)
# # plt.xlabel('t')
# # plt.ylabel("Average MLE")
# # print(f"Average error: {np.mean((np.array(u_list)[-1] - y_vals)):2f} ({np.abs(np.mean((np.array(u_list)[-1] - y_vals) * 100 / y_vals)):2f}%)")
# # plt.plot(x_vals, y_vals, label='CSV data')
# # plt.plot(x_vals, u_last, label='Simulation')
# # plt.xlim(x_vals.min(), x_vals.max())
# # plt.ylim(-20, 20)
# # plt.xlabel('x')
# # plt.ylabel('u(x,t)')
# # plt.legend()
# # plt.tight_layout()
# plt.show()
