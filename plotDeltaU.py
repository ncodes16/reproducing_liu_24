from load_data import load, load_parameters
import numpy as np
import matplotlib.pyplot as plt
from ks_liu2024 import Lx, Nx, parula_cmap, start_sim_time

# stop_sim_time = 105
# epsilon = 0.0426
# seed = 1
# transient_time = 11
# T_f = 0.5
quasi, seed, epsilon, T_f, stop_sim_time, transient_time, path = load_parameters()
rng = np.random.default_rng(seed=seed)
theta = rng.uniform()

periodic_data = load(False, seed, epsilon, T_f, stop_sim_time, transient_time)
quasi_data = load(True, seed, epsilon, T_f, stop_sim_time, transient_time)

t_list = periodic_data['t']
u_p = periodic_data['u']
u_q = quasi_data['u']

x_grid = np.linspace(0, Lx, Nx, endpoint=False)
plt.figure(figsize=(6,4))
plt.pcolormesh(np.array(t_list), x_grid, np.array(u_p).T - np.array(u_q).T, cmap='bwr', rasterized=True,)
colorbar = plt.colorbar()
plt.xlim(start_sim_time, stop_sim_time)
plt.ylim(0, Lx)
plt.xlabel('t')
plt.ylabel('x')
plt.title(f'fKSe, (epsilon,theta)=({epsilon},{theta})')
plt.tight_layout()
plt.savefig(f"{path}/periodic_minus_quasi.png")
plt.close()