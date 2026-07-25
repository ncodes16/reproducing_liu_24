import sys
import numpy as np
import matplotlib.pyplot as plt
from ks_liu2024 import Lx, Nx, parula_cmap, start_sim_time

stop_sim_time = 500
epsilon = 0.21
seed = 1
rng = np.random.default_rng(seed=seed)
theta = rng.uniform()
transient_time = 400
T_f = 70

#load data
try:
    data = np.load(f"runs/data_seed={seed}_epsilon={epsilon}_Tf={T_f}_simtime={stop_sim_time}_transInt={transient_time}.npz")
except FileNotFoundError:
    print("Data with these parameters does not exist. Go run the simulation on ks_liu2024.py to create it.")
    # print(f"runs/data_seed={seed}_epsilon={epsilon}_Tf={T_f}_simtime={stop_sim_time}_transInt={transient_time}")
    sys.exit()
except OSError, ValueError:
    print("Data corrupted. Go run the simulation on ks_liu2024.py to repair it.")
    sys.exit()
t_list = data['t']
u_list = data['u']

x_grid = np.linspace(0, Lx, Nx, endpoint=False)
plt.figure(figsize=(6,4))
plt.pcolormesh(np.array(t_list), x_grid, np.array(u_list).T, cmap=parula_cmap, rasterized=True,)
colorbar = plt.colorbar()
plt.xlim(start_sim_time, stop_sim_time)
plt.ylim(0, Lx)
plt.xlabel('t')
plt.ylabel('x')
plt.title(f'fKSe, (epsilon,theta)=({epsilon},{theta})')
plt.tight_layout()
plt.show()