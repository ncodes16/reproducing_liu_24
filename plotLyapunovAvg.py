from load_data import load, load_parameters
import numpy as np
import matplotlib.pyplot as plt
from ks_liu2024 import Lx, Nx, parula_cmap, start_sim_time

# quasi = True
# stop_sim_time = 500
# epsilon = 0.21
# seed = 1
# transient_time = 400
# T_f = 10
quasi, seed, epsilon, T_f, stop_sim_time, transient_time, path = load_parameters()
rng = np.random.default_rng(seed=seed)
theta = rng.uniform()

#load data
data = load(quasi, seed, epsilon, T_f, stop_sim_time, transient_time)

t_list = data['t']
lyapunov_list = data['lyapunov']


plt.figure(figsize=(7,7))
plt.plot(t_list, lyapunov_list)
plt.xlabel('t')
plt.ylabel("Average MLE")
plt.tight_layout()
plt.savefig(f"{path}/{"quasi_lyapunov.png" if quasi else "periodic_lyapunov.png"}")
plt.close()