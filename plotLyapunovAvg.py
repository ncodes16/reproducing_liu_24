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
except OSError, ValueError:
    print("Data corrupted. Go run the simulation on ks_liu2024.py to repair it.")
t_list = data['t']
lyapunov_list = data['lyapunov']


plt.figure(figsize=(7,7))
plt.plot(t_list, lyapunov_list)
plt.xlabel('t')
plt.ylabel("Average MLE")
plt.tight_layout()
plt.show()