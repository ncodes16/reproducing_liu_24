import sys
import numpy as np
import matplotlib.pyplot as plt

stop_sim_time = 500
epsilon = 0.0426
seed = 1
rng = np.random.default_rng(seed=seed)
theta = rng.uniform()
transient_time = 100
T_f = 0.5

#load data
try:
    data = np.load(f"runs/data_seed={seed}_epsilon={epsilon}_Tf={T_f}_simtime={stop_sim_time}_transInt={transient_time}.npz")
except FileNotFoundError:
    print("Data with these parameters does not exist. Go run the simulation on ks_liu2024.py to create it.")
    sys.exit()
except OSError, ValueError:
    print("Data corrupted. Go run the simulation on ks_liu2024.py to repair it.")
    sys.exit()
t_list = data['t']
lyapunov_inst_list = data['lyapunov_inst']


plt.figure(figsize=(14,7))
plt.plot(t_list, lyapunov_inst_list)
plt.xlabel('t')
plt.ylabel("MLE")
plt.tight_layout()
plt.show()