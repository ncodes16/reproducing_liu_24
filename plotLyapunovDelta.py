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
lyapunov_list = data['lyapunov']

stop_sim_time_2 = 500
epsilon_2 = 0.0426
seed_2 = 1
rng_2 = np.random.default_rng(seed=seed)
theta_2 = rng.uniform()
transient_time_2 = 100
T_f_2 = 1

#load data
try:
    data_2 = np.load(f"runs/data_seed={seed_2}_epsilon={epsilon_2}_Tf={T_f_2}_simtime={stop_sim_time_2}_transInt={transient_time_2}.npz")
except FileNotFoundError:
    print("Data with these parameters does not exist. Go run the simulation on ks_liu2024.py to create it.")
    sys.exit()
except OSError, ValueError:
    print("Data corrupted. Go run the simulation on ks_liu2024.py to repair it.")
    sys.exit()
lyapunov_list_2 = data_2['lyapunov']


plt.figure(figsize=(7,7))
plt.plot(t_list, lyapunov_list - lyapunov_list_2)
plt.xlabel('t')
plt.ylabel("Periodic avg. MLE - Quasi-Periodic avg. MLE")
plt.tight_layout()
plt.show()