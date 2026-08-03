from load_data import load, load_parameters
import numpy as np
import matplotlib.pyplot as plt

# stop_sim_time = 500
# epsilon = 0.0426
# seed = 1
# transient_time = 100
# T_f = 0.5
quasi, seed, epsilon, T_f, stop_sim_time, transient_time, path = load_parameters()
rng = np.random.default_rng(seed=seed)
theta = rng.uniform()

#load data
periodic_data = load(False, seed, epsilon, T_f, stop_sim_time, transient_time)
quasi_data = load(True, seed, epsilon, T_f, stop_sim_time, transient_time)

lyapunov_list_p = periodic_data['lyapunov']
lyapunov_list_q = quasi_data['lyapunov']
t_list = periodic_data['t']

plt.figure(figsize=(7,7))
plt.plot(t_list, lyapunov_list_p - lyapunov_list_q)
plt.xlabel('t')
plt.ylabel("Periodic avg. MLE - Quasi-Periodic avg. MLE")
plt.tight_layout()
plt.savefig(f"{path}/periodic_mle_minus_quasi_mle")
plt.close()