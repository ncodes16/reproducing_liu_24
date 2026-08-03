from load_data import load, load_parameters
import numpy as np
import matplotlib.pyplot as plt

# quasi = True
# stop_sim_time = 500
# epsilon = 0.0426
# seed = 1
# transient_time = 100
# T_f = 0.5

quasi, seed, epsilon, T_f, stop_sim_time, transient_time, path = load_parameters()
rng = np.random.default_rng(seed=seed)
theta = rng.uniform()

#load data
data = load(quasi, seed, epsilon, T_f, stop_sim_time, transient_time)
t_list = data['t']
lyapunov_inst_list = data['lyapunov_inst']


plt.figure(figsize=(14,7))
plt.plot(t_list, lyapunov_inst_list)
plt.xlabel('t')
plt.ylabel("MLE")
plt.tight_layout()
plt.show()