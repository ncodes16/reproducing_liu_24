import numpy as np
import matplotlib.pyplot as plt
from ks_liu2024 import Lx, Nx, parula_cmap, start_sim_time
from load_data import load, load_parameters

# stop_sim_time = 105
# epsilon = 0.0426
# seed = 1
# transient_time = 11
# T_f = 0.5
quasi, seed, epsilon, T_f, stop_sim_time, transient_time, path = load_parameters()
rng = np.random.default_rng(seed=seed)
theta = rng.uniform()
#load data
periodic_data = load(False, seed, epsilon, T_f, stop_sim_time, transient_time)

u_p = periodic_data['u']

u_hat_p = np.fft.rfft(u_p, axis=1)
e_hat_p = np.abs(u_hat_p) ** 2
wavenumbers = np.fft.rfftfreq(n=Nx, d=Lx / Nx) * Lx
e_hat_avg_p = np.mean(e_hat_p, axis=0) / ( Nx)
# print(e_hat_avg_1)

# # u_time_averaged = np.mean(u_list, axis=0)
# # wavenumbers = np.linspace(-Nx/2, Nx/2, Nx)
# # e_spectra = np.fft.fft(0.5 * u_time_averaged * u_time_averaged, n=Nx).real

# # print(e_spectra)


#load data
quasi_data = load(True, seed, epsilon, T_f, stop_sim_time, transient_time)

u_list_q = quasi_data['u']

u_hat_q = np.fft.rfft(u_list_q, axis=1)
e_hat_q = np.abs(u_hat_q) ** 2
e_hat_avg_q = np.mean(e_hat_q, axis=0) / ( Nx)


plt.figure(figsize=(7,7))
plt.loglog(wavenumbers, e_hat_avg_p,linestyle='--', color='blue', label="Periodic")
plt.loglog(wavenumbers, e_hat_avg_q,linestyle='--', color='red', label="Quasi-Periodic")
# plt.plot(wavenumbers_1, e_hat_avg_2/e_hat_avg_1, linestyle='--', color='green', label=r"Q/P")
plt.xlabel(r'$k$')
plt.ylabel(r"$S(k)$")
plt.legend()
plt.ylim(10**(-20), 10**2)
plt.yticks([10**(-20), 10**(-10), 10**0])
# plt.tight_layout()
plt.savefig(f"{path}/energy_spectrum")
plt.close()