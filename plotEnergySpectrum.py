import numpy as np
import matplotlib.pyplot as plt
from ks_liu2024 import Lx, Nx, parula_cmap, start_sim_time, timestep
import sys

stop_sim_time_1 = 35
epsilon_1 = 0.0426
seed_1 = 1
rng_1 = np.random.default_rng(seed=seed_1)
theta_1 = rng_1.uniform()
transient_time_1 = 10
T_f_1 = 0.5

#load data
try:
    data_1 = np.load(f"runs/data_seed={seed_1}_epsilon={epsilon_1}_Tf={T_f_1}_simtime={stop_sim_time_1}_transInt={transient_time_1}.npz")
except FileNotFoundError:
    print("Data with these parameters does not exist. Go run the simulation on ks_liu2024.py to create it.")
    sys.exit()
except OSError, ValueError:
    print("Data corrupted. Go run the simulation on ks_liu2024.py to repair it.")
    sys.exit()

u_list_1 = data_1['u']

u_hat_1 = np.fft.rfft(u_list_1, axis=1)
e_hat_1 = 0.5 * u_hat_1 * u_hat_1.conj() / Nx
e_hat_avg_1 = np.mean(e_hat_1, axis=0)

wavenumbers_1 = np.fft.rfftfreq(n=Nx, d=Lx/Nx) * Lx
# u_time_averaged = np.mean(u_list, axis=0)
# wavenumbers = np.linspace(-Nx/2, Nx/2, Nx)
# e_spectra = np.fft.fft(0.5 * u_time_averaged * u_time_averaged, n=Nx).real

# print(e_spectra)

stop_sim_time_2 = 35
epsilon_2 = 0.0426
seed_2 = 1
rng_2 = np.random.default_rng(seed=seed_2)
theta_2 = rng_2.uniform()
transient_time_2 = 9
T_f_2 = 0.5

#load data
try:
    data_2 = np.load(f"runs/data_seed={seed_2}_epsilon={epsilon_2}_Tf={T_f_2}_simtime={stop_sim_time_2}_transInt={transient_time_2}.npz")
except FileNotFoundError:
    print("Data with these parameters does not exist. Go run the simulation on ks_liu2024.py to create it.")
    sys.exit()
except OSError, ValueError:
    print("Data corrupted. Go run the simulation on ks_liu2024.py to repair it.")
    sys.exit()

u_list_2 = data_2['u']

u_hat_2 = np.fft.rfft(u_list_2, axis=1)
e_hat_2 = 0.5 * u_hat_2 * u_hat_2.conj() / Nx
e_hat_avg_2 = np.mean(e_hat_2, axis=0)

wavenumbers_2 = np.fft.rfftfreq(n=Nx, d=Lx/Nx) * Lx

plt.figure(figsize=(7,7))
plt.loglog(wavenumbers_1, e_hat_avg_1,linestyle='--', color='blue', label="Periodic")
plt.loglog(wavenumbers_2, e_hat_avg_2,linestyle='--', color='red', label="Quasi-Periodic")
plt.xlabel(r'$k$')
plt.ylabel(r"$S(k)$")
plt.legend()
plt.ylim(10**(-20), 10**2)
plt.yticks([10**(-20), 10**(-10), 10**0])
# plt.tight_layout()
plt.show()