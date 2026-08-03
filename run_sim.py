import multiprocessing  
import time
import subprocess
import os

stop_sim_time = [500]
epsilon = [0.0426]
seed = [1]
transient_time = [100]
T_f = [0.5]


total_trials = len(stop_sim_time) * len(epsilon) * len(seed) * len(T_f)
parameters = []
for i, stoptime  in enumerate(stop_sim_time):
    for eps in epsilon:
        for sd in seed:
            for t in T_f:
                parameters.append((sd, eps, t, stoptime, transient_time[i]))
#transient time is correlated with stop time; each transient time goes with its specific stop time, there is no mixing and matching

def run_sim(parameter:tuple[float]):
    sim_ran_periodic = subprocess.run(
        ["python3", "ks_liu2024.py", "False", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4])],
        capture_output=True,
        text=True,
        check=True
    )
    sim_ran_quasi = subprocess.run(
        ["python3", "ks_liu2024.py", "True", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4])],
        capture_output=True,
        text=True,
        check=True
    )
    #save results
    os.makedirs(f"image_comparisons/e_{parameter[1]}/tf_{parameter[2]}/seed_{parameter[0]}/stoptime_{parameter[3]}", exist_ok=True)
    path = f"image_comparisons/e_{parameter[1]}/tf_{parameter[2]}/seed_{parameter[0]}/stoptime_{parameter[3]}"
    u_p = subprocess.run(
            ["python3", "plotU.py", "False", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4]), path],
            capture_output=True,
            text=True,
            check=True
        )
    u_q = subprocess.run(
            ["python3", "plotU.py", "True", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4]), path],
            capture_output=True,
            text=True,
            check=True
        )
    u_p_q = subprocess.run(
            ["python3", "plotDeltaU.py", "True", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4]), path],
            capture_output=True,
            text=True,
            check=True
        )
    l_p = subprocess.run(
            ["python3", "plotLyapunovAvg.py", "False", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4]), path],
            capture_output=True,
            text=True,
            check=True
        )
    l_q = subprocess.run(
            ["python3", "plotLyapunovAvg.py", "True", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4]), path],
            capture_output=True,
            text=True,
            check=True
        )
    l_p_q = subprocess.run(
            ["python3", "plotLyapunovDelta.py", "True", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4]), path],
            capture_output=True,
            text=True,
            check=True
        )
    s_k = subprocess.run(
            ["python3", "plotEnergySpectrum.py", "True", str(parameter[0]), str(parameter[1]), str(parameter[2]), str(parameter[3]), str(parameter[4]), path],
            capture_output=True,
            text=True,
            check=True
        )
    


if __name__ == '__main__':
    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_sim, parameters)
    end_time = time.time()
    hours, r = divmod (end_time - start_time, 3600)
    minutes, seconds = divmod (r, 60)
    print(f"Simulation finished in {hours:02d}:{minutes:02d}:{seconds:05.2f}")