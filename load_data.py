import numpy as np
import sys
from pathlib import Path


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _runs_dir() -> Path:
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    return runs_dir


def load(quasi: bool, seed: int, epsilon: float, T_f: float, simtime: int | float, transtime: int | float) -> np.ndarray:
    runs_dir = _runs_dir()
    filename = f"QP={quasi}_data_seed={seed}_epsilon={epsilon}_Tf={T_f}_simtime={simtime}_transInt={int(transtime)}.npz"
    path = runs_dir / filename
    try:
        data = np.load(path)
    except FileNotFoundError:
        legacy_path = runs_dir / f"data_seed={seed}_epsilon={epsilon}_Tf={T_f}_simtime={simtime}_transInt={int(transtime)}.npz"
        try:
            data = np.load(legacy_path)
        except FileNotFoundError:
            print("Data with these parameters does not exist. Go run the simulation on ks_liu2024.py to create it.")
            sys.exit()
    except (OSError, ValueError):
        print("Data corrupted. Go run the simulation on ks_liu2024.py to repair it.")
        sys.exit()
    return data


def load_parameters() -> tuple[bool, int, float, float, int | float, int | float, str]:
    if len(sys.argv) > 1:
        a = sys.argv
        if len(a) > 7:
            parameters = (_parse_bool(a[1]), int(a[2]), float(a[3]), float(a[4]), float(a[5]), float(a[6]), a[7])
        else:
            parameters = (_parse_bool(a[1]), int(a[2]), float(a[3]), float(a[4]), float(a[5]), float(a[6]))
        return parameters
    else:
        print("No args provided, sim failed")
        sys.quit()

if __name__ == '__main__':
    print("This is a module, do not run this file.")