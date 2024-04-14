from pathlib import Path
import json, hashlib

PARAM_HAMILTONIAN = ["basis", "order", "NW", "muL", "muR", "muS", "dmuS", "vS", "U", "w0", "sym"]

PARAM_GS = PARAM_HAMILTONIAN.copy()
PARAM_GS.extend(["occS", "occLR"])

PARAM_QUENCH = PARAM_GS.copy()
PARAM_QUENCH.extend(["D0", "D1", "dt", "tolS", "time"])

def param_hamiltonian(param):
    return {k: param[k] for k in PARAM_HAMILTONIAN}

def fname_hamiltonian(param):
    return add_fhash(Path("./results/hamiltonians"), param_hamiltonian(param))

def param_gs(param):
    return {k: param[k] for k in PARAM_GS}

def fname_gs(param):
    return add_fhash(Path("./results/gs"), param_gs(param))

def param_quench(param):
    return {k: param[k] for k in PARAM_QUENCH}

def fname_quench(param0, param1):
    return add_fhash2(Path("./results/quench"), param_gs(param0), param_quench(param1))

def add_fhash(path, param):
    path.mkdir(parents=True, exist_ok=True)
    s = hashlib.sha224(json.dumps(param, sort_keys=True).encode('utf-8')).hexdigest()
    return path / f"{s}.npy"

def add_fhash2(path, param0, param1):
    path.mkdir(parents=True, exist_ok=True)
    s = hashlib.sha224(json.dumps(param0, sort_keys=True).encode('utf-8') + json.dumps(param1, sort_keys=True).encode('utf-8')).hexdigest()
    return path / f"{s}.npy"