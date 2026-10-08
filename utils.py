import numpy as np
from yastn.operators import SpinlessFermions
from yastn import from_dict
from yastn.tn import mps

def load_psi(file, sym, message = "", **config_kwargs):
    

    psi0data  = np.load( file, allow_pickle=True).item()
    ops = SpinlessFermions(sym=sym, **config_kwargs)

    try:
        psi0 = from_dict(psi0data, ops.config )
        
    except Exception as e:
        print(f"Error loading psi0 from {file}: {e}, legacy fallback to mps.load_from_dict")
        psi0 = mps.load_from_dict(ops.config, psi0data)
        

    return psi0