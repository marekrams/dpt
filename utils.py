import numpy as np
from yastn.operators import SpinlessFermions
import yastn.tn.mps as mps

def load_psi(file, sym, message = "", **config_kwargs):
    

    psi0data  = np.load( file, allow_pickle=True).item()
    ops = SpinlessFermions(sym=sym, **config_kwargs)
    psi0 = mps.load_from_dict(ops.config, psi0data)

    return psi0