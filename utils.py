import numpy as np
from yastn.operators import SpinlessFermions
import yastn.tn.mps as mps

def load_psi(file, sym, message = ""):
    print(message)
    psi0data  = np.load( file, allow_pickle=True).item()
    ops = SpinlessFermions(sym=sym)
    psi0 = mps.load_from_dict(ops.config, psi0data)

    return psi0