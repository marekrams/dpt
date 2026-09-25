from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import yastn.tn.mps as mps
import json
from pathlib import Path
from hamiltonians import local_operators, Hamiltonian_dpt_position, Hamiltonian_dpt_momentum, Hamiltonian_dpt_mixed, imaginary_time_evolution
from auxilliary import merge_sites, op1site, get_current, gpu_report
from sites import L, S, D, R, order_sites
from os import getcwd, mkdir
import os
#import yastn.backend.backend_torch as backend
#import cupy as cp

from utils import *
from threadpoolctl import threadpool_info
from copy import deepcopy
import time
from pprint import pprint


def fprint(*msg):
    print(*msg, flush=True)

def Hamiltonian( key):
    ham = {"position": Hamiltonian_dpt_position,
               "mixed": Hamiltonian_dpt_mixed,
               "momentum": Hamiltonian_dpt_momentum}
    
    return ham[key]


def init_occupations(mapping, NW, NS):
    """ Initial guess of occupations before DMRG. """
    occ = {}
    assert NW % 2 == 0, "Assume even NW for convinience"
    assert NS % 2 == 0, "Assume even NS"

    if mapping == "position":
        occ[D(1)] = 0
        for k in range(1, NS + 1):
            occ[S(k)] = 0.5
        for k in range(1, NW + 1):
            occ[L(k)] = 0.5
            occ[R(k)] = 0.5

    if mapping == "mixed":
        NW1 = NW + 1
        occ[D(1)] = 0
        for k in range(1, NS + 1):
            occ[S(k)] = 0.5
        for k in range(1, NW1):
            occ[L(k)] = np.heaviside(k - NW1 / 2, 0.5)
            occ[R(k)] = np.heaviside(k - NW1 / 2, 0.5)

    if mapping == "momentum":
        NW1 = NW + NS // 2 + 1
        occ[D(1)] = 0
        for k in range(1, NW1):
            occ[L(k)] = np.heaviside(k - NW1 / 2, 0.5)
            occ[R(k)] = np.heaviside(k - NW1 / 2, 0.5)

    assert sum(occ.values()) == NW + NS // 2, "We should have half-filling."
    return occ


def initial_state(NW, NS, U, muL, muR, vS0, alpha, mapping, order, merge, sym, D_total, curpath = '', muDs=[0, 10000], max2 = 4, max1 = 256, sites = [], Hdebug = False, rtype = 'sin-transform', Lambda = None, imaginary_time = False, **config_kwargs):

    print("init config: ", config_kwargs)

    s2i = {s: i for i, s in enumerate(sites)}
    if os.path.isfile( f'{curpath}init.npy'):
        psi = load_psi(f'{curpath}init.npy', sym, message = "loading init", **config_kwargs)
        energy = np.loadtxt(f'{curpath}initenergy' )
        return psi, energy
    

    init_occ = init_occupations(mapping, NW, NS)
    #pprint(sites)
    with open(f'{curpath}sites', 'w') as f:
        np.savetxt(f, sites, fmt = '%s')

    if sym == 'U1':
        n_profile = [NW + NS // 2 - x for x in accumulate([init_occ[site] for site in sites], initial=0)]
    elif sym == 'Z2':
        n_profile = 0
    else:
        raise ValueError("Only sym = 'U1' or 'Z2' suported.")

    if alpha is None:
        effU = U
    else:
        effU = U * (2 * alpha - 1)

    H0, M = Hamiltonian(mapping)(NW, NS, muL, muR, muDs, vS0, effU, Hdebug = Hdebug, sym=sym, order=sites, rtype = rtype, Lambda = Lambda, **config_kwargs)
    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym, **config_kwargs)

    if Hdebug:
        with open(f'{curpath}HamInit', 'w') as f:
            np.savetxt(f, M, fmt = '%s')

    psi = mps.random_mps(H0, n=n_profile, D_total=16, sigma=1, distribution='normal')

    H0 = merge_sites(H0, s2i, merge)
    psi = merge_sites(psi, s2i, merge)
    psi.canonize_(to='last').canonize_(to='first')


    if imaginary_time:
        psi = imaginary_time_evolution(D_total, psi, H0)

    #info = mps.dmrg_(psi, H0, method='1site', max_sweeps=8, Schmidt_tol=1e-12)
    #fprint(info)
    for opts_svd in [#{"D_total": D_total // 2, 'tol': 1e-12},
                     {"D_total": D_total, 'tol': 1e-12}]:
        
        fprint("Running 1-site DMRG ... ")
        info = mps.dmrg_(psi, H0, method='1site', opts_svd=opts_svd, max_sweeps=8, Schmidt_tol=1e-12)
        fprint(info)

        fprint("Running 2-site DMRG ... ")
        info = mps.dmrg_(psi, H0, method='2site', opts_svd=opts_svd, max_sweeps=max2, Schmidt_tol=1e-12)
        fprint(info)
        fprint("Running 1-site DMRG ... ")
        info = mps.dmrg_(psi, H0, method='1site', max_sweeps=max1, Schmidt_tol=1e-12)
        fprint(info)

    

    On1 = merge_sites(op1site(dn1, 'D1', s2i, qI, dI), s2i, merge)
    On2 = merge_sites(op1site(dn2, 'D1', s2i, qI, dI), s2i, merge)

    n1 = mps.vdot(psi, On1, psi)
    n2 = mps.vdot(psi, On2, psi)

    fprint("Before: n1 = ", n1, "n2 = ", n2)

    if alpha is not None:

        O = (np.sqrt(alpha) * dn1 + np.sqrt(1 - alpha) * m12)
        O = op1site(O, 'D1', s2i, qI, dI)
        O = merge_sites(O, s2i, merge)

        psi = O @ psi
        psi.canonize_(to='last')
        psi.canonize_(to='first')

    n1 = mps.vdot(psi, On1, psi)
    n2 = mps.vdot(psi, On2, psi)
    fprint("Done. n1 = ", n1, "n2 = ", n2)

    psidata = psi.save_to_dict()
    with open(f'{curpath}init.npy', 'wb') as f:
        np.save(f, psidata, allow_pickle=True)
    
    with open(f'{curpath}initenergy', 'w') as f:
        np.savetxt(f, [info.energy])

    return psi, n1



def groundstate(para: dict):


    L = int(para['L'])
    NS = 4
    U = float(para['U'])
    muL = float(para['biasLR'])
    muR = -float(para['biasLR'])
    mixed = bool(para['mixed'])
    merge = bool(para["merge"])
    SB = float(para.get('SB', 0.0))
    mapping = 'mixed' if mixed else 'position'
    sym = 'U1'
    order = para['order']
    D = int(para['TEdim'])
    vs = float(para['vs'])
    max2 = int(para["max2"])
    max1 = int(para["max1"])
    rtype = para.get('rtype', 'sin-transform')
    Lambda = para.get('Lambda', None)

    curpath = f'{getcwd()}/GS/'

    if not os.path.isdir(curpath):
        mkdir(curpath)


    sites = order_sites(mapping, order, L, NS=NS, muL = muL, muR = muR, rtype = rtype, Lambda = Lambda)

    _ , n1 = initial_state(L, NS, U, 0.0, 0.0, vs, None, mapping, order, merge, sym, D, curpath=curpath, max1 = max1, max2 = max2, sites = sites,  rtype = rtype, Lambda = Lambda, muDs=[0, 0])

    np.savetxt( f'{curpath}n1', [n1])

    return None








#     return None



#singlerun()
if __name__ == '__main__':


    with open( getcwd() + '/dptpara.json', 'r') as io:
        para = json.load(io)
    fprint("GROUND STATE")
    groundstate(para)
        

