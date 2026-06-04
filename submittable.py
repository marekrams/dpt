from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
import yastn.tn.mps as mps
import json
from hamiltonians import local_operators, Hamiltonian_dpt_position, Hamiltonian_dpt_momentum, Hamiltonian_dpt_mixed
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
               "momentum": Hamiltonian_dpt_momentum,}
    
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


def initial_state(NW, NS, U, muL, muR, vS0, alpha, mapping, order, merge, sym, D_total, curpath = '', muDs=[0, 10000], max2 = 4, max1 = 256, sites = [], Hdebug = False, **config_kwargs):

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

    H0, M = Hamiltonian(mapping)(NW, NS, muL, muR, muDs, vS0, U * (2 * alpha - 1), Hdebug = Hdebug, sym=sym, order=sites, **config_kwargs)
    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym, **config_kwargs)

    if Hdebug:
        with open(f'{curpath}HamInit', 'w') as f:
            np.savetxt(f, M, fmt = '%s')
    
    psi = mps.random_mps(H0, n=n_profile, D_total=D_total, sigma=2, distribution='normal')

    H0 = merge_sites(H0, s2i, merge)
    psi = merge_sites(psi, s2i, merge)
    psi.canonize_(to='last').canonize_(to='first')

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

    fprint("Before: n1 = ", mps.vdot(psi, On1, psi), "n2 = ", mps.vdot(psi, On2, psi))

    O = (np.sqrt(alpha) * dn1 + np.sqrt(1 - alpha) * m12)
    O = op1site(O, 'D1', s2i, qI, dI)
    O = merge_sites(O, s2i, merge)

    psi = O @ psi
    psi.canonize_(to='last')
    psi.canonize_(to='first')

    fprint("Done. n1 = ", mps.vdot(psi, On1, psi), "n2 = ", mps.vdot(psi, On2, psi))

    psidata = psi.save_to_dict()
    with open(f'{curpath}init.npy', 'wb') as f:
        np.save(f, psidata, allow_pickle=True)
    
    with open(f'{curpath}initenergy', 'w') as f:
        np.savetxt(f, [info.energy])

    return psi, info.energy


def run_evolution(psi, NW, NS, U, muL, muR, vS0, vS1, mapping, order, merge, sym, D_total, tswitch, tfin, dt, lasttime = 0.0, 
                  muDs=[0.0, 0.0], curpath = '', verbose=0, tdvptol = 1e-8, sites = [], Hdebug = False, **config_kwargs):

    
    print("dynamics config: ", config_kwargs)
    # statistics
    total = 0
    cnt = 0


    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym, **config_kwargs)
    s2i = {s: i for i, s in enumerate(sites)}
    On1 = merge_sites(op1site(dn1, 'D1', s2i, qI, dI), s2i, merge)
    #On2 not relevant
    #On2 = merge_sites(op1site(dn2, 'D1', s2i, qI, dI), s2i, merge)
    Om12 = merge_sites(op1site(m12, 'D1', s2i, qI, dI), s2i, merge)

    QPC = [s for s in s2i.keys() if s.startswith("S")]

    print(f"QPC : {QPC}")
    Ons = {ss: merge_sites(op1site(qn, ss, s2i, qI, dI), s2i, merge) for ss in QPC}


    opts_svd = {"D_total": D_total, 'tol': tdvptol}
    fprint("Running time evolution ... ")

    for t0, t1, stage in [(0, tswitch, 1), (tswitch, tfin, 2)]:

        if stage == 1:
            H, M = Hamiltonian(mapping)(NW, NS, muL, muR, muDs, vS0, U, sym=sym, Hdebug = Hdebug, order=sites, **config_kwargs)
            if Hdebug:
                with open(f'{curpath}HamT1', 'w') as f:
                    np.savetxt(f, M, fmt = '%s')

        else:
            H, M = Hamiltonian(mapping)(NW, NS, muL, muR, muDs, vS1, U, sym=sym, Hdebug = Hdebug, order=sites, **config_kwargs)
            if Hdebug:
                with open(f'{curpath}HamT2', 'w') as f:
                    np.savetxt(f, M, fmt = '%s')

        H = merge_sites(H, s2i, merge)

        times = np.linspace(t0, t1, int((t1 - t0) /dt) + 1)
        times = [t0, t0+dt/128, t0+dt/64, t0+dt/32, t0+dt/16, t0+dt/8, t0+dt/4, t0+dt/2] + list(times)[1:]

        times = np.array(times)
        times = times[ times >= lasttime]

        if len(times) == 0:
            fprint(f"last time = {lasttime}, Skipping stage 1")
            continue

        start_time = time.time()
        #fprint(times)
        for step in mps.tdvp_(psi, H, times, method='12site', dt=dt, opts_svd=opts_svd, 
                              yield_initial=True if times[0] == 0 else False, 
                              subtract_E=True):


            if config_kwargs['backend'] == 'torch' and cnt % 20 == 0:
                gpu_report(f'step = {cnt}')
                
            cnt += 1
            if verbose:
                fprint(step)

            n1 = mps.vdot(psi, On1, psi).real
            m12 = mps.vdot(psi, Om12, psi).real

            occs = [ mps.vdot(psi, Ons[s], psi).real for s in QPC]

            ent = psi.get_entropy()

            if config_kwargs['backend'] != 'torch':
                current1 = get_current(psi, s2i, qc, qcp, qI, dI, merge)

                with open(f'{curpath}current1', 'a') as f:
                    np.savetxt( f, [current1])

            else:
                ent = [ val.cpu() for val in ent]
                occs = [ val.cpu() for val in occs]
                n1 = n1.cpu()
                m12 = m12.cpu()
                

            #current2 = get_current(psi, s2i, qc, qcp, qI, dI, merge, method = 'inner')

            #current = get_current()

            effE = np.log( np.power( np.sum( np.exp( 3 * ent)) / ( len(ent) ), 1/3))

            end_time = time.time()
            fprint(f"TDVP end: {step.tf}, elapsed time: {end_time - start_time}")
            
            total += end_time - start_time
            fprint(f"Rolling average TDVP: {total/cnt}")

            psidata = psi.save_to_dict()
            #fprint(psidata)
            with open(f'{curpath}TDVPlast.npy', 'wb') as f:
                np.save(f, psidata, allow_pickle=True)

            with open(f'{curpath}TDVPlastback.npy', 'wb') as f:
                np.save(f, psidata, allow_pickle=True)

            with open(f'{curpath}times', 'a') as f:
                np.savetxt( f, [step.tf])

            # with open(f'{curpath}current2', 'a') as f:
            #     np.savetxt( f, [current2])

            with open(f'{curpath}occs', 'a') as f:
                np.savetxt( f, [occs], fmt = '%.7g')

            with open(f'{curpath}n1', 'a') as f:
                np.savetxt( f, [n1])

            with open(f'{curpath}m12', 'a') as f:
                np.savetxt( f, [m12])

            with open(f'{curpath}SvN', 'a') as f:
                np.savetxt( f, [ent])

            with open(f'{curpath}MaxEnt', 'a') as f:
                np.savetxt( f, [max(ent)])

            with open(f'{curpath}Seff', 'a') as f:
                np.savetxt( f, [effE])

            del psidata
            start_time = time.time()

    fprint("Done.")
    return None # psi, ts, traces



def singlerun(para, config_kwargs):


    L = int(para['L'])
    NS = 4
    U = float(para['U'])
    muL = float(para['biasLR'])
    muR = -float(para['biasLR'])
    alpha = float(para['n1init'])
    mixed = bool(para['mixed'])
    merge = bool(para["merge"])
    mapping = 'mixed' if mixed else 'position'
    sym = 'U1'
    order = para['order']
    D = int(para['TEdim'])
    vs = float(para['vs'])
    dt = float(para['timestep'])
    tswitch = float(para['tswitch'])
    tfin = float(para['tfin'])
    repeat = int(para['repeat'])
    max2 = int(para["max2"])
    max1 = int(para["max1"])
    finaltol = float(para['finaltol'])


    tdvptol = 1e-6
    for i in range(repeat):

        fprint(f"repeat: {i}")

        curpath = f'{getcwd()}/iter_repeat{i}/'

        if not os.path.isdir(curpath):
            mkdir(curpath)

        if os.path.isfile( f'{curpath}times'):

            lasttime = np.loadtxt( f'{curpath}times')[-1]
            sites = np.loadtxt( f'{curpath}sites', dtype = 'str')

            # finish!
            if lasttime == tfin:
                fprint( f"iter {i} exists, skip!")
                alpha = np.mean(np.loadtxt( f'{curpath}n1')[-8:])
                fprint( f"current alpha = {alpha}")

                if np.abs(float(para['n1init']) - alpha) < finaltol:
                    return True, new - alpha
                
                continue
            
            # we continue
            else:

                try:
                    psi0 = load_psi(f'{curpath}TDVPlast.npy', sym, message="Loading last", **config_kwargs)
                except:
                    psi0 = load_psi(f'{curpath}TDVPlastback.npy', sym, message="Loading last backup", **config_kwargs)
            
        # no time file, starting new!
        else:
            fprint("Starting new")
            sites = order_sites(mapping, order, L, NS=NS, muL = muL, muR = muR)

            psi0 , _ = initial_state(L, NS, U, 0.0, 0.0, 0, alpha, mapping, order, merge, sym, D, curpath=curpath, max1 = max1, max2 = max2, sites = sites, **config_kwargs)

            if config_kwargs['backend'] == 'torch':
                gpu_report('DMRG')

            lasttime = 0.0

        run_evolution(psi0, L, NS, U, muL, muR, 0, vs, mapping, order, merge, sym, D, tswitch, tfin, dt, 
                      lasttime = lasttime, curpath = curpath, tdvptol= tdvptol, verbose=0, sites = sites, **config_kwargs)

        n1 = np.loadtxt(f'{curpath}/n1')
        new = np.mean( n1[-8:])

        fprint("new n1 last: ", new)

        if np.abs(new - alpha) < finaltol:
            return True, new - alpha

        alpha = new

    return False, new - alpha


def singlerun_binary_search(para, config_kwargs):


    L = int(para['L'])
    NS = 4
    U = float(para['U'])
    muL = float(para['biasLR'])
    muR = -float(para['biasLR'])
    #alpha = float(para['n1init'])
    lo = float(para["lo"])
    hi = float(para["hi"])
    mixed = bool(para['mixed'])
    merge = bool(para["merge"])
    mapping = 'mixed' if mixed else 'position'
    sym = 'U1'
    n1init = float(para["n1init"])
    order = para['order']
    D = int(para['TEdim'])
    vs = float(para['vs'])
    dt = float(para['timestep'])
    tswitch = float(para['tswitch'])
    tfin = float(para['tfin'])
    repeat = int(para['repeat'])
    max2 = int(para["max2"])
    max1 = int(para["max1"])
    finaltol = float(para['finaltol'])


    tdvptol = 1e-6
    for i in range(repeat):

        fprint(f"repeat: {i}")

        curpath = f'{getcwd()}/bs_repeat{i}_lo{lo}_hi{hi}/'

        if not os.path.isdir(curpath):
            mkdir(curpath)

        if os.path.isfile( f'{curpath}times'):

            lasttime = np.loadtxt( f'{curpath}times')[-1]
            sites = np.loadtxt( f'{curpath}sites', dtype = 'str')

            # finish!
            if lasttime == tfin:
                fprint( f"iter {i} exists, skip!")
                #alpha = np.mean(np.loadtxt( f'{curpath}n1')[-8:])
                lo = np.loadtxt( f'{curpath}lo')
                hi = np.loadtxt( f'{curpath}hi')
                #points = [lo, hi]

                #lo = min(points)
                #hi = max(points)
                
                fprint( f"current lo = {lo}, hi = {hi}")
                continue
            
            # we continue
            else:

                try:
                    psi0 = load_psi(f'{curpath}TDVPlast.npy', sym, message="Loading last", **config_kwargs)
                except:
                    psi0 = load_psi(f'{curpath}TDVPlastback.npy', sym, message="Loading last backup", **config_kwargs)
            
        # no time file, starting new!
        else:
            fprint("Starting new")

            if i == 0:
                alpha = n1init
            else:
                alpha = (lo + hi) / 2

            sites = order_sites(mapping, order, L, NS=NS, muL = muL, muR = muR)

            psi0 , _ = initial_state(L, NS, U, 0.0, 0.0, 0, alpha, mapping, order, merge, sym, D, curpath=curpath, max1 = max1, max2 = max2, sites = sites, **config_kwargs)

            if config_kwargs['backend'] == 'torch':
                gpu_report('DMRG')

            lasttime = 0.0
        
        if i == 0:
            alpha = n1init
        else:
            alpha = (lo + hi) / 2

        run_evolution(psi0, L, NS, U, muL, muR, 0, vs, mapping, order, merge, sym, D, tswitch, tfin, dt, 
                      lasttime = lasttime, curpath = curpath, tdvptol= tdvptol, verbose=0, sites = sites, **config_kwargs)

        n1 = np.loadtxt(f'{curpath}/n1')
        new = np.mean( n1[-8:])

        fprint("new n1 last: ", new)

        if np.abs(new - alpha) < finaltol:
            fprint("final tol ", np.abs(new - alpha))
            #np.savetxt( f'{curpath}TOL_REACHED', [])
            np.savetxt( f'{getcwd()}/TOL_REACHED', [])
            return True
        
        # determine condition for choosing interval
        
        # choose lower
        if new < alpha :

            points = [lo, new]
            lo = min(points)
            hi = max(points)

        # choose higher
        else:

            points = [new, hi]
            lo = min(points)
            hi = max(points)


        np.savetxt( f'{curpath}lo', [lo], fmt = '%.7f')
        np.savetxt( f'{curpath}hi', [hi], fmt = '%.7f')

    return False



#singlerun()
if __name__ == '__main__':

    #print(threadpool_info())
    #np.__config__.show()

    if os.path.isfile(f'{getcwd()}/TOL_REACHED'):
        print("TOL REACHED")
        exit()


    with open( getcwd() + '/dptpara.json', 'r') as io:
        para = json.load(io)

    mode = para['searchmode']

    try:
        gpu = para['GPU']


        if gpu == True:
            pprint("USING GPU")
            #np = cp
            config_kwargs = {"backend": "torch", "default_device" : "cuda"}
        
        else:
            pprint("USING CPU")
            config_kwargs = {"backend": "np"}
    except:
        pprint("USING CPU")
        config_kwargs = {"backend": "np"}


    if mode == 'iterative':

        fprint("ITERATIVE")
        singlerun(para, config_kwargs)

    elif mode == 'binarysearch':
        fprint("BINARY SEARCH")

        # temppara = deepcopy(para)
        # temppara["repeat"] = 1
        # temppara["n1init"] = 0.501
        # temppara['finaltol'] = 1e-3
        # check, val = singlerun(temppara, config_kwargs)

        # #print(val)

        # if check and val < 0:
        #     fprint("tol reached")

        # else:
        #     fprint("tol not reached: bs")
        #     singlerun_binary_search(para, config_kwargs)

        singlerun_binary_search(para, config_kwargs)
            
        
    else:
        raise ValueError("not recognized searchmode")
