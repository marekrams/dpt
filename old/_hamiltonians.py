import numpy as np
import yastn
import yastn.tn.mps as mps
from .spinless_fermions_2ch import SpinlessFermions2ch


def L(k):
    """ label for L modes """
    return f'L{k}'

def R(k):
    """ label for R modes """
    return f'R{k}'

def S(k):
    """ label for S modes """
    return f'S{k}'


def local_operators(sym='U1'):
    if len(sym) == 2:  # U1  Z2
        # without distinguishing 2 fermionic spicies on a level of symmetry
        ops = yastn.operators.SpinlessFermions(sym=sym)
        op = {"config": ops.config,
              'I0': ops.I(),
              'I1': ops.I(),
              'c0': ops.c(),
              'c1': ops.c(),
              'cp0': ops.cp(),
              'cp1': ops.cp(),
              'n0': ops.n(),
              'n1': ops.n(),
              'v00' : ops.vec_n(val=0),
              'v10' : ops.vec_n(val=0),
              'v01' : ops.vec_n(val=1),
              'v11' : ops.vec_n(val=1)
              }
    elif len(sym) == 5:  # U1xU1  Z2xZ2
        # distinguishing 2 fermionic spicies on a level of symmetry
        ops = SpinlessFermions2ch(sym=sym)
        op = {"config": ops.config,
              'I0': ops.I(ch=0),
              'I1': ops.I(ch=1),
              'c0': ops.c(ch=0),
              'c1': ops.c(ch=1),
              'cp0': ops.cp(ch=0),
              'cp1': ops.cp(ch=1),
              'n0': ops.n(ch=0),
              'n1': ops.n(ch=1),
              'v00' : ops.vec_n(ch=0, val=0),
              'v10' : ops.vec_n(ch=1, val=0),
              'v01' : ops.vec_n(ch=0, val=1),
              'v11' : ops.vec_n(ch=1, val=1)
              }
    else:
        raise ValueError(" Wrong symmetry. ")
    return op


def order_generate_H(terms, s2i, N, one):
    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    I = mps.product_mpo(one, N)
    return mps.generate_mpo(I, Hterms)


def Hamiltonian_dpt_2U_position(NW, muL, muR, muS, dmuS, vS, U, w0=1, order='SLR', sym='U1'):
    """ generate MPO for DPT in position basis. """
    ops = local_operators(sym=sym)  # this is a dict
    NS = 2
    vL = vR = vLR = w0

    sites = [S(k) for k in range(NS, 0, -1)]  # will have SLR geomery
    sites.extend([L(k) for k in range(NW, 0, -1)])  # 'L1' is for L mode at the junction
    sites.extend([R(k) for k in range(1, NW + 1)])  # 'R1' is for R mode at the junction

    # here sites are ordered in position
    if order == 'SLR':
        s2i = {s: i for i, s in enumerate(sites)}
        i2s = {i: s for i, s in enumerate(sites)}
        I = mps.product_mpo([ops['I0']] * NS + [ops['I1']] * (NW + NW))
    else:
        raise ValueError("Hamiltonian_dpt_2U_position only SLR order defined.")

    terms = []
    for k in range(1, NW+1): # on-site energies
        terms.append((muL, [L(k)], [ops['n1']]))
        terms.append((muR, [R(k)], [ops['n1']]))
    terms.append((muS + dmuS, [S(1)], [ops['n0']]))
    terms.append((muS - dmuS, [S(2)], [ops['n0']]))

    terms.append((vS, [S(1), S(2)], [ops['cp0'], ops['c0']]))
    terms.append((vS, [S(2), S(1)], [ops['cp0'], ops['c0']]))
    terms.append((vLR, [L(1), R(1)], [ops['cp1'], ops['c1']]))
    terms.append((vLR, [R(1), L(1)], [ops['cp1'], ops['c1']]))
    for k in range(1, NW):
        terms.append((vL, [L(k), L(k+1)], [ops['cp1'], ops['c1']]))
        terms.append((vL, [L(k+1), L(k)], [ops['cp1'], ops['c1']]))
        terms.append((vR, [R(k), R(k+1)], [ops['cp1'], ops['c1']]))
        terms.append((vR, [R(k+1), R(k)], [ops['cp1'], ops['c1']]))

    # add Coulomb interactions
    terms.append((U, (S(1), L(1)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))
    terms.append((U, (S(1), R(1)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(I, Hterms)
    return H, s2i, i2s


def Hamiltonian_dpt_4U_position(NW, muL, muR, muS, dmuS, vS, U, w0=1, order='SLR', sym='U1'):
    """ generate MPO for DPT in position basis. first dot is interacting with 4 sites """
    ops = local_operators(sym=sym)  # this is a dict
    NS = 2
    vL = vR = vLR = w0

    sites = [S(k) for k in range(NS, 0, -1)]  # will have SLR geomery
    sites.extend([L(k) for k in range(NW, 0, -1)])  # 'L1' is for L mode at the junction
    sites.extend([R(k) for k in range(1, NW + 1)])  # 'R1' is for R mode at the junction

    # here sites are ordered in position
    if order == 'SLR':
        s2i = {s: i for i, s in enumerate(sites)}
        i2s = {i: s for i, s in enumerate(sites)}
        I = mps.product_mpo([ops['I0']] * NS + [ops['I1']] * (NW + NW))
    else:
        raise ValueError("Hamiltonian_dpt_4U_position only SLR order defined.")

    terms = []
    for k in range(1, NW+1): # on-site energies
        terms.append((muL, [L(k)], [ops['n1']]))
        terms.append((muR, [R(k)], [ops['n1']]))
    terms.append((muS + dmuS, [S(1)], [ops['n0']]))
    terms.append((muS - dmuS, [S(2)], [ops['n0']]))

    terms.append((vS, [S(1), S(2)], [ops['cp0'], ops['c0']]))
    terms.append((vS, [S(2), S(1)], [ops['cp0'], ops['c0']]))
    terms.append((vLR, [L(1), R(1)], [ops['cp1'], ops['c1']]))
    terms.append((vLR, [R(1), L(1)], [ops['cp1'], ops['c1']]))
    for k in range(1, NW):
        terms.append((vL, [L(k), L(k+1)], [ops['cp1'], ops['c1']]))
        terms.append((vL, [L(k+1), L(k)], [ops['cp1'], ops['c1']]))
        terms.append((vR, [R(k), R(k+1)], [ops['cp1'], ops['c1']]))
        terms.append((vR, [R(k+1), R(k)], [ops['cp1'], ops['c1']]))

    # add Coulomb interactions
    terms.append((U, (S(1), L(1)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))
    terms.append((U, (S(1), R(1)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))
    terms.append((U, (S(1), L(2)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))
    terms.append((U, (S(1), R(2)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(I, Hterms)
    return H, s2i, i2s


def Hamiltonian_dpt_RLM_position(NW, muL, muR, muS, dmuS, vS, U, w0=1, order='SLR', sym='U1'):
    """ generate MPO for DPT in position basis. """
    ops = local_operators(sym=sym)  # this is a dict
    NS = NW
    vL = vR = vLR = w0

    sites = [S(k) for k in range(NS, 0, -1)]  # will have SLR geomery
    sites.extend([L(k) for k in range(NW, 0, -1)])  # 'L1' is for L mode at the junction
    sites.extend([R(k) for k in range(1, NW + 1)])  # 'R1' is for R mode at the junction

    # here sites are ordered in position
    if order == 'SLR':
        s2i = {s: i for i, s in enumerate(sites)}
        i2s = {i: s for i, s in enumerate(sites)}
        I = mps.product_mpo([ops['I0']] * NS + [ops['I1']] * (NW + NW))
    else:
        raise ValueError("Hamiltonian_dpt_RLM_position only SLR order defined.")

    terms = []
    for k in range(1, NW+1): # on-site energies
        terms.append((muL, [L(k)], [ops['n1']]))
        terms.append((muR, [R(k)], [ops['n1']]))
    terms.append((muS + dmuS, [S(1)], [ops['n0']]))
    for k in range(2, NS + 1):
        terms.append((muS - dmuS, [S(k)], [ops['n0']]))

    terms.append((vLR, [L(1), R(1)], [ops['cp1'], ops['c1']]))
    terms.append((vLR, [R(1), L(1)], [ops['cp1'], ops['c1']]))
    for k in range(1, NW):
        terms.append((vL, [L(k), L(k+1)], [ops['cp1'], ops['c1']]))
        terms.append((vL, [L(k+1), L(k)], [ops['cp1'], ops['c1']]))
        terms.append((vR, [R(k), R(k+1)], [ops['cp1'], ops['c1']]))
        terms.append((vR, [R(k+1), R(k)], [ops['cp1'], ops['c1']]))
    for k in range(1, NS):
        terms.append((vS, [S(k), S(k+1)], [ops['cp0'], ops['c0']]))
        terms.append((vS, [S(k+1), S(k)], [ops['cp0'], ops['c0']]))

    # add Coulomb interactions
    terms.append((U, (S(1), L(1)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))
    terms.append((U, (S(1), R(1)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(I, Hterms)
    return H, s2i, i2s


def Hamiltonian_dpt_RLM_4U_position(NW, muL, muR, muS, dmuS, vS, U, w0=1, order='SLR', sym='U1'):
    """ generate MPO for DPT in position basis. """
    ops = local_operators(sym=sym)  # this is a dict
    NS = NW
    vL = vR = vLR = w0

    sites = [S(k) for k in range(NS, 0, -1)]  # will have SLR geomery
    sites.extend([L(k) for k in range(NW, 0, -1)])  # 'L1' is for L mode at the junction
    sites.extend([R(k) for k in range(1, NW + 1)])  # 'R1' is for R mode at the junction

    # here sites are ordered in position
    if order == 'SLR':
        s2i = {s: i for i, s in enumerate(sites)}
        i2s = {i: s for i, s in enumerate(sites)}
        I = mps.product_mpo([ops['I0']] * NS + [ops['I1']] * (NW + NW))
    else:
        raise ValueError("Hamiltonian_dpt_RLM_4U_position only SLR order defined.")

    terms = []
    for k in range(1, NW+1): # on-site energies
        terms.append((muL, [L(k)], [ops['n1']]))
        terms.append((muR, [R(k)], [ops['n1']]))
    terms.append((muS + dmuS, [S(1)], [ops['n0']]))
    for k in range(2, NS + 1):
        terms.append((muS - dmuS, [S(k)], [ops['n0']]))

    terms.append((vLR, [L(1), R(1)], [ops['cp1'], ops['c1']]))
    terms.append((vLR, [R(1), L(1)], [ops['cp1'], ops['c1']]))
    for k in range(1, NW):
        terms.append((vL, [L(k), L(k+1)], [ops['cp1'], ops['c1']]))
        terms.append((vL, [L(k+1), L(k)], [ops['cp1'], ops['c1']]))
        terms.append((vR, [R(k), R(k+1)], [ops['cp1'], ops['c1']]))
        terms.append((vR, [R(k+1), R(k)], [ops['cp1'], ops['c1']]))
    for k in range(1, NS):
        terms.append((vS, [S(k), S(k+1)], [ops['cp0'], ops['c0']]))
        terms.append((vS, [S(k+1), S(k)], [ops['cp0'], ops['c0']]))

    # add Coulomb interactions
    terms.append((U, (S(1), L(1)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))
    terms.append((U, (S(1), R(1)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))
    terms.append((U, (S(1), L(2)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))
    terms.append((U, (S(1), R(2)), [ops['n0'] - (ops['I0'] / 2), ops['n1'] - (ops['I1'] / 2)]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(I, Hterms)
    return H, s2i, i2s


def initial_position(H, param):
    """ This extends mps.random_mps, distributing 2 fermionic spicies separatly in two parts of the mps;
    assumes that first NS mps sites is for first fermionic type, and the rest is for the second."""

    NW = param['NW']
    NS = param['NW'] if param['basis'] in ("RLM_position", "RLM_4U_position") else 2

    psi = mps.Mps(NS + NW + NW)

    config = H.config
    lr = yastn.Leg(config, s=1, t=((0, 0),), D=(1,),)

    if "position" in param['basis']:
        ll_occ = [param["occLR"] * i / (2 * NW)  for i in range(1, 2 * NW + 1)]
        ll_occ = {site: oc for site, oc in zip(psi.sweep(to='first'), ll_occ)}
    elif "mixed" in param['basis']:
        NW1 = NW + 1
        vL, vR = param['w0'], param['w0']
        muL, muR = param["muL"], param["muR"]
        _, mu_o =  param["order"].split("_")
        mu_o = float(mu_o)
        oe = [(muL + 2 * vL * np.cos(np.pi * k / NW1), mu_o / 2 + 2 * vL * np.cos(np.pi * k / NW1)) for k in range(1, NW+1)]
        oe += [(muR + 2 * vR * np.cos(np.pi * k / NW1), -mu_o/2 + 2 * vR * np.cos(np.pi * k / NW1)) for k in range(1, NW+1)]
        oe.sort()
        oe = [(e1, int(ii < param["occLR"])) for ii, (_, e1) in enumerate(oe)]
        oe.sort()
        oe = [oo for (_, oo) in oe]
        ll_occ = np.cumsum(oe[::-1])
        ll_occ = {site: oc for site, oc in zip(psi.sweep(to='first'), ll_occ)}

    for site in psi.sweep(to='first'):
        lp = H[site].get_legs(axes=1)  # ket leg of MPS/MPO

        if site >= NS:
            nl = (0, ll_occ[site])
            config.sym.spanning_vectors = [[1000, 0], [0, 1]]
        else:
            nl = (param["occS"] * (NS - site) / NS, param["occLR"])
            config.sym.spanning_vectors = [[1, 0], [0, 1000]]
        if site != psi.first:
            ll = yastn.random_leg(config, s=-1, n=nl, D_total=param["Ds"][0], sigma=1, legs=[lp, lr])
        else:
            ll = yastn.Leg(config, s=-1, t=((param["occS"], param["occLR"]),), D=(1,),)
        psi.A[site] = yastn.rand(config, legs=[ll, lp, lr])
        lr = psi.A[site].get_legs(axes=0).conj()
    if sum(lr.D) == 1:
        return psi


def Hamiltonian_dpt_4U_mixed(NW, muL, muR, muS, dmuS, vS, U, w0=1, order='mixed_0', sym='U1'):
    """ generate mpo for dpt model in mixed basis """

    ops = local_operators(sym=sym)  # this is a dict
    NS = 2

    vL = vR = vLR = w0
    NW1 = NW + 1

    if "mixed" in order:
        sites = ['S2', 'S1']

        _, mu_o = order.split("_")
        mu_o = float(mu_o)
        oe = [(mu_o / 2 + 2 * vL * np.cos(np.pi * k / NW1), L(k)) for k in range(1, NW+1)]
        oe += [(-mu_o/2 + 2 * vR * np.cos(np.pi * k / NW1), R(k)) for k in range(1, NW+1)]
        oe.sort()
        sites += [st for _, st in oe]

        s2i = {s: i for i, s in enumerate(sites)}
        i2s = {i: s for i, s in enumerate(sites)}
        I = mps.product_mpo([ops['I0']] * NS + [ops['I1']] * (NW + NW))
    else:
        raise ValueError("Hamiltonian_dpt_4U_mixed only mixed order defined.")

    terms = []

    for k in range(1, NW+1):  # on-site energies
        terms.append((muL + 2 * vL * np.cos(np.pi * k / NW1), [L(k)], [ops['n1']]))
        terms.append((muR + 2 * vR * np.cos(np.pi * k / NW1), [R(k)], [ops['n1']]))

    terms.append((muS + dmuS, [S(1)], [ops['n0']]))
    for k in range(2, NS + 1):
        terms.append((muS - dmuS, [S(k)], [ops['n0']]))

    terms.append((vS, [S(1), S(2)], [ops['cp0'], ops['c0']]))
    terms.append((vS, [S(2), S(1)], [ops['cp0'], ops['c0']]))
    for kl in range(1, NW1):
        for kr in range(1, NW1):
            amp = vLR * (2 / NW1) * np.sin(np.pi * 1 * kl / NW1) * np.sin(np.pi * 1 * kr / NW1)
            terms.append((amp, [L(kl), R(kr)], [ops['cp1'], ops['c1']]))
            terms.append((amp, [R(kr), L(kl)], [ops['cp1'], ops['c1']]))

    for k1 in range(1, NW1):
       for k2 in range(1, NW1):
            amp = U * (2 / NW1) * np.sin(np.pi * 1 * k1 / NW1) * np.sin(np.pi * 1 * k2 / NW1)
            terms.append((amp, [S(1), L(k1), L(k2)], [ops['n0'] - ops['I0'] / 2, ops['cp1'], ops['c1']]))
            terms.append((amp, [S(1), R(k1), R(k2)], [ops['n0'] - ops['I0'] / 2, ops['cp1'], ops['c1']]))
            amp = U * (2 / NW1) * np.sin(np.pi * 2 * k1 / NW1) * np.sin(np.pi * 2 * k2 / NW1)
            terms.append((amp, [S(1), L(k1), L(k2)], [ops['n0'] - ops['I0'] / 2, ops['cp1'], ops['c1']]))
            terms.append((amp, [S(1), R(k1), R(k2)], [ops['n0'] - ops['I0'] / 2, ops['cp1'], ops['c1']]))
    terms.append((-4 * U / 2, [S(1)], [ops['n0'] - ops['I0'] / 2]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(I, Hterms)
    return H, s2i, i2s
