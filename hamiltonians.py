import numpy as np
import yastn
import yastn.tn.mps as mps
from sites import S, D, L, R
from pprint import pprint



def local_operators(sym='U1'):
    ops = yastn.operators.SpinlessFermions(sym=sym)
    qI, qc, qcp, qn = ops.I(), ops.c(), ops.cp(), ops.n()
    dx = yastn.Tensor(config=qI.config, s=qI.s)
    dx.set_block(ts=(0, 0), val=[[0, 1], [1, 0]], Ds=(2, 2))
    dn1 = yastn.Tensor(config=qI.config, s=qI.s)
    dn1.set_block(ts=(0, 0), val=[[0, 0], [0, 1]], Ds=(2, 2))
    dn2 = yastn.Tensor(config=qI.config, s=qI.s)
    dn2.set_block(ts=(0, 0), val=[[1, 0], [0, 0]], Ds=(2, 2))
    dI = yastn.Tensor(config=qI.config, s=qI.s)
    dI.set_block(ts=(0, 0), val=[[1, 0], [0, 1]], Ds=(2, 2))
    m12 = yastn.Tensor(config=qI.config, s=qI.s)
    m12.set_block(ts=(0, 0), val=[[0, 1], [0, 0]], Ds=(2, 2))
    m21 = yastn.Tensor(config=qI.config, s=qI.s)
    m21.set_block(ts=(0, 0), val=[[0, 0], [1, 0]], Ds=(2, 2))
    return qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21


def Hamiltonian_dpt_position(NW, NS, muL, muR, muDs, vS, U, w0=1, order='DLSR', sym='U1'):
    """ generate MPO for DPT in position basis. first dot is interacting with 4 sites """
    #
    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym)
    NS = 4
    #
    vL = vR = vLR = w0
    #
    if order == 'DLSR':
        sites = [D(1)]
        sites += [L(k) for k in range(NW, 0, -1)]  # 'L1' is for L mode at the junction
        sites += [S(k) for k in range(1, NS + 1)]  # 'S1' connected to L1
        sites += [R(k) for k in range(1, NW + 1)]  # 'R1' is for R mode at the junction
    elif order == 'LDSR':
        sites = [L(k) for k in range(NW, 0, -1)]   # 'L1' is for L mode at the junction
        sites += [S(1), S(2), D(1), S(3), S(4)]
        sites += [R(k) for k in range(1, NW + 1)]  # 'R1' is for R mode at the junction
    else:
        sites = order

    s2i = {s: i for i, s in enumerate(sites)}
    i2s = {i: s for i, s in enumerate(sites)}
    II = mps.product_mpo([dI if 'D' in site else qI for site in sites])

    try:
        lD = len(muDs)
        assert lD == 2
    except TypeError:
        muDs = [muDs] * 2

    terms = []
    for k in range(1, NW + 1): # on-site energies
        terms.append((muL, [L(k)], [qn]))
        terms.append((muR, [R(k)], [qn]))

    for k in range(1, NS // 2 + 1): # on-site energies
        terms.append((muL, [S(k)], [qn]))
    for k in range(NS // 2 + 1, NS + 1): # on-site energies
        terms.append((muR, [S(k)], [qn]))

    terms.append((muDs[0], [D(1)], [dn1]))
    terms.append((muDs[1], [D(1)], [dn2]))
    terms.append((vS, [D(1)], [dx]))

    terms.append((vLR, [S(1), R(1)], [qcp, qc]))
    terms.append((vLR, [R(1), S(1)], [qcp, qc]))
    terms.append((vLR, [L(1), S(NS)], [qcp, qc]))
    terms.append((vLR, [S(NS), L(1)], [qcp, qc]))
    for k in range(1, NS):
        terms.append((vLR, [S(k), S(k+1)], [qcp, qc]))
        terms.append((vLR, [S(k+1), S(k)], [qcp, qc]))
    for k in range(1, NW):
        terms.append((vL, [L(k), L(k+1)], [qcp, qc]))
        terms.append((vL, [L(k+1), L(k)], [qcp, qc]))
        terms.append((vR, [R(k), R(k+1)], [qcp, qc]))
        terms.append((vR, [R(k+1), R(k)], [qcp, qc]))

    # add Coulomb interactions
    for k in range(1, NS + 1):
        terms.append((U, (D(1), S(k)), [dn1 - dI / 2, qn - qI / 2]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(II, Hterms)
    return H, s2i, i2s


def Hamiltonian_dpt_momentum(NW, NS, muL, muR, muDs, vS, U, w0=1, order='DLR', sym='U1'):
    """ generate mpo for dpt model in mixed basis """
    #
    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym)
    #
    vL = vR = vLR = w0
    #
    NW1 = NW + NS // 2 + 1
    #
    if order == 'DLR':
        sites = [D(1)]
        sites += [L(k) for k in range(1, NW1)]
        sites += [R(k) for k in range(1, NW1)]
    else:
        sites = order

    s2i = {s: i for i, s in enumerate(sites)}
    i2s = {i: s for i, s in enumerate(sites)}
    II = mps.product_mpo([dI if 'D' in site else qI for site in sites])

    try:
        lD = len(muDs)
        assert lD == 2
    except TypeError:
        muDs = [muDs] * 2

    terms = []

    for k in range(1, NW1):  # on-site energies
        terms.append((muL + 2 * vL * np.cos(np.pi * k / NW1), [L(k)], [qn]))
        terms.append((muR + 2 * vR * np.cos(np.pi * k / NW1), [R(k)], [qn]))

    terms.append((muDs[0], [D(1)], [dn1]))
    terms.append((muDs[1], [D(1)], [dn2]))
    terms.append((vS, [D(1)], [dx]))

    for kl in range(1, NW1):
        for kr in range(1, NW1):
            amp = vLR * (2 / NW1) * np.sin(np.pi * 1 * kl / NW1) * np.sin(np.pi * 1 * kr / NW1)
            terms.append((amp, [L(kl), R(kr)], [qcp, qc]))
            terms.append((amp, [R(kr), L(kl)], [qcp, qc]))

    for k1 in range(1, NW1):
       for k2 in range(1, NW1):
           for ii in range(1, NS // 2 + 1):
                amp = U * (2 / NW1) * np.sin(np.pi * ii * k1 / NW1) * np.sin(np.pi * ii * k2 / NW1)
                terms.append((amp, [D(1), L(k1), L(k2)], [dn1 - dI / 2, qcp, qc]))
                terms.append((amp, [D(1), R(k1), R(k2)], [dn1 - dI / 2, qcp, qc]))
    terms.append((-NS * U / 2, [D(1)], [dn1 - dI / 2]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(II, Hterms)
    return H, s2i, i2s



def Hamiltonian_dpt_mixed(NW, NS, muL, muR, muDs, vS, U, w0=1, order = [], sym='U1'):
    """ generate mpo for dpt model in mixed basis """

    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym)
    #
    vL = vR = vLR = w0
    #
    NW1 = NW + 1
    #
    # sites = [D(1)]  # will have SLR geomery
    # sites += [L(k) for k in range(1, NW1)]  # 'L1' is for L mode at the junction
    # sites += [S(k) for k in range(1, NS + 1)]  # 'S1' connected to L1
    # sites += [R(k) for k in range(1, NW1)]  # 'R1' is for R mode at the junction

    # legacy code that was not used, as sites are now given by order_sites() function

    # if order == 'LRDSLR':
    #     sites = []
    #     for k in range(1, NW1):
    #         sites.append(L(k))
    #         sites.append(R(k))
    #     ss = [S(k) for k in range(1, NS + 1)]
    #     sites = sites[:NW1] + ss[:NS//2] + ['D1'] + ss[NS//2:] + sites[NW1:]
    # elif order == 'DLRSLR':
    #     sites = []
    #     for k in range(1, NW1):
    #         sites.append(L(k))
    #         sites.append(R(k))
    #     sites = [D(1)] + sites[:NW1] + [S(k) for k in range(1, NS + 1)] + sites[NW1:]
    # else:
    #     sites = order
    sites = order

    # here sites are ordered in position
    s2i = {s: i for i, s in enumerate(sites)}
    i2s = {i: s for i, s in enumerate(sites)}
    II = mps.product_mpo([dI if 'D' in site else qI for site in sites])

    try:
        lD = len(muDs)
        assert lD == 2
    except TypeError:
        muDs = [muDs] * 2

    terms = []
    for k in range(1, NW1):  # on-site energies
        terms.append((muL + 2 * vL * np.cos(np.pi * k / NW1), [L(k)], [qn]))
        terms.append((muR + 2 * vR * np.cos(np.pi * k / NW1), [R(k)], [qn]))
    for k in range(1, NS // 2 + 1): # on-site energies
        terms.append((muL, [S(k)], [qn]))
    for k in range(NS // 2 + 1, NS + 1): # on-site energies
        terms.append((muR, [S(k)], [qn]))

    terms.append((muDs[0], [D(1)], [dn1]))
    terms.append((muDs[1], [D(1)], [dn2]))
    terms.append((vS, [D(1)], [dx]))

    for k in range(1, NW1):  #  hopping to sys
        terms.append((vLR * np.sin(np.pi * k / NW1) * np.sqrt(2 / NW1), [S(1), L(k)], [qcp, qc]))
        terms.append((vLR * np.sin(np.pi * k / NW1) * np.sqrt(2 / NW1), [L(k), S(1)], [qcp, qc]))
        terms.append((vLR * np.sin(np.pi * k / NW1) * np.sqrt(2 / NW1), [R(k), S(NS)], [qcp, qc]))
        terms.append((vLR * np.sin(np.pi * k / NW1) * np.sqrt(2 / NW1), [S(NS), R(k)], [qcp, qc]))

    for k in range(1, NS):
        terms.append((vLR, [S(k), S(k+1)], [qcp, qc]))
        terms.append((vLR, [S(k+1), S(k)], [qcp, qc]))

    # add Coulomb interactions
    for k in range(1, NS + 1):
        terms.append((U, (D(1), S(k)), [dn1 - dI / 2, qn - qI / 2]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]

    H = mps.generate_mpo(II, Hterms)
    return H, s2i, i2s
