import numpy as np
import yastn
import yastn.tn.mps as mps


def s(n):  # system
    return f"s{n}"


def a(n):  # ancila
    return f"a{n}"


def Lindbladian_dpt_markovian(vS, U, gamma, w0=1):
    """ generate MPO for DPT in position basis. first dot is interacting with 4 sites """
    #
    ops = yastn.operators.SpinlessFermions(sym='U1')
    sI, sc, scp, sn = ops.I(), ops.c(), ops.cp(), ops.n()
    aI, ac, acp, an = sI.conj(), sc.conj(), scp.conj(), sn.conj()
    #
    sdx = yastn.Tensor(config=sI.config, s=sI.s)
    sdx.set_block(ts=(0, 0), val=[[0, 1], [1, 0]], Ds=(2, 2))
    sdn = yastn.Tensor(config=sI.config, s=sI.s)
    sdn.set_block(ts=(0, 0), val=[[0, 0], [0, 1]], Ds=(2, 2))
    sdI = yastn.Tensor(config=sI.config, s=sI.s)
    sdI.set_block(ts=(0, 0), val=[[1, 0], [0, 1]], Ds=(2, 2))
    adI, adx, adn  = sdI.conj(), sdx.conj(), sdn.conj()
    #
    N = 5
    s2i = {}
    for j in range(1, 6):
        s2i[s(j)] = 2 * j - 2
        s2i[a(j)] = 2 * j - 1
    #
    terms = []
    #
    # 1j H rho
    terms.append((1j * vS, [s(1)], [sdx]))
    for j in range(2, 5):
        terms.append((1j * w0, [s(j), s(j+1)], [scp, sc]))
        terms.append((1j * w0, [s(j+1), s(j)], [scp, sc]))
    for j in range(2, 6):
        terms.append((1j * U,  [s(1), s(j)], [sdn - sdI / 2, sn - sI / 2]))
    #
    # -1j rho H
    terms.append((-1j * vS, [a(1)], [adx]))
    for j in range(2, 5):
        terms.append((-1j * w0, [a(j), a(j+1)], [acp, ac]))
        terms.append((-1j * w0, [a(j+1), a(j)], [acp, ac]))
    for j in range(2, 6):
        terms.append((-1j * U,  [a(1), a(j)], [adn - adI / 2, an - aI / 2]))
    #
    terms.append((gamma, [s(5)], [sI]))
    #
    # injection
    terms.append((-gamma, [s(2), a(2)], [scp, acp]))
    terms.append((-gamma / 2, [s(2)], [sn]))
    terms.append((-gamma / 2, [a(2)], [an]))
    #
    # depletion
    terms.append((gamma, [s(5), a(5)], [sc, ac]))
    terms.append((gamma / 2, [s(5)], [sn]))
    terms.append((gamma / 2, [a(5)], [an]))
    #
    terms = [mps.Hterm(amp, [s2i[p] for p in pos], oprs) for amp, pos, oprs in terms]
    #
    one = mps.product_mpo([sdI, adI] + [sI, aI] * (N - 1))
    LL = mps.generate_mpo(one, terms)
    return merge_sa(LL)



def Lindbladian_dpt_markovian2(vS, U, gamma, w0=1):
    """ generate MPO for DPT in position basis. first dot is interacting with 4 sites """
    #
    ops = yastn.operators.SpinlessFermions(sym='U1')
    sI, sc, scp, sn = ops.I(), ops.c(), ops.cp(), ops.n()
    aI, ac, acp, an = sI.conj(), sc.conj(), scp.conj(), sn.conj()
    #
    sdx = yastn.Tensor(config=sI.config, s=sI.s)
    sdx.set_block(ts=(0, 0), val=[[0, 1], [1, 0]], Ds=(2, 2))
    sdn = yastn.Tensor(config=sI.config, s=sI.s)
    sdn.set_block(ts=(0, 0), val=[[0, 0], [0, 1]], Ds=(2, 2))
    sdI = yastn.Tensor(config=sI.config, s=sI.s)
    sdI.set_block(ts=(0, 0), val=[[1, 0], [0, 1]], Ds=(2, 2))
    adI, adx, adn  = sdI.conj(), sdx.conj(), sdn.conj()
    #
    N = 7
    s2i = {}
    for j in range(1, N + 1):
        s2i[s(j)] = 2 * j - 2
        s2i[a(j)] = 2 * j - 1
    #
    terms = []
    #
    # 1j H rho
    terms.append((1j * vS, [s(1)], [sdx]))
    for j in range(2, N):
        terms.append((1j * w0, [s(j), s(j+1)], [scp, sc]))
        terms.append((1j * w0, [s(j+1), s(j)], [scp, sc]))
    for j in range(3, 7):
        terms.append((1j * U,  [s(1), s(j)], [sdn - sdI / 2, sn - sI / 2]))
    #
    # -1j rho H
    terms.append((-1j * vS, [a(1)], [adx]))
    for j in range(2, N):
        terms.append((-1j * w0, [a(j), a(j+1)], [acp, ac]))
        terms.append((-1j * w0, [a(j+1), a(j)], [acp, ac]))
    for j in range(3, 7):
        terms.append((-1j * U,  [a(1), a(j)], [adn - adI / 2, an - aI / 2]))
    #
    terms.append(( gamma, [s(N)], [sI]))
    #
    # injection
    terms.append((-gamma, [s(2), a(2)], [scp, acp]))
    terms.append((-gamma / 2, [s(2)], [sn]))
    terms.append((-gamma / 2, [a(2)], [an]))
    #
    # depletion
    terms.append((gamma, [s(N), a(N)], [sc, ac]))
    terms.append((gamma / 2, [s(N)], [sn]))
    terms.append((gamma / 2, [a(N)], [an]))
    #
    # terms.append(( gamma, [s(N)], [sI]))
    # injection
    # terms.append((-gamma, [s(N), a(N)], [scp, acp]))
    # terms.append((-gamma / 2, [s(N)], [sn]))
    # terms.append((-gamma / 2, [a(N)], [an]))
    # #
    # # depletion
    # terms.append((gamma, [s(2), a(2)], [sc, ac]))
    # terms.append((gamma / 2, [s(2)], [sn]))
    # terms.append((gamma / 2, [a(2)], [an]))
    #
    terms = [mps.Hterm(amp, [s2i[p] for p in pos], oprs) for amp, pos, oprs in terms]
    #
    one = mps.product_mpo([sdI, adI] + [sI, aI] * (N - 1))
    LL = mps.generate_mpo(one, terms)
    return merge_sa(LL)


def merge_sa(psi):
    N = psi.N // 2
    phi = mps.MpsMpoOBC(N, psi.nr_phys)
    axes = (0, (1, 2), 3) if psi.nr_phys == 1 else (0, (1, 3), 4, (2, 5))
    for n in range(N):
        tmp = yastn.tensordot(psi[2*n], psi[2*n + 1], axes=(2, 0))
        phi[n] = tmp.fuse_legs(axes=axes)
    return phi


def vectorI(N):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    I2 = mps.product_mps([ops.vec_n(0), ops.vec_n(0).conj()])
    cc = mps.product_mpo([ops.cp(), ops.cp().conj()])
    psi2 = (1 / np.sqrt(2)) * (I2 + cc @ I2)
    psi = mps.Mps(2 * N)
    for n in range(1, N):
        psi[2*n]   = psi2[0]
        psi[2*n+1] = psi2[1]

    v1 = yastn.Tensor(config=I2.config, s=ops.s[0])
    v1.set_block(ts=(0, ), val=[1, 0], Ds=(2,))
    v2 = yastn.Tensor(config=I2.config, s=ops.s[0])
    v2.set_block(ts=(0, ), val=[0, 1], Ds=(2,))
    v11 = mps.product_mps([v1, v1.conj()])
    v22 = mps.product_mps([v2, v2.conj()])
    psi3 = (1 / np.sqrt(2)) * (v11 + v22)
    psi[0] = psi3[0]
    psi[1] = psi3[1]

    return merge_sa(psi).canonize_(to='first')


def occupation(N):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    sdn = yastn.Tensor(config=ops.config, s=ops.s)
    sdn.set_block(ts=(0, 0), val=[[0, 0], [0, 1]], Ds=(2, 2))
    sdI = yastn.Tensor(config=ops.config, s=ops.s)
    sdI.set_block(ts=(0, 0), val=[[1, 0], [0, 1]], Ds=(2, 2))
    tmp = [sdn, sdI.conj()] + [ops.I(), ops.I().conj()] * (N - 1)
    return merge_sa(mps.product_mpo(tmp))



def L(k):
    """ label for L modes """
    return f'L{k}'

def R(k):
    """ label for R modes """
    return f'R{k}'

def S(k):
    """ label for S modes """
    return f'S{k}'

def D(k):
    """ label for D modes """
    return f'D{k}'


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


def Hamiltonian_dpt_4U_position(NW, muL, muR, muDs, vS, U, w0=1, order='DLSR', sym='U1'):
    """ generate MPO for DPT in position basis. first dot is interacting with 4 sites """
    #
    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym)
    NS = 4
    #
    vL = vR = vLR = w0
    #
    if order == 'DLSR':
        sites = [D(1)]  # will have SLR geomery
        sites += [L(k) for k in range(NW, 0, -1)]  # 'L1' is for L mode at the junction
        sites += [S(k) for k in range(1, NS + 1)]  # 'S1' connected to L1
        sites += [R(k) for k in range(1, NW + 1)]  # 'R1' is for R mode at the junction
    else:
        sites = [L(k) for k in range(NW, 0, -1)]  # 'L1' is for L mode at the junction
        sites += [S(1), S(2), D(1), S(3), S(4)]
        sites += [R(k) for k in range(1, NW + 1)]  # 'R1' is for R mode at the junction


    # here sites are ordered in position
    s2i = {s: i for i, s in enumerate(sites)}
    i2s = {i: s for i, s in enumerate(sites)}
    II = mps.product_mpo([dI if 'D' in site else qI for site in sites])

    try:
        lD = len(muDs)
        assert lD == 2
    except TypeError:
        muDs = [muDs] * 2

    muSs = [muL, muL, muR, muR]

    terms = []
    for k in range(1, NW + 1): # on-site energies
        terms.append((muL, [L(k)], [qn]))
        terms.append((muR, [R(k)], [qn]))
    for k, ms in enumerate(muSs, start=1): # on-site energies
        terms.append((ms, [S(k)], [qn]))

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


def Hamiltonian_dpt_4U_momentum(NW, muL, muR, muDs, vS, U, w0=1, order='DLSR', sym='U1'):
    """ generate mpo for dpt model in mixed basis """

    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym)
    NS = 4
    #
    vL = vR = vLR = w0
    #
    NW1 = NW + NS // 2 + 1
    #
    sites = [D(1)]  # will have SLR geomery
    sites += [L(k) for k in range(1, NW1)]  # 'L1' is for L mode at the junction
    sites += [R(k) for k in range(1, NW1)]  # 'R1' is for R mode at the junction
    # here sites are ordered in position
    s2i = {s: i for i, s in enumerate(sites)}
    i2s = {i: s for i, s in enumerate(sites)}
    II = mps.product_mpo([dI if 'D' in site else qI for site in sites])

    try:
        lD = len(muDs)
        assert lD == 2
    except TypeError:
        muDs = [muDs] * 2

    muSs = [muL, muL, muR, muR]

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
            amp = U * (2 / NW1) * np.sin(np.pi * 1 * k1 / NW1) * np.sin(np.pi * 1 * k2 / NW1)
            terms.append((amp, [D(1), L(k1), L(k2)], [dn1 - dI / 2, qcp, qc]))
            terms.append((amp, [D(1), R(k1), R(k2)], [dn1 - dI / 2, qcp, qc]))
            amp = U * (2 / NW1) * np.sin(np.pi * 2 * k1 / NW1) * np.sin(np.pi * 2 * k2 / NW1)
            terms.append((amp, [D(1), L(k1), L(k2)], [dn1 - dI / 2, qcp, qc]))
            terms.append((amp, [D(1), R(k1), R(k2)], [dn1 - dI / 2, qcp, qc]))
    terms.append((-4 * U / 2, [D(1)], [dn1 - dI / 2]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(II, Hterms)
    return H, s2i, i2s



def Hamiltonian_dpt_4U_mixed(NW, muL, muR, muDs, vS, U, w0=1, order='DLSR', sym='U1'):
    """ generate mpo for dpt model in mixed basis """

    qI, qc, qcp, qn, dx, dn1, dn2, dI, m12, m21 = local_operators(sym=sym)
    NS = 4
    #
    vL = vR = vLR = w0
    #
    NW1 = NW + 1
    #
    # sites = [D(1)]  # will have SLR geomery
    # sites += [L(k) for k in range(1, NW1)]  # 'L1' is for L mode at the junction
    # sites += [S(k) for k in range(1, NS + 1)]  # 'S1' connected to L1
    # sites += [R(k) for k in range(1, NW1)]  # 'R1' is for R mode at the junction

    sites = [D(1)]  # will have SLR geomery

    sites = []
    for k in range(1, NW1):
        sites.append(L(k))
        sites.append(R(k))
    sites = sites[:NW1] + ['S1', 'S2', 'D1', 'S3', 'S4'] + sites[NW1:]

    # sites += [L(k) for k in range(1, NW1)]  # 'L1' is for L mode at the junction
    # sites += [S(k) for k in range(1, NS + 1)]  # 'S1' connected to L1
    # sites += [R(k) for k in range(1, NW1)]  # 'R1' is for R mode at the junction

    # here sites are ordered in position
    s2i = {s: i for i, s in enumerate(sites)}
    i2s = {i: s for i, s in enumerate(sites)}
    II = mps.product_mpo([dI if 'D' in site else qI for site in sites])

    try:
        lD = len(muDs)
        assert lD == 2
    except TypeError:
        muDs = [muDs] * 2

    muSs = [muL, muL, muR, muR]

    terms = []
    for k in range(1, NW1):  # on-site energies
        terms.append((muL + 2 * vL * np.cos(np.pi * k / NW1), [L(k)], [qn]))
        terms.append((muR + 2 * vR * np.cos(np.pi * k / NW1), [R(k)], [qn]))

    for k, ms in enumerate(muSs, start=1): # on-site energies
        terms.append((ms, [S(k)], [qn]))

    terms.append((muDs[0], [D(1)], [dn1]))
    terms.append((muDs[1], [D(1)], [dn2]))
    terms.append((vS, [D(1)], [dx]))

    for k in range(1, NW1):  # on-site energies
        terms.append((vLR * np.sin(np.pi * k / NW1) * np.sqrt(2 / NW1), [S(1), R(k)], [qcp, qc]))
        terms.append((vLR * np.sin(np.pi * k / NW1) * np.sqrt(2 / NW1), [R(k), S(1)], [qcp, qc]))
        terms.append((vLR * np.sin(np.pi * k / NW1) * np.sqrt(2 / NW1), [L(k), S(NS)], [qcp, qc]))
        terms.append((vLR * np.sin(np.pi * k / NW1) * np.sqrt(2 / NW1), [S(NS), L(k)], [qcp, qc]))

    for k in range(1, NS):
        terms.append((vLR, [S(k), S(k+1)], [qcp, qc]))
        terms.append((vLR, [S(k+1), S(k)], [qcp, qc]))

    # add Coulomb interactions
    for k in range(1, NS + 1):
        terms.append((U, (D(1), S(k)), [dn1 - dI / 2, qn - qI / 2]))

    Hterms = [mps.Hterm(v, tuple(s2i[x] for x in p), o) for v, p, o in terms]
    H = mps.generate_mpo(II, Hterms)
    return H, s2i, i2s
