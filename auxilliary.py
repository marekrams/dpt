import numpy as np
import yastn
import yastn.tn.mps as mps


def op1site(op, site, s2i, qI, dI):
    ii = s2i[site]
    sd = s2i['D1']
    oo = sd * [qI] + [dI] + (len(s2i) - 1 - sd) * [qI]
    oo[ii] = op
    return mps.product_mpo(oo)


def op2site(op1, site1, op2, site2, s2i, qI, dI):


    i1 = s2i[site1]
    i2 = s2i[site2]
    sd = s2i['D1']

    combs = sorted( [ (i1, op1), (i2, op2), (sd, dI)])

    inds = [ i for (i, _) in combs]
    ops = [ op for (_, op) in combs]
    terms =  [mps.Hterm( amplitude=1, positions= inds, operators=ops)]
    H = mps.generate_mpo(qI, terms, N=len(s2i))
    return H



def merge_sites(O, s2i, merge=True):
    if not merge:
        return O

    ii = sorted([v for k, v in s2i.items() if k[0] in 'DS'])  # try to merge 'DS' sites
    mi = min(ii)
    if ii != list(range(mi, mi + len(ii))):  # try to merge 'S' sites
        ii = sorted([v for k, v in s2i.items() if k[0] in 'S'])
        mi = min(ii)
        if ii != list(range(mi, mi + len(ii))):
            return O

    M = len(ii)
    Onew = mps.MpsMpoOBC(N=len(O) - M + 1, nr_phys=O.nr_phys)

    for i in range(mi):
        Onew[i] = O[i]
    for i in range(mi + M, len(O)):
        Onew[i - M + 1] = O[i]

    tensors = [O[i] for i in ii]

    if O.nr_phys == 2:
        inds = [[i, -2 * i - 1, i + 1, -2 * i - 2] for i in range(M)]
        inds[-1][-2] = -2 * M - 1
        axes = (0, [2*i+1 for i in range(M)], 2*M+1, [2*i+2 for i in range(M)])
    elif O.nr_phys == 1:
        inds = [[i, - i - 1, i + 1] for i in range(M)]
        inds[-1][-1] = - M - 1
        axes = (0, [i+1 for i in range(M)], M+1)

    X = yastn.ncon(tensors, inds)
    Onew[mi] = yastn.fuse_legs(X, axes=axes)
    return Onew


def get_current(psi,  s2i, qc, qcp, qI, dI, merge):

    op = merge_sites(op2site(qcp, 'S2', qc, 'S3', s2i, qI, dI), s2i, merge)
    current = 2 * mps.vdot(psi, op, psi).imag
    
    return current