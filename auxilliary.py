import numpy as np
import yastn
import yastn.tn.mps as mps


def op1site(op, site, s2i, qI, dI):
    ii = s2i[site]
    sd = s2i['D1']
    oo = sd * [qI] + [dI] + (len(s2i) - 1 - sd) * [qI]
    oo[ii] = op
    return mps.product_mpo(oo)


def merge_mpo(O, s2i, merge=True):
    if not merge:
        return O
    ii = sorted([s2i['D1'], s2i['S1'], s2i['S2'], s2i['S3'], s2i['S4']])
    mi = min(ii)

    assert ii == list(range(mi, mi+5)), "Sites should be neighbouring"
    Onew = mps.Mpo(N=len(O) - 4)

    for ii in range(mi):
        Onew[ii] = O[ii]
    for ii in range(mi + 5, len(O)):
        Onew[ii - 4] = O[ii]
    X = yastn.ncon([O[mi], O[mi + 1], O[mi + 2], O[mi + 3], O[mi + 4]], [[-0, -1, 1, -2], [1, -3, 2, -4], [2, -5, 3, -6], [3, -7, 4, -8], [4, -9, -10, -11]])
    Onew[mi] = yastn.fuse_legs(X, axes=(0, (1, 3, 5, 7, 9), 10, (2, 4, 6, 8, 11)))
    return Onew


def merge_mps(O, s2i, merge=True):
    if not merge:
        return O
    ii = sorted([s2i['D1'], s2i['S1'], s2i['S2'], s2i['S3'], s2i['S4']])
    mi = min(ii)

    assert ii == list(range(mi, mi+5)), "Sites should be neighbouring"
    Onew = mps.Mps(N=len(O) - 4)

    for ii in range(mi):
        Onew[ii] = O[ii]
    for ii in range(mi + 5, len(O)):
        Onew[ii - 4] = O[ii]
    X = yastn.ncon([O[mi], O[mi + 1], O[mi + 2], O[mi + 3], O[mi + 4]], [[-0, -1, 1], [1, -2, 2], [2, -3, 3], [3, -4, 4], [4, -5, -6]])
    Onew[mi] = yastn.fuse_legs(X, axes=(0, (1, 2, 3, 4, 5), 6))
    return Onew
