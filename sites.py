import numpy as np

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



def mixed_order(NW, muL, muR, vL, vR):

    NW1 = NW + 1
    raw = [(muR + 2 * vR * np.cos(np.pi * k / NW1), R(k)) for k in range(1, NW1)] + \
        [(muL + 2 * vL * np.cos(np.pi * k / NW1), L(k)) for k in range(1, NW1)] 
    
    sites = [s for _, s in sorted(raw, reverse=True)]
    return sites



def SDS_sites(NS):
    sites =  [S(k) for k in range(1, NS // 2 + 1)]
    sites += [D(1)]
    sites += [S(k) for k in range(NS // 2 + 1, NS + 1)]
    return sites

def S_sites(NS):
    return [S(k) for k in range(1, NS + 1)]

def order_sites(mapping, order, NW, NS=4, muL =0.0, muR = 0.0, vL = 1.0, vR = 1.0):
    """ predefined ordering of sites """
    if mapping == 'position':
        if order == 'DLSR':
            sites =  [D(1)]
            sites += [L(k) for k in range(NW, 0, -1)]  # 'L1' is for L mode at the junction
            sites += S_sites(NS)  # 'S1' connected to L1
            sites += [R(k) for k in range(1, NW + 1)]  # 'R1' is for R mode at the junction
        elif order == 'LSDSR':
            sites =  [L(k) for k in range(NW, 0, -1)]  # 'L1' is for L mode at the junction
            sites += SDS_sites(NS)
            sites += [R(k) for k in range(1, NW + 1)]  # 'R1' is for R mode at the junction
        else:
            raise ValueError("For mapping='position' select order from 'DLSR', LSDSR'.")
        return sites

    if mapping == 'mixed':
        if order in ['DLSR', 'LSDSR']:
            return order_sites('position', order, NW, NS)
        if order in ['LRSDSLR', 'DLRSLR']:
            # sites = []
            # for k in range(1, NW + 1):
            #     sites.append(L(k))
            #     sites.append(R(k))
            sites = mixed_order(NW, muL, muR, vL, vR)
            if order == 'LRSDSLR':
                return sites[:NW] + SDS_sites(NS) + sites[NW:]
            if order == 'DLRSLR':
                return [D(1)] + sites[:NW] + S_sites(NS) + sites[NW:]
        else:
            raise ValueError("For 'mixed' select order from 'DLSR', LSDSR', 'LRSDSLR', 'DLRSLR'.")

    if mapping == 'momentum':
        if order == 'LDR':
            return order_sites('position', "LSDSR", NW + NS // 2, NS=0)
        if order == 'DLR':
            return order_sites('position', "DLSR", NW + NS // 2, NS=0)
        if order == 'LRDLR':
            return order_sites('mixed', "LRSDSLR", NW + NS // 2, NS=0)
        if order == 'DLRLR':
            return order_sites('mixed', "DLRSLR", NW + NS // 2, NS=0)
        raise ValueError("For 'momentum' select order from 'DLR', LDR', 'LRDLR', 'DLRLR'.")

    raise ValueError("mapping should be 'mixed', 'momentum', or 'position'.")

