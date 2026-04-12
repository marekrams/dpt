import sys
import json
from copy import deepcopy
import numpy as np
from itertools import product
import os
from shutil import copy2
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from os import path
import pickle
from glob import glob

def changerepeat():

    pwd = os.getcwd()
    ds = glob( f'{pwd}/U*' )

    for d in ds:

        with open(f'{d}/dptpara.json', "r") as f:
            old = json.load(f)
            
        old['repeat'] = 40
        with open(f'{d}/dptpara.json', "w") as f:
            json.dump(old, f, indent = 4)

    



def gen_time(new, timecontrol):

    L = new['Ls']
    t = 0.25

    if timecontrol == "onestage":

        timepara = {
            "t" : t,
            "fin" : float(L)
        }

    

    elif timecontrol == "twostage":

        fin1 = L//128 * t
        timepara = {
            "t1" : t,
            "t2" : 8 * t,
            "fin1" : fin1,
            "fin2" : float(L)
        }

    elif timecontrol == "transient":

        timepara = {
            "t" : 1/32,
            "fin" : 5.0
        }


    elif timecontrol == "short":

        timepara = {
            "t" : 1/4,
            "fin" : 150.0
        }

    else:

        try:
            timepara = {
                "t" : t,
                "fin" : float(timecontrol)
            }
        except:
            raise ValueError("Unrecognized timecontrol!")

    return timepara 


def gen_dpt_cluster(categories : dict, toplevel = ''):


    dptpara = {
        "U": 3.2,
        "L": 8,
        "R": 8,
        "tswitch": 16.0,
        "tfin": 48.0,
        "timestep": 0.125,
        "TEdim": 128,
        "mixed": True,
        "vs": 0.25,
        "biasLR": 0.0,
        "n1init": 0.75,
        "repeat": 1,
        "order": "LRSDSLR",
        "lo" : 0.5,
        "hi" : 1.0,
        "searchmode" : "iterative",
        "max1" : 256,
        "max2" : 64,
        "merge" : True,
        "finaltol" : 1e-5
    }
    keys = list(categories.keys())
    prods = product(*categories.values())
    target_dicts = [ {key : prod[i] for i, key in enumerate(keys)} for prod in prods ]



    for target in target_dicts:


        new = deepcopy(dptpara)

        target["R"] = target["L"] if "R" not in target else target["R"]

        # this is for consistency reason, we remove the previously generated keys
        string_dict = deepcopy(target)
        string =  toplevel + '_'.join([ key + str(val) for key, val in string_dict.items()])

        #print(target)

        for key, val in target.items():

            changed = 0
            if key in new:
                
                changed = 1
                new[key] = val


            if changed == 0:
                raise(ValueError("category key '{}' does not match paras!".format(key)))



        if not os.path.exists(string):
            os.mkdir(string)

        # for file in os.listdir(target_path):

        #     #print(target_path + file)
        #     if os.path.isfile(target_path + file):
        #         copy2( target_path+ file, string)

        with open(string +'/dptpara.json', 'w') as f:
            json.dump(new, f, indent=4)


        # with open(string +'/dptpara.json', 'w') as f:
        #     json.dump(new_sd, f, indent=4)

        # with open(string +'/transportpara.json', 'w') as f:
        #     json.dump(new_transport, f, indent=4)



def inference_pts(nLs, ref):

    f = lambda x, a, b, c : a/x**c + b
    midpoints = []
    Ls = [34, 66, 130]
    for L in Ls:

        Us, inits = load_reference(L, 200)
        indx = np.argwhere( inits > 0.75).flatten()[0]
        U = Us[indx]

        midpoints.append(U)

    p, cov = curve_fit(f, Ls, midpoints)


    print("p: ", p)
    
    res = f(nLs, *p) 

    print("midpoints: ", midpoints)
    print("inferred midpoints: ", res)
    print("ref: ", f(ref, *p))

    return res - f(ref, *p)


def load_reference(f, pts = None, Uref = [], plusminus = 0.0):

    data = np.loadtxt(f)

    U = data[:, 0]
    init = data[:, 1]
    spl = CubicSpline( U, init)
    
    if len(Uref) == 0:
        x = np.linspace( U[0], U[-1], pts)
    else:
        x = list(filter(lambda x: x>= U[0] - 1e-6 and x <=U[-1] + 1e-6, Uref))

    print('U to fit: ', x)
    y = spl(x)

    if plusminus != 0.0:
        y = [[max(0.5, val - plusminus), min(1.0, val + plusminus)] for val in y]
    else:
        y = [[val] for val in y]

    print(' n1 init to fit: ', y)
    
    return x, y

def direct_load(f):

    data = np.loadtxt(f)

    return data[:, 0], data[:, 1:]






def get_n1init(L, U, key, base = None, dim = None, mixed = None) :

    # Guesses for Nov 3 test

    if key == 'Nov3':
        guess = {
            2.95 : [0.51, 0.55],
            3.0 : [0.51, 0.6],
            3.05 : [0.55, 0.6],
            3.1 : [0.55, 0.65],
            3.15 : [0.65, 0.75],
            3.2 : [0.75, 0.85],
            3.3: [0.8, 0.9]
        }

        return guess[U]
    
    elif key == 'Nov9':

        return 0.5, 1.0
    
    elif key == 'Nov11':

        fittingdada = np.loadtxt('../fittingdata/Nov10L32')

        fittingdada = np.round(fittingdada, decimals=10)
        Us = fittingdada[:, 0]
        inits = fittingdada[:, 1]

        d = { Us[i] : [max(0.53, init - 0.03), min(0.98, init + 0.03)] for i, init in enumerate(inits)}

        return d[U]
    
    elif key == 'Nov12':

        fittingdada = np.loadtxt(f'../fittingdata/Nov11L32dim64mixed{mixed}')

        fittingdada = np.round(fittingdada, decimals=10)
        Us = fittingdada[:, 0]
        inits = fittingdada[:, 1]

        d = { Us[i] : [max(0.51, init - 0.03), min(0.98, init + 0.03)] for i, init in enumerate(inits)}

        return d[U]
    
    elif key == 'Nov13':

        fittingdada = np.loadtxt(f'../fittingdata/Nov12L32dim128mixed{mixed}')

        fittingdada = np.round(fittingdada, decimals=10)
        Us = fittingdada[:, 0]
        inits = fittingdada[:, 1]

        d = { Us[i] : [max(0.51, init - 0.04), min(0.98, init + 0.04)] for i, init in enumerate(inits)}
        return d[U]
    
    elif key == 'Nov14':

        fittingdada = np.loadtxt(f'../fittingdata/Nov13fillmixed{mixed}')

        fittingdada = np.round(fittingdada, decimals=10)
        Us = fittingdada[:, 0]
        inits = fittingdada[:, 1]

        d = { Us[i] : [max(0.51, init - 0.06), min(0.98, init + 0.06)] for i, init in enumerate(inits)}

        return d[U]
    
    elif key == 'Nov20':

        # this is the first L = 64, 128 example
        # so the init is pushed up

        fittingdada = np.loadtxt(f'../fittingdata/Nov20L{L}dim128mixed{mixed}')

        fittingdada = np.round(fittingdada, decimals=10)
        Us = fittingdada[:, 0]
        val = fittingdada[:, 1]
        offset = 0.1
        d = { U : [max(0.47, val[i] - offset), min(1.0, val[i] + offset)] for i, U in enumerate(Us)}

        return d[U]
    

    elif key == 'Jan8':

        guess = {
            2.95: [0.6, 0.67],
            2.9: [0.55, 0.63]
        }

        return guess[U]
    
    elif key == 'Jan10':

        guess = {
            2.9 : 0.52,
            2.95 : 0.57,
            3.0 : 0.7016032584841878,
            3.05 : 0.776824925340917,
            3.1 : 0.8388404862710616,
            3.15 : 0.8846358030703229,
            3.2 : 0.9179932059175007,
            3.3 : 0.9534961288439816,
            3.4 : 0.9678187875179863,
            3.5 : 0.9761269329599717,
        }

        return [guess[U] - 0.05, guess[U] + 0.05]
    
    elif key == 'Jan31':

        with open(f'../fittingdata/Jan31results.pkl', 'rb') as f: # Use 'rb' for read binary mode
            data = pickle.load(f)

        for key in data:

            if L in key and U in key:
                arr = data[key]
                mid = np.mean(arr)

                return [mid - 0.01, mid + 0.01]
            
    elif key == 'highdimFeb5':

        with open(f'../fittingdata/Jan31results.pkl', 'rb') as f: # Use 'rb' for read binary mode
            data = pickle.load(f)
        
        if L == 96: 
            L = 128

        for key in data:
            if f'L{L}' in key and f'U{U}' in key:
                arr = data[key]
                mid = np.mean(arr)

                return [mid]    

    elif key == "Feb10vs0.75":

        # hardcoded init data
        Us = [3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0]
        vals = [0.51, 0.52, 0.6, 0.88, 0.93, 0.96, 0.98]

        spl = CubicSpline(Us, vals)
        assert U > min(Us) and U < max(Us)

        val = spl(U)

        guess = [val - 0.05, val + 0.05]
        print(U, guess)
        return np.round(guess, decimals=5)

    elif key == 'Feb12':

        with open(f'../fittingdata/Feb11results.pkl', 'rb') as f: # Use 'rb' for read binary mode
            data = pickle.load(f)
        
        for key in data:
            if f'L{L}' in key and f'U{U}' in key:
                return [data[key] ]
            
    elif key == 'Feb17short':

        assert L == 16

        Lsub = 32
        Usub = np.round(U - 0.2, decimals=5)

        print(Lsub, Usub)

        if Usub > 3.9:
            return [0.9]
        
        with open(f'../fittingdata/Feb17results.pkl', 'rb') as f: # Use 'rb' for read binary mode
            data = pickle.load(f)
        
        for key in data:
            if f'L{Lsub}' in key and f'U{Usub}' in key:
                return [data[key] ]
            
    elif key == 'Feb18':

        assert L == 256

        Lsub = 128
        Usub = np.round(U + 0.05, decimals=5)


        if Usub < 3.3:
            return [0.51]
        
        with open(f'../fittingdata/Feb17results.pkl', 'rb') as f: # Use 'rb' for read binary mode
            data = pickle.load(f)
        
        for key in data:
            if f'L{Lsub}' in key and f'U{Usub}' in key:
                return [data[key] ]
            

    elif key == 'Mar3':

        # we only have L64 data
        Lsub = 64

        if L == 16:
            Usub = np.round(U - 0.3, decimals=1)
        elif L == 32:
            Usub = np.round(U - 0.1, decimals=1)

        if Usub > 4.0:
            return [0.7, 0.95]
        
        with open(f'../fittingdata/Feb17results.pkl', 'rb') as f: # Use 'rb' for read binary mode
            data = pickle.load(f)
        
        for key in data:
            if f'L{Lsub}' in key and f'U{Usub}' in key:
                return [data[key] ]
            

    else:
        raise ValueError("Unrecognized type")


def U_determine(L, bias, tag):

    if bias == 0:

        Us = {
            32 : np.arange(3.4, 3.9, 0.025),
            64 : np.arange(3.3, 3.8, 0.025),
            128: np.arange(3.3, 3.7, 0.025)
        }

    elif tag == 'prodApr9testbias':
        return np.round(np.arange(3.0, 7.0), decimals = 5)


    return np.round(Us[L], decimals = 5)


def bias_determine(L):

    def count_num(L, bias):
        
        k = np.arange(1, L + 1)
        r = 2  * np.cos(np.pi * k / (L + 1)) 
        w = np.concat( (r + bias, r - bias))
        mids = w[ (w >= -bias) & (w <= bias)]
        N = len(mids)

        return N
    
    def search(L, lo, hi):

        mid = (lo + hi) / 2
        N = count_num(L, mid)

        if N == Nref:
            return mid

        if N > Nref:
            return search(L, lo, mid)
        
        else:
            return search(L, mid, hi)
        
        

    # baseline : L32 , bias = 0.25
    Nref = count_num(32, 0.25)
    bnew = search(L, 0, 2)
    return bnew

    



# def DPT_bias_test():

#     Ls = [16, 32]
#     dims = [128]
#     Us = np.round(np.arange(3.5, 4.5, 0.05), decimals=5)
#     #Us = [3.025, 3.05, 3.075]
#     biases = [#0.0,
#               0.25
#               ]
#     repeat = 80
#     vss  = [3/4]
#     taus = [1/8]
#     merge = [True]

#     for _, L in enumerate(Ls):

#         for _, bias in enumerate(biases):

#             #tfin = L * 0.9
#             tfin = L * 7/8

#             for tswitch in [L/4]:

#                 for k, U in enumerate(Us):
                    
#                     dpt_single = {
#                         "U": [U],
#                         "L" : [L],
#                         "tfin" : [tfin],
#                         "TEdim": dims,
#                         "mixed": [True],
#                         "vs" : vss,
#                         "merge" : merge,
#                         "biasLR" : [bias],
#                         "n1init" : get_n1init(L, U, 'Mar3'),
#                         "tswitch" : [tswitch],
#                         "timestep": taus,
#                         "order" : [ 'DLRSLR'],
#                         "repeat" : [repeat],
#                         "searchmode" : ['iterative']
#                     }

#                     #print( dpt_single["U"], dpt_single["n1init"])
#                     gen_dpt_cluster(dpt_single)


def DPT_bias_bs():

    Ls = [32, 48, 64, 96, 128]
    dims = [64, 128, 256]
    
    repeat = 40
    vss  = [3/4]
    taus = [1/8]
    merge = [True]

    for _, L in enumerate(Ls):

        biases = [#0.0,
              #0.25,
              0.5,
              0.75
              ]
        
        
        for _, bias in enumerate(biases):

            Us = U_determine(L, bias, 'prodApr9testbias')
            #tfin = L * 0.9
            tfin = L * 7/8

            #Us = U_determine(L, bias)

            for tswitch in [tfin/4, tfin/2]:

                for k, U in enumerate(Us):
                    
                    dpt_single = {
                        "U": [U],
                        "L" : [L],
                        "tfin" : [tfin],
                        "TEdim": dims,
                        "mixed": [True],
                        "vs" : vss,
                        "merge" : merge,
                        "biasLR" : [bias],
                        #"n1init" : get_n1init(L, U, 'Mar3'),
                        "lo" : [0.5],
                        "hi" : [1.0],
                        "tswitch" : [tswitch],
                        "timestep": taus,
                        "order" : [ 'DLRSLR'],
                        "repeat" : [repeat],
                        "searchmode" : ['binarysearch'],
                        "finaltol" : [1e-3, 1e-5]
                    }

                    #print( dpt_single["U"], dpt_single["n1init"])
                    gen_dpt_cluster(dpt_single)




if __name__ == '__main__':
    

    #generic()
    #scaling()
    #ring()

    # sub = 'tunneling'
    # NF(sub = sub) 

    #inference_pts([34, 66, 130, 258, 514], 34)
    #DPT_single()
    #DPT_repeat()
    #DPT_check()
    #DPT_yastn_comp()
    #DPT_yastn()
    #DPT_bias_test()
    DPT_bias_bs()
    #bias_determine()
    #changerepeat()
    #DPT_yastn_binary()
    #DPT_yastn_test()
    #DPT_local_Trotter()
    #Mar()
    #transient_bias()
    #bias_scan()
    #load_reference(130, 50)
    #DPT()
