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
        "U": 3.0,
        "L": 64,
        "R": 64,
        "tswitch": 0,
        "tfin" : 64.0,
        "timestep": 0.25,
        "TEdim": 64,
        "mixed": True,
        "ordering": "SORTED",
        "QPCmixed": False,
        "sweepcnt" : 20,
        "ddposition": "R",
        "vs" : 0.25,
        "avg" : False,
        "switchinterval" : 1,
        "initdd" : "UPPER",
        "QN" : True,
        "biasLR" : 0.0,
        "mode" : "disconnectDD",
        "fitmode" : "linear",
        "n1init" : 1.0,
        "repeat" : 10,
        "stagetype" : "uniform",
        "ifshuffle" : False,
        "Trotterfirst" : False,
        "method" : "TDVP",
        'lo' : 0.5,
        'hi' : 1.0,
        'searchmode' : 'iterative'
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

        print(target)

        for key, val in target.items():

            changed = 0
            if key in new:
                
                changed = 1
                new[key] = val


            if changed == 0:
                raise(ValueError("category key '{}' does not match paras!".format(key)))



        #target_path = '/home1/10407/kyvine/MPS/src'
        target_path = '/wrk/knl20/TN/src/'
        #target_path = '/wrk/knl20/ITensor/newsrc/'
        #target_path = '/home/knl20/Desktop/github/Disordered-Hubbard-ITensor/src/'
        #target_path = '/wrk/knl20/ITensor/ref/'
        #target_path = '/Users/knl20/Desktop/Code/TN/src/'

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




def get_n1init(L, U, key, dim = None, mixed = None) :

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

    else:
        raise ValueError("Unrecognized type")

def DPT_yastn():

    Ls = [32]
    dims = [128]
    #Us = [2.9, 2.95, 2.975, 3.0, 3.025, 3.05, 3.1, 3.15, 3.2 ,3.3]
    Us = [3.025, 3.05, 3.075]
    biases = [0.0]
    repeat = 40
    vss  = [1/4]
    taus = [1/8]
    
    for _, L in enumerate(Ls):

        for _, bias in enumerate(biases):

            #tfin = L * 0.9
            tfin = L * 3 / 4
            tswitch = L /4

            for k, U in enumerate(Us):

                for mixed in [True, False]:
                    dpt_single = {
                        "U": [U], #np.round(np.arange(0.1, 3.0, 0.05), 4),
                        "L" : [L],
                        "tfin" : [tfin],
                        "TEdim": dims,
                        "mixed": [mixed],
                        "vs" : vss,
                        "biasLR" : [bias],
                        "n1init" : get_n1init(L, U, 'Nov14', mixed = mixed),
                        "tswitch" : [tswitch],
                        "timestep": taus,
                        # "lo" : [0.5],
                        # 'hi' : [1.0],
                        "repeat" : [repeat],
                        "searchmode" : ['iterative']
                        #"QN": [True, False],
                        #"ddposition" : ["M", "R", "avg"],
                        #"stagetype": ["expadiabatic"],
                        #"ifshuffle" : [True]
                    }

                    print(dpt_single)
                    gen_dpt_cluster(dpt_single)

# def DPT():
def DPT_yastn_comp():

    Ls = [18, 34, 66]
    dims = [128]
    Us = [2.5, 3.0, 3.5]
    biases = [0.0]
    repeat = 40
    vss  = [1/4]
    taus = [1/8]
    
    for _, L in enumerate(Ls):

        for _, bias in enumerate(biases):

            #tfin = L * 0.9
            tfin = (L - 2) * 3 / 4
            tswitch = (L - 2) /4

            for k, U in enumerate(Us):

                dpt_single = {
                    "U": [U], #np.round(np.arange(0.1, 3.0, 0.05), 4),
                    "L" : [L],
                    "tfin" : [tfin],
                    "TEdim": dims,
                    "mixed": [True],
                    "vs" : vss,
                    "biasLR" : [bias],
                    "tswitch" : [tswitch],
                    "timestep": taus,
                    "n1init" : [0.6, 0.75, 0.9],
                    "repeat" : [repeat],
                    #"QN": [True, False],
                    "ddposition" : ["R"],
                    "stagetype": ["expadiabatic"],
                    #"ifshuffle" : [True]
                }

                print(dpt_single)
                gen_dpt_cluster(dpt_single)



def DPT_local_Trotter():

    Ls = [10]
    dims = [64]
    Us = [3.05]
    biases = [0.0]
    repeat = 1
    vss  = [1/4]
    taus = [1/8]
    
    for _, L in enumerate(Ls):

        ts = [ [ (L -2 )/4, (L - 2) * 3/ 4], [0, (L - 2 )/2]]

        for _, bias in enumerate(biases):

            for k, U in enumerate(Us):

                for tswitch, tfin in ts:

                    dpt_single = {
                        "U": [U], #np.round(np.arange(0.1, 3.0, 0.05), 4),
                        "L" : [L],
                        "tfin" : [tfin],
                        "TEdim": dims,
                        "mixed": [True],
                        "vs" : vss,
                        "biasLR" : [bias],
                        "tswitch" : [tswitch],
                        "timestep": taus,
                        "n1init" : [0.7],
                        "repeat" : [repeat],
                        #"QN": [True, False],
                        "ddposition" : ["M", "R", "MR"],
                        "stagetype": ["expadiabatic"],
                        #"ifshuffle" : [True]
                    }

                    print(dpt_single)
                    gen_dpt_cluster(dpt_single)

                    dpt_single = {
                        "U": [U], #np.round(np.arange(0.1, 3.0, 0.05), 4),
                        "L" : [L],
                        "tfin" : [tfin],
                        "TEdim": dims,
                        "mixed": [True],
                        "vs" : vss,
                        "biasLR" : [bias],
                        "tswitch" : [tswitch],
                        "timestep": [1/128],
                        "n1init" : [0.7],
                        "repeat" : [repeat],
                        #"QN": [True, False],
                        "ddposition" : ["M", "R", "MR"],
                        "stagetype": ["expadiabatic"],
                        "method" : ["TEBD"]
                    }

                    print(dpt_single)
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
    DPT_yastn()
    #DPT_local_Trotter()
    #Mar()
    #transient_bias()
    #bias_scan()
    #load_reference(130, 50)
    #DPT()
