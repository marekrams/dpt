import numpy as np
import ray
import yastn
import yastn.tn.mps as mps
from ._hamiltonians import SpinlessFermions2ch, local_operators,  initial_position

from ._hamiltonians import Hamiltonian_dpt_4U_mixed, Hamiltonian_dpt_2U_position,  Hamiltonian_dpt_4U_position, Hamiltonian_dpt_RLM_position, Hamiltonian_dpt_RLM_4U_position
from ._files import param_hamiltonian, fname_hamiltonian, param_gs, fname_gs, fname_quench

@ray.remote
def generate_Hamiltonian_ray(param):
    return generate_Hamiltonian(param)

def generate_Hamiltonian(param):
    """ main function to generate Hamiltonian"""

    fname = fname_hamiltonian(param)

    if fname.is_file():
        with open(fname, 'rb') as f:
            data = np.load(f, allow_pickle=True).item()
        H = clear_mpsmpo(data["H"], param)
        s2i = data["s2i"]
        i2s = data["i2s"]
        return H, s2i, i2s

    gHs = {'4U_mixed': Hamiltonian_dpt_4U_mixed,
           '2U_position': Hamiltonian_dpt_2U_position,
           '4U_position': Hamiltonian_dpt_4U_position,
           'RLM_position': Hamiltonian_dpt_RLM_position,
           'RLM_4U_position': Hamiltonian_dpt_RLM_4U_position}
    gH = gHs[param["basis"]]

    H, s2i, i2s = gH(NW=param["NW"],
                     muL=param["muL"],
                     muR=param["muR"],
                     muS=param["muS"],
                     dmuS=param["dmuS"],
                     vS=param["vS"],
                     U=param["U"],
                     w0=param["w0"],
                     order=param["order"],
                     sym=param["sym"])

    data = {"param": param_hamiltonian(param),
            "H": H.save_to_dict(),
            "s2i": s2i,
            "i2s": i2s}

    with open(fname, 'wb') as f:
        np.save(f, data, allow_pickle=True)

    return H, s2i, i2s


def clear_mpsmpo(psi, param):
    if isinstance(psi, dict):
        if len(param["sym"]) == 2:
            ops = yastn.operators.SpinlessFermions(sym=param["sym"])
        else:
            ops = SpinlessFermions2ch(sym=param["sym"])
        psi = mps.load_from_dict(ops.config, psi)
    return psi


def generate_gs(param, reset=False):
    fname = fname_gs(param)

    if not reset and fname.is_file():
        with open(fname, 'rb') as f:
            data = np.load(f, allow_pickle=True).item()
    else:
        data = {'param': param_gs(param),
                'states': {}}

    H, s2i, i2s = generate_Hamiltonian(param)
    ops = local_operators(sym=param["sym"])
    Ons = {k: ops['n0' if v[0] == 'S' else 'n1'] for k, v in i2s.items()}

    project = []
    for state_index in range(param["states"]):
        print(f" Calculating state = {state_index} ")
        for D in sorted(param["Ds"]):
            print(f" Calculating D = {D} ")
            Dkeys = [d for (d, i) in data['states'].keys() if i == state_index]
            if Dkeys:
                dinit = max(dd for dd in Dkeys if dd <= D)
                psi = data['states'][dinit]
            else:
                psi = initial_position(H, param) #
            psi = clear_mpsmpo(psi, param)

            opts_svd = {"D_total": D}
            info = mps.dmrg_(psi, H, project=project, method='2site', opts_svd=opts_svd,
                            max_sweeps=param["max_sweeps2"], Schmidt_tol=param["Schmidt_tol"])
            print(info.energy)

            info = mps.dmrg_(psi, H, project=project, method='1site',
                             max_sweeps=param["max_sweeps1"], Schmidt_tol=param["Schmidt_tol"])
            print(info.energy)

            psi0 = psi.save_to_dict()
            entropy = psi.get_entropy()
            occ = mps.measure_1site(psi, Ons, psi)
            occ = {i2s[k]: v for k, v in occ.items()}

            psi0["info"] = info
            psi0["energy"] = info.energy
            psi0["entropy"] = entropy
            psi0["occ"] = occ
            data['states'][state_index, D] = psi0

        project.append(psi.copy())

    with open(fname, 'wb') as f:
        np.save(f, data, allow_pickle=True)

    return psi


def generate_quench(param0, param1):
    fname0 = fname_gs(param0)

    with open(fname0, 'rb') as f:
        data = np.load(f, allow_pickle=True).item()

    D0 = param1["D0"]
    if not isinstance(D0, int):
        D0 = D0[-1]

    psi = clear_mpsmpo(data['states'][0, D0], param0)

    H, s2i, i2s = generate_Hamiltonian(param1)
    ops = local_operators(sym=param0["sym"])
    Ons = {k: ops['n0' if v[0] == 'S' else 'n1'] for k, v in i2s.items()}

    opts_svd = {"D_total": param1["D1"], "tol": param1["tolS"]}
    dt = param1["dt"]
    times = [0, dt/16, dt/8, dt/4, dt/2] + list(np.arange(dt, param1["time"] + dt/2, dt))

    ents = np.zeros((len(times), len(H) + 1), dtype=np.float64)
    occs = np.zeros((len(times), len(H)), dtype=np.float64)

    ii = 0
    entropy = psi.get_entropy()
    occ = mps.measure_1site(psi, Ons, psi)
    ents[ii] = entropy
    for k, v in occ.items():
        occs[ii, k] = v
    ii += 1

    for info in mps.tdvp_(psi, H, method='12site', opts_svd=opts_svd, dt=dt, times=times):
        entropy = psi.get_entropy()
        occ = mps.measure_1site(psi, Ons, psi)
        ents[ii] = entropy
        for k, v in occ.items():
            occs[ii, k] = v
        ii += 1
        print(info)

    data = {"param0": param0, "param1": param1, "times": times, "occs": occs, "ents": ents}

    fname1 = fname_quench(param0, param1)
    with open(fname1, 'wb') as f:
        np.save(f, data, allow_pickle=True)

    return psi