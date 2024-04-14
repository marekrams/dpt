import ray
import time
from routines import generate_Hamiltonian, generate_gs, generate_quench



def main_gs():
    to_scan = []
    for vS in [0.0]:
        for U in [2.0,]:
            for NW in [16]:
                param0 = {'basis': '4U_mixed',
                          'order': 'mixed_0.5',
                          # 'basis': '4U_position',
                          # 'order': 'SLR',
                          'NW': NW,
                          'muL': 0.0,
                          'muR': 0.0,
                          'muS': 0.0,
                          'dmuS': -99.0,
                          'vS': vS,
                          'U': U,
                          'sym': 'U1xU1',
                          'w0': 1.0,
                          'occS': 1,
                          'occLR': NW,
                          'states': 1,
                          'max_sweeps2': 8,
                          'max_sweeps1': 64,
                          'Schmidt_tol': 1e-6,
                          'Ds': [16, 32]}
                generate_gs(param0)
                to_scan.append(generate_gs_ray.remote(param0))
    ray.get(to_scan)

@ray.remote
def generate_gs_ray(param0):
    return generate_gs(param0)

@ray.remote
def generate_quench_ray(param0, param1):
    return generate_quench(param0, param1)

def main_evol():
    ray.shutdown()
    ray.init()

    refs = []

    for vS in [0.0]:
        for U in [2.0, 5.0]:
            for NW in [16]:
                param0 = {'basis': '4U_mixed',
                          'order': 'mixed_0.5',
                          # 'basis': '4U_position',
                          # 'order': 'SLR',
                          'NW': NW,
                          'muL': 0.0,
                          'muR': 0.0,
                          'muS': 0.0,
                          'dmuS': -99.0,
                          'vS': vS,
                          'U': U,
                          'sym': 'U1xU1',
                          'w0': 1.0,
                          'occS': 1,
                          'occLR': NW,
                          'states': 1,
                          'max_sweeps2': 8,
                          'max_sweeps1': 64,
                          'Schmidt_tol': 1e-6,
                          'Ds': [16, 32]}

                param1 = param0.copy()
                param1["U"] = U
                param1["dmuS"] = 0.0
                param1["muL"] = -0.25
                param1["muR"] = 0.25

                param1["tolS"] = 1e-6
                param1["D0"] = 32
                param1["D1"] = 32
                param1["dt"] = 0.125
                param1["time"] = 16
                refs.append(generate_quench_ray.remote(param0, param1))

    ray.get(refs)



if __name__ == '__main__':
    main_gs()
    # main_evol()
