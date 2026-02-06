
import numpy as np
import yastn
import yastn.tn.mps as mps
config_kwargs = {"backend": "torch"}


def run_dmrg(phi, H, O_occ, E_target, occ_target, opts_svd, tol, precompute=False):
    r"""
    Run mps.dmrg_ to find the ground state and
    a few low-energy states of the Hamiltonian H.
    Verify resulting energies against known reference solutions.
    """
    #
    # DMRG can look for solutions in a space
    # orthogonal to provided MPSs. We start with empty list,
    # project = [], and keep adding to it previously found eigenstates.
    # This allows us to find the ground state
    # and a few consequative lowest-energy eigenstates.
    #
    project, states = [], []
    for ref_eng, ref_occ  in zip(E_target, occ_target):
        #
        # We find a state and check that its energy and total occupation
        # matches the expected reference values.
        #
        # We copy initial random MPS psi, as
        # yastn.dmrg_ modifies provided input state in place.
        #
        psi = phi.shallow_copy()
        #
        # We set up dmrg_ to terminate iterations
        # when energy is converged within some tolerance.
        # precompute bool argument controls contraction order, see dmrg_ docs.
        #
        out = mps.dmrg_(psi, H, project=project, method='2site',
                        energy_tol=tol / 10, max_sweeps=20, opts_svd=opts_svd,
                        precompute=precompute)
        #
        # Output of _dmrg is a nametuple with information about the run,
        # including the final energy.
        # Occupation number has to be calcuted using measure_mpo.
        #
        eng = out.energy
        occ = mps.measure_mpo(psi, O_occ, psi)
        #
        # Print the result:
        #
        print(f"2site DMRG; energy: {eng:{1}.{8}} / {ref_eng:{1}.{8}}; "
              + f"charge: {occ:{1}.{8}} / {ref_occ}")
        assert abs(eng - ref_eng) < tol * 100
        assert abs(occ - ref_occ) < tol
        #
        # We further iterate with '1site' DMRG
        # and stricter convergence criterion.
        #
        out = mps.dmrg_(psi, H, project=project, method='1site',
                        Schmidt_tol=tol / 10, max_sweeps=20, precompute=precompute)

        eng = mps.measure_mpo(psi, H, psi)
        occ = mps.measure_mpo(psi, O_occ, psi)
        print(f"1site DMRG; energy: {eng:{1}.{8}} / {ref_eng:{1}.{8}}; "
              + f"occupation: {occ:{1}.{8}} / {ref_occ}")
        # test that energy outputed by dmrg is correct
        assert abs(eng - ref_eng) < tol
        assert abs(occ - ref_occ) < tol
        assert abs(eng - out.energy) < tol  # test dmrg_ output information
        #
        # Finally, we add the found state psi to the list of states
        # to be projected out in the next step of the loop.
        #
        penalty = 100
        project.append((penalty, psi))
        states.append(psi)
    return states


def test_dmrg_XX_model_dense(config_kwargs, tol=1e-6):
    """
    Initialize random MPS of dense tensors and
    runs a few sweeps of DMRG with the Hamiltonian of XX model.
    """
    # Knowing the exact solution we can compare it with the DMRG result.
    # In this test we will consider sectors of different occupations.
    #
    # In this example we use yastn.Tensor with no symmetry imposed.
    #
    # Here, the Hamiltonian for N = 7 sites
    # is obtained with the automatic generator.
    #
    N = 7
    #
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    generate = mps.Generator(N=N, operators=ops)
    parameters = {"t": 1.0, "mu": 0.2,
                  "rN": range(N),
                  "rNN": [(i, i+1) for i in range(N - 1)]}
    H_str = r"\sum_{i,j \in rNN} t ( sp_{i} sm_{j} + sp_{j} sm_{i} )"
    H_str += r" + \sum_{j \in rN} mu sp_{j} sm_{j}"
    H = generate.mpo_from_latex(H_str, parameters)
    #
    # and MPO to measure occupation:
    #
    O_occ = generate.mpo_from_latex(r"\sum_{j \in rN} sp_{j} sm_{j}",
                                    parameters)
    #
    # Known energies and occupations of low-energy eigenstates.
    #
    Eng_gs = [-3.427339492125, -3.227339492125, -2.861972627395]
    occ_gs = [3, 4, 2]
    #
    # To standardize this test we fix a seed for random MPS we use.
    #
    generate.random_seed(seed=0)
    #
    # Set options for truncation for '2site' method of mps.dmrg_.
    #
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}
    #
    # Finally run DMRG starting from random MPS psi
    #
    psi = generate.random_mps(D_total=Dmax)
    #
    # A single run for a ground state can be done using:
    # mps.dmrg_(psi, H, method=method,energy_tol=tol/10,
    #           max_sweeps=20, opts_svd=opts_svd)
    #
    # We create a subfunction run_dmrg to explain how to
    # target some sectors for occupation. This is not necessary
    # but we do it for the sake of clarity and testing.
    #
    psis = run_dmrg(psi, H, O_occ, Eng_gs, occ_gs, opts_svd, tol)
    #
    # _dmrg can be executed as a generator to monitor states
    # between dmrg sweeps. This is done by setting `iterator=True`.
    #
    psi = generate.random_mps(D_total=Dmax)
    for step in mps.dmrg_(psi, H, method='1site',
                          max_sweeps=20, iterator=True):
        occ = mps.measure_mpo(psi, O_occ, psi)  # measure occupation
        # here breaks if it is close to the known result.
        if abs(occ.item() - occ_gs[0]) < tol:
            break
    assert step.sweeps < 20

for _ in range(20):
    test_dmrg_XX_model_dense(config_kwargs)