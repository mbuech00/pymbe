import os
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, symm, mcscf, fci, ao2mo
from pymbe import MBE


def mbe_example(rst=True):

    # create mol object
    mol = gto.Mole()
    mol.build(
        verbose=0,
        output=None,
        atom="""
        C  0.00000  0.00000  0.00000
        H  0.98920  0.42714  0.00000
        H -0.98920  0.42714  0.00000
        """,
        basis="631g",
        symmetry="c2v",
        spin=2,
    )

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # frozen core
        ncore = 1

        # hf calculation
        hf = scf.RHF(mol).run(conv_tol=1e-10)

        # orbsym
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)

        # reference space
        ref_space = np.array([1, 2, 3, 4], dtype=np.int64)

        # expansion space
        exp_space = np.array(
            [i for i in range(ncore, mol.nao) if i not in ref_space],
            dtype=np.int64,
        )

        # mo coefficients for casscf orbitals
        mo_coeff = hf.mo_coeff.copy()

        # electrons in active space
        act_nelec = np.array(
            [
                np.count_nonzero(hf.mo_occ[ref_space] > 0.0),
                np.count_nonzero(hf.mo_occ[ref_space] > 1.0),
            ]
        )

        # sorter for active space
        n_core_inact = np.array(
            [i for i in range(max(mol.nelec)) if i not in ref_space], dtype=np.int64
        )
        n_virt_inact = np.array(
            [a for a in range(max(mol.nelec), mol.nao) if a not in ref_space],
            dtype=np.int64,
        )
        sort_casscf = np.concatenate((n_core_inact, ref_space, n_virt_inact))

        # mo coefficients for active space
        mo_coeff_cas = mo_coeff[:, sort_casscf]

        # orbsym for active space
        orbsym_cas = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_cas)

        # initialize casscf
        cas = mcscf.CASSCF(hf, ref_space.size, act_nelec).set(
            conv_tol=1.0e-10,
            max_cycle_macro=500,
            frozen=ncore,
            fcisolver=fci.direct_spin1_symm.FCI(mol).set(
                conv_tol=1.0e-10,
                orbsym=orbsym_cas[ref_space],
                wfnsym="b1",
            ),
        )

        # hf starting guess
        na = fci.cistring.num_strings(ref_space.size, act_nelec[0])
        nb = fci.cistring.num_strings(ref_space.size, act_nelec[1])
        ci0 = np.zeros((na, nb))
        ci0[0, 0] = 1

        # run casscf
        cas.kernel(mo_coeff_cas, ci0=ci0)

        # reorder mo_coeff
        mo_coeff = cas.mo_coeff[:, np.argsort(sort_casscf)]

        # orbital symmetries
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)

        # hcore
        hcore_ao = hf.get_hcore()
        hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

        # eri
        eri_ao = mol.intor("int2e_sph", aosym="s8")
        eri = ao2mo.incore.full(eri_ao, mo_coeff)

        # create mbe object
        mbe = MBE(
            mol=mol,
            orbsym=orbsym,
            fci_state_sym="b1",
            orb_type="casscf",
            hcore=hcore,
            eri=eri,
            ref_space=ref_space,
            exp_space=exp_space,
            rst=rst,
        )

    else:

        # create mbe object
        mbe = MBE()

    # perform calculation
    elec_energy = mbe.kernel()

    # get total energy
    tot_energy = mbe.final_prop(prop_type="total", nuc_prop=mol.energy_nuc().item())

    return tot_energy


if __name__ == "__main__":

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
