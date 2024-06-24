import os
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, symm, ao2mo, scf, symm
from pymbe import MBE
from pymbe.tools import E_IRREPS


def mbe_example(rst=True):
    # create mol object
    mol = gto.Mole()
    mol.build(
        verbose=0,
        output=None,
        atom="""
        C  0.  0.  .7
        C  0.  0. -.7
        """,
        basis="631g",
        symmetry="d2h",
    )

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):
        # frozen core
        ncore = 2

        # hf calculation
        hf = scf.RHF(mol).run(conv_tol=1e-10)

        # orbsym
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)

        # reference space
        ref_space = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)

        # expansion space (no partial pi pairs possible with pi-pruning)
        exp_space = np.array(
            [i for i in range(ncore, mol.nao) if i not in ref_space],
            dtype=np.int64,
        )

        # orbsym of linear point group for pi-pruning
        mol_linear = mol.copy()
        mol_linear = mol_linear.build(0, 0, symmetry="Dooh")
        orbsym_linear = symm.label_orb_symm(
            mol_linear, mol_linear.irrep_id, mol_linear.symm_orb, hf.mo_coeff
        )

        # cluster pi orbitals
        e_mask = np.in1d(orbsym_linear[exp_space], E_IRREPS)
        exp_space = [np.array(orb, dtype=np.int64) for orb in exp_space[~e_mask]] + [
            e_pair for e_pair in exp_space[e_mask].reshape(-1, 2)
        ]

        # hcore
        hcore_ao = hf.get_hcore()
        hcore = np.einsum("pi,pq,qj->ij", hf.mo_coeff, hcore_ao, hf.mo_coeff)

        # eri
        eri_ao = mol.intor("int2e_sph", aosym="s8")
        eri = ao2mo.incore.full(eri_ao, hf.mo_coeff)

        # create mbe object
        mbe = MBE(
            mol=mol,
            orbsym=orbsym,
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
