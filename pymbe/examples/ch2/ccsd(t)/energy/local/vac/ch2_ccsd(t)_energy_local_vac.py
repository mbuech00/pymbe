import os
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, lo, ao2mo
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

        # mo coefficients for pipek-mezey localized orbitals
        mo_coeff = hf.mo_coeff.copy()

        # occupied - occupied block
        mask = hf.mo_occ == 2.0
        mask[:ncore] = False
        if np.any(mask):
            loc = lo.PM(mol, mo_coeff=mo_coeff[:, mask]).set(conv_tol=1.0e-10)
            mo_coeff[:, mask] = loc.kernel()

        # singly occupied - singly occupied block
        mask = hf.mo_occ == 1.0
        if np.any(mask):
            loc = lo.PM(mol, mo_coeff=mo_coeff[:, mask]).set(conv_tol=1.0e-10)
            mo_coeff[:, mask] = loc.kernel()

        # virtual - virtual block
        mask = hf.mo_occ == 0.0
        if np.any(mask):
            loc = lo.PM(mol, mo_coeff=mo_coeff[:, mask]).set(conv_tol=1.0e-10)
            mo_coeff[:, mask] = loc.kernel()

        # reference space
        ref_space = np.array([3, 4], dtype=np.int64)

        # expansion space
        exp_space = np.array(
            [i for i in range(ncore, mol.nao) if i not in ref_space],
            dtype=np.int64,
        )

        # hcore
        hcore_ao = hf.get_hcore()
        hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

        # eri
        eri_ao = mol.intor("int2e_sph", aosym="s8")
        eri = ao2mo.incore.full(eri_ao, mo_coeff)

        # create mbe object
        mbe = MBE(
            method="ccsd(t)",
            mol=mol,
            orb_type="local",
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
