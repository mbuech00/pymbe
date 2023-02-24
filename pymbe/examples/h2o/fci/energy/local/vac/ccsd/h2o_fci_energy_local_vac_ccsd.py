import os
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, cc, lo, ao2mo
from pymbe import MBE


def mbe_example(rst=True):

    # create mol object
    mol = gto.Mole()
    mol.build(
        verbose=0,
        output=None,
        atom="""
        O  0.00000000  0.00000000  0.10840502
        H -0.75390364  0.00000000 -0.47943227
        H  0.75390364  0.00000000 -0.47943227
        """,
        basis="631g",
        symmetry="c2v",
    )

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # frozen core
        ncore = 1

        # hf calculation
        hf = scf.RHF(mol).run(conv_tol=1e-10)

        # base model
        ccsd = cc.CCSD(hf).run(
            conv_tol=1.0e-10, conv_tol_normt=1.0e-10, max_cycle=500, frozen=ncore
        )
        base_energy = ccsd.e_corr

        # mo coefficients for pipek-mezey localized orbitals
        mo_coeff = hf.mo_coeff.copy()

        # occupied - occupied block
        mask = hf.mo_occ == 2.0
        mask[:ncore] = False
        if np.any(mask):
            loc = lo.PM(mol, mo_coeff=mo_coeff[:, mask]).set(conv_tol=1.0e-10)
            mo_coeff[:, mask] = loc.kernel()

        # virtual - virtual block
        mask = hf.mo_occ == 0.0
        if np.any(mask):
            loc = lo.PM(mol, mo_coeff=mo_coeff[:, mask]).set(conv_tol=1.0e-10)
            mo_coeff[:, mask] = loc.kernel()

        # expansion space
        exp_space = np.arange(ncore, mol.nao, dtype=np.int64)

        # hcore
        hcore_ao = hf.get_hcore()
        hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

        # eri
        eri_ao = mol.intor("int2e_sph", aosym="s8")
        eri = ao2mo.incore.full(eri_ao, mo_coeff)

        # create mbe object
        mbe = MBE(
            mol=mol,
            orb_type="local",
            hcore=hcore,
            eri=eri,
            exp_space=exp_space,
            base_method="ccsd",
            base_prop=base_energy,
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