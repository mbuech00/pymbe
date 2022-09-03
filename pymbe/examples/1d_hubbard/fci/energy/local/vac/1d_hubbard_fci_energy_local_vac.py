import os
import numpy as np
from math import floor
from mpi4py import MPI
from pyscf import gto, scf, lo, ao2mo
from pymbe import MBE


def mbe_example(rst=True):

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # create mol object
        mol = gto.M(verbose=0)

        # hubbard hamiltonian
        u = 2.0
        n = 1.0
        matrix = (1, 10)

        # number of orbitals
        mol.nao = matrix[1]

        # number of electrons
        mol.nelectron = floor(matrix[0] * matrix[1] * n)

        # number of sites
        nsites = matrix[0] * matrix[1]

        # hcore_ao
        hcore_ao = np.zeros([nsites] * 2, dtype=np.float64)
        for i in range(nsites - 1):
            hcore_ao[i, i + 1] = hcore_ao[i + 1, i] = -1.0  # adjacent sites
        hcore_ao[-1, 0] = hcore_ao[0, -1] = -1.0  # pbc

        # eri_ao
        eri_ao = np.zeros([nsites] * 4, dtype=np.float64)
        for i in range(nsites):
            eri_ao[i, i, i, i] = u

        # initialize restricted hf calc
        hf = scf.RHF(mol).set(conv_tol=1e-10)
        hf.get_ovlp = lambda *args: np.eye(matrix[0] * matrix[1])
        hf.get_hcore = lambda *args: hcore_ao
        hf._eri = eri_ao
        hf.kernel()

        # mo coefficients for pipek-mezey localized orbitals
        mo_coeff = hf.mo_coeff.copy()

        class _hubbard_PM(lo.pipek.PM):
            """
            this class constructs the site-population tensor for each orbital-pair
            density, see: pyscf example - 40-hubbard_model_PM_localization.py
            """

            def atomic_pops(self, mol, mo_coeff, method=None) -> np.ndarray:
                """
                this function overwrites the tensor used in the pm cost function and
                its gradients
                """
                return np.einsum("pi,pj->pij", mo_coeff, mo_coeff)

        # occupied - occupied block
        mask = hf.mo_occ == 2.0
        if np.any(mask):
            loc = _hubbard_PM(mol, mo_coeff=mo_coeff[:, mask]).set(conv_tol=1.0e-10)
            mo_coeff[:, mask] = loc.kernel()

        # virtual - virtual block
        mask = hf.mo_occ == 0.0
        if np.any(mask):
            loc = _hubbard_PM(mol, mo_coeff=mo_coeff[:, mask]).set(conv_tol=1.0e-10)
            mo_coeff[:, mask] = loc.kernel()

        # hcore
        hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

        # eri
        eri = ao2mo.incore.full(eri_ao, mo_coeff)

        # create mbe object
        mbe = MBE(mol=mol, hcore=hcore, eri=eri, rst=rst)

    else:

        # create mbe object
        mbe = MBE()

    # perform calculation
    elec_energy = mbe.kernel()

    return elec_energy


if __name__ == "__main__":

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
