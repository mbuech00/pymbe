import os
import numpy as np
from math import floor
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ref_mo, ints


def mbe_example(rst=True):

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # create mol object
        mol = gto.M(verbose=0)

        # hubbard hamiltonian
        u = 2.0
        n = 1.0
        matrix = (1, 10)
        pbc = True

        # number of orbitals
        mol.nao = np.array(matrix[1], dtype=np.int64)

        # number of electrons
        mol.nelectron = floor(matrix[0] * matrix[1] * n)

        # hf calculation
        hf_object, orbsym, mo_coeff = hf(mol, u=u, matrix=matrix, pbc=pbc)

        # pipek-mezey localized orbitals
        mo_coeff, orbsym = ref_mo("local", mol, hf_object, mo_coeff, orbsym, 0)

        # integral calculation
        hcore, eri = ints(mol, mo_coeff, u=u, matrix=matrix, pbc=pbc)

        # create mbe object
        mbe = MBE(mol=mol, orbsym=orbsym, hcore=hcore, eri=eri, rst=rst)

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
