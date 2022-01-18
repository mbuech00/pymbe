import os
from math import floor
import numpy as np
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, ints

def mbe_example() -> Optional[Union[float, np.ndarray]]:

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd()+'/rst'):

        # create mol object
        mol = gto.M(verbose=0)

        # hubbard hamiltonian
        u = 2.0
        n = 1.0
        matrix = (1, 10)
        pbc = True

        # number of electrons
        mol.nelectron = floor(matrix[0] * matrix[1] * n)

        # hf calculation
        nocc, _, norb, _, hf_energy, _, occup, orbsym, \
        mo_coeff = hf(mol, u=u, matrix=matrix, pbc=pbc)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc, u=u, matrix=matrix, \
                               pbc=pbc)

        # create mbe object
        mbe = MBE(mol=mol, nocc=nocc, norb=norb, hf_prop=hf_energy, \
                  occup=occup, orbsym=orbsym, hcore=hcore, vhf=vhf, eri=eri)

        # perform calculation
        energy = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        energy = mbe.kernel()

    return energy

if __name__ == '__main__':

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
