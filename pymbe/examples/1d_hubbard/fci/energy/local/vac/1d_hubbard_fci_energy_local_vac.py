import os
import numpy as np
from math import floor
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, ref_mo, ints


def mbe_example(rst=True) -> Optional[Union[float, np.ndarray]]:

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

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
        nocc, nvirt, norb, hf_object, hf_energy, _, occup, orbsym, mo_coeff = hf(
            mol, u=u, matrix=matrix, pbc=pbc
        )

        # pipek-mezey localized orbitals
        mo_coeff, orbsym = ref_mo(
            "local", mol, hf_object, mo_coeff, occup, orbsym, norb, 0, nocc, nvirt
        )

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc, u=u, matrix=matrix, pbc=pbc)

        # create mbe object
        mbe = MBE(
            mol=mol,
            nocc=nocc,
            norb=norb,
            hf_prop=hf_energy,
            occup=occup,
            orbsym=orbsym,
            hcore=hcore,
            vhf=vhf,
            eri=eri,
            rst=rst,
        )

        # perform calculation
        energy = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        energy = mbe.kernel()

    return energy


if __name__ == "__main__":

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
