import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints


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
        _, orbsym, mo_coeff = hf(mol)

        # expansion space
        exp_space = np.arange(ncore, mol.nao, dtype=np.int64)

        # integral calculation
        hcore, eri = ints(mol, mo_coeff)

        # create mbe object
        mbe = MBE(
            method="ccsd",
            target="rdm12",
            mol=mol,
            orbsym=orbsym,
            hcore=hcore,
            eri=eri,
            exp_space=exp_space,
            rst=rst,
        )

        # perform calculation
        rdm12 = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        rdm12 = mbe.kernel()

    return rdm12


if __name__ == "__main__":

    # call example function
    rdm12 = mbe_example()

    # finalize mpi
    MPI.Finalize()
