import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints


def mbe_example(rst=True):

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

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

        # frozen core
        ncore = 1

        # hf calculation
        nocc, _, norb, _, hf_prop, occup, orbsym, mo_coeff = hf(mol, target="rdm12")

        # reference space
        ref_space = np.array([1, 2, 3, 4], dtype=np.int64)

        # integral calculation
        hcore, eri, vhf = ints(mol, mo_coeff, norb, nocc)

        # create mbe object
        mbe = MBE(
            method="fci",
            target="rdm12",
            mol=mol,
            ncore=ncore,
            nocc=nocc,
            norb=norb,
            orbsym=orbsym,
            hf_prop=hf_prop,
            occup=occup,
            hcore=hcore,
            eri=eri,
            vhf=vhf,
            ref_space=ref_space,
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
