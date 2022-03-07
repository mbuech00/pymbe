import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ref_mo, ints


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
        nocc, nvirt, norb, hf_object, hf_prop, occup, orbsym, mo_coeff = hf(mol)

        # natural orbitals
        mo_coeff, orbsym = ref_mo(
            "ccsd(t)", mol, hf_object, mo_coeff, occup, orbsym, norb, ncore, nocc, nvirt
        )

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # create mbe object
        mbe = MBE(
            method="fci",
            mol=mol,
            ncore=ncore,
            nocc=nocc,
            norb=norb,
            orbsym=orbsym,
            hf_prop=hf_prop,
            occup=occup,
            orb_type="ccsd(t)",
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
