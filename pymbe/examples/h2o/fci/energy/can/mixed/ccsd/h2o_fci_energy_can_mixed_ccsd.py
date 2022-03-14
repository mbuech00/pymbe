import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, base, ints, ref_prop


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
        nocc, _, norb, hf_object, hf_prop, occup, orbsym, mo_coeff = hf(mol)

        # base model
        base_energy = base(
            "ccsd", mol, hf_object, mo_coeff, occup, orbsym, norb, ncore, nocc
        )

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # integral calculation
        hcore, eri, vhf = ints(mol, mo_coeff, norb, nocc)

        # reference property
        ref_energy = ref_prop(
            mol,
            hcore,
            eri,
            occup,
            orbsym,
            nocc,
            norb,
            ref_space,
            base_method="ccsd",
            hf_prop=hf_prop,
            vhf=vhf,
        )

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
            hcore=hcore,
            eri=eri,
            vhf=vhf,
            ref_space=ref_space,
            ref_prop=ref_energy,
            base_method="ccsd",
            base_prop=base_energy,
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
