import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints, dipole_ints, ref_prop


def mbe_example(rst=True):

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

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

        # frozen core
        ncore = 1

        # hf calculation
        nocc, _, norb, _, _, occup, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # integral calculation
        hcore, eri, vhf = ints(mol, mo_coeff, norb, nocc)

        # gauge origin
        gauge_origin = np.array([0.0, 0.0, 0.0])

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # reference property
        ref_trans = ref_prop(
            mol,
            hcore,
            eri,
            occup,
            orbsym,
            nocc,
            norb,
            ref_space,
            fci_solver="pyscf_spin1",
            target="trans",
            fci_state_sym="b2",
            fci_state_root=1,
            vhf=vhf,
            dipole_ints=dip_ints,
        )

        # create mbe object
        mbe = MBE(
            method="fci",
            target="trans",
            fci_solver="pyscf_spin1",
            mol=mol,
            ncore=ncore,
            nocc=nocc,
            norb=norb,
            orbsym=orbsym,
            fci_state_sym="b2",
            fci_state_root=1,
            occup=occup,
            hcore=hcore,
            eri=eri,
            vhf=vhf,
            dipole_ints=dip_ints,
            ref_space=ref_space,
            ref_prop=ref_trans,
            rst=rst,
        )

        # perform calculation
        trans = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        trans = mbe.kernel()

    return energy


if __name__ == "__main__":

    # call example function
    trans = mbe_example()

    # finalize mpi
    MPI.Finalize()
