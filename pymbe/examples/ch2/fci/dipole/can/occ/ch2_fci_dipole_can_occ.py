import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, ints, dipole_ints, ref_prop


def mbe_example(rst=True) -> Optional[Union[float, np.ndarray]]:

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
        nocc, _, norb, _, _, hf_dipole, occup, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([1, 2, 3, 4], dtype=np.int64)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # gauge origin
        gauge_origin = np.array([0.0, 0.0, 0.0])

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # reference property
        ref_dipole = ref_prop(
            mol,
            hcore,
            vhf,
            eri,
            occup,
            orbsym,
            nocc,
            ref_space,
            fci_solver="pyscf_spin1",
            fci_state_sym="b2",
            target="dipole",
            hf_prop=hf_dipole,
            dipole_ints=dip_ints,
        )

        # create mbe object
        mbe = MBE(
            method="fci",
            fci_solver="pyscf_spin1",
            target="dipole",
            mol=mol,
            ncore=ncore,
            nocc=nocc,
            norb=norb,
            orbsym=orbsym,
            fci_state_sym="b2",
            hf_prop=hf_dipole,
            occup=occup,
            hcore=hcore,
            vhf=vhf,
            eri=eri,
            dipole_ints=dip_ints,
            ref_space=ref_space,
            ref_prop=ref_dipole,
            rst=rst,
        )

        # perform calculation
        dipole = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        dipole = mbe.kernel()

    return dipole


if __name__ == "__main__":

    # call example function
    dipole = mbe_example()

    # finalize mpi
    MPI.Finalize()
