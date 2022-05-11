import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints, dipole_ints, nuc_dipole


def mbe_example(rst=True):

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

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # frozen core
        ncore = 1

        # hf calculation
        _, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # integral calculation
        hcore, eri = ints(mol, mo_coeff)

        # gauge origin
        gauge_origin = np.array([0.0, 0.0, 0.0])

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # create mbe object
        mbe = MBE(
            target="dipole",
            mol=mol,
            ncore=ncore,
            orbsym=orbsym,
            fci_state_sym="b2",
            hcore=hcore,
            eri=eri,
            dipole_ints=dip_ints,
            ref_space=ref_space,
            rst=rst,
        )

    else:

        # create mbe object
        mbe = MBE()

    # perform calculation
    elec_dipole = mbe.kernel()

    # get total dipole moment
    tot_dipole = mbe.final_prop(prop_type="total", nuc_prop=nuc_dipole(mol))

    return tot_dipole


if __name__ == "__main__":

    # call example function
    dipole = mbe_example()

    # finalize mpi
    MPI.Finalize()
