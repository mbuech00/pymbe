import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ref_mo, ints, dipole_ints, ref_prop


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
        hf_object, hf_prop, orbsym, mo_coeff = hf(mol, target="dipole")

        # natural orbitals
        mo_coeff, orbsym = ref_mo("ccsd(t)", mol, hf_object, mo_coeff, orbsym, ncore)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # integral calculation
        hcore, eri, vhf = ints(mol, mo_coeff)

        # gauge origin
        gauge_origin = np.array([0.0, 0.0, 0.0])

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # reference property
        ref_dipole = ref_prop(
            mol,
            hcore,
            eri,
            ref_space,
            target="dipole",
            orbsym=orbsym,
            fci_state_sym="b2",
            hf_prop=hf_prop,
            vhf=vhf,
            dipole_ints=dip_ints,
            orb_type="ccsd(t)",
        )

        # create mbe object
        mbe = MBE(
            target="dipole",
            mol=mol,
            ncore=ncore,
            orbsym=orbsym,
            fci_state_sym="b2",
            hf_prop=hf_prop,
            orb_type="ccsd(t)",
            hcore=hcore,
            eri=eri,
            vhf=vhf,
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
