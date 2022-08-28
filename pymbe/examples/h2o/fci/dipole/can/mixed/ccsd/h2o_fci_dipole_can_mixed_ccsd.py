import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, base, ints, dipole_ints, nuc_dipole


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
        hf_object, orbsym, mo_coeff = hf(mol)

        # gauge origin
        gauge_origin = np.array([0.0, 0.0, 0.0])

        # base model
        base_dipole = base(
            "ccsd",
            mol,
            hf_object,
            mo_coeff,
            orbsym,
            ncore,
            target="dipole",
            gauge_origin=gauge_origin,
        )

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # expansion space
        exp_space = np.array(
            [i for i in range(ncore, mol.nao) if i not in ref_space],
            dtype=np.int64,
        )

        # integral calculation
        hcore, eri = ints(mol, mo_coeff)

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # create mbe object
        mbe = MBE(
            target="dipole",
            mol=mol,
            orbsym=orbsym,
            hcore=hcore,
            eri=eri,
            dipole_ints=dip_ints,
            ref_space=ref_space,
            exp_space=exp_space,
            base_method="ccsd",
            base_prop=base_dipole,
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
