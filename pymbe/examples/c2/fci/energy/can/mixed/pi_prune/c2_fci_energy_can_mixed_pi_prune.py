import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints, linear_orbsym


def mbe_example(rst=True):

    # create mol object
    mol = gto.Mole()
    mol.build(
        verbose=0,
        output=None,
        atom="""
        C  0.  0.  .7
        C  0.  0. -.7
        """,
        basis="631g",
        symmetry="d2h",
    )

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd() + "/rst"):

        # frozen core
        ncore = 2

        # hf calculation
        _, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)

        # expansion space
        exp_space = np.array(
            [i for i in range(2, mol.nao) if i not in ref_space],
            dtype=np.int64,
        )

        # integral calculation
        hcore, eri = ints(mol, mo_coeff)

        # pi_pruning
        orbsym_linear = linear_orbsym(mol, mo_coeff)

        # create mbe object
        mbe = MBE(
            mol=mol,
            orbsym=orbsym,
            hcore=hcore,
            eri=eri,
            ref_space=ref_space,
            exp_space=exp_space,
            rst=rst,
            pi_prune=True,
            orbsym_linear=orbsym_linear,
        )

    else:

        # create mbe object
        mbe = MBE()

    # perform calculation
    elec_energy = mbe.kernel()

    # get total energy
    tot_energy = mbe.final_prop(prop_type="total", nuc_prop=mol.energy_nuc().item())

    return tot_energy


if __name__ == "__main__":

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
