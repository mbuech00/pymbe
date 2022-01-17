import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from pymbe import MBE, hf, ints

def mbe_example() -> MBE:

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd()+'/rst'):

        # create mol object
        mol = gto.Mole()
        mol.build(
        verbose = 0,
        output = None,
        atom = '''
        O  0.00000000  0.00000000  0.10840502
        H -0.75390364  0.00000000 -0.47943227
        H  0.75390364  0.00000000 -0.47943227
        ''',
        basis = '631g',
        symmetry = 'c2v'
        )

        # hf calculation
        nocc, _, norb, _, hf_energy, _, occup, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([1, 2, 3, 4], dtype=np.int64)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # create mbe object
        mbe = MBE(method='ccsd', mol=mol, ncore=1, nocc=nocc, norb=norb, \
                  orbsym=orbsym, hf_prop=hf_energy, occup=occup, hcore=hcore, \
                  vhf=vhf, eri=eri, ref_space=ref_space)

        # perform calculation
        mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        mbe.kernel()

    return mbe

if __name__ == '__main__':

    # call example function
    mbe = mbe_example()

    # finalize mpi
    MPI.Finalize()
