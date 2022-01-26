import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, base, ints, dipole_ints

def mbe_example() -> Optional[Union[float, np.ndarray]]:

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

        # frozen core
        ncore = 1

        # hf calculation
        nocc, _, norb, hf_object, _, hf_dipole, occup, orbsym, \
        mo_coeff = hf(mol)

        # gauge origin
        gauge_origin = np.array([0., 0., 0.])

        # base model
        base_dipole = base('ccsd(t)', mol, hf_object, mo_coeff, occup, orbsym, \
                           norb, ncore, nocc, target='dipole', \
                           hf_dipole=hf_dipole, gauge_origin=gauge_origin)

        # reference space
        ref_space = np.array([1, 2, 3, 4], dtype=np.int64)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # create mbe object
        mbe = MBE(method='fci', target='dipole', mol=mol, ncore=1, nocc=nocc, \
                  norb=norb, orbsym=orbsym, hf_prop=hf_dipole, occup=occup, \
                  hcore=hcore, vhf=vhf, eri=eri, dipole_ints=dip_ints, \
                  ref_space=ref_space, base_method='ccsd(t)', \
                  base_prop=base_dipole)

        # perform calculation
        dipole = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        dipole = mbe.kernel()

    return dipole

if __name__ == '__main__':

    # call example function
    dipole = mbe_example()

    # finalize mpi
    MPI.Finalize()
