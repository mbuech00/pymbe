import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, ints, dipole_ints

def mbe_example(rst=True) -> Optional[Union[float, np.ndarray]]:

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd()+'/rst'):

        # create mol object
        mol = gto.Mole()
        mol.build(
        verbose = 0,
        output = None,
        atom = '''
        C  0.00000  0.00000  0.00000
        H  0.98920  0.42714  0.00000
        H -0.98920  0.42714  0.00000
        ''',
        basis = '631g',
        symmetry = 'c2v',
        spin = 2
        )

        # hf calculation
        nocc, _, norb, _, _, hf_dipole, occup, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([3, 4], dtype=np.int64)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # gauge origin
        gauge_origin = np.array([0., 0., 0.])

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # create mbe object
        mbe = MBE(method='fci', fci_solver='pyscf_spin1', target='dipole', \
                  mol=mol, ncore=1, nocc=nocc, norb=norb, orbsym=orbsym, \
                  fci_state_sym='b2', hf_prop=hf_dipole, occup=occup, \
                  hcore=hcore, vhf=vhf, eri=eri, dipole_ints=dip_ints, \
                  ref_space=ref_space, rst=rst)

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
