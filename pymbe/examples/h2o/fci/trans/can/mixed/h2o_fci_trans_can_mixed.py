import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, ints, dipole_ints, ref_prop

def mbe_example(rst=True) -> Optional[Union[float, np.ndarray]]:

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
        nocc, _, norb, _, _, _, occup, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # gauge origin
        gauge_origin = np.array([0., 0., 0.])

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # reference property
        ref_trans = ref_prop(mol, hcore, vhf, eri, occup, orbsym, nocc, \
                             ref_space, target='trans', fci_state_root=1, \
                             dipole_ints=dip_ints)

        # create mbe object
        mbe = MBE(method='fci', target='trans', mol=mol, ncore=1, nocc=nocc, \
                  norb=norb, orbsym=orbsym, fci_state_root=1, occup=occup, \
                  hcore=hcore, vhf=vhf, eri=eri, dipole_ints=dip_ints, \
                  ref_space=ref_space, ref_prop=ref_trans, rst=rst)

        # perform calculation
        trans = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        trans = mbe.kernel()

    return trans

if __name__ == '__main__':

    # call example function
    trans = mbe_example()

    # finalize mpi
    MPI.Finalize()
