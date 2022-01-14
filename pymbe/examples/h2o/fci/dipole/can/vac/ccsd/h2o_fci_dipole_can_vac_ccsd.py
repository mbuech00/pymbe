import os
import numpy as np
from pyscf import gto
from pymbe import MBE, MPICls, mpi_finalize, hf, base, ints, dipole_ints

def mbe_example() -> MBE:

    # create mpi object
    mpi = MPICls()

    if mpi.global_master and not os.path.isdir(os.getcwd()+'/rst'):

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
        base_dipole = base('ccsd', mol, hf_object, mo_coeff, occup, orbsym, \
                           norb, ncore, nocc, target='dipole', \
                           hf_dipole=hf_dipole, gauge_origin=gauge_origin)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # create mbe object
        mbe = MBE(method='fci', target='dipole', mol=mol, ncore=1, nocc=nocc, \
                  norb=norb, orbsym=orbsym, hf_prop=hf_dipole, occup=occup, \
                  hcore=hcore, vhf=vhf, eri=eri, dipole_ints=dip_ints, \
                  base_method='ccsd', base_prop=base_dipole)

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
    mpi_finalize(mbe.mpi)
