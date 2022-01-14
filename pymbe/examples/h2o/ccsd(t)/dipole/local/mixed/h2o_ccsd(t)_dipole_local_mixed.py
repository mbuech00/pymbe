import os
import numpy as np
from pyscf import gto
from pymbe import MBE, MPICls, mpi_finalize, hf, ref_mo, ints, dipole_ints, \
                  ref_prop

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
        nocc, nvirt, norb, hf_object, _, hf_dipole, occup, orbsym, \
        mo_coeff = hf(mol)

        # pipek-mezey localized orbitals
        mo_coeff, orbsym = ref_mo('local', mol, hf_object, mo_coeff, occup, \
                                  orbsym, norb, ncore, nocc, nvirt)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # gauge origin
        gauge_origin = np.array([0., 0., 0.])

        # dipole integral calculation
        dip_ints = dipole_ints(mol, mo_coeff, gauge_origin)

        # reference property
        ref_dipole = ref_prop(mol, hcore, vhf, eri, occup, orbsym, nocc, \
                              ref_space, method='ccsd(t)', target='dipole', \
                              hf_prop=hf_dipole, dipole_ints=dip_ints, \
                              orb_type='local')

        # create mbe object
        mbe = MBE(method='ccsd(t)', target='dipole', mol=mol, ncore=ncore, \
                  nocc=nocc, norb=norb, hf_prop=hf_dipole, occup=occup, \
                  orb_type='local', hcore=hcore, vhf=vhf, eri=eri, \
                  dipole_ints=dip_ints, ref_space=ref_space, \
                  ref_prop=ref_dipole)

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
