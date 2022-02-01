import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, ints, ref_prop, linear_orbsym

def mbe_example(rst=True) -> Optional[Union[float, np.ndarray]]:

    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(os.getcwd()+'/rst'):

        # create mol object
        mol = gto.Mole()
        mol.build(
        verbose = 0,
        output = None,
        atom = '''
        C  0.  0.  .7
        C  0.  0. -.7
        ''',
        basis = '631g',
        symmetry = 'd2h'
        )

        # frozen core
        ncore = 2

        # hf calculation
        nocc, _, norb, _, hf_energy, _, occup, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        #reference property
        ref_energy = ref_prop(mol, hcore, vhf, eri, occup, orbsym, nocc, \
                              ref_space, hf_prop=hf_energy)

        # pi_pruning
        orbsym_linear = linear_orbsym(mol, mo_coeff)

        # create mbe object
        mbe = MBE(method='fci', mol=mol, ncore=ncore, nocc=nocc, norb=norb, \
                  orbsym=orbsym, hf_prop=hf_energy, occup=occup, hcore=hcore, \
                  vhf=vhf, eri=eri, ref_space=ref_space, ref_prop=ref_energy, \
                  rst=rst, pi_prune=True, orbsym_linear=orbsym_linear)

        # perform calculation
        energy = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        energy = mbe.kernel()

    return energy

if __name__ == '__main__':

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
