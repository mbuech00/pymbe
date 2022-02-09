import os
import numpy as np
from mpi4py import MPI
from pyscf import gto
from typing import Optional, Union
from pymbe import MBE, hf, ref_mo, ints, ref_prop


def mbe_example(rst=True) -> Optional[Union[float, np.ndarray]]:

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
        nocc, nvirt, norb, hf_object, hf_energy, _, occup, orbsym, mo_coeff = hf(mol)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # casscf orbitals
        mo_coeff, orbsym = ref_mo(
            "casscf",
            mol,
            hf_object,
            mo_coeff,
            occup,
            orbsym,
            norb,
            ncore,
            nocc,
            nvirt,
            ref_space,
            fci_solver="pyscf_spin1",
            wfnsym=["b2"],
            weights=[1.0],
        )

        # integral calculation
        hcore, vhf, eri = ints(mol, mo_coeff, norb, nocc)

        # reference property
        ref_energy = ref_prop(
            mol,
            hcore,
            vhf,
            eri,
            occup,
            orbsym,
            nocc,
            ref_space,
            fci_solver="pyscf_spin1",
            fci_state_sym="b2",
            hf_prop=hf_energy,
            orb_type="casscf",
        )

        # create mbe object
        mbe = MBE(
            method="fci",
            fci_solver="pyscf_spin1",
            mol=mol,
            ncore=ncore,
            nocc=nocc,
            norb=norb,
            orbsym=orbsym,
            fci_state_sym="b2",
            hf_prop=hf_energy,
            occup=occup,
            orb_type="casscf",
            hcore=hcore,
            vhf=vhf,
            eri=eri,
            ref_space=ref_space,
            ref_prop=ref_energy,
            rst=rst,
        )

        # perform calculation
        energy = mbe.kernel()

    else:

        # create mbe object
        mbe = MBE()

        # perform calculation
        energy = mbe.kernel()

    return energy


if __name__ == "__main__":

    # call example function
    energy = mbe_example()

    # finalize mpi
    MPI.Finalize()
