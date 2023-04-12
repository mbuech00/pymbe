import os
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, symm, ao2mo
from pymbe import MBE


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
        hf = scf.RHF(mol).run(conv_tol=1e-10)

        # orbsym
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # expansion space
        exp_space = np.array(
            [i for i in range(ncore, mol.nao) if i not in ref_space],
            dtype=np.int64,
        )

        # hcore
        hcore_ao = hf.get_hcore()
        hcore = np.einsum("pi,pq,qj->ij", hf.mo_coeff, hcore_ao, hf.mo_coeff)

        # eri
        eri_ao = mol.intor("int2e_sph", aosym="s8")
        eri = ao2mo.incore.full(eri_ao, hf.mo_coeff)

        # gauge origin
        gauge_origin = np.array([0.0, 0.0, 0.0])

        # dipole integral calculation
        with mol.with_common_origin(gauge_origin):
            ao_dipole_ints = mol.intor_symmetric("int1e_r", comp=3)
        dipole_ints = np.einsum(
            "pi,xpq,qj->xij", hf.mo_coeff, ao_dipole_ints, hf.mo_coeff
        )

        # create mbe object
        mbe = MBE(
            target="trans",
            mol=mol,
            orbsym=orbsym,
            fci_state_root=1,
            hcore=hcore,
            eri=eri,
            dipole_ints=dipole_ints,
            ref_space=ref_space,
            exp_space=exp_space,
            rst=rst,
        )

        # perform calculation
        trans = mbe.kernel()

    else:
        # create mbe object
        mbe = MBE()

        # perform calculation
        trans = mbe.kernel()

    return trans


if __name__ == "__main__":
    # call example function
    trans = mbe_example()

    # finalize mpi
    MPI.Finalize()
