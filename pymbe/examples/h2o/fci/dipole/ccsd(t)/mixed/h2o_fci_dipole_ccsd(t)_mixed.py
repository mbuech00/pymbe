import os
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, symm, cc, ao2mo
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

        # ccsd calculation
        ccsd = cc.CCSD(hf).run(
            conv_tol=1.0e-10,
            conv_tol_normt=1.0e-10,
            max_cycle=500,
            async_io=False,
            diis_start_cycle=4,
            diis_space=12,
            incore_complete=True,
            frozen=ncore,
        )

        # rdm1 calculation
        l1, l2 = cc.ccsd_t_lambda_slow.kernel(ccsd, verbose=0)[1:]
        rdm1 = cc.ccsd_t_rdm_slow.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2)

        # mo coefficients for natural orbitals
        mo_coeff = hf.mo_coeff.copy()

        # occupied - occupied block
        mask = hf.mo_occ == 2.0
        mask[:ncore] = False
        if np.any(mask):
            no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
            mo_coeff[:, mask] = np.einsum("ip,pj->ij", mo_coeff[:, mask], no[:, ::-1])

        # virtual - virtual block
        mask = hf.mo_occ == 0.0
        if np.any(mask):
            no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
            mo_coeff[:, mask] = np.einsum("ip,pj->ij", mo_coeff[:, mask], no[:, ::-1])

        # orbital symmetries
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)

        # reference space
        ref_space = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

        # expansion space
        exp_space = np.array(
            [i for i in range(ncore, mol.nao) if i not in ref_space],
            dtype=np.int64,
        )

        # hcore
        hcore_ao = hf.get_hcore()
        hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

        # eri
        eri_ao = mol.intor("int2e_sph", aosym="s8")
        eri = ao2mo.incore.full(eri_ao, mo_coeff)

        # gauge origin
        gauge_origin = np.array([0.0, 0.0, 0.0])

        # dipole integral calculation
        with mol.with_common_origin(gauge_origin):
            ao_dipole_ints = mol.intor_symmetric("int1e_r", comp=3)
            dipole_ints = np.einsum(
                "pi,xpq,qj->xij", mo_coeff, ao_dipole_ints, mo_coeff
            )

        # create mbe object
        mbe = MBE(
            target="dipole",
            mol=mol,
            orbsym=orbsym,
            orb_type="ccsd(t)",
            hcore=hcore,
            eri=eri,
            dipole_ints=dipole_ints,
            ref_space=ref_space,
            exp_space=exp_space,
            rst=rst,
        )

    else:

        # create mbe object
        mbe = MBE()

    # perform calculation
    elec_dipole = mbe.kernel()

    # get total dipole moment
    tot_dipole = mbe.final_prop(
        prop_type="total",
        nuc_prop=np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords()),
    )

    return tot_dipole


if __name__ == "__main__":

    # call example function
    dipole = mbe_example()

    # finalize mpi
    MPI.Finalize()