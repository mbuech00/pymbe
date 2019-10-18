#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
kernel module containing all electronic structure kernels
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import copy
import numpy as np
import scipy as sp
from mpi4py import MPI
from pyscf import gto, symm, scf, ao2mo, lib, lo, ci, cc, mcscf, fci
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from typing import Tuple, List, Dict, Union, Any
import warnings

import parallel
import system
import tools


MAX_MEM = 1e10
CONV_TOL = 1.e-10
SPIN_TOL = 1.e-05
DIPOLE_TOL = 1.e-14


def ints(mol: system.MolCls, mo_coeff: np.ndarray, global_master: bool, local_master: bool, \
            global_comm: MPI.Comm, local_comm: MPI.Comm, master_comm: MPI.Comm, \
            num_masters: int) -> Tuple[MPI.Win, ...]:
        """
        this function returns 1e and 2e mo integrals and effective fock potentials from individual occupied orbitals

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = '631g')
        >>> mol.norb = mol.nao_nr()
        >>> mol.nocc = mol.nelectron // 2
        >>> np.random.seed(1234)
        >>> mo_coeff = np.random.rand(mol.norb, mol.norb)
        >>> ints(mol, mo_coeff, True, True, MPI.COMM_WORLD, MPI.COMM_WORLD, MPI.COMM_WORLD, 1) # doctest: +ELLIPSIS
        (<mpi4py.MPI.Win object at 0x...>, <mpi4py.MPI.Win object at 0x...>, <mpi4py.MPI.Win object at 0x...>)
        """
        # hcore_ao and eri_ao w/o symmetry
        if global_master:
            hcore_tmp, eri_tmp = _ao_ints(mol)

        # allocate hcore in shared mem
        if local_master:
            hcore_win = MPI.Win.Allocate_shared(8 * mol.norb**2, 8, comm=local_comm)
        else:
            hcore_win = MPI.Win.Allocate_shared(0, 8, comm=local_comm)
        buf = hcore_win.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # compute hcore
        if global_master:
            hcore[:] = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_tmp, mo_coeff)

        # bcast hcore
        if num_masters > 1 and local_master:
            hcore[:] = parallel.bcast(master_comm, hcore)

        # eri_mo w/o symmetry
        if global_master:
            eri_tmp = ao2mo.incore.full(eri_tmp, mo_coeff)

        # allocate vhf in shared mem
        if local_master:
            vhf_win = MPI.Win.Allocate_shared(8 * mol.nocc*mol.norb**2, 8, comm=local_comm)
        else:
            vhf_win = MPI.Win.Allocate_shared(0, 8, comm=local_comm)
        buf = vhf_win.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # compute vhf
        if global_master:
            for i in range(mol.nocc):
                idx = np.asarray([i])
                vhf[i] = np.einsum('pqrs->rs', eri_tmp[idx[:, None], idx, :, :]) * 2.
                vhf[i] -= np.einsum('pqrs->ps', eri_tmp[:, idx[:, None], idx, :]) * 2. * .5

        # bcast vhf
        if num_masters > 1 and local_master:
            vhf[:] = parallel.bcast(master_comm, vhf)

        # allocate eri in shared mem
        if local_master:
            eri_win = MPI.Win.Allocate_shared(8 * (mol.norb * (mol.norb + 1) // 2) ** 2, 8, comm=local_comm)
        else:
            eri_win = MPI.Win.Allocate_shared(0, 8, comm=local_comm)
        buf = eri_win.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

        # restore 4-fold symmetry in eri_mo
        if global_master:
            eri[:] = ao2mo.restore(4, eri_tmp, mol.norb)

        # bcast eri
        if num_masters > 1 and local_master:
            eri[:] = parallel.bcast(master_comm, eri)

        # mpi barrier
        global_comm.Barrier()

        return hcore_win, vhf_win, eri_win


def _ao_ints(mol: system.MolCls) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns 1e and 2e ao integrals

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = '631g')
        >>> hcore, eri = _ao_ints(mol)
        >>> hcore.shape
        (13, 13)
        >>> eri.shape
        (13, 13, 13, 13)
        >>> mol = gto.M()
        >>> mol.matrix = (1, 6)
        >>> mol.n = 1.
        >>> mol.u = 2.
        >>> mol.pbc = True
        >>> hcore, eri = _ao_ints(mol)
        >>> hcore.shape
        (6, 6)
        >>> eri.shape
        (6, 6, 6, 6)
        """
        if mol.atom:

            # hcore_ao
            hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
            # eri_ao w/o symmetry
            if mol.cart:
                eri = mol.intor('int2e_cart', aosym=1)
            else:
                eri = mol.intor('int2e_sph', aosym=1)

        else:

            # hcore_ao
            hcore = hubbard_h1e(mol.matrix, mol.pbc)
            # eri_ao
            eri = hubbard_eri(mol.matrix, mol.u)

        return hcore, eri


def dipole_ints(mol: system.MolCls) -> np.ndarray:
        """
        this function returns dipole integrals (in AO basis)

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = 'sto-3g')
        >>> dipole = dipole_ints(mol)
        >>> dipole.shape
        (3, 7, 7)
        """
        # gauge origin
        with mol.with_common_origin([0., 0., 0.]):
            dipole = mol.intor_symmetric('int1e_r', comp=3)

        return dipole


def e_core_h1e(e_nuc: float, hcore: np.ndarray, vhf: np.ndarray, \
                core_idx: np.ndarray, cas_idx: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        this function returns core energy and cas space 1e integrals

        example:
        >>> e_nuc = 0.
        >>> np.random.seed(1234)
        >>> hcore = np.random.rand(6, 6)
        >>> np.random.seed(1234)
        >>> vhf = np.random.rand(3, 6, 6)
        >>> core_idx = np.array([0])
        >>> cas_idx = np.array([2, 4, 5])
        >>> e_core, h1e_cas = e_core_h1e(e_nuc, hcore, vhf, core_idx, cas_idx)
        >>> np.isclose(e_core, 0.5745583511366769)
        True
        >>> h1e_cas.shape
        (3, 3)
        >>> h1e_cas_ref = np.array([[0.74050151, 1.00616633, 0.02753690],
        ...                         [0.79440516, 0.63367224, 1.13619731],
        ...                         [1.60429528, 1.40852194, 1.40916262]])
        >>> np.allclose(h1e_cas, h1e_cas_ref)
        True
        """
        # init core energy
        e_core = e_nuc

        # determine effective core fock potential
        if core_idx.size > 0:
            core_vhf = np.sum(vhf[core_idx], axis=0)
        else:
            core_vhf = 0

        # calculate core energy
        e_core += np.trace((hcore + .5 * core_vhf)[core_idx[:, None], core_idx]) * 2.

        # extract cas integrals
        h1e_cas = (hcore + core_vhf)[cas_idx[:, None], cas_idx]

        return e_core, h1e_cas


def hubbard_h1e(matrix: Tuple[int, int], pbc: bool = False) -> np.ndarray:
        """
        this function returns the hubbard hopping hamiltonian

        example:
        >>> matrix = (1, 4)
        >>> h1e = hubbard_h1e(matrix, False)
        >>> h1e_ref = np.array([[ 0., -1.,  0.,  0.],
        ...                     [-1.,  0., -1.,  0.],
        ...                     [ 0., -1.,  0., -1.],
        ...                     [ 0.,  0., -1.,  0.]])
        >>> np.allclose(h1e, h1e_ref)
        True
        >>> h1e = hubbard_h1e(matrix, True)
        >>> h1e_ref[-1, 0] = h1e_ref[0, -1] = -1.
        >>> np.allclose(h1e, h1e_ref)
        True
        >>> matrix = (2, 2)
        >>> h1e = hubbard_h1e(matrix)
        >>> h1e_ref = np.array([[ 0., -1., -1.,  0.],
        ...                     [-1.,  0.,  0., -1.],
        ...                     [-1.,  0.,  0., -1.],
        ...                     [ 0., -1., -1.,  0.]])
        >>> np.allclose(h1e, h1e_ref)
        True
        >>> matrix = (2, 3)
        >>> h1e = hubbard_h1e(matrix)
        >>> h1e_ref = np.array([[ 0., -1.,  0.,  0., -1.,  0.],
        ...                     [-1.,  0., -1., -1.,  0., -1.],
        ...                     [ 0., -1.,  0.,  0., -1.,  0.],
        ...                     [ 0., -1.,  0.,  0., -1.,  0.],
        ...                     [-1.,  0., -1., -1.,  0., -1.],
        ...                     [ 0., -1.,  0.,  0., -1.,  0.]])
        >>> np.allclose(h1e, h1e_ref)
        True
        """
        # dimension
        if 1 in matrix:
            ndim = 1
        else:
            ndim = 2

        # nsites
        nsites = matrix[0] * matrix[1]

        # init h1e
        h1e = np.zeros([nsites] * 2, dtype=np.float64)

        if ndim == 1:

            # adjacent neighbours
            for i in range(nsites-1):
                h1e[i, i+1] = h1e[i+1, i] = -1.

            if pbc:
                h1e[-1, 0] = h1e[0, -1] = -1.

        elif ndim == 2:

            # number of x- and y-sites
            nx, ny = matrix[0], matrix[1]

            # adjacent neighbours
            for site_1 in range(nsites):

                site_1_xy = tools.mat_idx(site_1, nx, ny)
                nbrs = tools.near_nbrs(site_1_xy, nx, ny)

                for site_2 in range(site_1):

                    site_2_xy = tools.mat_idx(site_2, nx, ny)

                    if site_2_xy in nbrs:
                        h1e[site_1, site_2] = h1e[site_2, site_1] = -1.

        return h1e


def hubbard_eri(matrix: Tuple[int, int], u: float) -> np.ndarray:
        """
        this function returns the hubbard two-electron hamiltonian

        example:
        >>> matrix = (1, 2)
        >>> eri = hubbard_eri(matrix, 2.)
        >>> eri_ref = np.array([[[[2., 0.],
        ...                       [0., 0.]],
        ...                      [[0., 0.],
        ...                       [0., 0.]]],
        ...                     [[[0., 0.],
        ...                       [0., 0.]],
        ...                      [[0., 0.],
        ...                       [0., 2.]]]])
        >>> np.allclose(eri, eri_ref)
        True
        """
        # nsites
        nsites = matrix[0] * matrix[1]

        # init eri
        eri = np.zeros([nsites] * 4, dtype=np.float64)

        # compute eri
        for i in range(nsites):
            eri[i,i,i,i] = u

        return eri


class _hubbard_PM(lo.pipek.PM):
        """
        this class constructs the site-population tensor for each orbital-pair density
        see: pyscf example - 40-hubbard_model_PM_localization.py
        """
        def atomic_pops(self, mol: gto.Mole, mo_coeff: np.ndarray, method: str = None) -> np.ndarray:
            """
            this function overwrites the tensor used in the pm cost function and its gradients
            """
            return np.einsum('pi,pj->pij', mo_coeff, mo_coeff)


def hf(mol: system.MolCls, target_mbe: str) -> Tuple[int, int, int, scf.RHF, float, np.ndarray, \
                                                    np.ndarray, np.ndarray, np.ndarray]:
        """
        this function returns the results of a hartree-fock calculation

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = '631g', symmetry = 'C2v', verbose=0)
        >>> mol.hf_symmetry = mol.symmetry
        >>> mol.debug = 0
        >>> mol.hf_init_guess = 'minao'
        >>> mol.irrep_nelec = {}
        >>> nocc, nvirt, norb, pyscf_hf, e_hf, dipole, occup, orbsym, mo_coeff = hf(mol, 'energy')
        >>> nocc
        5
        >>> nvirt
        8
        >>> norb
        13
        >>> np.isclose(e_hf, -75.9838464521063)
        True
        >>> dipole is None
        True
        >>> occup
        array([2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> orbsym
        array([0, 0, 2, 0, 3, 0, 2, 2, 3, 0, 0, 2, 0])
        >>> mol.dipole = dipole_ints(mol)
        >>> dipole = hf(mol, 'dipole')[5]
        >>> np.allclose(dipole, np.array([0., 0., 0.8642558]))
        True
        """
        # initialize restricted hf calc
        mol_hf = mol.copy()
        mol_hf.build(0, 0, symmetry = mol.hf_symmetry)
        hf = scf.RHF(mol_hf)

        # hf settings
        if mol.debug >= 1:
            hf.verbose = 4

        hf.init_guess = mol.hf_init_guess
        hf.conv_tol = CONV_TOL
        hf.max_cycle = 1000

        if mol.atom:
            # ab initio hamiltonian
            hf.irrep_nelec = mol.irrep_nelec
        else:
            # model hamiltonian
            hf.get_ovlp = lambda *args: np.eye(mol.matrix[0] * mol.matrix[1])
            hf.get_hcore = lambda *args: hubbard_h1e(mol.matrix, mol.pbc)
            hf._eri = hubbard_eri(mol.matrix, mol.u)

        # perform hf calc
        for i in range(0, 12, 2):

            hf.diis_start_cycle = i

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hf.kernel()
            except sp.linalg.LinAlgError:
                pass

            if hf.converged:
                break

        # convergence check
        tools.assertion(hf.converged, 'HF error: no convergence')

        # dipole moment
        if target_mbe == 'dipole':
            dm = hf.make_rdm1()
            elec_dipole = np.einsum('xij,ji->x', mol.dipole, dm)
            elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > DIPOLE_TOL else 0. for i in range(elec_dipole.size)])
        else:
            elec_dipole = None

        # determine dimensions
        norb, nocc, nvirt = _dim(hf.mo_occ)

        # store energy, occupation, and orbsym
        e_hf = hf.e_tot
        occup = hf.mo_occ
        if mol.atom:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        else:
            orbsym = np.zeros(hf.mo_energy.size, dtype=np.int)

        # debug print of orbital energies
        if mol.debug >= 1:
            if mol.symmetry:
                gpname = mol.symmetry
            else:
                gpname = 'C1'
            print('\n HF:  mo   symmetry    energy')
            for i in range(hf.mo_energy.size):
                print('     {:>3d}   {:>5s}     {:>7.5f}'.format(i, symm.addons.irrep_id2name(gpname, orbsym[i]), hf.mo_energy[i]))
            print('\n')

        return nocc, nvirt, norb, hf, np.asscalar(e_hf), elec_dipole, occup, \
                orbsym, np.asarray(hf.mo_coeff, order='C')


def _dim(mo_occ: np.ndarray) -> Tuple[int, ...]:
        """
        this function determines the involved dimensions (number of occupied, virtual, and total orbitals)

        example:
        >>> mo_occ = np.array([2.] * 4 + [0.] * 6)
        >>> _dim(mo_occ)
        (10, 4, 6)
        >>> mo_occ = np.array([2.] * 4 + [1.] + [0.] * 6)
        >>> _dim(mo_occ)
        (11, 5, 6)
        """
        # occupied and virtual lists
        occ = np.where(mo_occ > 0.)[0]
        virt = np.where(mo_occ == 0.)[0]
        return occ.size + virt.size, occ.size, virt.size


def ref_mo(mol: system.MolCls, mo_coeff: np.ndarray, occup: np.ndarray, orbsym: np.ndarray, \
            orbs: Dict[str, str], ref: Dict[str, Any], model: Dict[str, str], pi_prune: bool, \
            hf: scf.RHF) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
        """
        this function returns a set of reference mo coefficients and symmetries plus the associated spaces

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='C 0. 0. 0.625; C 0. 0. -0.625',
        ...               basis = '631g', symmetry = 'D2h', verbose=0)
        >>> mol.ncore, mol.nocc, mol.nvirt, mol.norb = 2, 6, 12, 18
        >>> mol.debug = 0
        >>> hf = scf.RHF(mol)
        >>> _ = hf.kernel()
        >>> orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        >>> model = {'method': 'fci', 'solver': 'pyscf_spin0'}
        >>> ref = {'method': 'casci', 'hf_guess': True, 'active': 'manual',
        ...        'select': [i for i in range(2, 6)],
        ...        'wfnsym': ['Ag']}
        >>> orbs = {'type': 'can'}
        >>> mo_coeff, act_nelec, ref_space, exp_space = ref_mo(mol, hf.mo_coeff, hf.mo_occ, orbsym,
        ...                                                     orbs, ref, model, False, hf)
        >>> np.isclose(np.sum(mo_coeff), -4.995051198781287)
        True
        >>> np.isclose(np.amax(mo_coeff), 4.954270427681284)
        True
        >>> act_nelec
        (4, 4)
        >>> np.allclose(ref_space['occ'], np.array([2, 3, 4, 5], dtype=np.int16))
        True
        >>> np.all(ref_space['occ'] == ref_space['tot'])
        True
        >>> ref_space['virt']
        array([], dtype=int16)
        >>> np.allclose(exp_space['virt'], np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int16))
        True
        >>> np.all(exp_space['virt'] == exp_space['tot'])
        True
        >>> exp_space['occ']
        array([], dtype=int16)
        >>> np.all(exp_space['occ'] == exp_space['seed'])
        True
        >>> exp_space['pi_orbs'] is None and exp_space['pi_hashes'] is None
        True
        >>> exp_space = ref_mo(mol, hf.mo_coeff, hf.mo_occ, orbsym, orbs, ref, model, True, hf)[-1]
        >>> np.allclose(exp_space['pi_orbs'], np.array([7, 8, 14, 15, 11, 12], dtype=np.int16))
        True
        >>> np.allclose(exp_space['pi_hashes'], np.array([-7365615264797734692,  2711701422158015467,  4980488901507643489]))
        True
        >>> orbs['type'] = 'ccsd'
        >>> mo_coeff = ref_mo(mol, hf.mo_coeff, hf.mo_occ, orbsym, orbs, ref, model, False, hf)[0]
        >>> np.isclose(np.sum(mo_coeff), 1.4521896109624048)
        True
        >>> np.isclose(np.amax(mo_coeff), 6.953346258094149)
        True
        >>> orbs['type'] = 'local'
        >>> mo_coeff = ref_mo(mol, hf.mo_coeff, hf.mo_occ, orbsym, orbs, ref, model, False, hf)[0]
        >>> np.isclose(np.sum(mo_coeff), 3.5665242146990463)
        True
        >>> np.isclose(np.amax(mo_coeff), 5.510437607766403)
        True
        >>> orbs['type'] = 'can'
        >>> ref['method'] = 'casscf'
        >>> ref['select'] = [4, 5, 7, 8]
        >>> mo_coeff = ref_mo(mol, hf.mo_coeff, hf.mo_occ, orbsym, orbs, ref, model, False, hf)[0]
        >>> np.isclose(np.sum(mo_coeff), -5.0278490212621385)
        True
        >>> np.isclose(np.amax(mo_coeff), 4.947394624365791)
        True
        """
        # copy mo coefficients
        mo_coeff_out = np.copy(mo_coeff)

        if orbs['type'] != 'can':

            # set core and cas spaces
            core_idx, cas_idx = tools.core_cas(mol.nocc, np.arange(mol.ncore, mol.nocc), \
                                                np.arange(mol.nocc, mol.norb))

            # NOs
            if orbs['type'] in ['ccsd', 'ccsd(t)']:

                # compute rmd1
                res = _cc(occup, core_idx, cas_idx, orbs['type'], hf=hf, rdm1=True)
                rdm1 = res['rdm1']
                if mol.spin > 0:
                    rdm1 = rdm1[0] + rdm1[1]

                # occ-occ block
                no = symm.eigh(rdm1[:(mol.nocc-mol.ncore), :(mol.nocc-mol.ncore)], orbsym[mol.ncore:mol.nocc])[-1]
                mo_coeff_out[:, mol.ncore:mol.nocc] = np.einsum('ip,pj->ij', mo_coeff[:, mol.ncore:mol.nocc], no[:, ::-1])

                # virt-virt block
                no = symm.eigh(rdm1[-mol.nvirt:, -mol.nvirt:], orbsym[mol.nocc:])[-1]
                mo_coeff_out[:, mol.nocc:] = np.einsum('ip,pj->ij', mo_coeff[:, mol.nocc:], no[:, ::-1])

            # pipek-mezey localized orbitals
            elif orbs['type'] == 'local':

                # occ-occ block
                if mol.atom:
                    loc = lo.PM(mol, mo_coeff[:, mol.ncore:mol.nocc])
                else:
                    loc = _hubbard_PM(mol, mo_coeff[:, mol.ncore:mol.nocc])
                loc.conv_tol = CONV_TOL
                if mol.debug >= 1:
                    loc.verbose = 4
                mo_coeff_out[:, mol.ncore:mol.nocc] = loc.kernel()

                # virt-virt block
                if mol.atom:
                    loc = lo.PM(mol, mo_coeff[:, mol.nocc:])
                else:
                    loc = _hubbard_PM(mol, mo_coeff[:, mol.nocc:])
                loc.conv_tol = CONV_TOL
                if mol.debug >= 1:
                    loc.verbose = 4
                mo_coeff_out[:, mol.nocc:] = loc.kernel()

        # active space
        if ref['active'] == 'manual':

            # active orbitals
            ref['select'] = np.asarray(ref['select'], dtype=np.int16)

            # active electrons
            act_nelec = tools.nelec(occup, ref['select'])

        # reference (primary) space
        ref_space: Dict[str, np.ndarray] = {}
        ref_space['tot'] = ref['select']
        ref_space['occ'] = ref_space['tot'][occup[ref_space['tot']] > 0.]
        ref_space['virt'] = ref_space['tot'][occup[ref_space['tot']] == 0.]

        # secondary space
        sec_space = np.asarray([i for i in range(mol.ncore, mol.norb) if i not in ref_space['tot']], dtype=np.int16)

        # divide exp_space into occupied and virtual parts
        exp_space: Dict[str, np.ndarray] = {}
        exp_space['occ'] = sec_space[occup[sec_space] > 0.]
        exp_space['virt'] = sec_space[occup[sec_space] == 0.]

        # seed and total expansion spaces
        if ref_space['tot'].size == 0 or (ref_space['occ'].size > 0. and ref_space['virt'].size == 0):
            exp_space['seed'] = exp_space['occ']
            exp_space['tot'] = exp_space['virt']
        else:
            exp_space['seed'] = np.array([], dtype=np.int16)
            exp_space['tot'] = sec_space

        # casscf
        if ref['method'] == 'casscf':

            tools.assertion(ref_space['occ'].size > 0, \
                            'no singly/doubly occupied orbitals in CASSCF calculation')
            tools.assertion(ref_space['virt'].size > 0, \
                            'no virtual/singly occupied orbitals in CASSCF calculation')

            # sorter for active space
            sort_casscf = np.concatenate((np.arange(mol.ncore), ref_space['tot'], exp_space['tot']))
            # divide into inactive-reference-expansion
            mo_coeff_casscf = mo_coeff[:, sort_casscf]

            # update orbsym
            if mol.atom:
                orbsym_casscf = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_casscf)

            # run casscf
            mo_coeff_out = _casscf(mol, model['solver'], ref['wfnsym'], orbsym_casscf, \
                                ref['hf_guess'], hf, mo_coeff_casscf, ref_space['tot'], act_nelec)

            # reorder mo_coeff
            mo_coeff_out = mo_coeff_out[:, np.argsort(sort_casscf)]

        # pi-orbital space
        if pi_prune:

            # recast mol in dooh point group - make pi-space based on those symmetries
            mol_dooh = mol.copy()
            mol_dooh = mol_dooh.build(0, 0, symmetry='Dooh')
            orbsym_dooh = symm.label_orb_symm(mol_dooh, mol_dooh.irrep_id, mol_dooh.symm_orb, mo_coeff_out)

            # pi-space
            exp_space_all = np.concatenate((exp_space['occ'], exp_space['virt']))
            exp_space['pi_orbs'], \
                exp_space['pi_hashes'] = tools.pi_space(orbsym_dooh[exp_space_all], exp_space_all)

        else:

            # no pi-space
            exp_space['pi_orbs'] = exp_space['pi_hashes'] = None

        # debug print of reference and expansion spaces
        if mol.debug >= 1:
            print('\n reference nelec        = {:}'.format(act_nelec))
            print(' reference space [occ]  = {:}'.format(ref_space['occ']))
            print(' reference space [virt] = {:}'.format(ref_space['virt']))
            if pi_prune:
                print(' expansion space [pi]   =\n{:}'.format(exp_space['pi_orbs'].reshape(-1, 2)))
            print(' expansion space [occ]  = {:}'.format(exp_space['occ']))
            print(' expansion space [virt] = {:}\n'.format(exp_space['virt']))

        return np.asarray(mo_coeff_out, order='C'), act_nelec, ref_space, exp_space


def ref_prop(mol: system.MolCls, occup: np.ndarray, target_mbe: str, \
                orbsym: np.ndarray, hf_guess: bool, ref_space: Dict[str, np.ndarray], \
                model: Dict[str, str], state: Dict[str, Any], e_hf: float, mo_coeff: np.ndarray, \
                dipole_hf: np.ndarray, base_method: Union[str, None]) -> Union[float, np.ndarray]:
        """
        this function returns reference space properties

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = '631g', symmetry = 'C2v', verbose=0)
        >>> mol.hf_symmetry = mol.symmetry
        >>> mol.debug = 0
        >>> mol.hf_init_guess = 'h1e'
        >>> mol.irrep_nelec = {}
        >>> mol.dipole = dipole_ints(mol)
        >>> mol.e_nuc = mol.energy_nuc()
        >>> mol.nocc, mol.nvirt, mol.norb, _, e_hf, dipole_hf, occup, orbsym, mo_coeff = hf(mol, 'dipole')
        >>> orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)
        >>> mol.hcore, mol.vhf, mol.eri = ints(mol, mo_coeff, True, True,
        ...                                    MPI.COMM_WORLD, MPI.COMM_WORLD, MPI.COMM_WORLD, 1)
        >>> ref_space = {'occ': np.arange(5), 'virt': np.array([6, 8, 10]),
        ...              'tot': np.array([0, 1, 2, 3, 4, 6, 8, 10])}
        >>> state = {'root': 0, 'wfnsym': 'A1'}
        >>> model = {'method': 'fci', 'solver': 'pyscf_spin0'}
        >>> e = ref_prop(mol, occup, 'energy', orbsym, True, ref_space,
        ...              model, state, e_hf, mo_coeff, dipole_hf, None)
        >>> np.isclose(e, -0.03769780809258805)
        True
        >>> e = ref_prop(mol, occup, 'energy', orbsym, True, ref_space,
        ...              model, state, e_hf, mo_coeff, dipole_hf, 'ccsd')
        >>> np.isclose(e, -0.00036229313775759664)
        True
        >>> dipole = ref_prop(mol, occup, 'dipole', orbsym, True, ref_space,
        ...                   model, state, e_hf, mo_coeff, dipole_hf, None)
        >>> np.allclose(dipole, np.array([0., 0., -0.02732937]))
        True
        >>> dipole = ref_prop(mol, occup, 'dipole', orbsym, True, ref_space,
        ...                   model, state, e_hf, mo_coeff, dipole_hf, 'ccsd(t)')
        >>> np.allclose(dipole, np.array([0., 0., -5.09683894e-05]))
        True
        >>> state['root'] = 1
        >>> exc = ref_prop(mol, occup, 'excitation', orbsym, True, ref_space,
        ...                model, state, e_hf, mo_coeff, dipole_hf, None)
        >>> np.isclose(exc, 0.7060145137233889)
        True
        >>> trans = ref_prop(mol, occup, 'trans', orbsym, True, ref_space,
        ...                  model, state, e_hf, mo_coeff, dipole_hf, None)
        >>> np.allclose(trans, np.array([0., 0., 0.72582795]))
        True
        """
        # load hcore
        buf = mol.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # load vhf
        buf = mol.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # load eri
        buf = mol.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb*(mol.norb + 1) // 2,) * 2)

        # core_idx and cas_idx
        core_idx, cas_idx = tools.core_cas(mol.nocc, ref_space['tot'], np.array([], dtype=np.int16))

        # nelec
        nelec = np.asarray((np.count_nonzero(occup[cas_idx] > 0.), \
                            np.count_nonzero(occup[cas_idx] > 1.)), dtype=np.int16)

        if ref_space['occ'].size > 0 and ref_space['virt'].size > 0:

            # get cas space h2e
            cas_idx_tril = tools.cas_idx_tril(cas_idx)
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

            # exp model
            ref = main(model['method'], model['solver'], occup, target_mbe, \
                        state['wfnsym'], orbsym, hf_guess, state['root'], \
                        e_hf, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, mol.debug, \
                        mol.dipole if target_mbe in ['dipole', 'trans'] else None, \
                        mo_coeff if target_mbe in ['dipole', 'trans'] else None, \
                        dipole_hf if target_mbe in ['dipole', 'trans'] else None)[0]

            # base model
            if base_method is not None:
                ref -= main(base_method, '', occup, target_mbe, \
                            state['wfnsym'], orbsym, hf_guess, state['root'], \
                            e_hf, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, mol.debug, \
                            mol.dipole if target_mbe in ['dipole', 'trans'] else None, \
                            mo_coeff if target_mbe in ['dipole', 'trans'] else None, \
                            dipole_hf if target_mbe in ['dipole', 'trans'] else None)[0]

        else:

            # no correlation in expansion reference space
            if target_mbe in ['energy', 'excitation']:
                ref = 0.
            else:
                ref = np.zeros(3, dtype=np.float64)

        return ref


def main(method: str, solver: str, occup: np.ndarray, target_mbe: str, \
            state_wfnsym: str, orbsym: np.ndarray, hf_guess: bool, \
            state_root: int, e_hf: float, e_core: float, h1e: np.ndarray, \
            h2e: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray, \
            nelec: Tuple[int, int], debug: int, ao_dipole: Union[np.ndarray, None], \
            mo_coeff: Union[np.ndarray, None], dipole_hf: Union[np.ndarray, None]) -> Tuple[Union[float, np.ndarray], int]:
        """
        this function return the result property from a given method

        example:
        >>> occup = np.array([2.] * 3 + [0.] * 3)
        >>> orbsym = np.zeros(6, dtype=np.int)
        >>> e_hf = e_core = 0.
        >>> h1e = hubbard_h1e((1, 6), True)
        >>> h2e = hubbard_eri((1, 6), 2.)
        >>> h2e = ao2mo.restore(4, h2e, 6)
        >>> core_idx = np.array([0])
        >>> cas_idx = np.arange(1, 5)
        >>> h1e_cas = h1e[cas_idx[:, None], cas_idx]
        >>> cas_idx_tril = tools.cas_idx_tril(cas_idx)
        >>> h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
        >>> nelec = tools.nelec(occup, cas_idx)
        >>> e, ndets = main('fci', 'pyscf_spin0', occup, 'energy', 'A', orbsym, True, 0,
        ...                 0., 0., h1e_cas, h2e_cas, core_idx, cas_idx, nelec, 0, None, None, None)
        >>> np.isclose(e, -2.8759428090050676)
        True
        >>> ndets
        36
        >>> exc = main('fci', 'pyscf_spin0', occup, 'excitation', 'A', orbsym, True, 1,
        ...            0., 0., h1e_cas, h2e_cas, core_idx, cas_idx, nelec, 0, None, None, None)[0]
        >>> np.isclose(exc, 1.850774199956839)
        True
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = '631g', symmetry = 'C2v', verbose=0)
        >>> hf = scf.RHF(mol)
        >>> _ = hf.kernel()
        >>> orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        >>> hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
        >>> h1e = np.einsum('pi,pq,qj->ij', hf.mo_coeff, hcore_ao, hf.mo_coeff)
        >>> eri_ao = mol.intor('int2e_sph', aosym=4)
        >>> h2e = ao2mo.incore.full(eri_ao, hf.mo_coeff)
        >>> core_idx = np.array([])
        >>> cas_idx = np.array([0, 1, 2, 3, 4, 7, 9])
        >>> h1e_cas = h1e[cas_idx[:, None], cas_idx]
        >>> cas_idx_tril = tools.cas_idx_tril(cas_idx)
        >>> h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
        >>> nelec = tools.nelec(hf.mo_occ, cas_idx)
        >>> e_core = mol.energy_nuc()
        >>> e = main('ccsd', 'pyscf_spin0', hf.mo_occ, 'energy', 'A1', orbsym, True, 0,
        ...          hf.e_tot, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, 0,
        ...          None, None, None)[0]
        >>> np.isclose(e, -0.014118607610972691)
        True
        >>> ao_dipole = dipole_ints(mol)
        >>> dipole_hf = np.einsum('xij,ji->x', ao_dipole, hf.make_rdm1())
        >>> dipole = main('fci', 'pyscf_spin0', hf.mo_occ, 'dipole', 'A1', orbsym, True, 0,
        ...               hf.e_tot, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, 0,
        ...               ao_dipole, hf.mo_coeff, dipole_hf)[0]
        >>> np.allclose(dipole, np.array([0., 0., -7.97781259e-03]))
        True
        >>> trans = main('fci', 'pyscf_spin0', hf.mo_occ, 'trans', 'A1', orbsym, True, 1,
        ...              hf.e_tot, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, 0,
        ...              ao_dipole, hf.mo_coeff, dipole_hf)[0]
        >>> np.allclose(trans, np.array([0., 0., -0.26497816]))
        True
       """
        if method in ['ccsd', 'ccsd(t)']:

            res_tmp = _cc(occup, core_idx, cas_idx, method, h1e=h1e, h2e=h2e, rdm1=target_mbe == 'dipole')
            ndets = tools.ndets(occup, cas_idx, n_elec=nelec)

        elif method == 'fci':

            res_tmp = _fci(solver, target_mbe, state_wfnsym, \
                            orbsym, hf_guess, state_root, \
                            e_hf, e_core, h1e, h2e, \
                            occup, core_idx, cas_idx, nelec, debug)
            ndets = res_tmp['ndets']

        if target_mbe in ['energy', 'excitation']:

            res = res_tmp[target_mbe]

        elif target_mbe == 'dipole':

            res = _dipole(ao_dipole, occup, dipole_hf, mo_coeff, cas_idx, res_tmp['rdm1'])

        elif target_mbe == 'trans':

            res = _trans(ao_dipole, occup, dipole_hf, \
                            mo_coeff, cas_idx, res_tmp['t_rdm1'], \
                            res_tmp['hf_weight'][0], res_tmp['hf_weight'][1])

        return res, ndets


def _dipole(ao_dipole: np.ndarray, occup: np.ndarray, hf_dipole: np.ndarray, mo_coeff: np.ndarray, \
                cas_idx: np.ndarray, cas_rdm1: np.ndarray, trans: bool = False) -> np.ndarray:
        """
        this function returns an electronic (transition) dipole moment

        example:
        >>> occup = np.array([2.] * 3 + [0.] * 3)
        >>> hf_dipole = np.zeros(3, dtype=np.float64)
        >>> mo_coeff = np.eye(6, dtype=np.float64)
        >>> cas_idx = np.arange(1, 5)
        >>> np.random.seed(1234)
        >>> ao_dipole = np.random.rand(3, 6, 6)
        >>> np.random.seed(1234)
        >>> cas_rdm1 = np.random.rand(cas_idx.size, cas_idx.size)
        >>> dipole = _dipole(ao_dipole, occup, hf_dipole, mo_coeff, cas_idx, cas_rdm1)
        >>> np.allclose(dipole, np.array([5.90055525, 5.36437348, 6.40001788]))
        True
        """
        # init (transition) rdm1
        if trans:
            rdm1 = np.zeros_like(mo_coeff)
        else:
            rdm1 = np.diag(occup)

        # insert correlated subblock
        rdm1[cas_idx[:, None], cas_idx] = cas_rdm1

        # ao representation
        rdm1 = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1, mo_coeff)

        # compute elec_dipole
        elec_dipole = np.einsum('xij,ji->x', ao_dipole, rdm1)

        # remove noise
        elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > DIPOLE_TOL else 0. for i in range(elec_dipole.size)])

        # 'correlation' dipole
        if not trans:
            elec_dipole -= hf_dipole

        return elec_dipole


def _trans(ao_dipole: np.ndarray, occup: np.ndarray, hf_dipole: np.ndarray, mo_coeff: np.ndarray, \
            cas_idx: np.ndarray, cas_rdm1: np.ndarray, hf_weight_gs: float, hf_weight_ex: float) -> np.ndarray:
        """
        this function returns an electronic transition dipole moment

        example:
        >>> occup = np.array([2.] * 3 + [0.] * 3)
        >>> hf_dipole = np.zeros(3, dtype=np.float64)
        >>> mo_coeff = np.eye(6, dtype=np.float64)
        >>> cas_idx = np.arange(1, 5)
        >>> np.random.seed(1234)
        >>> ao_dipole = np.random.rand(3, 6, 6)
        >>> np.random.seed(1234)
        >>> cas_rdm1 = np.random.rand(cas_idx.size, cas_idx.size)
        >>> trans = _trans(ao_dipole, occup, hf_dipole, mo_coeff, cas_idx, cas_rdm1, .9, .4)
        >>> np.allclose(trans, np.array([5.51751635, 4.92678927, 5.45675281]))
        True
        """
        return _dipole(ao_dipole, occup, hf_dipole, mo_coeff, cas_idx, cas_rdm1, True) \
                        * np.sign(hf_weight_gs) * np.sign(hf_weight_ex)


def base(mol: system.MolCls, occup: np.ndarray, target_mbe: str, \
                method: str, mo_coeff: np.ndarray, dipole_hf: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        this function returns base model energy

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = '631g', symmetry = 'C2v', verbose=0)
        >>> mol.ncore = 1
        >>> mol.hf_symmetry = mol.symmetry
        >>> mol.debug = 0
        >>> mol.hf_init_guess = 'h1e'
        >>> mol.irrep_nelec = {}
        >>> mol.dipole = dipole_ints(mol)
        >>> mol.e_nuc = mol.energy_nuc()
        >>> mol.nocc, mol.nvirt, mol.norb, _, e_hf, dipole_hf, occup, orbsym, mo_coeff = hf(mol, 'dipole')
        >>> orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)
        >>> mol.hcore, mol.vhf, mol.eri = ints(mol, mo_coeff, True, True,
        ...                                    MPI.COMM_WORLD, MPI.COMM_WORLD, MPI.COMM_WORLD, 1)
        >>> e, dipole = base(mol, occup, 'energy', 'ccsd(t)', mo_coeff, dipole_hf)
        >>> np.isclose(e, -0.1353082155512597)
        True
        >>> np.allclose(dipole, np.zeros(3, dtype=np.float64))
        True
        >>> e, dipole = base(mol, occup, 'dipole', 'ccsd', mo_coeff, dipole_hf)
        >>> np.isclose(e, -0.13432841702437032)
        True
        >>> np.allclose(dipole, np.array([0., 0., -0.04312132]))
        True
        """
        # load hcore
        buf = mol.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # load vhf
        buf = mol.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # load eri
        buf = mol.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb*(mol.norb + 1) // 2,) * 2)

        # set core and cas spaces
        core_idx, cas_idx = tools.core_cas(mol.nocc, np.arange(mol.ncore, mol.nocc), np.arange(mol.nocc, mol.norb))

        # get cas space h2e
        cas_idx_tril = tools.cas_idx_tril(cas_idx)
        h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

        # get e_core and h1e_cas
        e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

        # run calc
        res_tmp = _cc(occup, core_idx, cas_idx, method, h1e=h1e_cas, h2e=h2e_cas, rdm1=target_mbe == 'dipole')

        # collect results
        energy = res_tmp['energy']
        if target_mbe == 'energy':
            dipole = np.zeros(3, dtype=np.float64)
        else:
            dipole = _dipole(mol.dipole, occup, dipole_hf, mo_coeff, cas_idx, res_tmp['rdm1'])

        return energy, dipole


def _casscf(mol: system.MolCls, solver: str, wfnsym: List[str], \
                orbsym: np.ndarray, hf_guess: bool, hf: scf.RHF, \
                mo_coeff: np.ndarray, ref_space: np.ndarray, nelec: Tuple[int, int]) -> np.ndarray:
        """
        this function returns the results of a casscf calculation

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='C 0. 0. 0.625; C 0. 0. -0.625',
        ...               basis = '631g', symmetry = 'D2h', verbose=0)
        >>> mol.ncore, mol.nocc, mol.nvirt, mol.norb = 2, 6, 12, 18
        >>> mol.debug = 0
        >>> hf = scf.RHF(mol)
        >>> _ = hf.kernel()
        >>> orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        >>> mo_coeff = _casscf(mol, 'pyscf_spin0', ['Ag'], orbsym, True,
        ...                    hf, hf.mo_coeff, np.arange(2, 10), (4, 4))
        >>> np.isclose(np.sum(mo_coeff), 2.2922857024683)
        True
        >>> np.isclose(np.amax(mo_coeff), 6.528333586540256)
        True
        >>> mo_coeff = _casscf(mol, 'pyscf_spin0', ['Ag', 'Ag', 'Ag', 'B1g'], orbsym, False,
        ...                    hf, hf.mo_coeff, np.arange(2, 10), (4, 4))
        >>> np.isclose(np.sum(mo_coeff), 2.700100458554667)
        True
        >>> np.isclose(np.amax(mo_coeff), 6.437087455128202)
        True
        """
        # init casscf
        cas = mcscf.CASSCF(hf, ref_space.size, nelec)

        # casscf settings
        cas.conv_tol = CONV_TOL
        cas.max_cycle_macro = 500
        cas.frozen = mol.ncore
        if mol.debug >= 1:
            cas.verbose = 4

        # init fcisolver
        if solver == 'pyscf_spin0':
            fcisolver = fci.direct_spin0_symm.FCI(mol)
        elif solver == 'pyscf_spin1':
            fcisolver = fci.direct_spin1_symm.FCI(mol)

        # fci settings
        fcisolver.conv_tol = CONV_TOL
        fcisolver.orbsym = orbsym[ref_space]
        fcisolver.wfnsym = wfnsym[0]
        cas.fcisolver = fcisolver

        # state-averaged casscf
        if len(wfnsym) > 1:

            # weights
            weights = np.array((1 / len(wfnsym),) * len(wfnsym), dtype=np.float64)

            if len(set(wfnsym)) == 1:

                # state average over all states of same symmetry
                cas.state_average_(weights)

            else:

                # nroots for first fcisolver
                fcisolver.nroots = np.count_nonzero(np.asarray(wfnsym) == list(set(wfnsym))[0])

                # init list of fcisolvers
                fcisolvers = [fcisolver]

                # loop over symmetries
                for i in range(1, len(set(wfnsym))):

                    # copy fcisolver
                    fcisolver_ = copy.copy(fcisolver)

                    # wfnsym for fcisolver_
                    fcisolver_.wfnsym = list(set(wfnsym))[i]

                    # nroots for fcisolver_
                    fcisolver_.nroots = np.count_nonzero(np.asarray(wfnsym) == list(set(wfnsym))[i])

                    # append to fcisolvers
                    fcisolvers.append(fcisolver_)

                # state average
                mcscf.state_average_mix_(cas, fcisolvers, weights)

        # hf starting guess
        if hf_guess:
            na = fci.cistring.num_strings(ref_space.size, nelec[0])
            nb = fci.cistring.num_strings(ref_space.size, nelec[1])
            ci0 = np.zeros((na, nb))
            ci0[0, 0] = 1
        else:
            ci0 = None

        # run casscf calc
        cas.kernel(mo_coeff, ci0=ci0)

        # collect ci vectors
        if len(wfnsym) == 1:
            c = [cas.ci]
        else:
            c = cas.ci

        # multiplicity check
        for root in range(len(c)):

            s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)

            if np.abs((mol.spin + 1) - mult) > SPIN_TOL:

                # fix spin by applyting level shift
                sz = np.abs(nelec[0]-nelec[1]) * 0.5
                cas.fix_spin_(shift=0.25, ss=sz * (sz + 1.))

                # run casscf calc
                cas.kernel(mo_coeff, ci0=ci0)

                # collect ci vectors
                if len(wfnsym) == 1:
                    c = [cas.ci]
                else:
                    c = cas.ci

                # verify correct spin
                for root in range(len(c)):
                    s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)
                    tools.assertion(np.abs((mol.spin + 1) - mult) < SPIN_TOL, \
                                    'spin contamination for root entry = {:} , 2*S + 1 = {:.6f}'. \
                                        format(root, mult))

        # convergence check
        tools.assertion(cas.converged, 'CASSCF error: no convergence')

        # debug print of orbital energies
        if mol.atom:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, cas.mo_coeff)
        else:
            orbsym = np.zeros(cas.mo_energy.size, dtype=np.int)

        if mol.debug >= 1:
            if mol.symmetry:
                gpname = mol.symmetry
            else:
                gpname = 'C1'
            print('\n CASSCF:  mo   symmetry    energy')
            for i in range(cas.mo_energy.size):
                print('         {:>3d}   {:>5s}     {:>7.3f}'.format(i, symm.addons.irrep_id2name(gpname, orbsym[i]), cas.mo_energy[i]))
            print('\n')

        return np.asarray(cas.mo_coeff, order='C')


def _fci(solver_type: str, target_mbe: str, wfnsym: str, orbsym: np.ndarray, \
            hf_guess: bool, root: int, e_hf: float, e_core: float, \
            h1e: np.ndarray, h2e: np.ndarray, occup: np.ndarray, \
            core_idx: np.ndarray, cas_idx: np.ndarray, \
            nelec: Tuple[int, int], debug: int) -> Dict[str, Union[float, np.ndarray]]:
        """
        this function returns the results of a fci calculation

        example:
        >>> occup = np.array([2.] * 4 + [0.] * 4)
        >>> orbsym = np.zeros(8, dtype=np.int)
        >>> h1e = hubbard_h1e((2, 4), True)
        >>> h2e = hubbard_eri((2, 4), 2.)
        >>> h2e = ao2mo.restore(4, h2e, 8)
        >>> res = _fci('pyscf_spin0', 'energy', 'A', orbsym, True, 0, 0., 0.,
        ...          h1e, h2e, occup, np.array([]), np.arange(8), (4, 4), 0)
        >>> res['ndets']
        4900
        >>> np.isclose(res['energy'], -5.246918061839909)
        True
        >>> res = _fci('pyscf_spin1', 'excitation', 'A', orbsym, True, 1, 0., 0.,
        ...          h1e, h2e, occup, np.array([]), np.arange(8), (4, 4), 0)
        >>> np.isclose(res['energy'], -4.179698414137736)
        True
        >>> np.isclose(res['excitation'], 1.0672196477046079)
        True
        >>> res = _fci('pyscf_spin0', 'dipole', 'A', orbsym, True, 1, 0., 0.,
        ...          h1e, h2e, occup, np.array([]), np.arange(8), (4, 4), 0)
        >>> np.isclose(np.sum(res['rdm1']), 15.544465598616451)
        True
        >>> np.isclose(np.amax(res['rdm1']), 1.)
        True
        >>> res = _fci('pyscf_spin1', 'trans', 'A', orbsym, True, 1, 0., 0.,
        ...          h1e, h2e, occup, np.array([]), np.arange(8), (4, 4), 0)
        >>> np.isclose(np.sum(res['t_rdm1']), 0.)
        True
        >>> np.isclose(np.amax(res['t_rdm1']), 0.1008447233008727)
        True
        >>> np.isclose(np.sum(res['hf_weight']), 0.)
        True
        """
        # spin
        spin = np.count_nonzero(occup[cas_idx] == 1.)

        # init fci solver
        if solver_type == 'pyscf_spin0':
            solver = fci.direct_spin0_symm.FCI()
        elif solver_type == 'pyscf_spin1':
            solver = fci.direct_spin1_symm.FCI()

        # settings
        solver.conv_tol = CONV_TOL
        if target_mbe in ['dipole', 'trans']:
            solver.conv_tol *= 1.e-04
            solver.lindep = solver.conv_tol * 1.e-01
        solver.max_memory = MAX_MEM
        solver.max_cycle = 5000
        solver.max_space = 25
        solver.davidson_only = True
        solver.pspace_size = 0
        if debug >= 3:
            solver.verbose = 10
        solver.wfnsym = wfnsym
        solver.orbsym = orbsym[cas_idx]
        solver.nroots = root + 1

        # hf starting guess
        if hf_guess:
            na = fci.cistring.num_strings(cas_idx.size, nelec[0])
            nb = fci.cistring.num_strings(cas_idx.size, nelec[1])
            ci0 = np.zeros((na, nb))
            ci0[0, 0] = 1
        else:
            ci0 = None

        # interface
        def _fci_kernel():
                """
                this function provides an interface to solver.kernel
                """
                # perform calc
                e, c = solver.kernel(h1e, h2e, cas_idx.size, nelec, ecore=e_core, \
                                        orbsym=solver.orbsym, ci0=ci0)

                # collect results
                if solver.nroots == 1:
                    return [e], [c]
                else:
                    return [e[0], e[-1]], [c[0], c[-1]]

        # perform calc
        energy, civec = _fci_kernel()

        # multiplicity check
        for root in range(len(civec)):

            s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)

            if np.abs((spin + 1) - mult) > SPIN_TOL:

                # fix spin by applyting level shift
                sz = np.abs(nelec[0]-nelec[1]) * 0.5
                solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.))

                # perform calc
                energy, civec = _fci_kernel()

                # verify correct spin
                for root in range(len(civec)):
                    s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
                    tools.assertion(np.abs((spin + 1) - mult) < SPIN_TOL, \
                                    'spin contamination for root entry = {:}\n2*S + 1 = {:.6f}\n'
                                    'cas_idx = {:}\ncas_sym = {:}'. \
                                    format(root, mult, cas_idx, orbsym[cas_idx]))

        # convergence check
        if solver.nroots == 1:

            tools.assertion(solver.converged, \
                                 'state {:} not converged\n'
                                 'cas_idx = {:}\ncas_sym = {:}'. \
                                 format(root, cas_idx, orbsym[cas_idx]))

        else:

            if target_mbe == 'excitation':

                for root in [0, solver.nroots-1]:
                    tools.assertion(solver.converged[root], \
                                         'state {:} not converged\n'
                                         'cas_idx = {:}\ncas_sym = {:}'. \
                                         format(root, cas_idx, orbsym[cas_idx]))

            else:

                tools.assertion(solver.converged[solver.nroots-1], \
                                     'state {:} not converged\n'
                                     'cas_idx = {:}\ncas_sym = {:}'. \
                                     format(solver.nroots-1, cas_idx, orbsym[cas_idx]))

        # collect results
        res: Dict[str, Union[int, float, np.ndarray]] = {'ndets': np.count_nonzero(civec[-1])}
        res['energy'] = energy[-1] - e_hf
        if target_mbe == 'excitation':
            res['excitation'] = energy[-1] - energy[0]
        elif target_mbe == 'dipole':
            res['rdm1'] = solver.make_rdm1(civec[-1], cas_idx.size, nelec)
        elif target_mbe == 'trans':
            res['t_rdm1'] = solver.trans_rdm1(civec[0], civec[-1], cas_idx.size, nelec)
            res['hf_weight'] = [civec[i][0, 0] for i in range(2)]

        return res


def _cc(occup, core_idx, cas_idx, method, h1e=None, h2e=None, hf=None, rdm1=False):
        """
        this function returns the results of a ccsd / ccsd(t) calculation

        :param occup: orbital occupation. numpy array of shape (n_orb,)
        :param core_idx: core space indices. numpy array of shape (n_inactive,)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :param method: cc model. string
        :param h1e: cas space 1e integrals. numpy array of shape (n_cas, n_cas)
        :param h2e: cas space 2e integrals. numpy array of shape (n_cas*(n_cas + 1) // 2, n_cas*(n_cas + 1) // 2)
        :param hf: pyscf hf object
        :param rdm1: rdm1 logical. bool
        :return: float or numpy array of shape (n_orb-n_core, n_orb-n_core) depending on rdm1 bool
        """
        spin = np.count_nonzero(occup[cas_idx] == 1.)
        singlet = spin == 0

        # init ccsd solver
        if h1e is not None and h2e is not None:

            mol_tmp = gto.M(verbose=0)
            mol_tmp.max_memory = MAX_MEM
            mol_tmp.incore_anyway = True

            if singlet:
                hf = scf.RHF(mol_tmp)
            else:
                hf = scf.UHF(mol_tmp)

            hf.get_hcore = lambda *args: h1e
            hf._eri = h2e

            if singlet:
                ccsd = cc.ccsd.CCSD(hf, mo_coeff=np.eye(cas_idx.size), mo_occ=occup[cas_idx])
            else:
                ccsd = cc.uccsd.UCCSD(hf, mo_coeff=np.array((np.eye(cas_idx.size), np.eye(cas_idx.size))), \
                                        mo_occ=np.array((occup[cas_idx] > 0., occup[cas_idx] == 2.), dtype=np.double))

        else:

            tools.assertion(hf is not None, 'in the absence of h1e and h2e, hf object must be passed to cc solver')
            ccsd = cc.CCSD(hf)
            ccsd.frozen = core_idx.size

        # settings
        ccsd.conv_tol = CONV_TOL
        if rdm1:
            ccsd.conv_tol_normt = ccsd.conv_tol
        ccsd.max_cycle = 500
        ccsd.async_io = False
        ccsd.incore_complete = True
        eris = ccsd.ao2mo()

        # calculate ccsd energy
        for i in range(0, 12, 2):

            ccsd.diis_start_cycle = i

            try:
                ccsd.kernel(eris=eris)
            except sp.linalg.LinAlgError:
                pass

            if ccsd.converged:
                break

        # convergence check
        tools.assertion(ccsd.converged, 'CCSD error: no convergence , core_idx = {:} , cas_idx = {:}'. \
                                        format(core_idx, cas_idx))

        # e_corr
        e_cc = ccsd.e_corr

        # calculate (t) correction
        if method == 'ccsd(t)':

            if np.amin(occup[cas_idx]) == 1.:
                if np.where(occup[cas_idx] == 1.)[0].size >= 3:
                    e_cc += ccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)

            else:

                e_cc += ccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)

        # collect results
        res: Dict[str, Union[float, np.ndarray]] = {'energy': e_cc}

        # rdm1
        if rdm1:
            if method == 'ccsd':
                ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
                res['rdm1'] = ccsd.make_rdm1()
            elif method == 'ccsd(t)':
                l1, l2 = ccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)[1:]
                res['rdm1'] = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)

        return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()#verbose=True)


