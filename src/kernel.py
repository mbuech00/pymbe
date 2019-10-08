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
from typing import Tuple, List, Union
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


def hf(mol: system.MolCls, target: str) -> Tuple[int, int, int, scf.RHF, float, np.ndarray, \
                                                    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        >>> nocc, nvirt, norb, pyscf_hf, e_hf, dipole, occup, orbsym, mo_energy, mo_coeff = hf(mol, 'energy')
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
        >>> mo_energy_ref = np.array([-20.56043822,  -1.35751378,  -0.71019175,  -0.56159433, -0.50164834,
        ...                         0.20395512,   0.30015235,   1.05581618, 1.16430251, 1.19036532,
        ...                         1.2167634 ,   1.37962513, 1.69745496])
        >>> np.allclose(mo_energy, mo_energy_ref)
        True
        >>> mol.dipole = dipole_ints(mol)
        >>> dipole = hf(mol, 'dipole')[5]
        >>> dipole_ref = np.array([0., 0., 0.8642558])
        >>> np.allclose(dipole, dipole_ref)
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
        if target == 'dipole':
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
                orbsym, hf.mo_energy, np.asarray(hf.mo_coeff, order='C')


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


def ref_mo(mol, calc):
        """
        this function returns a set of reference mo energies and coefficients plus the associated spaces

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: numpy array of shape (n_orb,) [mo_energy],
                 numpy array of shape (n_orb, n_orb) [mo_coeff],
                 tuple of integers [nelec],
                 numpy array of shape (n_orb) [ref_space],
                 dictionary with numpy arrays of shapes (n_exp_occ,), (n_exp_virt,), and (n_exp_tot)
        """
        # init mo_energy and mo_coeff
        mo_energy = calc.mo_energy
        mo_coeff = calc.mo_coeff

        if calc.orbs['type'] != 'can':

            # set core and cas spaces
            core_idx, cas_idx = tools.core_cas(mol.nocc, np.arange(mol.ncore), np.arange(mol.ncore, mol.norb))

            # NOs
            if calc.orbs['type'] in ['ccsd', 'ccsd(t)']:

                # compute rmd1
                rdm1 = _cc(calc.occup, core_idx, cas_idx, calc.orbs['type'], hf=calc.hf, rdm1=True)
                if mol.spin > 0:
                    rdm1 = rdm1[0] + rdm1[1]

                # occ-occ block
                occup, no = symm.eigh(rdm1[:(mol.nocc-mol.ncore), :(mol.nocc-mol.ncore)], calc.orbsym[mol.ncore:mol.nocc])
                calc.mo_coeff[:, mol.ncore:mol.nocc] = np.einsum('ip,pj->ij', calc.mo_coeff[:, mol.ncore:mol.nocc], no[:, ::-1])

                # virt-virt block
                occup, no = symm.eigh(rdm1[-mol.nvirt:, -mol.nvirt:], calc.orbsym[mol.nocc:])
                calc.mo_coeff[:, mol.nocc:] = np.einsum('ip,pj->ij', calc.mo_coeff[:, mol.nocc:], no[:, ::-1])

            # pipek-mezey localized orbitals
            elif calc.orbs['type'] == 'local':

                # occ-occ block
                if mol.atom:
                    loc = lo.PM(mol, calc.mo_coeff[:, mol.ncore:mol.nocc])
                else:
                    loc = _hubbard_PM(mol, calc.mo_coeff[:, mol.ncore:mol.nocc])
                loc.conv_tol = CONV_TOL
                if mol.debug >= 1:
                    loc.verbose = 4
                calc.mo_coeff[:, mol.ncore:mol.nocc] = loc.kernel()

                # virt-virt block
                if mol.atom:
                    loc = lo.PM(mol, calc.mo_coeff[:, mol.nocc:])
                else:
                    loc = _hubbard_PM(mol, calc.mo_coeff[:, mol.nocc:])
                loc.conv_tol = CONV_TOL
                if mol.debug >= 1:
                    loc.verbose = 4
                calc.mo_coeff[:, mol.nocc:] = loc.kernel()

        # sort orbitals
        if calc.ref['active'] == 'manual':

            # active orbs
            calc.ref['select'] = np.asarray(calc.ref['select'], dtype=np.int16)

            # electrons
            nelec = (np.count_nonzero(calc.occup[calc.ref['select']] > 0.), \
                        np.count_nonzero(calc.occup[calc.ref['select']] > 1.))

            # inactive orbitals
            inact_elec = mol.nelectron - (nelec[0] + nelec[1])
            tools.assertion(inact_elec % 2 == 0, 'odd number of inactive electrons')
            inact_orbs = inact_elec // 2

            # active orbitals
            act_orbs = calc.ref['select'].size

            # virtual orbitals
            virt_orbs = mol.norb - inact_orbs - act_orbs

            # divide into inactive-active-virtual
            idx = np.asarray([i for i in range(mol.norb) if i not in calc.ref['select']])
            if act_orbs > 0:

                mo_coeff = np.concatenate((mo_coeff[:, idx[:inact_orbs]], mo_coeff[:, calc.ref['select']], mo_coeff[:, idx[inact_orbs:]]), axis=1)

                # update orbsym
                if mol.atom:
                    calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)

        # reference and expansion spaces
        ref_space = np.arange(inact_orbs, inact_orbs+act_orbs, dtype=np.int16)
        exp_space = np.append(np.arange(mol.ncore, inact_orbs, dtype=np.int16), \
                              np.arange(inact_orbs+act_orbs, mol.norb, dtype=np.int16))

        # divide exp_space into occupied and virtual parts
        exp_space = {'tot': exp_space}
        exp_space['occ'] = exp_space['tot'][exp_space['tot'] < mol.nocc]
        exp_space['virt'] = exp_space['tot'][mol.nocc <= exp_space['tot']]

        # casci or casscf
        if calc.ref['method'] == 'casci':

            if act_orbs > 0:
                mo_energy = np.concatenate((mo_energy[idx[:inact_orbs]], \
                                            mo_energy[calc.ref['select']], \
                                            mo_energy[idx[inact_orbs:]]))

        elif calc.ref['method'] == 'casscf':

            tools.assertion(np.count_nonzero(calc.occup[calc.ref['select']] > 0.) != 0, \
                            'no singly/doubly occupied orbitals in CASSCF calculation')
            tools.assertion(np.count_nonzero(calc.occup[calc.ref['select']] < 2.) != 0, \
                            'no virtual/singly occupied orbitals in CASSCF calculation')

            mo_energy, mo_coeff = _casscf(mol, calc.model['solver'], calc.ref['wfnsym'], calc.orbsym, \
                                            calc.ref['hf_guess'], calc.hf, mo_coeff, ref_space, nelec)

        # pi-orbital space
        if calc.extra['pi_prune']:
            exp_space['pi_orbs'], exp_space['pi_hashes'] = tools.pi_space(mo_energy, exp_space['tot'])
        else:
            exp_space['pi_orbs'] = exp_space['pi_hashes'] = None

        # debug print of reference and expansion spaces
        if mol.debug >= 1:
            print('\n reference nelec        = {:}'.format(nelec))
            print(' reference space        = {:}'.format(ref_space))
            if calc.extra['pi_prune']:
                print(' expansion space [pi]   =\n{:}'.format(exp_space['pi_orbs'].reshape(-1, 2)))
            print(' expansion space [occ]  = {:}'.format(exp_space['occ']))
            print(' expansion space [virt] = {:}\n'.format(exp_space['virt']))

        return mo_energy, np.asarray(mo_coeff, order='C'), nelec, ref_space, exp_space


def ref_prop(mol, calc):
        """
        this function returns reference space properties

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: float or numpy array of shape (3,) depending on calc.target_mbe
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
        core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, np.array([], dtype=np.int16))

        # nelec
        nelec = np.asarray((np.count_nonzero(calc.occup[cas_idx] > 0.), \
                            np.count_nonzero(calc.occup[cas_idx] > 1.)), dtype=np.int16)

        if np.any(calc.occup[calc.ref_space] == 2.) and np.any(calc.occup[calc.ref_space] < 2.):

            # get cas space h2e
            cas_idx_tril = tools.cas_idx_tril(cas_idx)
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

            # exp model
            ref = main(calc.model['method'], calc.model['solver'], calc.occup, calc.target_mbe, \
                        calc.state['wfnsym'], calc.orbsym, calc.extra['hf_guess'], calc.state['root'], \
                        calc.prop['hf']['energy'], e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, mol.debug, \
                        mol.dipole if calc.target_mbe in ['dipole', 'trans'] else None, \
                        calc.mo_coeff if calc.target_mbe in ['dipole', 'trans'] else None, \
                        calc.prop['hf']['dipole'] if calc.target_mbe in ['dipole', 'trans'] else None)[0]

            # base model
            if calc.base['method'] is not None:
                ref -= main(calc.base['method'], '', calc.occup, calc.target_mbe, \
                            calc.state['wfnsym'], calc.orbsym, calc.extra['hf_guess'], calc.state['root'], \
                            calc.prop['hf']['energy'], e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, mol.debug, \
                            mol.dipole if calc.target_mbe in ['dipole', 'trans'] else None, \
                            calc.mo_coeff if calc.target_mbe in ['dipole', 'trans'] else None, \
                            calc.prop['hf']['dipole'] if calc.target_mbe in ['dipole', 'trans'] else None)[0]

        else:

            # no correlation in expansion reference space
            if calc.target_mbe in ['energy', 'excitation']:
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
        >>> nelec = (2, 2)
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
        >>> e = main('ccsd', '', occup, 'energy', 'A', orbsym, True, 0,
        ...          0., 0., h1e_cas, h2e_cas, core_idx, cas_idx, nelec, 0, None, None, None)[0]
        >>> np.isclose(e, 0.8234069541302586)
        True
        >>> np.random.seed(1234)
        >>> ao_dipole = np.random.rand(3, 6, 6)
        >>> mo_coeff = np.eye(6, dtype=np.float64)
        >>> dipole_hf = np.zeros(3, dtype=np.float64)
        >>> dipole = main('fci', 'pyscf_spin0', occup, 'dipole', 'A', orbsym, True, 0,
        ...               0., 0., h1e_cas, h2e_cas, core_idx, cas_idx, nelec, 0,
        ...               ao_dipole, mo_coeff, dipole_hf)[0]
        >>> dipole_ref = np.array([4.59824861, 4.7898921 , 5.10183542])
        >>> np.allclose(dipole, dipole_ref)
        True
        >>> trans = main('fci', 'pyscf_spin0', occup, 'trans', 'A', orbsym, True, 1,
        ...              0., 0., h1e_cas, h2e_cas, core_idx, cas_idx, nelec, 0,
        ...              ao_dipole, mo_coeff, dipole_hf)[0]
        >>> trans_ref = np.array([-0.39631621, -0.19800879, -0.31924946])
        >>> np.allclose(trans, trans_ref)
        True
        """
        if method in ['ccsd', 'ccsd(t)']:

            res = _cc(occup, core_idx, cas_idx, method, h1e=h1e, h2e=h2e)
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

                res = _dipole(ao_dipole, occup, dipole_hf, \
                                mo_coeff, cas_idx, res_tmp['rdm1'])

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
        >>> dipole_ref = np.array([5.90055525, 5.36437348, 6.40001788])
        >>> np.allclose(dipole, dipole_ref)
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
        >>> trans_ref = np.array([5.51751635, 4.92678927, 5.45675281])
        >>> np.allclose(trans, trans_ref)
        True
        """
        return _dipole(ao_dipole, occup, hf_dipole, mo_coeff, cas_idx, cas_rdm1, True) \
                        * np.sign(hf_weight_gs) * np.sign(hf_weight_ex)


def base(mol, occup, method):
        """
        this function returns base model energy

        :param mol: pymbe mol object
        :param occup: orbital occupation. numpy array of shape (n_orb,)
        :param method: base model. string
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

        return _cc(occup, core_idx, cas_idx, method, h1e=h1e_cas, h2e=h2e_cas)


def _casscf(mol, solver, wfnsym, orbsym, hf_guess, hf, mo_coeff, ref_space, nelec):
        """
        this function returns the results of a casscf calculation

        :param mol: pymbe mol object
        :param solver: fci solver. string
        :param wfnsym: wave function symmetries for involved states. list of strings
        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param hf_guess: hf as initial guess. bool
        :param hf: pyscf hf object
        :param mo_coeff: input mo coefficients. numpy array of shape (n_orb, n_orb)
        :param ref_space: reference space. numpy array of shape (n_ref,)
        :param nelec: number of correlated electrons. tuple of integers
        :return: numpy array of shape (n_orb,) [mo_energy],
                 numpy array of shape (n_orb, n_orb) [mo_coeff]
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

        return cas.mo_energy, np.asarray(cas.mo_coeff, order='C')


def _fci(solver, target, wfnsym, orbsym, hf_guess, root, hf_energy, \
            e_core, h1e, h2e, occup, core_idx, cas_idx, nelec, debug):
        """
        this function returns the results of a fci calculation

        :param mol: pymbe mol object
        :param solver: fci solver. string
        :param target: calculation target. string
        :param wfnsym: wave function symmetry. string
        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param hf_guess: hf as initial guess. bool
        :param root: state root of interest. integer
        :param hf_energy: hf energy. float
        :param e_core: core space energy. float
        :param h1e: cas space 1e integrals. numpy array of shape (n_cas, n_cas)
        :param h2e: cas space 2e integrals. numpy array of shape (n_cas*(n_cas + 1) // 2, n_cas*(n_cas + 1) // 2)
        :param core_idx: core space indices. numpy array of shape (n_inactive,)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :param nelec: number of correlated electrons. tuple of integers
        :return: dict of float [ndets],
                 floats [energy and excitation],
                 numpy array of shape (n_cas, n_cas) [dipole],
                 or numpy array of shape (n_cas, n_cas) and a list of floats [trans]
        """
        # spin
        spin = np.count_nonzero(occup[cas_idx] == 1.)

        # init fci solver
        if solver == 'pyscf_spin0':
            solver = fci.direct_spin0_symm.FCI()
        elif solver == 'pyscf_spin1':
            solver = fci.direct_spin1_symm.FCI()

        # settings
        solver.conv_tol = CONV_TOL
        if target in ['dipole', 'trans']:
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
                                    'core_idx = {:}\ncore_sym = {:}\ncas_idx = {:}\ncas_sym = {:}'. \
                                    format(root, mult, core_idx, orbsym[core_idx], cas_idx, orbsym[cas_idx]))

        # convergence check
        if solver.nroots == 1:

            tools.assertion(solver.converged, \
                                 'state {:} not converged\ncore_idx = {:}\ncore_sym = {:}\n'
                                 'cas_idx = {:}\ncas_sym = {:}'. \
                                 format(root, core_idx, orbsym[core_idx] if core_idx.size > 0 else core_idx, \
                                        cas_idx, orbsym[cas_idx]))

        else:

            if target == 'excitation':

                for root in [0, solver.nroots-1]:
                    tools.assertion(solver.converged[root], \
                                         'state {:} not converged\ncore_idx = {:}\ncore_sym = {:}\n'
                                         'cas_idx = {:}\ncas_sym = {:}'. \
                                         format(root, core_idx, orbsym[core_idx], cas_idx, orbsym[cas_idx]))

            else:

                tools.assertion(solver.converged[solver.nroots-1], \
                                     'state {:} not converged\ncore_idx = {:}\ncore_sym = {:}\n'
                                     'cas_idx = {:}\ncas_sym = {:}'. \
                                     format(solver.nroots-1, core_idx, orbsym[core_idx], cas_idx, orbsym[cas_idx]))

        # collect results
        res = {'ndets': np.count_nonzero(civec[-1])}
        if target == 'energy':
            res['energy'] = energy[-1] - hf_energy
        elif target == 'excitation':
            res['excitation'] = energy[-1] - energy[0]
        elif target == 'dipole':
            res['rdm1'] = solver.make_rdm1(civec[-1], cas_idx.size, nelec)
        elif target == 'trans':
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

        # rdm1
        if not rdm1:

            return e_cc

        else:

            if method == 'ccsd':

                ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
                rdm1 = ccsd.make_rdm1()

            elif method == 'ccsd(t)':

                l1, l2 = ccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)[1:]
                rdm1 = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)

            return rdm1


if __name__ == "__main__":
    import doctest
    doctest.testmod()#verbose=True)


