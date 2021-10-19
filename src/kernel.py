#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
kernel module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from pyscf import gto, symm, scf, ao2mo, lo, cc, mcscf, fci
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.mcscf import avas, PiOS
from copy import copy
from typing import Tuple, List, Dict, Union, Any
from warnings import catch_warnings, simplefilter

from parallel import mpi_bcast
from system import MolCls
from tools import assertion, suppress_stdout, mat_idx, near_nbrs, \
                  idx_tril, core_cas, nelec, ndets
from interface import mbecc_interface


MAX_MEM = 1e10
CONV_TOL = 1.e-10
SPIN_TOL = 1.e-05


def ints(mol: MolCls, mo_coeff: np.ndarray, global_master: bool, local_master: bool, \
         global_comm: MPI.Comm, local_comm: MPI.Comm, master_comm: MPI.Comm, \
         num_masters: int) -> Tuple[MPI.Win, ...]:
        """
        this function returns 1e and 2e mo integrals and effective fock potentials from individual occupied orbitals
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

        # mpi_bcast hcore
        if num_masters > 1 and local_master:
            hcore[:] = mpi_bcast(master_comm, hcore)

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

        # mpi_bcast vhf
        if num_masters > 1 and local_master:
            vhf[:] = mpi_bcast(master_comm, vhf)

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

        # mpi_bcast eri
        if num_masters > 1 and local_master:
            eri[:] = mpi_bcast(master_comm, eri)

        # mpi barrier
        global_comm.Barrier()

        return hcore_win, vhf_win, eri_win


def _ao_ints(mol: MolCls) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns 1e and 2e ao integrals
        """
        if mol.atom:

            # hcore_ao
            if mol.x2c:
                hf = scf.ROHF(mol).x2c()
            else:
                hf = scf.ROHF(mol)
            hcore = hf.get_hcore()
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


def gauge_origin(mol: MolCls) -> np.ndarray:
        """
        this function return dipole gauge origin
        """
        if mol.gauge == 'charge':
            charges = mol.atom_charges()
            coords  = mol.atom_coords()
            origin = np.einsum('z,zr->r', charges, coords) / charges.sum()
        else:
            origin = np.array([0., 0., 0.])
        return origin


def dipole_ints(mol: MolCls, mo: np.ndarray) -> np.ndarray:
        """
        this function returns dipole integrals (in AO basis)
        """
        with mol.with_common_origin(mol.gauge_origin):
            dipole = mol.intor_symmetric('int1e_r', comp=3)

        return np.einsum('pi,xpq,qj->xij', mo, dipole, mo)


def e_core_h1e(e_nuc: float, hcore: np.ndarray, vhf: np.ndarray, \
               core_idx: np.ndarray, cas_idx: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        this function returns core energy and cas space 1e integrals
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

                site_1_xy = mat_idx(site_1, nx, ny)
                nbrs = near_nbrs(site_1_xy, nx, ny)

                for site_2 in range(site_1):

                    site_2_xy = mat_idx(site_2, nx, ny)

                    if site_2_xy in nbrs:
                        h1e[site_1, site_2] = h1e[site_2, site_1] = -1.

        return h1e


def hubbard_eri(matrix: Tuple[int, int], u: float) -> np.ndarray:
        """
        this function returns the hubbard two-electron hamiltonian
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


def hf(mol: MolCls, hf_ref: Dict[str, Any]) -> Tuple[int, int, int, scf.RHF, float, np.ndarray, \
                                                     np.ndarray, np.ndarray, np.ndarray]:
        """
        this function returns the results of a restricted (open-shell) hartree-fock calculation
        """
        # initialize restricted hf calc
        mol_hf = mol.copy()
        mol_hf.build(0, 0, symmetry = hf_ref['symmetry'])
        if mol.x2c:
            hf = scf.RHF(mol_hf).x2c()
        else:
            hf = scf.RHF(mol_hf)

        # hf settings
        if mol.debug >= 2:
            hf.verbose = 4

        hf.init_guess = hf_ref['init_guess']
        if hf_ref['newton']:
            hf.conv_tol = 1.e-01
        else:
            hf.conv_tol = CONV_TOL
        hf.max_cycle = 1000

        if mol.atom:
            # ab initio hamiltonian
            hf.irrep_nelec = hf_ref['irrep_nelec']
        else:
            # model hamiltonian
            hf.get_ovlp = lambda *args: np.eye(mol.matrix[0] * mol.matrix[1])
            hf.get_hcore = lambda *args: hubbard_h1e(mol.matrix, mol.pbc)
            hf._eri = hubbard_eri(mol.matrix, mol.u)

        # hf calc
        with catch_warnings():
            simplefilter("ignore")
            hf.kernel()

        # convergence check
        assertion(hf.converged, 'HF error: no convergence')

        if hf_ref['newton']:

            # initial mo coefficients and occupation
            mo_coeff = hf.mo_coeff
            mo_occ = hf.mo_occ

            # new so-hf object
            hf = hf.newton()
            hf.conv_tol = CONV_TOL

            with catch_warnings():
                simplefilter("ignore")
                hf.kernel(mo_coeff, mo_occ)

            # convergence check
            assertion(hf.converged, 'HF error: no convergence')

        # dipole moment
        if mol.atom:
            dm = hf.make_rdm1()
            if mol.spin > 0:
                dm = dm[0] + dm[1]
            with mol.with_common_orig(mol.gauge_origin):
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)
            elec_dipole = np.einsum('xij,ji->x', ao_dip, dm)
        else:
            elec_dipole = np.zeros(3, dtype=np.float64)

        # determine dimensions
        norb, nocc, nvirt = _dim(hf.mo_occ)

        # store energy, occupation, and orbsym
        e_hf = hf.e_tot
        occup = hf.mo_occ
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        else:
            orbsym = np.zeros(hf.mo_energy.size, dtype=np.int64)

        # debug print of orbital energies
        if mol.debug >= 2:
            print('\n HF:  mo   symmetry    energy')
            for i in range(hf.mo_energy.size):
                print('     {:>3d}   {:>5s}     {:>7.5f}'.format(i, symm.addons.irrep_id2name(mol.groupname, orbsym[i]), hf.mo_energy[i]))
            print('\n')

        return nocc, nvirt, norb, hf, e_hf.item(), elec_dipole, occup, \
                orbsym, np.asarray(hf.mo_coeff, order='C')


def _dim(mo_occ: np.ndarray) -> Tuple[int, ...]:
        """
        this function determines the involved dimensions (number of occupied, virtual, and total orbitals)
        """
        # occupied and virtual lists
        occ = np.where(mo_occ > 0.)[0]
        virt = np.where(mo_occ == 0.)[0]
        return occ.size + virt.size, occ.size, virt.size


def ref_mo(mol: MolCls, mo_coeff: np.ndarray, occup: np.ndarray, orbsym: np.ndarray, \
           orbs: Dict[str, str], ref: Dict[str, Any], model: Dict[str, str], \
           hf: scf.RHF) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
        """
        this function returns a set of reference mo coefficients and symmetries plus the associated spaces
        """
        # copy mo coefficients
        mo_coeff_out = np.copy(mo_coeff)

        if orbs['type'] != 'can':

            # set core and cas spaces
            core_idx, cas_idx = core_cas(mol.nocc, np.arange(mol.ncore, mol.nocc), \
                                         np.arange(mol.nocc, mol.norb))

            # NOs
            if orbs['type'] in ['ccsd', 'ccsd(t)']:

                # compute rmd1
                res = _cc(mol.spin, occup, core_idx, cas_idx, orbs['type'], hf=hf, higher_amp_extrap=False, rdm1=True)
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
                if mol.debug >= 2:
                    loc.verbose = 4
                mo_coeff_out[:, mol.ncore:mol.nocc] = loc.kernel()

                # virt-virt block
                if mol.atom:
                    loc = lo.PM(mol, mo_coeff[:, mol.nocc:])
                else:
                    loc = _hubbard_PM(mol, mo_coeff[:, mol.nocc:])
                loc.conv_tol = CONV_TOL
                if mol.debug >= 2:
                    loc.verbose = 4
                mo_coeff_out[:, mol.nocc:] = loc.kernel()

        # active space
        if ref['active'] == 'manual':

            ref['select'] = np.asarray(ref['select'], dtype=np.int64)
            act_n_elec = nelec(occup, ref['select'])

        elif ref['active'] == 'avas':

            assertion(0 < len(ref['ao-labels']), 'empty ao-labels input')
            norbs, avas_n_elec, mo_coeff_casscf = avas.avas(hf, ref['ao-labels'], ncore=mol.ncore)
            ref['select'] = np.arange(mol.nocc - avas_n_elec // 2, mol.nocc - avas_n_elec // 2 + norbs, dtype=np.int64)
            act_n_elec = (int(avas_n_elec) // 2, int(avas_n_elec) // 2)

        elif ref['active'] == 'pios':

            assertion(0 < len(ref['pi-atoms']), 'empty pi-atoms input')
            with suppress_stdout():
                _, norbs, _, pios_n_elec, mo_coeff_casscf = PiOS.MakePiOS(mol, hf, ref['pi-atoms'])
            ref['select'] = np.arange(mol.nocc - pios_n_elec // 2, mol.nocc - pios_n_elec // 2 + norbs, dtype=np.int64)
            act_n_elec = (int(pios_n_elec) // 2, int(pios_n_elec) // 2)

        # reference (primary) space
        ref_space = ref['select']

        # expansion space
        exp_space = np.array([i for i in range(mol.ncore, mol.norb) if i not in ref_space], dtype=np.int64)

        # casscf
        if ref['method'] == 'casscf':

            assertion(ref_space[occup[ref_space] > 0.].size > 0, \
                      'no singly/doubly occupied orbitals in CASSCF calculation')
            assertion(ref_space[occup[ref_space] == 0.].size > 0, \
                      'no virtual/singly occupied orbitals in CASSCF calculation')

            # sorter for active space
            if ref['active'] == 'manual':
                n_core_inact = np.array([i for i in range(mol.nocc) if i not in ref_space], dtype=np.int64)
                n_virt_inact = np.array([a for a in range(mol.nocc, mol.norb) if a not in ref_space], dtype=np.int64)
                sort_casscf = np.concatenate((n_core_inact, ref_space, n_virt_inact))
                mo_coeff_casscf = mo_coeff_out[:, sort_casscf]

            # update orbsym
            if mol.symmetry:
                orbsym_casscf = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_casscf)

            # run casscf
            mo_coeff_out = _casscf(mol, model['solver'], ref['wfnsym'], ref['weights'], orbsym_casscf, \
                                ref['hf_guess'], hf, mo_coeff_casscf, ref_space, act_n_elec)

            # reorder mo_coeff
            if ref['active'] == 'manual':
                mo_coeff_out = mo_coeff_out[:, np.argsort(sort_casscf)]

        # debug print of reference and expansion spaces
        if mol.debug >= 1:
            print('\n reference nelec        = {:}'.format(act_n_elec))
            print(' reference space [occ]  = {:}'.format(ref_space[occup[ref_space] > 0.]))
            print(' reference space [virt] = {:}'.format(ref_space[occup[ref_space] == 0.]))
            print(' expansion space [occ]  = {:}'.format(exp_space[occup[exp_space] > 0.]))
            print(' expansion space [virt] = {:}\n'.format(exp_space[occup[exp_space] == 0.]))

        return np.asarray(mo_coeff_out, order='C'), act_n_elec, ref_space


def ref_prop(mol: MolCls, occup: np.ndarray, target_mbe: str, \
             orbsym: np.ndarray, hf_guess: bool, ref_space: np.ndarray, \
             model: Dict[str, str], orb_type: str, state: Dict[str, Any], e_hf: float, \
             dipole_hf: np.ndarray, base_method: Union[str, None]) -> Union[float, np.ndarray]:
        """
        this function returns reference space properties
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
        core_idx, cas_idx = core_cas(mol.nocc, ref_space, np.array([], dtype=np.int64))

        # n_elec
        n_elec = nelec(occup, cas_idx)

        if ((base_method is None and ref_space[occup[ref_space] > 0.].size > 0 and ref_space[occup[ref_space] == 0.].size > 0) or \
            (base_method in ['ccsd', 'ccsd(t)', 'ccsdt'] and ref_space[occup[ref_space] > 0.].size > 1 and ref_space[occup[ref_space] == 0.].size > 1) or \
            (base_method == 'ccsdtq' and ref_space[occup[ref_space] > 0.].size > 2 and ref_space[occup[ref_space] == 0.].size > 2) or \
            (mol.spin > 0 and ref_space.size > 0)):

            # get cas_space h2e
            cas_idx_tril = idx_tril(cas_idx)
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

            # exp model
            ref = main(model['method'], model['cc_backend'], model['solver'], orb_type, mol.spin, occup, target_mbe, \
                        state['wfnsym'], mol.groupname, orbsym, hf_guess, state['root'], \
                        e_hf, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, n_elec, mol.debug, \
                        mol.dipole_ints if target_mbe in ['dipole', 'trans'] else None, \
                        dipole_hf if target_mbe in ['dipole', 'trans'] else None, higher_amp_extrap=False)[0]

            # base model
            if base_method is not None:
                ref -= main(base_method, model['cc_backend'], '', orb_type, mol.spin, occup, target_mbe, \
                            state['wfnsym'], mol.groupname, orbsym, hf_guess, state['root'], \
                            e_hf, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, n_elec, mol.debug, \
                            mol.dipole_ints if target_mbe in ['dipole', 'trans'] else None, \
                            dipole_hf if target_mbe in ['dipole', 'trans'] else None, higher_amp_extrap=False)[0]

        else:

            # no correlation in expansion reference space
            if target_mbe in ['energy', 'excitation']:
                ref = 0.
            else:
                ref = np.zeros(3, dtype=np.float64)

        return ref


def main(method: str, cc_backend: str, solver: str, orb_type: str, spin: int, \
         occup: np.ndarray, target_mbe: str, state_wfnsym: str, point_group: str, \
         orbsym: np.ndarray, hf_guess: bool, state_root: int, e_hf: float, \
         e_core: float, h1e: np.ndarray, h2e: np.ndarray, core_idx: np.ndarray, \
         cas_idx: np.ndarray, n_elec: Tuple[int, int], debug: int, \
         dipole_ints: Union[np.ndarray, None], dipole_hf: Union[np.ndarray, None], \
         higher_amp_extrap: bool = False) -> Tuple[Union[float, np.ndarray], int]:
        """
        this function return the result property from a given method
        """
        if method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq']:

            res_tmp = _cc(spin, occup, core_idx, cas_idx, method, cc_backend=cc_backend, n_elec=n_elec, \
            orb_type=orb_type, point_group=point_group, orbsym=orbsym, h1e=h1e, h2e=h2e, \
            higher_amp_extrap=higher_amp_extrap, rdm1=target_mbe == 'dipole', debug=debug)
            n_dets = ndets(occup, cas_idx, n_elec=n_elec)

        elif method == 'fci':

            res_tmp = _fci(solver, spin, target_mbe, state_wfnsym, \
                            orbsym, hf_guess, state_root, \
                            e_hf, e_core, h1e, h2e, \
                            occup, core_idx, cas_idx, n_elec, debug)
            n_dets = res_tmp['n_dets']

        if target_mbe in ['energy', 'excitation']:

            res = res_tmp[target_mbe]

        elif target_mbe == 'dipole':

            res = _dipole(dipole_ints, occup, dipole_hf, cas_idx, res_tmp['rdm1'])

        elif target_mbe == 'trans':

            res = _trans(dipole_ints, occup, dipole_hf, cas_idx, \
                            res_tmp['t_rdm1'], res_tmp['hf_weight'][0], res_tmp['hf_weight'][1])

        return res, n_dets


def _dipole(dipole_ints: np.ndarray, occup: np.ndarray, hf_dipole: np.ndarray, \
            cas_idx: np.ndarray, cas_rdm1: np.ndarray, trans: bool = False) -> np.ndarray:
        """
        this function returns an electronic (transition) dipole moment
        """
        # init (transition) rdm1
        if trans:
            rdm1 = np.zeros([occup.size, occup.size], dtype=np.float64)
        else:
            rdm1 = np.diag(occup)

        # insert correlated subblock
        rdm1[cas_idx[:, None], cas_idx] = cas_rdm1

        # compute elec_dipole
        elec_dipole = np.einsum('xij,ji->x', dipole_ints, rdm1)

        # 'correlation' dipole
        if not trans:
            elec_dipole -= hf_dipole

        return elec_dipole


def _trans(dipole_ints: np.ndarray, occup: np.ndarray, hf_dipole: np.ndarray, \
           cas_idx: np.ndarray, cas_rdm1: np.ndarray, hf_weight_gs: float, \
           hf_weight_ex: float) -> np.ndarray:
        """
        this function returns an electronic transition dipole moment
        """
        return _dipole(dipole_ints, occup, hf_dipole, cas_idx, cas_rdm1, True) \
                        * np.sign(hf_weight_gs) * np.sign(hf_weight_ex)


def base(mol: MolCls, orb_type: str, occup: np.ndarray, mo_coeff: np.ndarray, target_mbe: str, \
         method: str, cc_backend: str, dipole_hf: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        this function returns base model energy
        """
        # load integrals for canonical orbitals
        if orb_type == 'can':

            # load hcore
            buf = mol.hcore.Shared_query(0)[0]
            hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

            # load vhf
            buf = mol.vhf.Shared_query(0)[0]
            vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

            # load eri
            buf = mol.eri.Shared_query(0)[0]
            eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb*(mol.norb + 1) // 2,) * 2)

        # calculate integrals of canonical orbitals for other orbital types
        else:

            # hcore_ao and eri_ao w/o symmetry
            hcore, eri = _ao_ints(mol)

            # compute hcore
            hcore = np.einsum('pi,pq,qj->ij', mo_coeff, hcore, mo_coeff)

            # eri_mo w/o symmetry
            eri = ao2mo.incore.full(eri, mo_coeff)

            # allocate vhf
            vhf = np.empty((mol.nocc, mol.norb, mol.norb), dtype=np.float64)

            # compute vhf
            for i in range(mol.nocc):
                idx = np.asarray([i])
                vhf[i] = np.einsum('pqrs->rs', eri[idx[:, None], idx, :, :]) * 2.
                vhf[i] -= np.einsum('pqrs->ps', eri[:, idx[:, None], idx, :]) * 2. * .5

            # restore 4-fold symmetry in eri_mo
            eri = ao2mo.restore(4, eri, mol.norb)

        # set core and cas spaces
        core_idx, cas_idx = core_cas(mol.nocc, np.arange(mol.ncore, mol.nocc), np.arange(mol.nocc, mol.norb))

        # get cas space h2e
        cas_idx_tril = idx_tril(cas_idx)
        h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

        # get e_core and h1e_cas
        e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

        # n_elec
        n_elec = nelec(occup, cas_idx)

        # use no symmetry for pyscf backend
        if cc_backend == 'pyscf':
            
            # point group
            point_group = mol.groupname

            # orbital symmetries
            orbsym = np.zeros(mol.norb, dtype=np.int64)

        # create new mol object with point group symmetry for other backends
        else:

            # mol object
            mol_fullsym = MolCls()
            mol_fullsym.defaults()

            # copy atom attribute
            mol_fullsym.atom = mol.atom

            # copy system attributes
            for key, val in mol.system.items():
                setattr(mol_fullsym, key, val)

            # use symmetry
            mol_fullsym.symmetry = True

            # make pyscf mol object
            mol_fullsym.make()

            # point group
            point_group = mol_fullsym.groupname

            # orbital symmetries
            orbsym = symm.label_orb_symm(mol_fullsym, mol_fullsym.irrep_id, mol_fullsym.symm_orb, mo_coeff)

        # run calc
        res_tmp = _cc(mol.spin, occup, core_idx, cas_idx, method, cc_backend=cc_backend, n_elec=n_elec, \
                      orb_type='can', point_group=point_group, orbsym=orbsym, h1e=h1e_cas, h2e=h2e_cas, \
                      higher_amp_extrap=False, rdm1=target_mbe == 'dipole')

        # collect results
        energy = res_tmp['energy']
        if target_mbe == 'energy':
            dipole = np.zeros(3, dtype=np.float64)
        else:
            dipole = _dipole(mol.dipole_ints, occup, dipole_hf, cas_idx, res_tmp['rdm1'])

        return energy, dipole


def _casscf(mol: MolCls, solver: str, wfnsym: List[str], \
            weights: List[float], orbsym: np.ndarray, hf_guess: bool, hf: scf.RHF, \
            mo_coeff: np.ndarray, ref_space: np.ndarray, n_elec: Tuple[int, int]) -> np.ndarray:
        """
        this function returns the results of a casscf calculation
        """
        # init casscf
        cas = mcscf.CASSCF(hf, ref_space.size, n_elec)

        # casscf settings
        cas.conv_tol = CONV_TOL
        cas.max_cycle_macro = 500
        cas.frozen = mol.ncore
        if mol.debug >= 2:
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
                    fcisolver_ = copy(fcisolver)

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
            na = fci.cistring.num_strings(ref_space.size, n_elec[0])
            nb = fci.cistring.num_strings(ref_space.size, n_elec[1])
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

            s, mult = fcisolver.spin_square(c[root], ref_space.size, n_elec)

            if np.abs((mol.spin + 1) - mult) > SPIN_TOL:

                # fix spin by applyting level shift
                sz = np.abs(n_elec[0] - n_elec[1]) * 0.5
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
                    s, mult = fcisolver.spin_square(c[root], ref_space.size, n_elec)
                    assertion(np.abs((mol.spin + 1) - mult) < SPIN_TOL, \
                              'spin contamination for root entry = {:} , 2*S + 1 = {:.6f}'.format(root, mult))

        # convergence check
        assertion(cas.converged, 'CASSCF error: no convergence')

        # debug print of orbital energies
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, cas.mo_coeff)
        else:
            orbsym = np.zeros(cas.mo_energy.size, dtype=np.int64)

        if mol.debug >= 2:
            print('\n CASSCF:  mo   symmetry    energy')
            for i in range(cas.mo_energy.size):
                print('         {:>3d}   {:>5s}     {:>7.3f}'.format(i, symm.addons.irrep_id2name(mol.groupname, orbsym[i]), cas.mo_energy[i]))
            print('\n')

        return np.asarray(cas.mo_coeff, order='C')


def _fci(solver_type: str, spin: int, target_mbe: str, wfnsym: str, orbsym: np.ndarray, \
         hf_guess: bool, root: int, e_hf: float, e_core: float, h1e: np.ndarray, \
         h2e: np.ndarray, occup: np.ndarray, core_idx: np.ndarray, \
         cas_idx: np.ndarray, n_elec: Tuple[int, int], debug: int) -> Dict[str, Any]:
        """
        this function returns the results of a fci calculation
        """
        # spin
        spin_cas = np.count_nonzero(occup[cas_idx] == 1.)
        assertion(spin_cas == spin, 'casci wrong spin in space: {:}'.format(cas_idx))

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
            na, nb = fci.cistring.num_strings(cas_idx.size, n_elec[0]), fci.cistring.num_strings(cas_idx.size, n_elec[1])
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
                e, c = solver.kernel(h1e, h2e, cas_idx.size, n_elec, ecore=e_core, ci0=ci0)

                # collect results
                if solver.nroots == 1:
                    return [e], [c]
                else:
                    return [e[0], e[-1]], [c[0], c[-1]]

        # perform calc
        energy, civec = _fci_kernel()

        # multiplicity check
        for root in range(len(civec)):

            s, mult = solver.spin_square(civec[root], cas_idx.size, n_elec)

            if np.abs((spin_cas + 1) - mult) > SPIN_TOL:

                # fix spin by applyting level shift
                sz = np.abs(n_elec[0] - n_elec[1]) * 0.5
                solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.))

                # perform calc
                energy, civec = _fci_kernel()

                # verify correct spin
                for root in range(len(civec)):
                    s, mult = solver.spin_square(civec[root], cas_idx.size, n_elec)
                    assertion(np.abs((spin_cas + 1) - mult) < SPIN_TOL, \
                              'spin contamination for root entry = {:}\n2*S + 1 = {:.6f}\n'
                              'cas_idx = {:}\ncas_sym = {:}'.format(root, mult, cas_idx, orbsym[cas_idx]))

        # convergence check
        if solver.nroots == 1:

            assertion(solver.converged, \
                     'state {:} not converged\ncas_idx = {:}\ncas_sym = {:}'.format(root, cas_idx, orbsym[cas_idx]))

        else:

            if target_mbe == 'excitation':

                for root in [0, solver.nroots-1]:
                    assertion(solver.converged[root], \
                              'state {:} not converged\ncas_idx = {:}\ncas_sym = {:}'.format(root, cas_idx, orbsym[cas_idx]))

            else:

                assertion(solver.converged[-1], \
                          'state {:} not converged\ncas_idx = {:}\ncas_sym = {:}'.format(root, cas_idx, orbsym[cas_idx]))

        # collect results
        res: Dict[str, Union[int, float, np.ndarray]] = {'n_dets': np.count_nonzero(civec[-1])}
        res['energy'] = energy[-1] - e_hf
        if target_mbe == 'excitation':
            res['excitation'] = energy[-1] - energy[0]
        elif target_mbe == 'dipole':
            res['rdm1'] = solver.make_rdm1(civec[-1], cas_idx.size, n_elec)
        elif target_mbe == 'trans':
            res['t_rdm1'] = solver.trans_rdm1(civec[0], civec[-1], cas_idx.size, n_elec)
            res['hf_weight'] = [civec[i][0, 0] for i in range(2)]

        return res


def _cc(spin: int, occup: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray, method: str, \
        cc_backend: str = 'pyscf', n_elec: Tuple[int, int] = None, orb_type: str = None, \
        point_group: str = None, orbsym: np.ndarray = None, h1e: np.ndarray = None, \
        h2e: np.ndarray = None, hf: scf.RHF = None, higher_amp_extrap: bool = True, \
        rdm1: bool = False, debug: int = 0) -> Dict[str, Any]:
        """
        this function returns the results of a ccsd / ccsd(t) calculation
        """
        spin_cas = np.count_nonzero(occup[cas_idx] == 1.)
        assertion(spin_cas == spin, 'cascc wrong spin in space: {:}'.format(cas_idx))
        singlet = spin_cas == 0

        if cc_backend == 'pyscf':

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

            elif hf is not None:

                ccsd = cc.CCSD(hf)
                frozen_orbs = np.asarray([i for i in range(hf.mo_coeff.shape[1]) if i not in cas_idx])
                if frozen_orbs.size > 0:
                    ccsd.frozen = frozen_orbs

            # settings
            ccsd.conv_tol = CONV_TOL
            if rdm1:
                ccsd.conv_tol_normt = ccsd.conv_tol
            ccsd.max_cycle = 500
            ccsd.async_io = False
            ccsd.diis_start_cycle = 4
            ccsd.diis_space = 12
            ccsd.incore_complete = True
            eris = ccsd.ao2mo()

            # calculate ccsd energy
            ccsd.kernel(eris=eris)

            # convergence check
            assertion(ccsd.converged, 'CCSD error: no convergence, core_idx = {:} , cas_idx = {:}'.format(core_idx, cas_idx))

            # e_corr
            e_cc = ccsd.e_corr

            # calculate (t) correction
            if method == 'ccsd(t)':

                if np.amin(occup[cas_idx]) == 1.:
                    if np.where(occup[cas_idx] == 1.)[0].size >= 3:
                        e_cc += ccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)

                else:

                    e_cc += ccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)

        elif (cc_backend in ['ecc', 'ncc']):

            # assume necessary variables are passed if MBECC is to be used
            assert isinstance(orb_type, str)
            assert isinstance(point_group, str)
            assert isinstance(orbsym, np.ndarray)
            assert isinstance(n_elec, tuple)

            # calculate cc energy
            cc_energy, success = mbecc_interface(method, cc_backend, orb_type, point_group, orbsym[cas_idx], h1e, h2e, \
                                                 n_elec, higher_amp_extrap, debug)

            # convergence check
            assertion(success == 1, \
            'MBECC error: no convergence, core_idx = {:} , cas_idx = {:}'.format(core_idx, cas_idx))

            # e_corr
            e_cc = cc_energy

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
