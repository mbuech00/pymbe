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

import tools


CONV_TOL = 1.0e-10
SPIN_TOL = 1.0e-05
DIPOLE_TOL = 1.0e-14


def ints(mpi, mol, mo_coeff):
        """
        this function returns 1e and 2e mo integrals and effective fock potentials from individual occupied orbitals

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param mo_coeff: mo coefficients. numpy array of shape (n_orb, n_orb)
        :return: MPI window handle to numpy array of shape (n_orb, n_orb) [hcore],
                 MPI window handle to numpy array of shape (nocc, norb, norb) [vhf],
                 MPI window handle to numpy array of shape (n_orb*(n_orb + 1) // 2, n_orb*(n_orb + 1) // 2) [eri]
        """
        # hcore_ao and eri_ao w/o symmetry
        if mpi.global_master:
            hcore_tmp, eri_tmp = _ao_ints(mol)

        # allocate hcore in shared mem
        if mpi.local_master:
            hcore_win = MPI.Win.Allocate_shared(8 * mol.norb**2, 8, comm=mpi.local_comm)
        else:
            hcore_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
        buf = hcore_win.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # compute hcore
        if mpi.global_master:
            hcore[:] = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_tmp, mo_coeff)

        # bcast hcore
        if mpi.num_masters > 1:
            hcore[:] = parallel.bcast(mpi.master_comm, hcore)

        # eri_mo w/o symmetry
        if mpi.global_master:
            eri_tmp = ao2mo.incore.full(eri_tmp, mo_coeff)

        # allocate vhf in shared mem
        if mpi.local_master:
            vhf_win = MPI.Win.Allocate_shared(8 * mol.nocc*mol.norb**2, 8, comm=mpi.local_comm)
        else:
            vhf_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
        buf = vhf_win.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # compute vhf
        if mpi.global_master:
            for i in range(mol.nocc):
                idx = np.asarray([i])
                vhf[i] = np.einsum('pqrs->rs', eri_tmp[idx[:, None], idx, :, :]) * 2.
                vhf[i] -= np.einsum('pqrs->ps', eri_tmp[:, idx[:, None], idx, :]) * 2. * .5

        # bcast vhf
        if mpi.num_masters > 1:
            vhf[:] = parallel.bcast(mpi.master_comm, vhf)

        # allocate eri in shared mem
        if mpi.local_master:
            eri_win = MPI.Win.Allocate_shared(8 * (mol.norb * (mol.norb + 1) // 2) ** 2, 8, comm=mpi.local_comm)
        else:
            eri_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
        buf = eri_win.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

        # restore 4-fold symmetry in eri_mo
        if mpi.global_master:
            eri[:] = ao2mo.restore(4, eri_tmp, mol.norb)

        # bcast eri
        if mpi.num_masters > 1:
            eri[:] = parallel.bcast(mpi.master_comm, eri)

        # mpi barrier
        mpi.global_comm.Barrier()

        return hcore_win, vhf_win, eri_win


def _ao_ints(mol):
        """
        this function returns 1e and 2e ao integrals

        :param mol: pymbe mol object
        :return: numpy array of shape (n_orb, n_orb) [hcore_tmp],
                 numpy array of shape (n_orb*(n_orb + 1) // 2, n_orb*(n_orb + 1) // 2) [eri_tmp]
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
            hcore = _hubbard_h1e(mol.matrix, mol.nsites, mol.pbc)
            # eri_ao
            eri = _hubbard_eri(mol)

        return hcore, eri


def dipole_ints(mol):
        """
        this function returns dipole integrals (in AO basis)

        :param mol: pymbe mol object
        :return: numpy array of shape (3, n_orb, n_orb)
        """
        # gauge origin at (0.0, 0.0, 0.0)
        with mol.with_common_origin([0.0, 0.0, 0.0]):
            dipole = mol.intor_symmetric('int1e_r', comp=3)

        return dipole


def e_core_h1e(e_nuc, hcore, vhf, core_idx, cas_idx):
        """
        this function returns core energy and cas space 1e integrals

        :param e_nuc: nuclear energy. float
        :param hcore: 1e integrals in mo basis. numpy array of shape (n_orb, n_orb)
        :param vhf: effective fock potentials. numpy array of shape (n_occ, n_orb, n_orb)
        :param core_idx: core space indices. numpy array of shape (n_inactive,)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :return: float [e_core],
                 numpy array of shape (n_cas, n_cas) [h1e_cas]
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


def _hubbard_h1e(matrix, nsites, pbc):
        """
        this function returns the hubbard hopping hamiltonian

        :param matrix: hubbard dimensions. tuple of integers
        :param nsites: number of lattice sites. integer
        :param pbc: periodic boundary condition logical. bool
        :return: numpy array of shape(matrix)
        """
        # dimension
        if 1 in matrix:
            ndim = 1
        else:
            ndim = 2

        # init h1e
        h1e = np.zeros([nsites] * 2, dtype=np.float64)

        if ndim == 1:

            # adjacent neighbours
            for i in range(nsites-1):
                h1e[i, i+1] = h1e[i+1, i] = -1.0

            if pbc:
                h1e[-1, 0] = h1e[0, -1] = -1.0

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
                        h1e[site_1, site_2] = h1e[site_2, site_1] = -1.0

            if pbc:

                # sideways
                for i in range(ny):
                    h1e[i, ny * (nx - 1) + i] = h1e[ny * (nx - 1) + i, i] = -1.0

                # up-down
                for i in range(nx):
                    h1e[i * ny, i * ny + (ny - 1)] = h1e[i * ny + (ny - 1), i * ny] = -1.0

        return h1e


def _hubbard_eri(mol):
        """
        this function returns the hubbard two-electron hamiltonian

        :param nsites: number of lattice sites. integer
        :param u: hubbard u/t. float
        :return: numpy array of shape (n_sites*(n_sites + 1) // 2, n_sites*(n_sites + 1) // 2)
        """
        # init eri
        eri = np.zeros([mol.nsites] * 4, dtype=np.float64)

        # compute eri
        for i in range(mol.nsites):
            eri[i,i,i,i] = mol.u

        return ao2mo.restore(4, eri, mol.nsites)


class _hubbard_PM(lo.pipek.PM):
        """
        this class constructs the site-population tensor for each orbital-pair density
        see: pyscf example - 40-hubbard_model_PM_localization.py
        """
        def atomic_pops(self, mol, mo_coeff, method=None):
            """
            this function returns the tensor used in the pm cost function and its gradients

            :param mol: pyscf mol object
            :param mo_coeff: mo coefficients. numpy array of shape (n_sites, n_sites)
            :param method: localization method. string
            :return: numpy array of shape (n_sites, n_sites, n_sites)
            """
            return np.einsum('pi,pj->pij', mo_coeff, mo_coeff)


def hf(mol, target):
        """
        this function returns the results of a hartree-fock calculation

        :param mol: pymbe mol object
        :param target: calculation target. string
        :return: integer [nocc],
                 integer [nvirt],
                 integer [norb],
                 pyscf hf object [hf],
                 float [e_hf],
                 numpy array of shape (3,) or None [elec_dipole],
                 numpy array of shape (n_orb,) [occup],
                 numpy array of shape (n_orb,) [orbsym],
                 numpy array of shape (n_orb,) [mo_energy],
                 numpy array of shape (n_orb, n_orb) [mo_coeff]
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
            hf.get_ovlp = lambda *args: np.eye(mol.nsites)
            hf.get_hcore = _hubbard_h1e(mol.matrix, mol.nsites, mol.pbc)
            hf._eri = _hubbard_eri(mol)

        # perform hf calc
        for i in range(0, 12, 2):

            hf.diis_start_cycle = i

            try:
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
            elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > DIPOLE_TOL else 0.0 for i in range(elec_dipole.size)])
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


def _dim(mo_occ):
        """
        this function determines the involved dimensions (number of occupied, virtual, and total orbitals)

        :param mo_occ: hf occupation. numpy array of shape (n_orb,)
        :return: integer [norb],
                 integer [nocc],
                 integer [nvirt]
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
                rdm1 = _cc(mol, calc.occup, core_idx, cas_idx, calc.orbs['type'], hf=calc.hf, rdm1=True)
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
                loc.conv_tol = 1.0e-10
                if mol.debug >= 1:
                    loc.verbose = 4
                calc.mo_coeff[:, mol.ncore:mol.nocc] = loc.kernel()

                # virt-virt block
                if mol.atom:
                    loc = lo.PM(mol, calc.mo_coeff[:, mol.nocc:])
                else:
                    loc = _hubbard_PM(mol, calc.mo_coeff[:, mol.nocc:])
                loc.conv_tol = 1.0e-10
                if mol.debug >= 1:
                    loc.verbose = 4
                calc.mo_coeff[:, mol.nocc:] = loc.kernel()

        # sort orbitals
        if calc.ref['active'] == 'manual':

            # active orbs
            calc.ref['select'] = np.asarray(calc.ref['select'], dtype=np.int32)

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
        ref_space = np.arange(inact_orbs, inact_orbs+act_orbs, dtype=np.int32)
        exp_space = np.append(np.arange(mol.ncore, inact_orbs, dtype=np.int32), \
                              np.arange(inact_orbs+act_orbs, mol.norb, dtype=np.int32))

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
            exp_space['pi_orbs'], exp_space['pi_hashes'] = tools.pi_space(mo_energy, exp_space)

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
        :return: float or numpy array of shape (3,) depending on calc.target
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
        core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, np.array([], dtype=np.int32))

        # nelec
        nelec = np.asarray((np.count_nonzero(calc.occup[cas_idx] > 0.), \
                            np.count_nonzero(calc.occup[cas_idx] > 1.)), dtype=np.int32)

        if np.any(calc.occup[calc.ref_space] == 2.) and np.any(calc.occup[calc.ref_space] < 2.):

            # get cas space h2e
            cas_idx_tril = tools.cas_idx_tril(cas_idx)
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

            # exp model
            ref = main(mol, calc, calc.model['method'], e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec)

            # base model
            if calc.base['method'] is not None:
                ref -= main(mol, calc, calc.base['method'], e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec)

        else:

            # no correlation in expansion reference space
            if calc.target in ['energy', 'excitation']:
                ref = 0.0
            else:
                ref = np.zeros(3, dtype=np.float64)

        return ref


def main(mol, calc, method, e_core, h1e, h2e, core_idx, cas_idx, nelec):
        """
        this function return the result property from a given method

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param method: correlated method. string
        :param e_core: core space energy. float
        :param h1e: cas space 1e integrals. numpy array of shape (n_cas, n_cas)
        :param h2e: cas space 2e integrals. numpy array of shape (n_cas*(n_cas + 1) // 2, n_cas*(n_cas + 1) // 2)
        :param core_idx: core space indices. numpy array of shape (n_inactive,)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :param nelec: number of correlated electrons. tuple of integers
        :return: float or numpy array of shape (3,) depending on target
        """
        if method in ['ccsd','ccsd(t)']:

            res = _cc(mol, calc.occup, core_idx, cas_idx, method, h1e=h1e, h2e=h2e)

        elif method == 'fci':

            res_tmp = _fci(mol, calc.model['solver'], calc.target, calc.state['wfnsym'], \
                            calc.orbsym, calc.extra['hf_guess'], calc.state['root'], \
                            calc.prop['hf']['energy'], e_core, h1e, h2e, core_idx, cas_idx, nelec)

            if calc.target in ['energy', 'excitation']:

                res = res_tmp[calc.target]

            elif calc.target == 'dipole':

                res = _dipole(mol.norb, mol.dipole, calc.occup, calc.prop['hf']['dipole'], \
                                calc.mo_coeff, cas_idx, res_tmp['rdm1'])

            elif calc.target == 'trans':

                res = _trans(mol.norb, mol.dipole, calc.occup, calc.prop['hf']['dipole'], \
                                calc.mo_coeff, cas_idx, res_tmp['t_rdm1'], \
                                res_tmp['hf_weight'][0], res_tmp['hf_weight'][1])

        return res


def _dipole(norb, ao_dipole, occup, hf_dipole, mo_coeff, cas_idx, cas_rdm1, trans=False):
        """
        this function returns an electronic (transition) dipole moment

        :param norb: number of orbitals. integer
        :param ao_dipole: dipole integrals in ao basis. numpy array of shape (3, n_orb, n_orb)
        :param occup: orbital occupation. numpy array of shape (n_orb,)
        :param hf_dipole: hf dipole moment. numpy array of shape (3,)
        :param mo_coeff: mo coefficient. numpy array of shape (n_orb, n_orb)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :param cas_rdm1: cas space rdm1. numpy array of shape (n_cas, n_cas)
        :param trans: transition dipole moment logical. bool
        :return: numpy array of shape (3,)
        """
        # init (transition) rdm1
        if trans:
            rdm1 = np.zeros([norb, norb], dtype=np.float64)
        else:
            rdm1 = np.diag(occup)

        # insert correlated subblock
        rdm1[cas_idx[:, None], cas_idx] = cas_rdm1

        # ao representation
        rdm1 = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1, mo_coeff)

        # compute elec_dipole
        elec_dipole = np.einsum('xij,ji->x', ao_dipole, rdm1)

        # remove noise
        elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > DIPOLE_TOL else 0.0 for i in range(elec_dipole.size)])

        # 'correlation' dipole
        if not trans:
            elec_dipole -= hf_dipole

        return elec_dipole


def _trans(norb, ao_dipole, occup, hf_dipole, mo_coeff, cas_idx, cas_rdm1, hf_weight_gs, hf_weight_ex):
        """
        this function returns an electronic transition dipole moment

        :param norb: number of orbitals. integer
        :param ao_dipole: dipole integrals in ao basis. numpy array of shape (n_orb, n_orb)
        :param occup: orbital occupation. numpy array of shape (n_orb,)
        :param hf_dipole: hf dipole moment. numpy array of shape (3,)
        :param mo_coeff: mo coefficient. numpy array of shape (n_orb, n_orb)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :param cas_rdm1: cas space rdm1. numpy array of shape (n_cas, n_cas)
        :param hf_weight_gs: weight of ground state in ci vector. float
        :param hf_weight_ex: weight of excited state in ci vector. float
        :return: numpy array of shape (3,)
        """
        return _dipole(norb, ao_dipole, occup, hf_dipole, mo_coeff, cas_idx, cas_rdm1, True) \
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

        return _cc(mol, occup, core_idx, cas_idx, method, h1e=h1e_cas, h2e=h2e_cas)


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


def _fci(mol, solver, target, wfnsym, orbsym, hf_guess, root, hf_energy, \
            e_core, h1e, h2e, core_idx, cas_idx, nelec):
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
        :return: dict of floats [energy and excitation],
                 numpy array of shape (n_cas, n_cas) [dipole],
                 or numpy array of shape (n_cas, n_cas) and a list of floats [trans]
        """
        # init fci solver
        if solver == 'pyscf_spin0':
            solver = fci.direct_spin0_symm.FCI(mol)
        elif solver == 'pyscf_spin1':
            solver = fci.direct_spin1_symm.FCI(mol)

        # settings
        solver.conv_tol = CONV_TOL
        if target in ['dipole', 'trans']:
            solver.conv_tol *= 1.0e-04
            solver.lindep = solver.conv_tol * 1.0e-01
        solver.max_cycle = 5000
        solver.max_space = 25
        solver.davidson_only = True
        solver.pspace_size = 0
        if mol.debug >= 3:
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

            if np.abs((mol.spin + 1) - mult) > SPIN_TOL:

                # fix spin by applyting level shift
                sz = np.abs(nelec[0]-nelec[1]) * 0.5
                solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.))

                # perform calc
                energy, civec = _fci_kernel()

                # verify correct spin
                for root in range(len(civec)):
                    s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
                    tools.assertion(np.abs((mol.spin + 1) - mult) < SPIN_TOL, \
                                    'spin contamination for root entry = {:}\n2*S + 1 = {:.6f}\n'
                                    'core_idx = {:}\ncore_sym = {:}\ncas_idx = {:}\ncas_sym = {:}'. \
                                    format(root, mult, core_idx, orbsym[core_idx], cas_idx, orbsym[cas_idx]))

        # convergence check
        if solver.nroots == 1:

            tools.assertion(solver.converged, \
                                 'state {:} not converged\ncore_idx = {:}\ncore_sym = {:}\n'
                                 'cas_idx = {:}\ncas_sym = {:}'. \
                                 format(root, core_idx, orbsym[core_idx], cas_idx, orbsym[cas_idx]))

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
        if target == 'energy':
            res = {'energy': energy[-1] - hf_energy}
        elif target == 'excitation':
            res = {'excitation': energy[-1] - energy[0]}
        elif target == 'dipole':
            res = {'rdm1': solver.make_rdm1(civec[-1], cas_idx.size, nelec)}
        elif target == 'trans':
            res = {'t_rdm1': solver.trans_rdm1(civec[0], civec[-1], cas_idx.size, nelec)}
            res['hf_weight'] = [civec[i][0, 0] for i in range(2)]

        return res


def _cc(mol, occup, core_idx, cas_idx, method, h1e=None, h2e=None, hf=None, rdm1=False):
        """
        this function returns the results of a ccsd / ccsd(t) calculation

        :param mol: pymbe mol object
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
        # init ccsd solver
        if h1e is not None and h2e is not None:

            mol_tmp = gto.M(verbose=0)
            mol_tmp.incore_anyway = mol.incore_anyway
            mol_tmp.max_memory = mol.max_memory

            if mol.spin == 0:
                hf = scf.RHF(mol_tmp)
            else:
                hf = scf.UHF(mol_tmp)

            hf.get_hcore = lambda *args: h1e
            hf._eri = h2e

            if mol.spin == 0:
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

            if np.amin(occup[cas_idx]) == 1.0:
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


