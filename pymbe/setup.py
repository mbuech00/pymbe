#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
start module
"""

from __future__ import annotations

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import numpy as np
from json import load, dump
from pyscf import symm
from typing import TYPE_CHECKING, cast

from pymbe.parallel import MPICls, kw_dist, system_dist
from pymbe.expansion import ExpCls
from pymbe.tools import RST, logger_config, assertion, nelec, nexc, \
                        ground_state_sym

if TYPE_CHECKING:

    from pymbe.pymbe import MBE


def main(mbe: MBE) -> MBE:

        # mpi object
        mbe.mpi = MPICls()

        # input handling
        if mbe.mpi.global_master:

            # check for restart folder
            if not os.path.isdir(RST):

                # copy attributes from mol object
                if mbe.mol:

                    # spin
                    if mbe.spin is None:
                        mbe.spin = mbe.mol.spin

                    # point group
                    if mbe.point_group is None and mbe.orb_type != 'local':
                        mbe.point_group = mbe.mol.groupname

                    # nuclear repulsion energy
                    if mbe.nuc_energy is None:
                        mbe.nuc_energy = mbe.mol.energy_nuc().item() if mbe.mol.atom else 0.

                    # nuclear dipole moment
                    if mbe.nuc_dipole is None and mbe.target == 'dipole':
                        if mbe.mol.atom:
                            charges = mbe.mol.atom_charges()
                            coords  = mbe.mol.atom_coords()
                            mbe.nuc_dipole = np.einsum('i,ix->x', charges, coords)
                        else:
                            mbe.nuc_dipole = np.zeros(3, dtype=np.float64)

                # set default value for spin
                if mbe.spin is None:
                    mbe.spin = 0

                # set default value for point group
                if mbe.point_group is None:
                    mbe.point_group = 'C1'
                mbe.point_group = symm.addons.std_symb(mbe.point_group)

                # set default value for orbital symmetry
                if mbe.orbsym is None and mbe.point_group == 'C1' and mbe.norb is not None:
                    mbe.orbsym = np.zeros(mbe.norb, dtype=np.int64)

                # set default value for fci wavefunction state symmetry
                if mbe.fci_state_sym is None:
                    if mbe.orbsym is not None and mbe.occup is not None:
                        mbe.fci_state_sym = ground_state_sym(mbe.orbsym, \
                                                             mbe.occup, \
                                                             cast(str, mbe.point_group))
                    else:
                        mbe.fci_state_sym = 0

                # set default value for fci wavefunction state root
                if mbe.fci_state_root is None:
                    if mbe.target in ['energy', 'dipole']:
                        mbe.fci_state_root = 0
                    elif mbe.target in ['excitation', 'trans']:
                        mbe.fci_state_root = 1

                # set default value for hartree-fock property
                if mbe.target == 'excitation':
                    mbe.hf_prop = 0.
                elif mbe.target == 'trans':
                    mbe.hf_prop = np.zeros(3, dtype=np.float64)

                # set default value for reference space property
                if mbe.ref_prop is None and mbe.occup is not None:

                    # n_elec
                    n_elec = nelec(mbe.occup, mbe.ref_space)

                    # n_exc
                    n_exc = nexc(n_elec, mbe.ref_space)

                    if n_exc <= 1 or \
                       (mbe.base_method in ['ccsd', 'ccsd(t)'] and n_exc <= 2) or \
                       (mbe.base_method == 'ccsdt' and n_exc <= 3) or \
                       (mbe.base_method == 'ccsdtq' and n_exc <= 4):
                        if mbe.target in ['energy', 'excitation']:
                            mbe.ref_prop = 0.
                        elif mbe.target in ['dipole', 'trans']:
                            mbe.ref_prop = np.zeros(3, dtype=np.float64)

                # set default value for base model property
                if mbe.base_prop is None and mbe.base_method is None:
                    if mbe.target in ['energy', 'excitation']:
                        mbe.base_prop = 0.
                    elif mbe.target in ['dipole', 'trans']:
                        mbe.base_prop = np.zeros(3, dtype=np.float64)

                # create restart folder
                if mbe.rst:
                    os.mkdir(RST)

                # restart logical
                mbe.restarted = False

            else:

                # read keywords
                mbe = restart_read_kw(mbe)

                # read system quantities
                mbe = restart_read_system(mbe)

                # restart logical
                mbe.restarted = True

            # configure logger on global master
            logger_config(mbe.verbose)

            # sanity check
            sanity_check(mbe)

        # bcast keywords
        mbe = kw_dist(mbe)

        # write keywords
        if not mbe.restarted and mbe.mpi.global_master and mbe.rst:
            restart_write_kw(mbe)

        # bcast system quantities
        mbe = system_dist(mbe)

        # write system quantities
        if not mbe.restarted and mbe.mpi.global_master and mbe.rst:
            restart_write_system(mbe)

        # configure logging on slaves
        if not mbe.mpi.global_master:
            logger_config(mbe.verbose)

        # exp object
        mbe.exp = ExpCls(mbe)

        return mbe


def sanity_check(mbe: MBE) -> None:
        """
        this function performs sanity checks of all mbe attributes
        """
        # expansion model
        assertion(isinstance(mbe.method, str), \
                        'electronic structure method (method keyword argument) must be a string')
        assertion(mbe.method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq', 'fci'], \
                        'valid electronic structure methods (method keyword argument) are: ccsd, ccsd(t), ccsdt, ccsdtq and fci')
        assertion(isinstance(mbe.cc_backend, str), \
                        'coupled-cluster backend (cc_backend keyword argument) must be a string')
        assertion(mbe.cc_backend in ['pyscf', 'ecc', 'ncc'], \
                        'valid coupled-cluster backends (cc_backend keyword argument) are: pyscf, ecc and ncc')
        assertion(isinstance(mbe.fci_solver, str), \
                        'fci solver (fci_solver keyword argument) must be a string')
        assertion(mbe.fci_solver in ['pyscf_spin0', 'pyscf_spin1'], \
                        'valid fci solvers (fci_solver keyword argument) are: pyscf_spin0 and pyscf_spin1')
        assertion(isinstance(mbe.hf_guess, bool), \
                        'hf initial guess for fci calculations (hf_guess keyword argument) must be a bool')
        if mbe.method != 'fci':
            assertion(mbe.fci_solver == 'pyscf_spin0', \
                            'setting a fci solver (fci_solver keyword argument) for a non-fci expansion model (method keyword argument) is not meaningful')
            assertion(mbe.hf_guess, \
                            'non-hf initial guess (hf_guess keyword argument) only valid for fci calcs (method keyword argument)')
            if mbe.method == 'ccsdt':
                assertion(mbe.cc_backend != 'pyscf', \
                            'ccsdt (method keyword argument) is not available with the pyscf coupled-cluster backend (cc_backend keyword argument)')
            if mbe.method == 'ccsdtq':
                assertion(mbe.cc_backend == 'ncc', \
                            'ccsdtq (method keyword argument) is not available with the pyscf and ecc coupled-cluster backends (cc_backend keyword argument)')

        # targets
        assertion(isinstance(mbe.target, str), \
                        'expansion target property (target keyword argument) must be a string')
        assertion(mbe.target in ['energy', 'excitation', 'dipole', 'trans'], \
                        'invalid choice for target property (target keyword argument). valid choices are: '
                        'energy, excitation energy (excitation), dipole, and transition dipole (trans)')
        if mbe.method != 'fci':
            assertion(mbe.target in ['energy', 'dipole'], \
                            'excited target states (target keyword argument) not implemented for chosen expansion model (method keyword argument)')
            if mbe.cc_backend in ['ecc', 'ncc']:
                assertion(mbe.target == 'energy', \
                                'calculation of targets (target keyword argument) other than energy are not possible using the ecc and ncc backends (cc_backend keyword argument)')

        # system
        assertion(isinstance(mbe.nuc_energy, float), \
                        'nuclear energy (nuc_energy keyword argument) must be a float')
        if mbe.target == 'dipole':
            assertion(isinstance(mbe.nuc_dipole, np.ndarray), \
                            'nuclear dipole (nuc_dipole keyword argument) must be a np.ndarray')
        assertion(isinstance(mbe.ncore, int) and mbe.ncore >= 0, \
                        'number of core orbitals (ncore keyword argument) must be an int >= 0')
        assertion(isinstance(mbe.nocc, int) and mbe.nocc > 0, \
                        'number of occupied orbitals (nocc keyword argument) must be an int > 0')
        assertion(isinstance(mbe.norb, int) and mbe.norb > 0, \
                        'number of orbitals (norb keyword argument) must be an int > 0')
        assertion(isinstance(mbe.spin, int) and mbe.spin >= 0, \
                        'spin (spin keyword argument) must be an int >= 0')
        if mbe.spin is not None and mbe.spin > 0:
            if mbe.method == 'fci':
                assertion(mbe.fci_solver != 'pyscf_spin0', \
                                'the pyscf_spin0 fci solver (fci_solver keyword argument) is designed for spin singlets only (spin keyword argument)')
            if mbe.method != 'fci' or mbe.base_method is not None:
                assertion(mbe.cc_backend == 'pyscf', \
                                'the ecc and ncc backends (cc_backend keyword argument) are designed for closed-shell systems only (spin keyword argument)')
        assertion(isinstance(mbe.point_group, str), \
                        'symmetry (point_group keyword argument) must be a str')
        assertion(isinstance(mbe.orbsym, np.ndarray), \
                        'orbital symmetry (orbsym keyword argument) must be a np.ndarray')
        assertion(isinstance(mbe.fci_state_sym, (str, int)), \
                        'state wavefunction symmetry (fci_state_sym keyword argument) must be a str or int')
        if isinstance(mbe.fci_state_sym, str):
            try:
                mbe.fci_state_sym = symm.addons.irrep_name2id(mbe.point_group, mbe.fci_state_sym)
            except Exception as err:
                raise ValueError('illegal choice of state wavefunction symmetry (fci_state_sym keyword argument) -- PySCF error: {:}'.format(err))
        assertion(isinstance(mbe.fci_state_root, int) and mbe.fci_state_root >= 0, \
                        'target state (root keyword argument) must be an int >= 0')
        if mbe.occup is not None:
            hf_wfnsym = ground_state_sym(cast(np.ndarray, mbe.orbsym), \
                                         mbe.occup, cast(str, mbe.point_group))
        else: 
            hf_wfnsym = 0
        if mbe.method == 'fci' and mbe.hf_guess:
                assertion(mbe.fci_state_sym == hf_wfnsym, \
                                'illegal choice of reference wfnsym (wfnsym keyword argument) when enforcing hf initial guess (hf_guess keyword argument)'
                                'wfnsym does not equal hf state symmetry')
        if mbe.method != 'fci' or mbe.base_method is not None:
                assertion(mbe.fci_state_sym == hf_wfnsym, \
                                'illegal choice of reference wfnsym (wfnsym keyword argument) for chosen expansion model (method or base_method keyword argument)'
                                'wfnsym does not equal hf state symmetry')
                assertion(mbe.fci_state_root == 0, \
                                'excited target states (root keyword argument) not implemented for chosen expansion model (method or base_method keyword argument)')
        if mbe.target in ['excitation', 'trans']:
            assertion(cast(int, mbe.fci_state_root) > 0, \
                            'calculation of excitation energies or transition dipole moments (target keyword argument) requires target state root (state_root keyword argument) >= 1')

        # hf calculation
        if mbe.target == 'energy':
            assertion(isinstance(mbe.hf_prop, float), \
                            'hartree-fock energy (hf_prop keyword argument) must be a float')
        elif mbe.target == 'dipole':
            assertion(isinstance(mbe.hf_prop, np.ndarray), \
                            'hartree-fock dipole moment (hf_prop keyword argument) must be a np.ndarray')
        assertion(isinstance(mbe.occup, np.ndarray), \
                        'orbital occupation (occup keyword argument) must be a np.ndarray')
        assertion(np.sum(mbe.occup == 1.) == mbe.spin, \
                        'only high-spin open-shell systems are currently possible')

        # orbital representation
        assertion(isinstance(mbe.orb_type, str), \
                        'orbital representation (orb_type keyword argument) must be a string')
        assertion(mbe.orb_type in ['can', 'local', 'ccsd', 'ccsd(t)', 'casscf'], \
                        'valid orbital representations (orb_type keyword argument) are currently: '
                        'canonical (can), pipek-mezey (local), natural (ccsd or ccsd(t) or casscf orbs (casscf))')
        if mbe.orb_type == 'local':
            assertion(mbe.point_group == 'C1', \
                            'the combination of local orbitals (orb_type keyword argument) and point group symmetry (point_group keyword argument) different from c1 is not allowed')

        # integrals
        assertion(isinstance(mbe.hcore, np.ndarray), \
                        'core hamiltonian integral (hcore keyword argument) must be a np.ndarray')
        assertion(isinstance(mbe.vhf, np.ndarray), \
                        'hartree-fock potential (vhf keyword argument) must be a np.ndarray')
        assertion(isinstance(mbe.eri, np.ndarray), \
                        'electron repulsion integral (eri keyword argument) must be a np.ndarray')
        if mbe.target in ['dipole', 'trans']:
            assertion(isinstance(mbe.dipole_ints, np.ndarray), \
                            'dipole integrals (dipole_ints keyword argument) must be a np.ndarray')

        # reference space
        assertion(isinstance(mbe.ref_space, np.ndarray), \
                        'reference space (ref_space keyword argument) must be a np.ndarray of orbital indices')
        assertion(not np.any(np.delete(cast(np.ndarray, mbe.occup), mbe.ref_space) == 1.), \
                        'all partially occupied orbitals have to be included in the reference space (ref_space keyword argument)')
        if mbe.target in ['energy', 'excitation']:
            assertion(isinstance(mbe.ref_prop, float), \
                            'reference (excitation) energy (ref_prop keyword argument) must be a float')
        elif mbe.target in ['dipole', 'trans']:
            assertion(isinstance(mbe.ref_prop, np.ndarray), \
                            'reference (transition) dipole moment (ref_prop keyword argument) must be a np.ndarray')
        
        # base model
        assertion(isinstance(mbe.base_method, (str, type(None))), \
                        'base model electronic structure method (base_method keyword argument) must be a str or None')
        if mbe.base_method is not None:
            assertion(mbe.base_method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq'], \
                            'valid base model electronic structure methods (base_method keyword argument) are currently: ccsd, ccsd(t), ccsdt and ccsdtq')
            if mbe.base_method == 'ccsdt':
                assertion(mbe.cc_backend != 'pyscf', \
                            'ccsdt (base_method keyword argument) is not available with pyscf coupled-cluster backend (cc_backend keyword argument)')
            if mbe.base_method == 'ccsdtq':
                assertion(mbe.cc_backend == 'ncc', \
                            'ccsdtq (base_method keyword argument) is not available with pyscf and ecc coupled-cluster backends (cc_backend keyword argument)')
            assertion(mbe.target in ['energy', 'dipole'], \
                            'excited target states (target keyword argument) not implemented for base model calculations (base_method keyword argument)')
            if mbe.cc_backend in ['ecc', 'ncc']:
                assertion(mbe.target == 'energy', \
                            'calculation of targets (target keyword argument) other than energy are not possible using the ecc and ncc coupled-cluster backends (cc_backend keyword argument)')
            assertion(mbe.fci_state_root == 0, \
                            'excited target states (root keyword argument) not implemented for base model (base_method keyword argument)')
            if mbe.target == 'energy':
                assertion(isinstance(mbe.base_prop, float), \
                            'base model energy (base_prop keyword argument) must be a float')
            elif mbe.target == 'dipole':
                assertion(isinstance(mbe.base_prop, np.ndarray), \
                            'base model dipole moment (base_prop keyword argument) must be a np.ndarray')

        # screening
        assertion(isinstance(mbe.screen_start, int) and mbe.screen_start >= 2, \
                        'screening start order (screen_start keyword argument) must be an int >= 2')
        assertion(isinstance(mbe.screen_perc, float) and mbe.screen_perc <= 1., \
                        'screening threshold (screen_perc keyword argument) must be a float <= 1.')
        if mbe.max_order is not None:
            assertion(isinstance(mbe.max_order, int) and mbe.max_order >= 1, \
                            'maximum expansion order (max_order keyword argument) must be an int >= 1')

        # restart
        assertion(isinstance(mbe.rst, bool), \
                        'restart logical (rst keyword argument) must be a bool')
        assertion(isinstance(mbe.rst_freq, int) and mbe.rst_freq >= 1, \
                        'restart frequency (rst_freq keyword argument) must be an int >= 1')

        # verbose
        assertion(isinstance(mbe.verbose, int) and mbe.verbose >= 0, \
                        'verbose option (verbose keyword argument) must be an int >= 0')
        
        # pi pruning
        assertion(isinstance(mbe.pi_prune, bool), \
                        'pruning of pi-orbitals (pi_prune keyword argument) must be a bool')
        if mbe.pi_prune:
            assertion(mbe.point_group in ['D2h', 'C2v'], \
                            'pruning of pi-orbitals (pi_prune keyword argument) is only implemented for linear D2h and C2v symmetries (point_group keyword argument)')
            assertion(isinstance(mbe.orbsym_linear, np.ndarray), \
                        'linear point group orbital symmetry (orbsym_linear keyword argument) must be a np.ndarray')


def restart_write_kw(mbe: MBE) -> None:
        """
        this function writes the keyword restart file
        """
        # define keywords
        keywords = {'method': mbe.method, 'fci_solver': mbe.fci_solver, \
                    'cc_backend': mbe.cc_backend, 'hf_guess': mbe.hf_guess, \
                    'target': mbe.target, 'point_group': mbe.point_group, \
                    'fci_state_sym': mbe.fci_state_sym, 'fci_state_root': mbe.fci_state_root, \
                    'orb_type': mbe.orb_type, 'base_method': mbe.base_method, \
                    'screen_start': mbe.screen_start, 'screen_perc': mbe.screen_perc, \
                    'max_order': mbe.max_order, 'rst': mbe.rst, \
                    'rst_freq': mbe.rst_freq, 'verbose': mbe.verbose, \
                    'pi_prune': mbe.pi_prune}

        # write keywords
        with open(os.path.join(RST, 'keywords.rst'), 'w') as f:
            dump(keywords, f)


def restart_read_kw(mbe: MBE) -> MBE:
        """
        this function reads the keyword restart file
        """
        # read keywords
        with open(os.path.join(RST, 'keywords.rst'), 'r') as f:
            keywords = load(f)
        
        # set keywords as MBE attributes
        for key, val in keywords.items():
            setattr(mbe, key, val)

        return mbe


def restart_write_system(mbe: MBE) -> None:
        """
        this function writes all system quantities restart files
        """
        # define system quantities
        system = {'nuc_energy': mbe.nuc_energy, 'ncore': mbe.ncore, 'nocc': mbe.nocc, \
                  'norb': mbe.norb, 'spin': mbe.spin, 'orbsym': mbe.orbsym, \
                  'hf_prop': mbe.hf_prop, 'occup': mbe.occup, 'hcore': mbe.hcore, \
                  'vhf': mbe.vhf, 'eri': mbe.eri, 'ref_space': mbe.ref_space, \
                  'ref_prop': mbe.ref_prop, 'base_prop': mbe.base_prop}

        if mbe.target == 'dipole':
            system['nuc_dipole'] = mbe.nuc_dipole

        if mbe.target in ['dipole', 'trans']:
            system['dipole_ints'] = mbe.dipole_ints

        if mbe.pi_prune:
            system['orbsym_linear'] = mbe.orbsym_linear

        # write system quantities
        np.savez(os.path.join(RST, 'system'), **system) # type: ignore


def restart_read_system(mbe: MBE) -> MBE:
        """
        this function reads all system quantities restart files
        """
        # read system quantities
        system_npz = np.load(os.path.join(RST, 'system.npz'))

        # create system dictionary
        system = {}
        for file in system_npz.files:
            system[file] = system_npz[file]

        # close npz object
        system_npz.close()

        # define scalar values
        scalars = ['nuc_energy', 'ncore', 'nocc', 'norb', 'spin']

        if mbe.target in ['energy', 'excitation']:
            scalars.append('hf_prop')
            scalars.append('ref_prop')
            scalars.append('base_prop')

        # convert to scalars
        for scalar in scalars:
            system[scalar] = system[scalar].item()

        # set system quantities as MBE attributes
        for key, val in system.items():
            setattr(mbe, key, val)

        return mbe


def settings() -> None:
        """
        this function sets and asserts some general settings
        """
        # only run with python3+
        assertion(3 <= sys.version_info[0], 'PyMBE only runs under python3+')

        # PYTHONHASHSEED = 0
        pythonhashseed = os.environ.get('PYTHONHASHSEED', -1)
        assertion(int(pythonhashseed) == 0, \
                  'environment variable PYTHONHASHSEED must be set to zero')


