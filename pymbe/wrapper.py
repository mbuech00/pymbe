#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
wrapper module
"""

from __future__ import annotations

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, scf, symm, lo, cc, mcscf, fci, ao2mo
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import uccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import uccsd_t_rdm
from copy import copy
from typing import TYPE_CHECKING, cast
from warnings import catch_warnings, simplefilter

from pymbe.kernel import main as kernel_main, e_core_h1e, _cc, _dipole
from pymbe.tools import assertion, mat_idx, near_nbrs, core_cas, nelec, nexc, \
                        idx_tril, ground_state_sym

if TYPE_CHECKING:

    from typing import Tuple, Dict, Union, List, Optional, Any


CONV_TOL = 1.e-10
SPIN_TOL = 1.e-05


def ints(mol: gto.Mole, mo_coeff: np.ndarray, norb: int, nocc: int, \
         x2c: bool = False, u: float = 1., matrix: Tuple[int, int] = (1, 6), \
         pbc: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        this function returns 1e and 2e mo integrals and effective fock potentials from individual occupied orbitals
        """
        # mol
        assertion(isinstance(mol, gto.Mole), \
                        'ints: mol (first argument) must be a gto.Mole object')
        # mo_coeff
        assertion(isinstance(mo_coeff, np.ndarray), \
                        'ints: mo coefficients (second argument) must be a np.ndarray')
        # norb
        assertion(isinstance(norb, int) and norb > 0, \
                        'ints: number of orbitals (third argument) must be an int > 0')
        # nocc
        assertion(isinstance(nocc, int) and nocc > 0, \
                        'ints: number of occupied orbitals (fourth argument) must be an int > 0')
        # x2c
        assertion(isinstance(x2c, bool), \
                        'ints: spin-free x2c relativistic hamiltonian option (x2c keyword argument) must be a bool')
        # hubbard
        if not mol.atom:
            # matrix
            assertion(isinstance(matrix, tuple) and len(matrix) == 2 and isinstance(matrix[0], int) and isinstance(matrix[1], int), \
                            'ints: hubbard matrix (matrix keyword argument) must be a tuple of ints with a dimension of 2')
            # u parameter
            assertion(isinstance(u, float), \
                            'ints: hubbard on-site repulsion parameter (u keyword argument) must be a float')
            assertion(u > 0., \
                            'ints: only repulsive hubbard models are implemented'
                            'hubbard on-site repulsion parameter (u keyword argument) must be > 0.')
            # periodic boundary conditions
            assertion(isinstance(pbc, bool), \
                            'ints: hubbard model periodic boundary conditions (pbc keyword argument) must be a bool')

        # hcore_ao and eri_ao w/o symmetry
        hcore_ao, eri_ao = _ao_ints(mol, x2c, u, matrix, pbc)

        # compute hcore
        hcore = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_ao, mo_coeff)

        # eri_mo w/o symmetry
        eri = ao2mo.incore.full(eri_ao, mo_coeff)

        # compute vhf
        vhf = np.empty((nocc, norb, norb), dtype=np.float64)
        for i in range(nocc):
            idx = np.asarray([i])
            vhf[i] = np.einsum('pqrs->rs', eri[idx[:, None], idx, :, :]) * 2.
            vhf[i] -= np.einsum('pqrs->ps', eri[:, idx[:, None], idx, :]) * 2. * .5

        # restore 4-fold symmetry in eri
        eri = ao2mo.restore(4, eri, norb)

        return hcore, vhf, eri


def _ao_ints(mol: gto.Mole, x2c: bool, u: float, matrix: Tuple[int, int], \
             pbc: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns 1e and 2e ao integrals
        """
        if mol.atom:

            # hcore_ao
            if x2c:
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
            hcore = _hubbard_h1e(matrix, pbc)
            # eri_ao
            eri = _hubbard_eri(matrix, u)

        return hcore, eri


def _hubbard_h1e(matrix: Tuple[int, int], pbc: bool = False) -> np.ndarray:
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


def _hubbard_eri(matrix: Tuple[int, int], u: float) -> np.ndarray:
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


def dipole_ints(mol: gto.Mole, mo_coeff: np.ndarray, gauge_origin: np.ndarray) -> np.ndarray:
        """
        this function returns dipole integrals (in AO basis)
        """
        # mol
        assertion(isinstance(mol, gto.Mole), \
                        'dipole_ints: mol (first argument) must be a gto.Mole object')
        # mo_coeff
        assertion(isinstance(mo_coeff, np.ndarray), \
                        'dipole_ints: mo coefficients (second argument) must be a np.ndarray')
        # gauge origin
        assertion(isinstance(gauge_origin, np.ndarray) and gauge_origin.size == 3, \
                        'dipole_ints: gauge origin (gauge_origin keyword argument) must be a np.ndarray of size 3')

        with mol.with_common_origin(gauge_origin):
            dipole = mol.intor_symmetric('int1e_r', comp=3)

        return np.einsum('pi,xpq,qj->xij', mo_coeff, dipole, mo_coeff)


def hf(mol: gto.Mole, init_guess: str = 'minao', newton: bool = False, \
       irrep_nelec: Dict[str, Any] = {}, x2c: bool = False, u: float = 1., \
       matrix: Tuple[int, int] = (1, 6), pbc: bool = False, \
       gauge_origin: np.ndarray = np.array([0., 0., 0.])) -> Tuple[int, int, int, scf.hf.SCF, float, np.ndarray, \
                                                     np.ndarray, np.ndarray, np.ndarray]:
        """
        this function returns the results of a restricted (open-shell) hartree-fock calculation
        """
        # mol
        assertion(isinstance(mol, gto.Mole), \
                        'hf: mol (first argument) must be a gto.Mole object')
        # init_guess
        assertion(isinstance(init_guess, str), \
                        'hf: hf initial guess (init_guess keyword argument) must be a str')
        assertion(init_guess in ['minao', 'atom', '1e'], \
                        'hf: valid hf initial guesses (init_guess keyword argument) are: '
                        'minao, atom, and 1e')
        # newton
        assertion(isinstance(newton, bool), \
                        'hf: newton option (newton keyword argument) must be a bool')
        # irrep_nelec
        assertion(isinstance(irrep_nelec, dict), \
                        'hf: irreducible representation occupation (irrep_nelec keyword argument) must be a dict')
        # x2c
        assertion(isinstance(x2c, bool), \
                        'hf: spin-free x2c relativistic hamiltonian option (x2c keyword argument) must be a bool')
        # hubbard
        if not mol.atom:
            # matrix
            assertion(isinstance(matrix, tuple) and len(matrix) == 2 and isinstance(matrix[0], int) and isinstance(matrix[1], int), \
                            'hf: hubbard matrix (matrix keyword argument) must be a tuple of ints with a dimension of 2')
            # u parameter
            assertion(isinstance(u, float), \
                            'hf: hubbard on-site repulsion parameter (u keyword argument) must be a float')
            assertion(u > 0., \
                            'hf: only repulsive hubbard models are implemented'
                            'hubbard on-site repulsion parameter (u keyword argument) must be > 0.')
            
            # periodic boundary conditions
            assertion(isinstance(pbc, bool), \
                            'hf: hubbard model periodic boundary conditions (pbc keyword argument) must be a bool')
        # gauge origin
        assertion(isinstance(gauge_origin, np.ndarray) and gauge_origin.size == 3, \
                        'hf: gauge origin (gauge_origin keyword argument) must be a np.ndarray of size 3')

        # initialize restricted hf calc
        if x2c:
            hf = scf.RHF(mol).x2c()
        else:
            hf = scf.RHF(mol)

        hf.init_guess = init_guess
        if newton:
            hf.conv_tol = 1.e-01
        else:
            hf.conv_tol = CONV_TOL
        hf.max_cycle = 1000

        if mol.atom:
            # ab initio hamiltonian
            hf.irrep_nelec = irrep_nelec
        else:
            # model hamiltonian
            hf.get_ovlp = lambda *args: np.eye(matrix[0] * matrix[1])
            hf.get_hcore = lambda *args: _hubbard_h1e(matrix, pbc)
            hf._eri = _hubbard_eri(matrix, u)

        # hf calc
        with catch_warnings():
            simplefilter("ignore")
            hf.kernel()

        if newton:

            # initial mo coefficients and occupation
            mo_coeff = hf.mo_coeff
            mo_occ = hf.mo_occ

            # new so-hf object
            hf = hf.newton()
            hf.conv_tol = CONV_TOL

            with catch_warnings():
                simplefilter("ignore")
                hf.kernel(mo_coeff, mo_occ)

        # dipole moment
        if mol.atom:
            dm = hf.make_rdm1()
            if mol.spin > 0:
                dm = dm[0] + dm[1]
            with mol.with_common_orig(gauge_origin):
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)
            elec_dipole = np.einsum('xij,ji->x', ao_dip, dm)
        else:
            elec_dipole = np.zeros(3, dtype=np.float64)

        # determine dimensions
        norb, nocc, nvirt = _dim(hf.mo_occ)

        # store energy, occupation, and orbsym
        hf_energy = hf.e_tot
        occup = hf.mo_occ
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        else:
            orbsym = np.zeros(norb, dtype=np.int64)

        return nocc, nvirt, norb, hf, hf_energy.item(), elec_dipole, occup, \
                orbsym, np.asarray(hf.mo_coeff, order='C')


def _dim(mo_occ: np.ndarray) -> Tuple[int, ...]:
        """
        this function determines the involved dimensions (number of occupied, virtual, and total orbitals)
        """
        # occupied and virtual lists
        occ = np.where(mo_occ > 0.)[0]
        virt = np.where(mo_occ == 0.)[0]
        return occ.size + virt.size, occ.size, virt.size


def ref_mo(orbs: str, mol: gto.Mole, hf: scf.hf.SCF, mo_coeff: np.ndarray, \
           occup: np.ndarray, orbsym: np.ndarray, norb: int, ncore: int, \
           nocc: int, nvirt: int, ref_space: np.ndarray = np.array([]), \
           fci_solver: str = 'pyscf_spin0', \
           wfnsym: Optional[List[Union[str, int]]] = None, \
           weights: List[float] = [1.], \
           hf_guess: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns a set of reference mo coefficients and symmetries plus the associated spaces
        """
        # orbs
        assertion(isinstance(orbs, str), \
                        'ref_mo: orbital representation (first argument) must be a str')
        assertion(orbs in ['ccsd', 'ccsd(t)', 'local', 'casscf'], \
                        'ref_mo: valid orbital representations (first argument) are: '
                        'natural (ccsd or ccsd(t)), pipek-mezey (local), or casscf orbs (casscf')
        # mol
        assertion(isinstance(mol, gto.Mole), \
                        'ref_mo: mol (second argument) must be a gto.Mole object')
        # hf
        assertion(isinstance(hf, scf.hf.SCF), \
                        'ref_mo: hf (third argument) must be a scf.hf.SCF object')
        # mo_coeff
        assertion(isinstance(mo_coeff, np.ndarray), \
                        'ref_mo: mo coefficients (fourth argument) must be a np.ndarray')
        # occup
        assertion(isinstance(occup, np.ndarray), \
                        'ref_mo: orbital occupation (fifth argument) must be a np.ndarray')
        assertion(np.sum(occup == 1.) == mol.spin, \
                        'only high-spin open-shell systems are currently possible')
        # orbsym
        assertion(isinstance(orbsym, np.ndarray), \
                        'ref_mo: orbital symmetry (sixth argument) must be a np.ndarray')
        # norb
        assertion(isinstance(norb, int) and norb > 0, \
                        'ref_mo: number of orbitals (seventh argument) must be an int > 0')
        # ncore
        assertion(isinstance(ncore, int), \
                        'ref_mo: number of core orbitals (eighth argument) must be an int')
        # nocc
        assertion(isinstance(nocc, int) and nocc > 0, \
                        'ref_mo: number of occupied orbitals (ninth argument) must be an int > 0')
        # nvirt
        assertion(isinstance(nvirt, int) and nvirt > 0, \
                        'ref_mo: number of virtual orbitals (tenth argument) must be an int > 0')
        # casscf
        if orbs == 'casscf':
            # set default casscf reference symmetry
            if wfnsym is None:
                wfnsym = [symm.addons.irrep_id2name(mol.groupname, 0)] if mol.groupname else [0]
            # ref_space
            assertion(isinstance(ref_space, np.ndarray), \
                            'ref_mo: reference space (ref_space keyword argument) must be a np.ndarray of orbital indices')
            assertion(np.any(occup[ref_space] > 0.), \
                            'ref_mo: no singly/doubly occupied orbitals in cas space (ref_space keyword argument) of casscf calculation')
            assertion(np.any(occup[ref_space] < 2.), \
                            'ref_mo: no virtual orbitals in cas space (ref_space keyword argument) of casscf calculation')
            # fci_solver
            assertion(isinstance(fci_solver, str), \
                            'ref_mo: fci solver (fci_solver keyword argument) must be a str')
            assertion(fci_solver in ['pyscf_spin0', 'pyscf_spin1'], \
                            'ref_mo: valid fci solvers (fci_solver keyword argument) are: '
                            'pyscf_spin0 and pyscf_spin1')
            if mol.spin > 0:
                assertion(fci_solver != 'pyscf_spin0', \
                                'ref_mo: the pyscf_spin0 fci solver (fci_solver keyword argument) is designed for spin singlets only')
            # wfnsym
            assertion(isinstance(wfnsym, list) and all(isinstance(i, str) for i in wfnsym), \
                            'ref_mo: casscf wavefunction symmetries (wfnsym keyword argument) must be a list of str')
            wfnsym_int: List[int] = []
            for i in range(len(wfnsym)):
                try:
                    wfnsym_int.append(symm.addons.irrep_name2id(mol.groupname, wfnsym[i]) if mol.groupname else 0)
                except Exception as err:
                    raise ValueError('ref_mo: illegal choice of ref wfnsym (wfnsym keyword argument) -- PySCF error: {:}'.format(err))
            # weights
            assertion(isinstance(weights, list) and all(isinstance(i, float) for i in weights), \
                            'ref_mo: casscf weights (weights keyword argument) must be a list of floats')
            assertion(len(wfnsym_int) == len(weights), \
                            'ref_mo: list of wfnsym (wfnsym keyword argument) and weights (weights keyword argument) for casscf calculation must be of same length')
            assertion(all(isinstance(i, float) for i in weights), \
                            'ref_mo: casscf weights (weights keyword argument) must be floats')
            assertion(abs(sum(weights) - 1.) < 1.e-3, \
                            'ref_mo: sum of weights for casscf calculation (weights keyword argument) must be equal to 1.')
            # hf_guess
            assertion(isinstance(hf_guess, bool), \
                            'ref_mo: hf initial guess (hf_guess keyword argument) must be a bool')
            if hf_guess:
                assertion(len(set(wfnsym_int)) == 1, \
                                'ref_mo: illegal choice of reference wfnsym (wfnsym keyword argument) when enforcing hf initial guess (hf_guess keyword argument)'
                                'wfnsym should be limited to one state')
                assertion(wfnsym_int[0] == ground_state_sym(orbsym, occup, mol.groupname), \
                                'ref_mo: illegal choice of reference wfnsym (wfnsym keyword argument) when enforcing hf initial guess (hf_guess keyword argument)'
                                'wfnsym does not equal hf state symmetry')

        # copy mo coefficients
        mo_coeff_out = np.copy(mo_coeff)

        # set core and cas spaces
        core_idx, cas_idx = core_cas(nocc, np.arange(ncore, nocc), \
                                        np.arange(nocc, norb))

        # NOs
        if orbs in ['ccsd', 'ccsd(t)']:

            # compute rmd1
            ccsd = cc.CCSD(hf)
            frozen_orbs = np.asarray([i for i in range(hf.mo_coeff.shape[1]) if i not in cas_idx])
            if frozen_orbs.size > 0:
                ccsd.frozen = frozen_orbs

            # settings
            ccsd.conv_tol = 1.e-10
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

            # rdm1
            if orbs == 'ccsd':
                ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
                rdm1 = ccsd.make_rdm1()
            elif orbs == 'ccsd(t)':
                if mol.spin == 0:
                    l1, l2 = ccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)[1:]
                    rdm1 = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
                else:
                    l1, l2 = uccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)[1:]
                    rdm1 = uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)

            if mol.spin > 0:
                rdm1 = rdm1[0] + rdm1[1]

            # occupied - occupied block
            mask = occup == 2.
            mask[:ncore] = False
            if np.any(mask):
                no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
                mo_coeff_out[:, mask] = np.einsum('ip,pj->ij', mo_coeff[:, mask], no[:, ::-1])

            # singly occupied - singly occupied block
            mask = occup == 1.
            mask[:ncore] = False
            if np.any(mask):
                no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
                mo_coeff_out[:, mask] = np.einsum('ip,pj->ij', mo_coeff[:, mask], no[:, ::-1])

            # virtual - virtual block
            mask = occup == 0.
            mask[:ncore] = False
            if np.any(mask):
                no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
                mo_coeff_out[:, mask] = np.einsum('ip,pj->ij', mo_coeff[:, mask], no[:, ::-1])

            # orbital symmetries
            if mol.symmetry:
                orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_out)

        # pipek-mezey localized orbitals
        elif orbs == 'local':

            # occupied - occupied block
            mask = occup == 2.
            mask[:ncore] = False
            if np.any(mask):
                if mol.atom:
                    loc = lo.PM(mol, mo_coeff[:, mask])
                else:
                    loc = _hubbard_PM(mol, mo_coeff[:, mask])
                loc.conv_tol = CONV_TOL
                mo_coeff_out[:, mask] = loc.kernel()

            # singly occupied - singly occupied block
            mask = occup == 1.
            mask[:ncore] = False
            if np.any(mask):
                if mol.atom:
                    loc = lo.PM(mol, mo_coeff[:, mask])
                else:
                    loc = _hubbard_PM(mol, mo_coeff[:, mask])
                loc.conv_tol = CONV_TOL
                mo_coeff_out[:, mask] = loc.kernel()

            # virtual - virtual block
            mask = occup == 0.
            mask[:ncore] = False
            if np.any(mask):
                if mol.atom:
                    loc = lo.PM(mol, mo_coeff[:, mask])
                else:
                    loc = _hubbard_PM(mol, mo_coeff[:, mask])
                loc.conv_tol = CONV_TOL
                mo_coeff_out[:, mask] = loc.kernel()

            # orbital symmetries
            if mol.symmetry:
                orbsym = np.zeros(norb, dtype=np.int64)

        # casscf
        elif orbs == 'casscf':

            # electrons in active space
            act_n_elec = nelec(occup, ref_space)

            # sorter for active space
            n_core_inact = np.array([i for i in range(nocc) if i not in ref_space], dtype=np.int64)
            n_virt_inact = np.array([a for a in range(nocc, norb) if a not in ref_space], dtype=np.int64)
            sort_casscf = np.concatenate((n_core_inact, ref_space, n_virt_inact))
            mo_coeff_casscf = mo_coeff_out[:, sort_casscf]

            # update orbsym
            if mol.symmetry:
                orbsym_casscf = symm.label_orb_symm(mol, mol.irrep_id, \
                                                    mol.symm_orb, \
                                                    mo_coeff_casscf)

            # run casscf
            mo_coeff_out = _casscf(mol, fci_solver, wfnsym_int, weights, \
                                   orbsym_casscf, hf_guess, hf, \
                                   mo_coeff_casscf, ref_space, act_n_elec, \
                                   ncore)

            # reorder mo_coeff
            mo_coeff_out = mo_coeff_out[:, np.argsort(sort_casscf)]

            # orbital symmetries
            if mol.symmetry:
                orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_out)

        return np.asarray(mo_coeff_out, order='C'), orbsym


class _hubbard_PM(lo.pipek.PM):
        """
        this class constructs the site-population tensor for each orbital-pair density
        see: pyscf example - 40-hubbard_model_PM_localization.py
        """
        def atomic_pops(self, mol: gto.Mole, mo_coeff: np.ndarray, \
                        method: Optional[str] = None) -> np.ndarray:
            """
            this function overwrites the tensor used in the pm cost function and its gradients
            """
            return np.einsum('pi,pj->pij', mo_coeff, mo_coeff)



def _casscf(mol: gto.Mole, solver: str, wfnsym: List[int], \
            weights: List[float], orbsym: np.ndarray, hf_guess: bool, \
            hf: scf.hf.SCF, mo_coeff: np.ndarray, ref_space: np.ndarray, \
            n_elec: Tuple[int, int], ncore: int) -> np.ndarray:
        """
        this function returns the results of a casscf calculation
        """
        # init casscf
        cas = mcscf.CASSCF(hf, ref_space.size, n_elec)

        # casscf settings
        cas.conv_tol = CONV_TOL
        cas.max_cycle_macro = 500
        cas.frozen = ncore

        # init fcisolver
        if solver == 'pyscf_spin0':
            fcisolver = fci.direct_spin0_symm.FCI(mol)
        elif solver == 'pyscf_spin1':
            fcisolver = fci.direct_spin1_symm.FCI(mol)

        # create unique list of wfnsym while maintaining order
        unique_wfnsym = list(dict.fromkeys(wfnsym))

        # fci settings
        fcisolver.conv_tol = CONV_TOL
        fcisolver.orbsym = orbsym[ref_space]
        fcisolver.wfnsym = unique_wfnsym[0]
        cas.fcisolver = fcisolver

        # state-averaged casscf
        if len(wfnsym) > 1:

            if len(unique_wfnsym) == 1:

                # state average over all states of same symmetry
                cas.state_average_(weights)

            else:

                # nroots for first fcisolver
                fcisolver.nroots = np.count_nonzero(np.asarray(wfnsym) == unique_wfnsym[0])
                

                # init list of fcisolvers
                fcisolvers = [fcisolver]

                # loop over symmetries
                for i in range(1, len(unique_wfnsym)):

                    # copy fcisolver
                    fcisolver_ = copy(fcisolver)

                    # wfnsym for fcisolver_
                    fcisolver_.wfnsym = unique_wfnsym[i]

                    # nroots for fcisolver_
                    fcisolver_.nroots = np.count_nonzero(np.asarray(wfnsym) == unique_wfnsym[i])

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

        return np.asarray(cas.mo_coeff, order='C')


def ref_prop(mol: gto.Mole, hcore: np.ndarray, vhf: np.ndarray, \
             eri: np.ndarray, occup: np.ndarray, orbsym: np.ndarray, \
             nocc: int, ref_space: np.ndarray, method: str = 'fci', \
             base_method: Optional[str] = None, \
             fci_solver: str = 'pyscf_spin0', cc_backend: str = 'pyscf', \
             fci_state_sym: Optional[Union[str, int]] = None, \
             fci_state_root: int = 0, hf_guess: bool = True, \
             target: str = 'energy', \
             hf_prop: Optional[Union[float, np.ndarray]] = None, \
             dipole_ints: Optional[np.ndarray] = None, \
             orb_type: str = 'can') -> Union[float, np.ndarray]:
        """
        this function returns reference space properties
        """
        # mol
        assertion(isinstance(mol, gto.Mole), \
                        'ref_prop: mol (first argument) must be a gto.Mole object')
        # hcore
        assertion(isinstance(hcore, np.ndarray), \
                        'ref_prop: core hamiltonian integrals (second argument) must be a np.ndarray')
        # vhf
        assertion(isinstance(vhf, np.ndarray), \
                        'ref_prop: hartree-fock potential (third argument) must be a np.ndarray')
        # eri
        assertion(isinstance(eri, np.ndarray), \
                        'ref_prop: electron repulsion integrals (fourth argument) must be a np.ndarray')
        # occup
        assertion(isinstance(occup, np.ndarray), \
                        'ref_prop: orbital occupation (fifth argument) must be a np.ndarray')
        assertion(np.sum(occup == 1.) == mol.spin, \
                        'ref_prop: only high-spin open-shell systems are currently possible')
        # orbsym
        assertion(isinstance(orbsym, np.ndarray), \
                        'ref_prop: orbital symmetry (sixth argument) must be a np.ndarray')
        # nocc
        assertion(isinstance(nocc, int) and nocc > 0, \
                        'ref_prop: number of occupied orbitals (seventh argument) must be an int > 0')
        # ref_space
        assertion(isinstance(ref_space, np.ndarray), \
                        'ref_prop: reference space (eighth argument) must be a np.ndarray of orbital indices')
        # method
        assertion(isinstance(method, str), \
                        'ref_prop: electronic structure method (method keyword argument) must be a string')
        assertion(method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq', 'fci'], \
                        'ref_prop: valid electronic structure methods (method keyword argument) are: '
                        'ccsd, ccsd(t), ccsdt, ccsdtq and fci')
        # base_method
        assertion(isinstance(base_method, (str, type(None))), \
                            'ref_prop: base model electronic structure method (base_method keyword argument) must be a str or None')
        if base_method is not None:
            assertion(base_method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq'], \
                            'ref_prop: valid base model electronic structure methods (base_method keyword argument) are: '
                            'ccsd, ccsd(t), ccsdt and ccsdtq')
        # fci
        if fci_state_sym is None:
            fci_state_sym = ground_state_sym(orbsym, occup, mol.groupname)
        if method == 'fci':
            # fci_solver
            assertion(isinstance(fci_solver, str), \
                            'ref_prop: fci solver (fci_solver keyword argument) must be a str')
            assertion(fci_solver in ['pyscf_spin0', 'pyscf_spin1'], \
                            'ref_prop: valid fci solvers (fci_solver keyword argument) are: '
                            'pyscf_spin0 and pyscf_spin1')
            if mol.spin > 0:
                assertion(fci_solver != 'pyscf_spin0', \
                                'ref_prop: the pyscf_spin0 fci solver (fci_solver keyword argument) is designed for spin singlets only')
            # fci_state_sym
            assertion(isinstance(fci_state_sym, (str, int)), \
                            'ref_prop: fci state symmetry (fci_state_sym keyword argument) must be a str or int')
            if isinstance(fci_state_sym, str):
                try:
                    fci_state_sym = symm.addons.irrep_name2id(mol.groupname, fci_state_sym) if mol.groupname else 0
                except Exception as err:
                    raise ValueError('ref_prop: illegal choice of fci state symmetry (fci_state_sym keyword argument) -- PySCF error: {:}'.format(err))
            # fci_state_root
            assertion(isinstance(fci_state_root, int), \
                            'ref_prop: fci state root (fci_state_root keyword argument) must be an int')
            assertion(fci_state_root >= 0, \
                            'ref_prop: choice of fci target state (fci_state_root keyword argument) must be an int >= 0')
            # hf_guess
            assertion(isinstance(hf_guess, bool), \
                            'ref_prop: hf initial guess (hf_guess keyword argument) must be a bool')
            if hf_guess:
                assertion(fci_state_sym == ground_state_sym(orbsym, occup, mol.groupname), \
                                'ref_prop: illegal choice of reference wfnsym (wfnsym keyword argument) when enforcing hf initial guess (hf_guess keyword argument)'
                                'wfnsym does not equal hf state symmetry')
        # cc methods
        elif method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq'] or base_method:
            assertion(isinstance(cc_backend, str), \
                            'ref_prop: coupled-cluster backend (cc_backend keyword argument) must be a string')
            assertion(cc_backend in ['pyscf', 'ecc', 'ncc'], \
                            'ref_prop: valid coupled-cluster backends (cc_backend keyword argument) are: '
                            'pyscf, ecc and ncc')
            if base_method == 'ccsdt':
                assertion(cc_backend != 'pyscf', \
                            'ref_prop: ccsdt is not available with pyscf coupled-cluster backend (cc_backend keyword argument)')
            if base_method == 'ccsdtq':
                assertion(cc_backend == 'ncc', \
                            'ref_prop: ccsdtq is not available with pyscf and ecc coupled-cluster backends (cc_backend keyword argument)')
            if mol.spin > 0:
                assertion(cc_backend == 'pyscf', \
                            'ref_prop: open-shell systems are not available with ecc and ncc coupled-cluster backends (cc_backend keyword argument)')
        # target
        assertion(isinstance(target, str), \
                        'ref_prop: target property (target keyword argument) must be str')
        assertion(target in ['energy', 'excitation', 'dipole', 'trans'], \
                        'ref_prop: valid target properties (target keyword argument) are: '
                        'energy, excitation energy (excitation), dipole, and transition dipole (trans)')
        if target in ['excitation', 'trans']:
            assertion(fci_state_root > 0, \
                            'ref_prop: calculation of excitation energies or transition dipole moments (target keyword argument) requires target state root (state_root keyword argument) >= 1')
        if (method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq'] or base_method):
            assertion(target in ['energy', 'dipole'], \
                            'ref_prop: calculation of excitation energies or transition dipole moments (target keyword argument) not possible with coupled-cluster methods (method keyword argument)')
            if cc_backend in ['ecc', 'ncc']:
                assertion(target == 'energy', \
                                'ref_prop: calculation of targets (target keyword argument) other than energy not possible with ecc and ncc coupled-cluster backends (cc_backend keyword argument)')
        # hf_prop
        if target == 'energy':
            assertion(isinstance(hf_prop, float), \
                            'ref_prop: hartree-fock energy (hf_prop keyword argument) must be a float')
        elif target == 'dipole':
            assertion(isinstance(hf_prop, np.ndarray), \
                            'ref_prop: hartree-fock dipole moment (hf_prop keyword argument) must be a np.ndarray')
        # dipole_ints
        if target in ['dipole', 'trans']:
            assertion(isinstance(dipole_ints, np.ndarray), \
                            'ref_prop: dipole integrals (dipole_ints keyword argument) must be a np.ndarray')
        # orbital representation
        assertion(isinstance(orb_type, str), \
                        'ref_prop: orbital representation (orbs keyword argument) must be a str')
        assertion(orb_type in ['can', 'ccsd', 'ccsd(t)', 'local', 'casscf'], \
                        'ref_prop: valid orbital representations (orbs keyword argument) are: '
                        'canonical (can), natural (ccsd or ccsd(t)), pipek-mezey (local), or casscf orbs (casscf')

        # hf_prop
        if target == 'excitation':
            hf_prop = 0.
        elif target == 'trans':
            hf_prop = np.zeros(3, dtype=np.float64)
        hf_prop = np.asarray(hf_prop)

        # core_idx and cas_idx
        core_idx, cas_idx = core_cas(nocc, ref_space, np.array([], dtype=np.int64))

        # n_elec
        n_elec = nelec(occup, cas_idx)

        # n_exc
        n_exc = nexc(n_elec, cas_idx)

        # ref_prop
        ref_prop: Union[float, np.ndarray]

        if n_exc <= 1 or \
           (base_method in ['ccsd', 'ccsd(t)'] and n_exc <= 2) or \
           (base_method == 'ccsdt' and n_exc <= 3) or \
           (base_method == 'ccsdtq' and n_exc <= 4):

            # no correlation in expansion reference space
            if target in ['energy', 'excitation']:
                ref_prop = 0.
            else:
                ref_prop = np.zeros(3, dtype=np.float64)

        else:

            # get cas_space h2e
            cas_idx_tril = idx_tril(cas_idx)
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(mol.energy_nuc().item(), hcore, vhf, core_idx, cas_idx)

            # exp model
            ref_prop = kernel_main(method, cc_backend, fci_solver, orb_type, \
                                   mol.spin, occup, target, \
                                   cast(int, fci_state_sym), mol.groupname, \
                                   orbsym, hf_guess, fci_state_root, hf_prop, \
                                   e_core, h1e_cas, h2e_cas, core_idx, cas_idx, \
                                   n_elec, 0, dipole_ints, \
                                   higher_amp_extrap=False)
                        
            # base model
            if base_method is not None:
                ref_prop -= kernel_main(base_method, cc_backend, '', orb_type, \
                                        mol.spin, occup, target, \
                                        cast(int, fci_state_sym), \
                                        mol.groupname, orbsym, hf_guess, \
                                        fci_state_root, hf_prop, e_core, \
                                        h1e_cas, h2e_cas, core_idx, cas_idx, \
                                        n_elec, 0, dipole_ints, \
                                        higher_amp_extrap=False)

        return ref_prop


def base(method: str, mol: gto.Mole, hf: scf.hf.SCF, mo_coeff: np.ndarray, \
         occup: np.ndarray, orbsym: np.ndarray, norb: int, ncore: int, \
         nocc: int, cc_backend: str = 'pyscf', target: str = 'energy', \
         hf_dipole: Optional[np.ndarray] = None, \
         gauge_origin: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
        """
        this function returns base model energy
        """
        # method
        assertion(isinstance(method, str), \
                        'base: base model electronic structure method (first argument) must be a str')
        assertion(method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq'], \
                        'base: valid base model electronic structure methods (first argument) are: '
                        'ccsd, ccsd(t), ccsdt, and ccsdtq')
        # mol
        assertion(isinstance(mol, gto.Mole), \
                        'base: mol (second argument) must be a gto.Mole object')
        # hf
        assertion(isinstance(hf, scf.hf.SCF), \
                        'base: hf (third argument) must be a scf.hf.SCF object')
        # mo_coeff
        assertion(isinstance(mo_coeff, np.ndarray), \
                        'base: mo coefficients (fourth argument) must be a np.ndarray')
        # occup
        assertion(isinstance(occup, np.ndarray), \
                        'base: orbital occupation (fifth argument) must be a np.ndarray')
        assertion(np.sum(occup == 1.) == mol.spin, \
                        'only high-spin open-shell systems are currently possible')
        # orbsym
        assertion(isinstance(orbsym, np.ndarray), \
                        'base: orbital symmetry (sixth argument) must be a np.ndarray')
        # norb
        assertion(isinstance(norb, int) and norb > 0, \
                        'base: number of orbitals (seventh argument) must be an int > 0')
        # ncore
        assertion(isinstance(ncore, int), \
                        'base: number of core orbitals (eighth argument) must be an int')
        # cc_backend
        assertion(isinstance(cc_backend, str), \
                        'base: coupled-cluster backend (cc_backend keyword argument) must be a string')
        assertion(cc_backend in ['pyscf', 'ecc', 'ncc'], \
                        'base: valid coupled-cluster backends (cc_backend keyword argument) are: '
                        'pyscf, ecc and ncc')
        if method == 'ccsdt':
            assertion(cc_backend != 'pyscf', \
                            'base: ccsdt (first argument) is not available with pyscf coupled-cluster backend (cc_backend keyword argument)')
        if method == 'ccsdtq':
            assertion(cc_backend == 'ncc', \
                            'base: ccsdtq (first argument) is not available with pyscf and ecc coupled-cluster backends (cc_backend keyword argument)')
        if mol.spin > 0:
            assertion(cc_backend == 'pyscf', \
                        'base: open-shell systems are not available with ecc and ncc coupled-cluster backends (cc_backend keyword argument)')
        # target
        assertion(isinstance(target, str), \
                        'base: target property (target keyword argument) must be str')
        assertion(target in ['energy', 'dipole'], \
                        'base: valid target properties (keyword argument) with coupled-cluster base methods are: '
                        'energy (energy) and dipole moment (dipole)')
        if cc_backend in ['ecc', 'ncc']:
            assertion(target == 'energy', \
                            'base: calculation of targets (target keyword argument) other than energy are not possible with ecc and ncc coupled-cluster backends (cc_backend keyword argument)')
        if target == 'dipole':
            # hf_dipole
            assertion(isinstance(hf_dipole, np.ndarray), \
                            'base: hartree-fock dipole moment (hf_dipole keyword argument) must be a np.ndarray')
            # gauge_dipole
            assertion(isinstance(gauge_origin, np.ndarray), \
                            'base: gauge origin (gauge_origin keyword argument) must be a np.ndarray')

        # hcore_ao and eri_ao with 8-fold symmetry
        hcore_ao = hf.get_hcore()
        eri_ao = hf._eri

        # remove symmetry from eri_ao
        eri_ao = ao2mo.restore(1, eri_ao, norb)

        # compute hcore
        hcore = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_ao, mo_coeff)

        # eri_mo w/o symmetry
        eri = ao2mo.incore.full(eri_ao, mo_coeff)

        # allocate vhf
        vhf = np.empty((nocc, norb, norb), dtype=np.float64)

        # compute vhf
        for i in range(nocc):
            idx = np.asarray([i])
            vhf[i] = np.einsum('pqrs->rs', eri[idx[:, None], idx, :, :]) * 2.
            vhf[i] -= np.einsum('pqrs->ps', eri[:, idx[:, None], idx, :]) * 2. * .5

        # restore 4-fold symmetry in eri_mo
        eri = ao2mo.restore(4, eri, norb)

        # compute dipole integrals
        if target == 'dipole' and mol.atom:
            dip_ints = dipole_ints(mol, mo_coeff, \
                                   cast(np.ndarray, gauge_origin))
        else:
            dip_ints = None

        # set core and cas spaces
        core_idx, cas_idx = core_cas(nocc, np.arange(ncore, nocc), np.arange(nocc, norb))

        # get cas space h2e
        cas_idx_tril = idx_tril(cas_idx)
        h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

        # get e_core and h1e_cas
        e_core, h1e_cas = e_core_h1e(mol.energy_nuc().item(), hcore, vhf, core_idx, cas_idx)

        # n_elec
        n_elec = nelec(occup, cas_idx)

        # run calc
        res = _cc(mol.spin, occup, core_idx, cas_idx, method, cc_backend, \
                  n_elec, 'can', mol.groupname, orbsym, h1e_cas, h2e_cas, \
                  False, target == 'dipole', 0)

        # collect results
        if target == 'energy':
            base_prop = res['energy']
        elif target == 'dipole':
            base_prop = _dipole(cast(np.ndarray, dip_ints), occup, cas_idx, \
                                res['rdm1'], hf_dipole=cast(np.ndarray, \
                                                            hf_dipole))

        return base_prop


def linear_orbsym(mol: gto.Mole, mo_coeff: np.ndarray) -> np.ndarray:
        """
        returns orbsym in linear point groups for pi pruning
        """
        # mol
        assertion(isinstance(mol, gto.Mole), \
                        'linear orbsym: mol (first argument) must be a gto.Mole object')
        assertion(symm.addons.std_symb(mol.groupname) in ['D2h', 'C2v'], \
                        'linear orbsym: only works for linear D2h and C2v symmetries')
        # mo_coeff
        assertion(isinstance(mo_coeff, np.ndarray), \
                        'linear orbsym: mo coefficients (second argument) must be a np.ndarray')

        # recast mol in parent point group (dooh/coov)
        mol_parent = mol.copy()
        parent_group = 'Dooh' if mol.groupname == 'D2h' else 'Coov'
        mol_parent = mol_parent.build(0, 0, symmetry=parent_group)

        orbsym_parent = symm.label_orb_symm(mol_parent, mol_parent.irrep_id, \
                                            mol_parent.symm_orb, mo_coeff)

        return orbsym_parent
