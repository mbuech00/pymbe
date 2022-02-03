#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
kernel module
"""

from __future__ import annotations

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, scf, cc, fci
from pyscf.cc import ccsd_t
from pyscf.cc import uccsd_t
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import uccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import uccsd_t_rdm
from typing import TYPE_CHECKING, cast

from pymbe.tools import assertion, nexc
from pymbe.interface import mbecc_interface

if TYPE_CHECKING:

    from typing import Tuple, Dict, Union, Any, Optional, List


MAX_MEM = 1e10
CONV_TOL = 1.e-10
SPIN_TOL = 1.e-05


def e_core_h1e(nuc_energy: float, hcore: np.ndarray, vhf: np.ndarray, \
               core_idx: np.ndarray, cas_idx: np.ndarray) -> Tuple[float, \
                                                                   np.ndarray]:
        """
        this function returns core energy and cas space 1e integrals
        """
        # init core energy
        e_core = nuc_energy

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


def main(method: str, cc_backend: str, solver: str, orb_type: str, spin: int, \
         occup: np.ndarray, target_mbe: str, state_wfnsym: int, \
         point_group: str, orbsym: np.ndarray, hf_guess: bool, \
         state_root: int, hf_prop: np.ndarray, e_core: float, h1e: np.ndarray, \
         h2e: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray, \
         n_elec: Tuple[int, int], verbose: int, \
         dipole_ints: Optional[np.ndarray], \
         higher_amp_extrap: bool = False) -> Union[float, np.ndarray]:
        """
        this function return the result property from a given method
        """
        if method in ['ccsd', 'ccsd(t)', 'ccsdt', 'ccsdtq']:

            res_tmp = _cc(spin, occup, core_idx, cas_idx, method, cc_backend, \
                          n_elec, orb_type, point_group, orbsym, h1e, h2e, \
                          higher_amp_extrap, target_mbe == 'dipole', verbose)

        elif method == 'fci':

            res_tmp = _fci(solver, spin, target_mbe, state_wfnsym, orbsym, \
                           hf_guess, state_root, hf_prop, e_core, h1e, h2e, \
                           occup, cas_idx, n_elec, verbose)

        if target_mbe in ['energy', 'excitation']:

            res = res_tmp[target_mbe]

        elif target_mbe == 'dipole':

            res = _dipole(cast(np.ndarray, dipole_ints), occup, cas_idx, \
                          res_tmp['rdm1'], hf_dipole=hf_prop)

        elif target_mbe == 'trans':

            res = _trans(cast(np.ndarray, dipole_ints), occup, cas_idx, \
                         res_tmp['t_rdm1'], res_tmp['hf_weight'][0], \
                         res_tmp['hf_weight'][1])

        return res


def _dipole(dipole_ints: np.ndarray, occup: np.ndarray, cas_idx: np.ndarray, \
            cas_rdm1: np.ndarray, \
            hf_dipole: np.ndarray = np.zeros(3, dtype=np.float64), \
            trans: bool = False) -> np.ndarray:
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


def _trans(dipole_ints: np.ndarray, occup: np.ndarray, cas_idx: np.ndarray, \
           cas_rdm1: np.ndarray, hf_weight_gs: float, \
           hf_weight_ex: float) -> np.ndarray:
        """
        this function returns an electronic transition dipole moment
        """
        return _dipole(dipole_ints, occup, cas_idx, cas_rdm1, trans=True) \
                        * np.sign(hf_weight_gs) * np.sign(hf_weight_ex)


def _fci(solver_type: str, spin: int, target_mbe: str, wfnsym: int, \
         orbsym: np.ndarray, hf_guess: bool, root: int, \
         hf_prop: np.ndarray, e_core: float, h1e: np.ndarray, h2e: np.ndarray, \
         occup: np.ndarray, cas_idx: np.ndarray, n_elec: Tuple[int, int], \
         verbose: int) -> Dict[str, Any]:
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
        if verbose >= 3:
            solver.verbose = 10
        solver.wfnsym = wfnsym
        solver.orbsym = orbsym[cas_idx]
        solver.nroots = root + 1

        # hf starting guess
        if hf_guess:
            na, nb = fci.cistring.num_strings(cas_idx.size, n_elec[0]), \
                     fci.cistring.num_strings(cas_idx.size, n_elec[1])
            ci0 = np.zeros((na, nb))
            ci0[0, 0] = 1
        else:
            ci0 = None

        # interface
        def _fci_kernel() -> Tuple[List[float], List[np.ndarray]]:
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

                # fix spin by applying level shift
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
        res: Dict[str, Union[int, float, np.ndarray]] = {}
        if target_mbe == 'energy':
            res['energy'] = energy[-1] - hf_prop.item()
        elif target_mbe == 'excitation':
            res['excitation'] = energy[-1] - energy[0]
        elif target_mbe == 'dipole':
            res['rdm1'] = solver.make_rdm1(civec[-1], cas_idx.size, n_elec)
        elif target_mbe == 'trans':
            res['t_rdm1'] = solver.trans_rdm1(civec[0], civec[-1], cas_idx.size, n_elec)
            res['hf_weight'] = np.array([civec[i][0, 0] for i in range(2)])

        return res


def _cc(spin: int, occup: np.ndarray, core_idx: np.ndarray, \
        cas_idx: np.ndarray, method: str, cc_backend: str, \
        n_elec: Tuple[int, int], orb_type: str, point_group: str, \
        orbsym: np.ndarray, h1e: np.ndarray, h2e: np.ndarray, \
        higher_amp_extrap: bool, rdm1: bool, verbose: int) -> Dict[str, Any]:
        """
        this function returns the results of a ccsd / ccsd(t) calculation
        """
        spin_cas = np.count_nonzero(occup[cas_idx] == 1.)
        assertion(spin_cas == spin, 'cascc wrong spin in space: {:}'.format(cas_idx))
        singlet = spin_cas == 0

        if cc_backend == 'pyscf':

            # init ccsd solver
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

                # n_exc
                n_exc = nexc(n_elec, cas_idx)

                # ensure that more than two excitations are possible
                if n_exc > 2:
                    if singlet:
                        e_cc += ccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)
                    else:
                        e_cc += uccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)

        elif (cc_backend in ['ecc', 'ncc']):

            # calculate cc energy
            cc_energy, success = mbecc_interface(method, cc_backend, orb_type, \
                                                 point_group, orbsym[cas_idx], \
                                                 h1e, h2e, n_elec, \
                                                 higher_amp_extrap, verbose)

            # convergence check
            assertion(success == 1, \
            'MBECC error: no convergence, core_idx = {:} , cas_idx = {:}'.format(core_idx, cas_idx))

            # e_corr
            e_cc = cc_energy

        # collect results
        res: Dict[str, Union[float, np.ndarray]] = {'energy': e_cc}

        # rdm1
        if rdm1:
            if method == 'ccsd' or n_exc <= 2:
                ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
                rdm1s = ccsd.make_rdm1()
            elif method == 'ccsd(t)':
                if singlet:
                    l1, l2 = ccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)[1:]
                    rdm1s = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
                else:
                    l1, l2 = uccsd_t_lambda.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)[1:]
                    rdm1s = uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
            if singlet:
                res['rdm1'] = rdm1s
            else: 
                res['rdm1'] = rdm1s[0] + rdm1s[1]

        return res
