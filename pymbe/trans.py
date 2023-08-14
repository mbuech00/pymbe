#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
transition dipole moment expansion module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from pyscf import fci
from typing import TYPE_CHECKING

from pymbe.expansion import MAX_MEM, CONV_TOL, SPIN_TOL
from pymbe.dipole import DipoleExpCls
from pymbe.results import DIVIDER, results_plt

if TYPE_CHECKING:
    import matplotlib
    from typing import Tuple, Union, List


class TransExpCls(DipoleExpCls):
    """
    this class contains the pymbe expansion attributes for expansions of the transition
    dipole moment
    """

    def plot_results(
        self, y_axis: str, nuc_prop: np.ndarray = np.zeros(3, dtype=np.float64)
    ) -> matplotlib.figure.Figure:
        """
        this function plots the transition dipole moment
        """
        # array of total MBE transition dipole moment
        trans = self._prop_conv()
        trans_arr = np.empty(trans.shape[0], dtype=np.float64)
        for i in range(trans.shape[0]):
            trans_arr[i] = np.linalg.norm(trans[i, :])

        return results_plt(
            trans_arr,
            self.min_order,
            self.final_order,
            "s",
            "xkcd:dark magenta",
            f"excitation 0 -> {self.fci_state_root}",
            "Transition dipole (in au)",
        )

    def _calc_hf_prop(self, *args: np.ndarray) -> np.ndarray:
        """
        this function calculates the hartree-fock property
        """
        return np.zeros(3, dtype=np.float64)

    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
    ) -> np.ndarray:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """
        # perform main calc
        trans = self._kernel(
            self.method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
        )

        trans -= self.ref_prop

        return trans

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
    ) -> np.ndarray:
        """
        this function returns the results of a fci calculation
        """
        # spin
        spin_cas = abs(nelec[0] - nelec[1])
        if spin_cas != self.spin:
            raise RuntimeError(f"casci wrong spin in space: {cas_idx}")

        # init fci solver
        if spin_cas == 0:
            solver = fci.direct_spin0_symm.FCI()
        else:
            solver = fci.direct_spin1_symm.FCI()

        # settings
        solver.conv_tol = CONV_TOL * 1.0e-04
        solver.lindep = solver.conv_tol * 1.0e-01
        solver.max_memory = MAX_MEM
        solver.max_cycle = 5000
        solver.max_space = 25
        solver.davidson_only = True
        solver.pspace_size = 0
        if self.verbose >= 4:
            solver.verbose = 10
        solver.wfnsym = self.fci_state_sym
        solver.orbsym = self.orbsym[cas_idx]
        solver.nroots = self.fci_state_root + 1

        # hf starting guess
        if self.hf_guess:
            na = fci.cistring.num_strings(cas_idx.size, nelec[0])
            nb = fci.cistring.num_strings(cas_idx.size, nelec[1])
            ci0 = np.zeros((na, nb))
            ci0[0, 0] = 1
        else:
            ci0 = None

        # interface
        def _fci_interface() -> Tuple[List[float], List[np.ndarray]]:
            """
            this function provides an interface to solver.kernel
            """
            # perform calc
            e, c = solver.kernel(h1e, h2e, cas_idx.size, nelec, ecore=e_core, ci0=ci0)

            # collect results
            return [e[0], e[-1]], [c[0], c[-1]]

        # perform calc
        _, civec = _fci_interface()

        # multiplicity check
        for root in range(len(civec)):
            s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)

            if np.abs((spin_cas + 1) - mult) > SPIN_TOL:
                # fix spin by applying level shift
                sz = np.abs(nelec[0] - nelec[1]) * 0.5
                solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.0))

                # perform calc
                _, civec = _fci_interface()

                # verify correct spin
                for root in range(len(civec)):
                    s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
                    raise RuntimeError(
                        f"spin contamination for root entry = {root}\n"
                        f"2*S + 1 = {mult:.6f}\n"
                        f"cas_idx = {cas_idx}\n"
                        f"cas_sym = {self.orbsym[cas_idx]}"
                    )

        # convergence check
        for root in [0, solver.nroots - 1]:
            if not solver.converged[root]:
                raise RuntimeError(
                    f"state {root} not converged\n"
                    f"cas_idx = {cas_idx}\n"
                    f"cas_sym = {self.orbsym[cas_idx]}"
                )

        # init transition rdm1
        t_rdm1 = np.zeros([self.occup.size, self.occup.size], dtype=np.float64)

        # insert correlated subblock
        t_rdm1[cas_idx[:, None], cas_idx] = solver.trans_rdm1(
            np.sign(civec[0][0, 0]) * civec[0],
            np.sign(civec[-1][0, 0]) * civec[-1],
            cas_idx.size,
            nelec,
        )

        return np.einsum("xij,ji->x", self.dipole_ints, t_rdm1)

    def _cc_kernel(
        self,
        method: str,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        higher_amp_extrap: bool,
    ) -> np.ndarray:
        """
        this function returns the results of a cc calculation
        """
        raise NotImplementedError

    def _prop_summ(
        self,
    ) -> Tuple[
        Union[float, np.floating], Union[float, np.floating], Union[float, np.floating]
    ]:
        """
        this function returns the norm of the hf, base and total transition dipole
        moment
        """
        hf_prop = 0.0
        base_prop = 0.0
        mbe_tot_prop = np.linalg.norm(self._prop_conv()[-1, :])

        return hf_prop, base_prop, mbe_tot_prop

    def _results_prt(self) -> str:
        """
        this function returns the transition dipole moments table
        """
        string: str = DIVIDER[:83] + "\n"
        string += (
            f"MBE trans. dipole moment (roots 0 > {self.fci_state_root})".center(87)
            + "\n"
        )

        string += DIVIDER[:83] + "\n"
        string += (
            f"{'':3}{'MBE order':^14}{'|':1}"
            f"{'dipole components (x,y,z)':^43}{'|':1}"
            f"{'dipole moment':^21}\n"
        )

        string += DIVIDER[:83] + "\n"
        tot_ref_trans: np.ndarray = self.ref_prop
        string += (
            f"{'':3}{'ref':^14s}{'|':1}"
            f"{tot_ref_trans[0]:>13.6f}"
            f"{tot_ref_trans[1]:>13.6f}"
            f"{tot_ref_trans[2]:>13.6f}{'':4}{'|':1}"
            f"{np.linalg.norm(tot_ref_trans[:]):>14.6f}{'':7}\n"
        )

        string += DIVIDER[:83] + "\n"
        trans = self._prop_conv()
        for i, j in enumerate(range(self.min_order, self.final_order + 1)):
            string += (
                f"{'':3}{j:>8d}{'':6}{'|':1}"
                f"{trans[i, 0]:>13.6f}"
                f"{trans[i, 1]:>13.6f}"
                f"{trans[i, 2]:>13.6f}{'':4}{'|':1}"
                f"{np.linalg.norm(trans[i, :]):>14.6f}{'':7}\n"
            )

        string += DIVIDER[:83] + "\n"

        return string

    def _prop_conv(
        self,
        nuc_prop: np.ndarray = np.zeros(3, dtype=np.float64),
        hf_prop: np.ndarray = np.zeros(3, dtype=np.float64),
    ) -> np.ndarray:
        """
        this function returns the total transition dipole moment
        """
        tot_trans = np.array(self.mbe_tot_prop)
        tot_trans += self.ref_prop

        return tot_trans
