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
from typing import TYPE_CHECKING

from pymbe.expansion import CONV_TOL
from pymbe.output import DIVIDER as DIVIDER_OUTPUT
from pymbe.dipole import DipoleExpCls
from pymbe.results import DIVIDER as DIVIDER_RESULTS, results_plt

if TYPE_CHECKING:
    import matplotlib
    from typing import Tuple, Union, List


class TransExpCls(DipoleExpCls):
    """
    this class contains the pymbe expansion attributes for expansions of the transition
    dipole moment
    """

    def prop(self, *args: Union[str, np.ndarray]) -> np.ndarray:
        """
        this function returns the final transition dipole moment
        """
        if len(self.mbe_tot_prop) > 0:
            tot_trans = self.mbe_tot_prop[-1].copy()
        else:
            tot_trans = self._init_target_inst(0.0)
        tot_trans += self.base_prop
        tot_trans += self.ref_prop

        return tot_trans

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

    def _ref_results(self, ref_prop: np.ndarray) -> str:
        """
        this function prints reference space results for a target calculation
        """
        header = (
            f"reference space transition dipole moment for excitation 0 -> "
            f"{self.fci_state_root}"
        )
        trans = f"(total increment = {np.linalg.norm(ref_prop):.4e})"

        string = DIVIDER_OUTPUT + "\n"
        string += f" RESULT: {header:^80}\n"
        string += f" RESULT: {trans:^80}\n"
        string += DIVIDER_OUTPUT

        return string

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
        _core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        ref_guess: bool,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        this function returns the results of a fci calculation
        """
        # spin
        spin_cas = abs(nelec[0] - nelec[1])
        if spin_cas != self.spin:
            raise RuntimeError(f"casci wrong spin in space: {cas_idx}")

        # run fci calculation
        _, civec, solver = self._fci_driver(
            e_core,
            h1e,
            h2e,
            cas_idx,
            nelec,
            spin_cas,
            self.fci_state_sym,
            [0, self.fci_state_root],
            ref_guess,
            conv_tol=CONV_TOL * 1.0e-04,
        )

        # init transition rdm1
        t_rdm1 = np.zeros([self.occup.size, self.occup.size], dtype=np.float64)

        # insert correlated subblock
        t_rdm1[cas_idx[:, None], cas_idx] = solver.trans_rdm1(
            np.sign(civec[0][0, 0]) * civec[0],
            np.sign(civec[1][0, 0]) * civec[1],
            cas_idx.size,
            nelec,
        )

        return np.einsum("xij,ji->x", self.dipole_ints, t_rdm1), civec

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
        string: str = DIVIDER_RESULTS[:83] + "\n"
        string += (
            f"MBE trans. dipole moment (roots 0 > {self.fci_state_root})".center(87)
            + "\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        string += (
            f"{'':3}{'MBE order':^14}{'|':1}"
            f"{'dipole components (x,y,z)':^43}{'|':1}"
            f"{'dipole moment':^21}\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        tot_ref_trans: np.ndarray = self.ref_prop
        string += (
            f"{'':3}{'ref':^14s}{'|':1}"
            f"{tot_ref_trans[0]:>13.6f}"
            f"{tot_ref_trans[1]:>13.6f}"
            f"{tot_ref_trans[2]:>13.6f}{'':4}{'|':1}"
            f"{np.linalg.norm(tot_ref_trans[:]):>14.6f}{'':7}\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        trans = self._prop_conv()
        for i, j in enumerate(range(self.min_order, self.final_order + 1)):
            string += (
                f"{'':3}{j:>8d}{'':6}{'|':1}"
                f"{trans[i, 0]:>13.6f}"
                f"{trans[i, 1]:>13.6f}"
                f"{trans[i, 2]:>13.6f}{'':4}{'|':1}"
                f"{np.linalg.norm(trans[i, :]):>14.6f}{'':7}\n"
            )

        string += DIVIDER_RESULTS[:83] + "\n"

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
