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

from pymbe.dipole import DipoleExpCls
from pymbe.kernel import main_kernel, dipole_kernel
from pymbe.tools import get_nelec
from pymbe.results import DIVIDER, results_plt

if TYPE_CHECKING:

    import matplotlib
    from typing import Tuple, Union


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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function calculates the current-order contribution to the increment associated
        with a given tuple
        """
        # nelec
        nelec = get_nelec(self.occup, cas_idx)

        # perform main calc
        res = main_kernel(
            self.method,
            self.cc_backend,
            self.orb_type,
            self.spin,
            self.occup,
            self.target,
            self.fci_state_sym,
            self.point_group,
            self.orbsym,
            self.hf_guess,
            self.fci_state_root,
            self.hf_prop,
            e_core,
            h1e_cas,
            h2e_cas,
            core_idx,
            cas_idx,
            nelec,
            self.verbose,
        )

        res_full = dipole_kernel(
            self.dipole_ints, self.occup, cas_idx, res["t_rdm1"], trans=True
        )

        res_full -= self.ref_prop

        return res_full, nelec

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
