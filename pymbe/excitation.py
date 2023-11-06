#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
excitation energy expansion module
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

from pymbe.energy import EnergyExpCls
from pymbe.results import DIVIDER, results_plt

if TYPE_CHECKING:
    import matplotlib
    from typing import Tuple, Union, List


class ExcExpCls(EnergyExpCls):
    """
    this class contains the pymbe expansion attributes for expansions of the excitation
    energy
    """

    def plot_results(
        self, y_axis: str, hf_prop: float = 0.0
    ) -> matplotlib.figure.Figure:
        """
        this function plots the excitation energy
        """
        return results_plt(
            self._prop_conv(),
            self.min_order,
            self.final_order,
            "x",
            "xkcd:dull blue",
            f"excitation 0 -> {self.fci_state_root}",
            "Excitation energy (in au)",
        )

    def _calc_hf_prop(self, *args: np.ndarray) -> float:
        """
        this function calculates the hartree-fock property
        """
        return 0.0

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        _core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        ref_guess: bool,
    ) -> Tuple[float, List[np.ndarray]]:
        """
        this function returns the results of a fci calculation
        """
        # spin
        spin_cas = abs(nelec[0] - nelec[1])
        if spin_cas != self.spin:
            raise RuntimeError(f"casci wrong spin in space: {cas_idx}")

        # run fci calculation
        energy, civec, _ = self._fci_driver(
            e_core,
            h1e,
            h2e,
            cas_idx,
            nelec,
            spin_cas,
            self.fci_state_sym,
            [0, self.fci_state_root],
            ref_guess,
        )

        return energy[-1] - energy[0], civec

    def _cc_kernel(
        self,
        method: str,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        higher_amp_extrap: bool,
    ) -> float:
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
        this function returns the hf, base and total excitation energy
        """
        return 0.0, 0.0, self._prop_conv()[-1]

    def _results_prt(self) -> str:
        """
        this function returns the excitation energies table
        """
        string: str = DIVIDER[:43] + "\n"
        string += (
            f"MBE excitation energy (roots = 0 > {self.fci_state_root})".center(49)
            + "\n"
        )

        string += DIVIDER[:43] + "\n"
        string += f"{'':3}{'MBE order':^14}{'|':1}{'excitation energy':^25}\n"

        string += DIVIDER[:43] + "\n"
        string += f"{'':3}{'ref':^14s}{'|':1}{self.ref_prop:>18.5e}{'':7}\n"

        string += DIVIDER[:43] + "\n"
        excitation = self._prop_conv()
        for i, j in enumerate(range(self.min_order, self.final_order + 1)):
            string += f"{'':3}{j:>8d}{'':6}{'|':1}{excitation[i]:>18.5e}{'':7}\n"

        string += DIVIDER[:43] + "\n"

        return string

    def _prop_conv(self, nuc_prop: float = 0.0, hf_prop: float = 0.0) -> np.ndarray:
        """
        this function returns the total excitation energy
        """
        tot_exc = np.array(self.mbe_tot_prop)
        tot_exc += self.ref_prop

        return tot_exc
