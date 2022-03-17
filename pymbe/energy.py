#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
energy expansion module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import os
import logging
import numpy as np
from mpi4py import MPI
from typing import TYPE_CHECKING, cast

from pymbe.expansion import ExpCls, SingleTargetExpCls
from pymbe.kernel import main_kernel
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import RST, write_file, get_nelec
from pymbe.parallel import mpi_reduce, open_shared_win
from pymbe.results import DIVIDER as DIVIDER_RESULTS, results_plt

if TYPE_CHECKING:

    import matplotlib
    from typing import List, Optional, Tuple, Union

    from pymbe.pymbe import MBE


# get logger
logger = logging.getLogger("pymbe_logger")


class EnergyExpCls(SingleTargetExpCls, ExpCls[float, np.ndarray, MPI.Win]):
    """
    this class contains the pymbe expansion attributes for expansions of the electronic
    energy
    """

    def __init__(self, mbe: MBE) -> None:
        """
        init expansion attributes
        """
        super().__init__(
            mbe,
            cast(float, mbe.hf_prop),
            cast(float, mbe.ref_prop),
            cast(float, mbe.base_prop),
        )

    def tot_prop(self) -> float:
        """
        this function returns the final total energy
        """
        return self._prop_conv()[-1]

    def plot_results(self) -> matplotlib.figure.Figure:
        """
        this function plots the energy
        """
        return results_plt(
            self._prop_conv(),
            self.min_order,
            self.final_order,
            "x",
            "xkcd:kelly green",
            f"state {self.fci_state_root}",
            "Energy (in au)",
        )

    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
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

        res_full = res[self.target]

        # perform base calc
        if self.base_method is not None:

            res = main_kernel(
                self.base_method,
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

            res_base = res["energy"]

            res_full -= res_base

        res_full -= self.ref_prop

        return res_full, nelec

    @staticmethod
    def _write_target_file(order: Optional[int], prop: float, string: str) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        write_file(order, np.array(prop, dtype=np.float64), string)

    @staticmethod
    def _read_target_file(file: str) -> float:
        """
        this function reads files of attributes with the target type
        """
        return np.load(os.path.join(RST, file)).item()

    @staticmethod
    def _init_target_inst(value: float, norb: int) -> float:
        """
        this function initializes an instance of the target type
        """
        return value

    @staticmethod
    def _zero_target_arr(length: int):
        """
        this function initializes an array of the target type with value zero
        """
        return np.zeros(length, dtype=np.float64)

    @staticmethod
    def _mpi_reduce_target(comm: MPI.Comm, values: float, op: MPI.Op) -> float:
        """
        this function performs a MPI reduce operation on values of the target type
        """
        return mpi_reduce(
            comm, np.array(values, dtype=np.float64), root=0, op=op
        ).item()

    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm
    ) -> MPI.Win:
        """
        this function allocates a shared increment window
        """
        return MPI.Win.Allocate_shared(
            8 * size if allocate else 0, 8, comm=comm  # type: ignore
        )

    @staticmethod
    def _open_shared_inc(
        window: MPI.Win, n_tuples: int, idx: Optional[int] = None
    ) -> np.ndarray:
        """
        this function opens a shared increment window
        """
        return open_shared_win(window, np.float64, (n_tuples,))

    @staticmethod
    def _flatten_inc(inc_lst: List[np.ndarray], order: int) -> np.ndarray:
        """
        this function flattens the supplied increment arrays
        """
        return np.array(inc_lst, dtype=np.float64)

    @staticmethod
    def _screen(inc_tup: float, screen: np.ndarray, tup: np.ndarray) -> np.ndarray:
        """
        this function modifies the screening array
        """
        return np.maximum(screen[tup], np.abs(inc_tup))

    @staticmethod
    def _total_inc(inc: np.ndarray, mean_inc: float) -> float:
        """
        this function calculates the total increment at a certain order
        """
        return np.sum(inc, axis=0)

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: float,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> str:
        """
        this function prints mbe debug information
        """
        string = mbe_debug(
            self.point_group, self.orbsym, nelec_tup, self.order, cas_idx, tup
        )
        string += f"      increment for root {self.fci_state_root:d} = {inc_tup:.4e}\n"

        return string

    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for an energy or excitation energy
        calculation
        """
        # calculate total inc
        if order == self.min_order:
            tot_inc = self.mbe_tot_prop[order - self.min_order]
        else:
            tot_inc = (
                self.mbe_tot_prop[order - self.min_order]
                - self.mbe_tot_prop[order - self.min_order - 1]
            )

        # set header
        if self.target == "energy":
            header = f"energy for root {self.fci_state_root} (total increment = {tot_inc:.4e})"
        elif self.target == "excitation":
            header = f"excitation energy for root {self.fci_state_root} (total increment = {tot_inc:.4e})"

        # set string
        string: str = FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{header:^81}\n"
        string += DIVIDER_OUTPUT + "\n"

        # set string
        string += DIVIDER_OUTPUT + "\n"
        string += (
            f" RESULT-{order:d}:      mean increment     |      "
            "min. abs. increment     |     max. abs. increment\n"
        )
        string += DIVIDER_OUTPUT + "\n"
        string += (
            f" RESULT-{order:d}:     "
            f"{self.mean_inc[order - self.min_order]:>13.4e}       |        "
            f"{self.min_inc[order - self.min_order]:>13.4e}         |       "
            f"{self.max_inc[order - self.min_order]:>13.4e}\n"
        )

        # set string
        string += DIVIDER_OUTPUT + "\n"
        string += FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT

        return string

    def _prop_summ(
        self,
    ) -> Tuple[
        Union[float, np.floating], Union[float, np.floating], Union[float, np.floating]
    ]:
        """
        this function returns the hf, base and total energy
        """
        return self.hf_prop, self.hf_prop + self.base_prop, self._prop_conv()[-1]

    def _results_prt(self) -> str:
        """
        this function returns the energies table
        """
        string: str = DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"MBE-{self.method.upper()} energy (root = {self.fci_state_root})"
        ).center(73) + "\n"

        string += DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"{'':3}{'MBE order':^14}{'|':1}"
            f"{'total energy':^22}{'|':1}"
            f"{'correlation energy':^26}\n"
        )

        string += DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"{'':3}{'ref':^14s}{'|':1}"
            f"{(self.hf_prop + self.ref_prop):^22.6f}{'|':1}"
            f"{self.ref_prop:>19.5e}{'':7}\n"
        )

        string += DIVIDER_RESULTS[:67] + "\n"
        energy = self._prop_conv()
        for i, j in enumerate(range(self.min_order, self.final_order + 1)):
            string += (
                f"{'':3}{j:>8d}{'':6}{'|':1}"
                f"{energy[i]:^22.6f}{'|':1}"
                f"{(energy[i] - self.hf_prop):>19.5e}{'':7}\n"
            )

        string += DIVIDER_RESULTS[:67] + "\n"

        return string

    def _prop_conv(self) -> np.ndarray:
        """
        this function returns the total energy
        """
        tot_energy = np.array(self.mbe_tot_prop)
        tot_energy += self.hf_prop
        tot_energy += self.base_prop
        tot_energy += self.ref_prop

        return tot_energy
