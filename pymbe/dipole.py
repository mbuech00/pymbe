#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
dipole moment expansion module
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
from pymbe.kernel import main_kernel, dipole_kernel
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


class DipoleExpCls(SingleTargetExpCls, ExpCls[np.ndarray, np.ndarray, MPI.Win]):
    """
    this class contains the pymbe expansion attributes for expansions of the dipole
    moment
    """

    def __init__(self, mbe: MBE) -> None:
        """
        init expansion attributes
        """
        # integrals
        self.dipole_ints = cast(np.ndarray, mbe.dipole_ints)

        super().__init__(mbe, cast(np.ndarray, mbe.base_prop))

    def prop(
        self, prop_type: str, nuc_prop: np.ndarray = np.zeros(3, dtype=np.float64)
    ) -> np.ndarray:
        """
        this function returns the final dipole moment
        """
        return self._prop_conv(
            nuc_prop,
            self.hf_prop
            if prop_type in ["electronic", "total"]
            else self._init_target_inst(0.0, self.norb),
        )[-1, :]

    def plot_results(
        self, y_axis: str, nuc_prop: np.ndarray = np.zeros(3, dtype=np.float64)
    ) -> matplotlib.figure.Figure:
        """
        this function plots the dipole moment
        """
        # array of total MBE dipole moment
        dipole = self._prop_conv(
            nuc_prop,
            self.hf_prop
            if y_axis in ["electronic", "total"]
            else self._init_target_inst(0.0, self.norb),
        )
        dipole_arr = np.empty(dipole.shape[0], dtype=np.float64)
        for i in range(dipole.shape[0]):
            dipole_arr[i] = np.linalg.norm(dipole[i, :])

        return results_plt(
            dipole_arr,
            self.min_order,
            self.final_order,
            "*",
            "xkcd:salmon",
            f"state {self.fci_state_root}",
            "Dipole moment (in au)",
        )

    def _calc_hf_prop(self, *args: np.ndarray) -> np.ndarray:
        """
        this function calculates the hartree-fock property
        """
        return np.einsum("p,xpp->x", self.occup, self.dipole_ints)

    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        res_full = dipole_kernel(
            self.dipole_ints, self.occup, cas_idx, res["rdm1"], hf_dipole=self.hf_prop
        )

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

            res_base = dipole_kernel(
                self.dipole_ints,
                self.occup,
                cas_idx,
                res["rdm1"],
                hf_dipole=self.hf_prop,
            )

            res_full -= res_base

        res_full -= self.ref_prop

        return res_full, nelec

    @staticmethod
    def _write_target_file(order: Optional[int], prop: np.ndarray, string: str) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        write_file(order, prop, string)

    @staticmethod
    def _read_target_file(file: str) -> np.ndarray:
        """
        this function reads files of attributes with the target type
        """
        return np.load(os.path.join(RST, file))

    @staticmethod
    def _init_target_inst(value: float, norb: int) -> np.ndarray:
        """
        this function initializes an instance of the target type
        """
        return np.full(3, value, dtype=np.float64)

    @staticmethod
    def _zero_target_arr(length: int):
        """
        this function initializes an array of the target type with value zero
        """
        return np.zeros((length, 3), dtype=np.float64)

    @staticmethod
    def _mpi_reduce_target(
        comm: MPI.Comm, values: np.ndarray, op: MPI.Op
    ) -> np.ndarray:
        """
        this function performs a MPI reduce operation on values of the target type
        """
        return mpi_reduce(comm, values, root=0, op=op)

    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm
    ) -> MPI.Win:
        """
        this function allocates a shared increment window
        """
        return MPI.Win.Allocate_shared(
            8 * size * 3 if allocate else 0, 8, comm=comm  # type: ignore
        )

    @staticmethod
    def _open_shared_inc(
        window: MPI.Win, n_tuples: int, idx: Optional[int] = None
    ) -> np.ndarray:
        """
        this function opens a shared increment window
        """
        return open_shared_win(window, np.float64, (n_tuples, 3))

    @staticmethod
    def _flatten_inc(inc_lst: List[np.ndarray], order: int) -> np.ndarray:
        """
        this function flattens the supplied increment arrays
        """
        return np.array(inc_lst, dtype=np.float64).reshape(-1)

    @staticmethod
    def _screen(inc_tup: np.ndarray, screen: np.ndarray, tup: np.ndarray) -> np.ndarray:
        """
        this function modifies the screening array
        """
        return np.maximum(screen[tup], np.max(np.abs(inc_tup)))

    @staticmethod
    def _total_inc(inc: np.ndarray, mean_inc: np.ndarray) -> np.ndarray:
        """
        this function calculates the total increment at a certain order
        """
        return mean_inc.copy()

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: np.ndarray,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> str:
        """
        this function prints mbe debug information
        """
        string = mbe_debug(
            self.point_group, self.orbsym, nelec_tup, self.order, cas_idx, tup
        )
        string += (
            f"      increment for root {self.fci_state_root:d} = ({inc_tup[0]:.4e}, "
            f"{inc_tup[1]:.4e}, {inc_tup[2]:.4e})\n"
        )

        return string

    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for a dipole or transition dipole
        calculation
        """
        # calculate total inc
        if order == self.min_order:
            tot_inc = np.linalg.norm(self.mbe_tot_prop[order - self.min_order])
        else:
            tot_inc = np.linalg.norm(
                self.mbe_tot_prop[order - self.min_order]
            ) - np.linalg.norm(self.mbe_tot_prop[order - self.min_order - 1])

        # set header
        if self.target == "dipole":
            header = f"dipole moment for root {self.fci_state_root} (total increment = {tot_inc:.4e})"
        elif self.target == "trans":
            header = (
                f"transition dipole moment for excitation 0 -> {self.fci_state_root} (total increment = "
                f"{tot_inc:.4e})"
            )

        # set string
        string: str = FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{header:^81}\n"
        string += DIVIDER_OUTPUT + "\n"

        # set components
        string += DIVIDER_OUTPUT
        comp = ("x-component", "y-component", "z-component")

        # loop over x, y, and z
        for k in range(3):

            # set string
            string += f"\n RESULT-{order:d}:{comp[k]:^81}\n"
            string += DIVIDER_OUTPUT + "\n"
            string += (
                f" RESULT-{order:d}:      mean increment     |      "
                "min. abs. increment     |     max. abs. increment\n"
            )
            string += DIVIDER_OUTPUT + "\n"
            string += (
                f" RESULT-{order:d}:     "
                f"{self.mean_inc[order - self.min_order][k]:>13.4e}       |        "
                f"{self.min_inc[order - self.min_order][k]:>13.4e}         |       "
                f"{self.max_inc[order - self.min_order][k]:>13.4e}\n"
            )
            if k < 2:
                string += DIVIDER_OUTPUT

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
        this function returns the norm of the hf, base and total electronic dipole moment
        """
        hf_prop = np.linalg.norm(self.hf_prop)
        base_prop = np.linalg.norm(self.hf_prop + self.base_prop)
        mbe_tot_prop = np.linalg.norm(self._prop_conv(hf_prop=self.hf_prop)[-1, :])

        return hf_prop, base_prop, mbe_tot_prop

    def _results_prt(self) -> str:
        """
        this function returns the dipole moments table
        """
        string: str = DIVIDER_RESULTS[:83] + "\n"
        string += (
            f"MBE electronic dipole moment (root = {self.fci_state_root})".center(87)
            + "\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        string += (
            f"{'':3}{'MBE order':^14}{'|':1}"
            f"{'elec. dipole components (x,y,z)':^43}{'|':1}"
            f"{'elec. dipole moment':^21}\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        tot_ref_dipole: np.ndarray = -(self.hf_prop + self.base_prop + self.ref_prop)
        string += (
            f"{'':3}{'ref':^14s}{'|':1}{tot_ref_dipole[0]:>13.6f}"
            f"{tot_ref_dipole[1]:>13.6f}{tot_ref_dipole[2]:>13.6f}{'':4}{'|':1}"
            f"{np.linalg.norm(tot_ref_dipole):>14.6f}{'':7}\n"
        )

        string += DIVIDER_RESULTS[:83] + "\n"
        dipole = self._prop_conv(hf_prop=self.hf_prop)
        for i, j in enumerate(range(self.min_order, self.final_order + 1)):
            string += (
                f"{'':3}{j:>8d}{'':6}{'|':1}{dipole[i, 0]:>13.6f}"
                f"{dipole[i, 1]:>13.6f}{dipole[i, 2]:>13.6f}{'':4}{'|':1}"
                f"{np.linalg.norm(dipole[i, :]):>14.6f}{'':7}\n"
            )

        string += DIVIDER_RESULTS[:83] + "\n"

        return string

    def _prop_conv(
        self,
        nuc_prop: np.ndarray = np.zeros(3, dtype=np.float64),
        hf_prop: np.ndarray = np.zeros(3, dtype=np.float64),
    ) -> np.ndarray:
        """
        this function returns the total dipole moment
        """
        tot_dipole = -np.array(self.mbe_tot_prop)
        tot_dipole -= self.base_prop
        tot_dipole -= self.ref_prop
        tot_dipole -= hf_prop
        tot_dipole += nuc_prop

        return tot_dipole
