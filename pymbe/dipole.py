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
from pyscf import gto, scf, fci, cc
from pyscf.cc import (
    ccsd_t_lambda_slow as ccsd_t_lambda,
    ccsd_t_rdm_slow as ccsd_t_rdm,
    uccsd_t_lambda,
    uccsd_t_rdm,
)
from typing import TYPE_CHECKING, cast

from pymbe.expansion import ExpCls, SingleTargetExpCls, MAX_MEM, CONV_TOL, SPIN_TOL
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import RST, write_file, get_nelec, get_nhole, get_nexc, assertion
from pymbe.parallel import mpi_reduce, open_shared_win
from pymbe.results import DIVIDER as DIVIDER_RESULTS, results_plt

if TYPE_CHECKING:
    import matplotlib
    from typing import List, Optional, Tuple, Union

    from pymbe.pymbe import MBE


# get logger
logger = logging.getLogger("pymbe_logger")


class DipoleExpCls(SingleTargetExpCls[np.ndarray]):
    """
    this class contains the pymbe expansion attributes for expansions of the dipole
    moment
    """

    def __init__(self, mbe: MBE) -> None:
        """
        init expansion attributes
        """
        super().__init__(mbe, cast(np.ndarray, mbe.base_prop))

        # additional integrals
        self.dipole_ints = cast(np.ndarray, mbe.dipole_ints)

        # initialize dependent attributes
        self._init_dep_attrs(mbe)

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
            else self._init_target_inst(0.0, self.norb, self.nocc),
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
            else self._init_target_inst(0.0, self.norb, self.nocc),
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
        this function calculates the hartree-fock electronic dipole moment
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
        dipole = self._kernel(
            self.method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
        )

        # perform base calc
        if self.base_method is not None:
            dipole -= self._kernel(
                self.base_method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
            )

        dipole -= self.ref_prop

        return dipole, nelec

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
        assertion(spin_cas == self.spin, f"casci wrong spin in space: {cas_idx}")

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
        if self.verbose >= 3:
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
        def _fci_interface() -> Tuple[float, np.ndarray]:
            """
            this function provides an interface to solver.kernel
            """
            # perform calc
            e, c = solver.kernel(h1e, h2e, cas_idx.size, nelec, ecore=e_core, ci0=ci0)

            # collect results
            if solver.nroots == 1:
                return e, c
            else:
                return e[-1], c[-1]

        # perform calc
        _, civec = _fci_interface()

        # multiplicity check
        s, mult = solver.spin_square(civec, cas_idx.size, nelec)

        if np.abs((spin_cas + 1) - mult) > SPIN_TOL:
            # fix spin by applying level shift
            sz = np.abs(nelec[0] - nelec[1]) * 0.5
            solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.0))

            # perform calc
            _, civec = _fci_interface()

            # verify correct spin
            s, mult = solver.spin_square(civec, cas_idx.size, nelec)
            assertion(
                np.abs((spin_cas + 1) - mult) < SPIN_TOL,
                f"spin contamination for root = {self.fci_state_root}\n"
                f"2*S + 1 = {mult:.6f}\n"
                f"cas_idx = {cas_idx}\n"
                f"cas_sym = {self.orbsym[cas_idx]}",
            )

        # convergence check
        if solver.nroots == 1:
            assertion(
                solver.converged,
                f"state {self.fci_state_root} not converged\n"
                f"cas_idx = {cas_idx}\n"
                f"cas_sym = {self.orbsym[cas_idx]}",
            )

        else:
            assertion(
                solver.converged[-1],
                f"state {self.fci_state_root} not converged\n"
                f"cas_idx = {cas_idx}\n"
                f"cas_sym = {self.orbsym[cas_idx]}",
            )

        # init rdm1
        rdm1 = np.diag(self.occup.astype(np.float64))

        # insert correlated subblock
        rdm1[cas_idx[:, None], cas_idx] = solver.make_rdm1(civec, cas_idx.size, nelec)

        # compute elec_dipole
        elec_dipole = np.einsum("xij,ji->x", self.dipole_ints, rdm1)

        return elec_dipole - self.hf_prop

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
        spin_cas = abs(nelec[0] - nelec[1])
        assertion(spin_cas == self.spin, f"cascc wrong spin in space: {cas_idx}")
        singlet = spin_cas == 0

        # number of holes in cas space
        nhole = get_nhole(nelec, cas_idx)

        # number of possible excitations in cas space
        nexc = get_nexc(nelec, nhole)

        # init ccsd solver
        mol_tmp = gto.Mole(verbose=0)
        mol_tmp._built = True
        mol_tmp.max_memory = MAX_MEM
        mol_tmp.incore_anyway = True

        if singlet:
            hf = scf.RHF(mol_tmp)
        else:
            hf = scf.UHF(mol_tmp)

        hf.get_hcore = lambda *args: h1e
        hf._eri = h2e

        if singlet:
            ccsd = cc.ccsd.CCSD(
                hf, mo_coeff=np.eye(cas_idx.size), mo_occ=self.occup[cas_idx]
            )
        else:
            ccsd = cc.uccsd.UCCSD(
                hf,
                mo_coeff=np.array((np.eye(cas_idx.size), np.eye(cas_idx.size))),
                mo_occ=np.array(
                    (self.occup[cas_idx] > 0.0, self.occup[cas_idx] == 2.0),
                    dtype=np.double,
                ),
            )

        # settings
        ccsd.conv_tol = CONV_TOL
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
        assertion(
            ccsd.converged,
            f"CCSD error: no convergence, core_idx = {core_idx}, cas_idx = {cas_idx}",
        )

        # rdms
        if method == "ccsd" or nexc <= 2:
            ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
            cas_rdm1 = ccsd.make_rdm1()
        elif method == "ccsd(t)":
            if singlet:
                l1, l2 = ccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                cas_rdm1 = ccsd_t_rdm.make_rdm1(
                    ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                )
            else:
                l1, l2 = uccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                cas_rdm1 = uccsd_t_rdm.make_rdm1(
                    ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                )

        if not singlet:
            cas_rdm1 = cas_rdm1[0] + cas_rdm1[1]

        # init rdm1
        rdm1 = np.diag(self.occup.astype(np.float64))

        # insert correlated subblock
        rdm1[cas_idx[:, None], cas_idx] = cas_rdm1

        # compute elec_dipole
        elec_dipole = np.einsum("xij,ji->x", self.dipole_ints, rdm1)

        return elec_dipole - self.hf_prop

    @staticmethod
    def _write_target_file(prop: np.ndarray, string: str, order: int) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        write_file(prop, string, order=order)

    @staticmethod
    def _read_target_file(file: str) -> np.ndarray:
        """
        this function reads files of attributes with the target type
        """
        return np.load(os.path.join(RST, file))

    def _init_target_inst(self, value: float, *args: int) -> np.ndarray:
        """
        this function initializes an instance of the target type
        """
        return np.full(3, value, dtype=np.float64)

    def _zero_target_arr(self, length: int):
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
        self, size: int, allocate: bool, comm: MPI.Comm, *args: int
    ) -> MPI.Win:
        """
        this function allocates a shared increment window
        """
        return MPI.Win.Allocate_shared(
            8 * size * 3 if allocate else 0, 8, comm=comm  # type: ignore
        )

    def _open_shared_inc(
        self, window: MPI.Win, n_tuples: int, *args: int
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
    def _screen(
        inc_tup: np.ndarray, screen: np.ndarray, tup: np.ndarray, screen_func: str
    ) -> np.ndarray:
        """
        this function modifies the screening array
        """
        if screen_func == "sum":
            return screen[tup] + np.sum(np.abs(inc_tup))
        else:
            return np.maximum(screen[tup], np.max(np.abs(inc_tup)))

    @staticmethod
    def _total_inc(inc: List[np.ndarray], mean_inc: np.ndarray) -> np.ndarray:
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
            header = (
                f"dipole moment for root {self.fci_state_root} "
                + f"(total increment = {tot_inc:.4e})"
            )
        elif self.target == "trans":
            header = (
                f"transition dipole moment for excitation 0 -> {self.fci_state_root} "
                + f"(total increment = {tot_inc:.4e})"
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
