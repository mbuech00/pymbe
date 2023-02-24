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
from pyscf import ao2mo, gto, scf, fci, cc
from typing import TYPE_CHECKING, cast

from pymbe.expansion import ExpCls, SingleTargetExpCls, MAX_MEM, CONV_TOL, SPIN_TOL
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import (
    RST,
    write_file,
    get_nelec,
    idx_tril,
    get_nhole,
    get_nexc,
    assertion,
)
from pymbe.parallel import mpi_reduce, open_shared_win
from pymbe.results import DIVIDER as DIVIDER_RESULTS, results_plt
from pymbe.interface import mbecc_interface

if TYPE_CHECKING:

    import matplotlib
    from typing import List, Optional, Tuple, Union

    from pymbe.pymbe import MBE


# get logger
logger = logging.getLogger("pymbe_logger")


class EnergyExpCls(SingleTargetExpCls[float]):
    """
    this class contains the pymbe expansion attributes for expansions of the electronic
    energy
    """

    def __init__(self, mbe: MBE) -> None:
        """
        init expansion attributes
        """
        super().__init__(mbe, cast(float, mbe.base_prop))

        # initialize dependent attributes
        self._init_dep_attrs(mbe)

    def prop(self, prop_type: str, nuc_prop: float = 0.0) -> float:
        """
        this function returns the final energy
        """
        return self._prop_conv(
            nuc_prop,
            self.hf_prop
            if prop_type in ["electronic", "total"]
            else self._init_target_inst(0.0, self.norb),
        )[-1]

    def plot_results(
        self, y_axis: str, nuc_prop: float = 0.0
    ) -> matplotlib.figure.Figure:
        """
        this function plots the energy
        """
        return results_plt(
            self._prop_conv(
                nuc_prop,
                self.hf_prop
                if y_axis in ["electronic", "total"]
                else self._init_target_inst(0.0, self.norb),
            ),
            self.min_order,
            self.final_order,
            "x",
            "xkcd:kelly green",
            f"state {self.fci_state_root}",
            "Energy (in au)",
        )

    def _calc_hf_prop(
        self, hcore: np.ndarray, eri: np.ndarray, vhf: np.ndarray
    ) -> float:
        """
        this function calculates the hartree-fock electronic energy
        """
        # add one-electron integrals
        hf_energy = np.sum(self.occup * np.diag(hcore))

        # set closed- and open-shell indices
        cs_idx = np.where(self.occup == 2)[0]
        os_idx = np.where(self.occup == 1)[0]

        # add closed-shell and coupling electron repulsion terms
        hf_energy += np.trace((np.sum(vhf, axis=0))[cs_idx.reshape(-1, 1), cs_idx])

        # check if system is open-shell
        if self.spin > 0:

            # get indices for eris that only include open-shell orbitals
            os_eri_idx = idx_tril(os_idx)

            # retrieve eris of open-shell orbitals and unpack these
            os_eri = ao2mo.restore(
                1, eri[os_eri_idx.reshape(-1, 1), os_eri_idx], os_idx.size
            )

            # add open-shell electron repulsion terms
            hf_energy += 0.5 * (
                np.einsum("pqrr->", os_eri) - np.einsum("pqrp->", os_eri)
            )

        return hf_energy

    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """
        # nelec
        nelec = get_nelec(self.occup, cas_idx)

        # perform main calc
        energy = self._kernel(
            self.method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
        )

        # perform base calc
        if self.base_method is not None:

            energy -= self._kernel(
                self.base_method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
            )

        energy -= self.ref_prop

        return energy, nelec

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
    ) -> float:
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
        solver.conv_tol = CONV_TOL
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
        energy, civec = _fci_interface()

        # multiplicity check
        s, mult = solver.spin_square(civec, cas_idx.size, nelec)

        if np.abs((spin_cas + 1) - mult) > SPIN_TOL:

            # fix spin by applying level shift
            sz = np.abs(nelec[0] - nelec[1]) * 0.5
            solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.0))

            # perform calc
            energy, civec = _fci_interface()

            # verify correct spin
            s, mult = solver.spin_square(civec, cas_idx.size, nelec)
            assertion(
                np.abs((spin_cas + 1) - mult) < SPIN_TOL,
                f"spin contamination for root entry = {self.fci_state_root}\n"
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

        return energy - self.hf_prop

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
        spin_cas = abs(nelec[0] - nelec[1])
        assertion(spin_cas == self.spin, f"cascc wrong spin in space: {cas_idx}")
        singlet = spin_cas == 0

        if self.cc_backend == "pyscf":

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

            # e_corr
            e_cc = ccsd.e_corr

            # calculate (t) correction
            if method == "ccsd(t)":

                # number of holes in cas space
                nhole = get_nhole(nelec, cas_idx)

                # nexc
                nexc = get_nexc(nelec, nhole)

                # ensure that more than two excitations are possible
                if nexc > 2:
                    e_cc += ccsd.ccsd_t()

        elif self.cc_backend in ["ecc", "ncc"]:

            # calculate cc energy
            cc_energy, success = mbecc_interface(
                method,
                self.cc_backend,
                self.orb_type,
                self.point_group,
                self.orbsym[cas_idx],
                h1e,
                h2e,
                nelec,
                higher_amp_extrap,
                self.verbose,
            )

            # convergence check
            assertion(
                success == 1,
                f"MBECC error: no convergence, core_idx = {core_idx}, cas_idx = {cas_idx}",
            )

            # e_corr
            e_cc = cc_energy

        return e_cc

    @staticmethod
    def _write_target_file(prop: float, string: str, order: Optional[int]) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        write_file(np.array(prop, dtype=np.float64), string, order)

    @staticmethod
    def _read_target_file(file: str) -> float:
        """
        this function reads files of attributes with the target type
        """
        return np.load(os.path.join(RST, file)).item()

    def _init_target_inst(self, value: float, *args: int) -> float:
        """
        this function initializes an instance of the target type
        """
        return value

    def _zero_target_arr(self, length: int):
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

    def _open_shared_inc(
        self, window: MPI.Win, n_tuples: int, *args: int
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
    def _screen(
        inc_tup: float, screen: np.ndarray, tup: np.ndarray, screen_func: str
    ) -> np.ndarray:
        """
        this function modifies the screening array
        """
        if screen_func == "max":
            return np.maximum(screen[tup], np.abs(inc_tup))
        elif screen_func == "abs_sum":
            return screen[tup] + np.abs(inc_tup)
        elif screen_func == "sum":
            return screen[tup] + inc_tup
        else:
            raise ValueError

    @staticmethod
    def _total_inc(inc: np.ndarray, mean_inc: float) -> float:
        """
        this function calculates the total increment at a certain order
        """
        return mean_inc

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
            header = (
                f"energy for root {self.fci_state_root} "
                + f"(total increment = {tot_inc:.4e})"
            )
        elif self.target == "excitation":
            header = (
                f"excitation energy for root {self.fci_state_root} "
                + f"(total increment = {tot_inc:.4e})"
            )

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
        this function returns the hf, base and total electronic energy
        """
        return (
            self.hf_prop,
            self.hf_prop + self.base_prop,
            self._prop_conv(hf_prop=self.hf_prop)[-1],
        )

    def _results_prt(self) -> str:
        """
        this function returns the energies table
        """
        string: str = DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"MBE-{self.method.upper()} electronic energy (root = {self.fci_state_root})"
        ).center(73) + "\n"

        string += DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"{'':3}{'MBE order':^14}{'|':1}"
            f"{'electronic energy':^22}{'|':1}"
            f"{'correlation energy':^26}\n"
        )

        string += DIVIDER_RESULTS[:67] + "\n"
        string += (
            f"{'':3}{'ref':^14s}{'|':1}"
            f"{(self.hf_prop + self.ref_prop):^22.6f}{'|':1}"
            f"{self.ref_prop:>19.5e}{'':7}\n"
        )

        string += DIVIDER_RESULTS[:67] + "\n"
        energy = self._prop_conv(hf_prop=self.hf_prop)
        for i, j in enumerate(range(self.min_order, self.final_order + 1)):
            string += (
                f"{'':3}{j:>8d}{'':6}{'|':1}"
                f"{energy[i]:^22.6f}{'|':1}"
                f"{(energy[i] - self.hf_prop):>19.5e}{'':7}\n"
            )

        string += DIVIDER_RESULTS[:67] + "\n"

        return string

    def _prop_conv(self, nuc_prop: float = 0.0, hf_prop: float = 0.0) -> np.ndarray:
        """
        this function returns the total energy
        """
        tot_energy = np.array(self.mbe_tot_prop)
        tot_energy += self.base_prop
        tot_energy += self.ref_prop
        tot_energy += hf_prop
        tot_energy += nuc_prop

        return tot_energy
