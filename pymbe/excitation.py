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
from pyscf import fci
from typing import TYPE_CHECKING

from pymbe.expansion import MAX_MEM, CONV_TOL, SPIN_TOL
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
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
    ) -> float:
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
        solver.conv_tol = CONV_TOL
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
        energy, civec = _fci_interface()

        # multiplicity check
        for root in range(len(civec)):
            s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)

            if np.abs((spin_cas + 1) - mult) > SPIN_TOL:
                # fix spin by applying level shift
                sz = np.abs(nelec[0] - nelec[1]) * 0.5
                solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.0))

                # perform calc
                energy, civec = _fci_interface()

                # verify correct spin
                for root in range(len(civec)):
                    s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
                    if np.abs((spin_cas + 1) - mult) > SPIN_TOL:
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

        return energy[-1] - energy[0]

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
