#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
generalized Fock matrix expansion module
(occ-general and active-general)
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
from typing import TYPE_CHECKING, cast

from pymbe.expansion import ExpCls, SingleTargetExpCls, MAX_MEM, CONV_TOL, SPIN_TOL
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import RST, get_nelec, write_file, get_nhole, get_nexc, assertion
from pymbe.parallel import mpi_reduce, open_shared_win
from pymbe import direct_spin0_symm, direct_spin1_symm

if TYPE_CHECKING:

    import matplotlib
    from typing import List, Optional, Union, Tuple

    from pymbe.pymbe import MBE


# get logger
logger = logging.getLogger("pymbe_logger")


class GenFockExpCls(SingleTargetExpCls, ExpCls[np.ndarray, np.ndarray, MPI.Win]):
    """
    this class contains the pymbe expansion attributes for the generalized Fock matrix
    elements
    """

    def __init__(self, mbe: MBE) -> None:
        """
        init expansion attributes
        """
        super().__init__(mbe, cast(np.ndarray, mbe.base_prop))

        # additional system parameters
        self.full_norb = cast(int, mbe.full_norb)
        self.full_nocc = cast(int, mbe.full_nocc)
        self.full_nvirt = self.full_norb - self.norb - self.full_nocc

        # additional integrals
        self.inact_fock = cast(np.ndarray, mbe.inact_fock)
        self.eri_goaa = cast(np.ndarray, mbe.eri_goaa)
        self.eri_gaao = cast(np.ndarray, mbe.eri_gaao)
        self.eri_gaaa = cast(np.ndarray, mbe.eri_gaaa)

        # additional settings
        self.no_singles = cast(bool, mbe.no_singles)

        # initialize dependent attributes
        self._init_dep_attrs(mbe)

    def prop(self, prop_type: str) -> np.ndarray:
        """
        this function returns the final generalized Fock matrix elements
        """
        tot_gen_fock = self.mbe_tot_prop[-1].copy()
        tot_gen_fock += self.base_prop
        tot_gen_fock += self.ref_prop
        tot_gen_fock += (
            self.hf_prop
            if prop_type in ["electronic", "total"]
            else self._init_target_inst(0.0, self.norb)
        )

        return tot_gen_fock

    def plot_results(self, *args: str) -> matplotlib.figure.Figure:
        """
        this function plots the 1- and 2-particle reduced density matrices
        """
        raise NotImplementedError

    def _calc_hf_prop(self, *args: np.ndarray) -> np.ndarray:
        """
        this function calculates the hartree-fock generalized fock matrix
        """
        # initialize generalized Fock matrix
        gen_fock = np.empty(
            (self.full_nocc + self.norb, self.full_norb), dtype=np.float64
        )

        # get alpha and beta occupation vectors
        occup_a = self.occup.copy()
        occup_a[occup_a > 0.0] = 1.0
        occup_b = self.occup - occup_a

        # calculate general-occupied active Fock matrix elements
        eri_piuu = np.einsum("piuu->piu", self.eri_goaa)
        eri_puui = np.einsum("puui->piu", self.eri_gaao)
        asymm_eri_piuu = eri_piuu - 0.5 * eri_puui
        act_fock_pi = np.einsum("u,piu->pi", self.occup, asymm_eri_piuu)

        # calculate occupied-general generalized Fock matrix elements
        gen_fock[: self.full_nocc] = (
            2 * (self.inact_fock[:, : self.full_nocc] + act_fock_pi).transpose()
        )

        # calculate occupied-active generalized Fock matrix elements
        gen_fock[self.full_nocc :] = (
            np.einsum("u,pu->up", self.occup, self.inact_fock[:, self.full_nocc :])
            + np.einsum("u,v,puvv->up", self.occup, self.occup, self.eri_gaaa)
            - (
                np.einsum("u,v,pvvu->up", occup_a, occup_a, self.eri_gaaa)
                - np.einsum("u,v,pvvu->up", occup_b, occup_b, self.eri_gaaa)
            )
        )

        return gen_fock

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
        gen_fock = self._kernel(
            self.method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
        )

        # perform base calc
        if self.base_method is not None:
            gen_fock -= self._kernel(
                self.base_method, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec
            )

        gen_fock -= self.ref_prop

        return gen_fock, nelec

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
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
        if not self.no_singles:
            if spin_cas == 0:
                solver = fci.direct_spin0_symm.FCI()
            else:
                solver = fci.direct_spin1_symm.FCI()
        else:
            if spin_cas == 0:
                solver = direct_spin0_symm.FCISolver()
            else:
                solver = direct_spin1_symm.FCISolver()

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
        def _fci_interface() -> Tuple[List[float], List[np.ndarray]]:
            """
            this function provides an interface to solver.kernel
            """
            # perform calc
            e, c = solver.kernel(h1e, h2e, cas_idx.size, nelec, ecore=e_core, ci0=ci0)

            # collect results
            if solver.nroots == 1:
                return [e], [c]
            else:
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
                    assertion(
                        np.abs((spin_cas + 1) - mult) < SPIN_TOL,
                        f"spin contamination for root entry = {root}\n"
                        f"2*S + 1 = {mult:.6f}\n"
                        f"cas_idx = {cas_idx}\n"
                        f"cas_sym = {self.orbsym[cas_idx]}",
                    )

        # convergence check
        if solver.nroots == 1:

            assertion(
                solver.converged,
                f"state {root} not converged\n"
                f"cas_idx = {cas_idx}\n"
                f"cas_sym = {self.orbsym[cas_idx]}",
            )

        else:

            assertion(
                solver.converged[-1],
                f"state {root} not converged\n"
                f"cas_idx = {cas_idx}\n"
                f"cas_sym = {self.orbsym[cas_idx]}",
            )

        rdm1, rdm2 = solver.make_rdm12(civec[-1], cas_idx.size, nelec)

        # calculate generalized Fock matrix elements
        gen_fock = self._calc_gen_fock(cas_idx, rdm1, rdm2)

        return gen_fock - self.hf_prop

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
            rdm1 = ccsd.make_rdm1()
            rdm2 = ccsd.make_rdm2()
        elif method == "ccsd(t)":
            if singlet:
                l1, l2 = cc.ccsd_t_lambda_slow.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = cc.ccsd_t_rdm_slow.make_rdm1(
                    ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                )
                rdm2 = cc.ccsd_t_rdm_slow.make_rdm2(
                    ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                )
            else:
                l1, l2 = cc.uccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = cc.uccsd_t_rdm.make_rdm1(
                    ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                )
                rdm2 = cc.uccsd_t_rdm.make_rdm2(
                    ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                )
        if not singlet:
            rdm1 = rdm1[0] + rdm1[1]
            rdm2 = rdm2[0] + rdm2[1] + rdm2[2] + rdm2[3]

        # calculate generalized Fock matrix elements
        gen_fock = self._calc_gen_fock(cas_idx, rdm1, rdm2)

        return gen_fock - self.hf_prop

    def _calc_gen_fock(self, cas_idx: np.ndarray, rdm1: np.ndarray, rdm2: np.ndarray):
        """
        this function calculates generalized Fock matrix elements from 1- and 2-particle
        reduced density matrices of a given CAS
        this could also be done by viewing all orbitals in CAS space as active and
        padding the 1-RDMs and 2-RDMs with HF values. Is this faster?
        """
        # occupied orbitals inside CAS space but outside iCAS
        occ_idx = np.setdiff1d(np.arange(self.nocc), cas_idx)

        # initialize inactive Fock matrix for all occupied orbitals outside iCAS, these
        # now include all contributions from occupied orbitals outside CAS
        inact_fock_pi = np.concatenate(
            (
                self.inact_fock[:, : self.full_nocc],
                self.inact_fock[:, self.full_nocc + occ_idx],
            ),
            axis=1,
        )

        # add occupied orbitals inside CAS but outside iCAS to inactive Fock matrix
        # elements for occupied orbitals outside CAS
        eri_pijj = np.einsum(
            "pijj->pij", self.eri_goaa[:, :, occ_idx.reshape(-1, 1), occ_idx]
        )
        eri_pjji = np.einsum(
            "pjji->pij", self.eri_gaao[:, occ_idx.reshape(-1, 1), occ_idx, :]
        )
        asymm_eri_pijj = eri_pijj - 0.5 * eri_pjji
        inact_fock_pi[:, : self.full_nocc] += 2 * np.einsum("pij->pi", asymm_eri_pijj)

        # add occupied orbitals inside CAS but outside iCAS to inactive Fock matrix
        # elements for occupied orbitals inside CAS but outside iCAS
        eri_pijj = np.einsum(
            "pijj->pij",
            self.eri_gaaa[
                :, occ_idx.reshape(-1, 1, 1), occ_idx.reshape(-1, 1), occ_idx
            ],
        )
        eri_pjji = np.einsum(
            "pjji->pij",
            self.eri_gaaa[
                :, occ_idx.reshape(-1, 1, 1), occ_idx.reshape(-1, 1), occ_idx
            ],
        )
        asymm_eri_pijj = eri_pijj - 0.5 * eri_pjji
        inact_fock_pi[:, self.full_nocc :] += 2 * np.einsum("pij->pi", asymm_eri_pijj)

        # calculate active Fock matrix elements
        eri_piuv = np.concatenate(
            (
                self.eri_goaa[:, :, cas_idx.reshape(-1, 1), cas_idx],
                self.eri_gaaa[
                    :, occ_idx.reshape(-1, 1, 1), cas_idx.reshape(-1, 1), cas_idx
                ],
            ),
            axis=1,
        )
        eri_pvui = np.concatenate(
            (
                self.eri_gaao[:, cas_idx.reshape(-1, 1), cas_idx, :],
                self.eri_gaaa[
                    :, cas_idx.reshape(-1, 1, 1), cas_idx.reshape(-1, 1), occ_idx
                ],
            ),
            axis=3,
        )
        asymm_eri_piuv = eri_piuv - 0.5 * eri_pvui.transpose(0, 3, 2, 1)
        act_fock_pi = np.einsum("uv,piuv->pi", rdm1, asymm_eri_piuv)

        # calculate occupied-general generalized Fock matrix elements
        gen_fock_ip = 2 * (inact_fock_pi + act_fock_pi).transpose()

        # initialize generalized Fock matrix
        # the size of this matrix could be reduced because contributions from virtuals
        # inside CAS but outside iCAS are zero
        gen_fock = np.zeros(
            (self.full_nocc + self.norb, self.full_norb), dtype=np.float64
        )
        gen_fock[: self.full_nocc] = gen_fock_ip[: self.full_nocc]
        gen_fock[self.full_nocc + occ_idx] = gen_fock_ip[self.full_nocc :]

        # initialize inactive Fock matrix for all active orbitals inside iCAS, these
        # now include all contributions from occupied orbitals outside CAS
        inact_fock_pv = self.inact_fock[:, self.full_nocc + cas_idx]

        # add occupied orbitals inside CAS but outside iCAS to inactive Fock matrix
        # elements for active orbitals inside iCAS
        eri_pxjj = np.einsum(
            "pxjj->pxj",
            self.eri_gaaa[
                :, cas_idx.reshape(-1, 1, 1), occ_idx.reshape(-1, 1), occ_idx
            ],
        )
        eri_pjjx = np.einsum(
            "pjjx->pxj",
            self.eri_gaaa[
                :, occ_idx.reshape(-1, 1, 1), occ_idx.reshape(-1, 1), cas_idx
            ],
        )
        asymm_eri_pxjj = eri_pxjj - 0.5 * eri_pjjx
        inact_fock_pv += 2 * np.einsum("pxj->px", asymm_eri_pxjj)

        # calculate active-general generalized Fock matrix elements
        gen_fock[self.full_nocc + cas_idx] = np.einsum(
            "uv,pv->up", rdm1, inact_fock_pv
        ) + np.einsum(
            "uvxy,pvxy->up",
            rdm2,
            self.eri_gaaa[
                :, cas_idx.reshape(-1, 1, 1), cas_idx.reshape(-1, 1), cas_idx
            ],
        )

        return gen_fock

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

    def _init_target_inst(self, value: float, *args: int) -> np.ndarray:
        """
        this function initializes an instance of the target type
        """
        return np.full(
            (self.full_nocc + self.norb, self.full_norb), value, dtype=np.float64
        )

    def _zero_target_arr(self, length: int):
        """
        this function initializes an array of the target type with value zero
        """
        return np.zeros(
            (length, self.full_nocc + self.norb, self.full_norb), dtype=np.float64
        )

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
            8 * size * (self.full_nocc + self.norb) * self.full_norb if allocate else 0,
            8,
            comm=comm,  # type: ignore
        )

    def _open_shared_inc(
        self, window: MPI.Win, n_tuples: int, *args: int
    ) -> np.ndarray:
        """
        this function opens a shared increment window
        """
        return open_shared_win(
            window, np.float64, (n_tuples, self.full_nocc + self.norb, self.full_norb)
        )

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
            f"      generalized Fock matrix increment for root {self.fci_state_root:d} "
            + "= "
            + np.array2string(inc_tup, max_line_width=59, precision=4)
            + "\n"
        )

        return string

    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for an generalized Fock matrix
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

        # set headers
        gen_Fock = (
            f"generalized Fock matrix for root {self.fci_state_root} (total "
            + f"increment norm = {np.linalg.norm(tot_inc):.4e})"
        )

        # set string
        string: str = FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{gen_Fock:^81}\n"

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
        this function returns the hf, base and total 1- and 2-particle reduced density
        matrices
        """
        raise NotImplementedError

    def _results_prt(self) -> str:
        """
        this function returns the 1- and 2-particle reduced density matrices table
        """
        raise NotImplementedError
