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
from pyscf import ao2mo, gto, scf, fci, cc
from typing import TYPE_CHECKING, cast, Tuple

from pymbe.expansion import ExpCls, MAX_MEM, CONV_TOL, SPIN_TOL
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import (
    RST,
    GenFockCls,
    GenFockArrayCls,
    get_nelec,
    tuples,
    hash_1d,
    hash_lookup,
    get_nhole,
    get_nexc,
    assertion,
    idx_tril,
)
from pymbe.parallel import (
    mpi_reduce,
    mpi_allreduce,
    mpi_bcast,
    mpi_gatherv,
    open_shared_win,
)
from pymbe import direct_spin0_symm, direct_spin1_symm

if TYPE_CHECKING:

    import matplotlib
    from typing import List, Optional, Union

    from pymbe.pymbe import MBE


# get logger
logger = logging.getLogger("pymbe_logger")


class GenFockExpCls(ExpCls[GenFockCls, GenFockArrayCls, Tuple[MPI.Win, MPI.Win]]):
    """
    this class contains the pymbe expansion attributes for the generalized Fock matrix
    elements
    """

    def __init__(self, mbe: MBE) -> None:
        """
        init expansion attributes
        """
        super(GenFockExpCls, self).__init__(
            mbe, GenFockCls(*cast(tuple, mbe.base_prop))
        )

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
        self.no_singles = mbe.no_singles

        # initialize dependent attributes
        self._init_dep_attrs(mbe)

    def prop(self, prop_type: str) -> Tuple[float, np.ndarray]:
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

        return tot_gen_fock.energy, tot_gen_fock.gen_fock

    def plot_results(self, *args: str) -> matplotlib.figure.Figure:
        """
        this function plots the generalized Fock matrix elements
        """
        raise NotImplementedError

    def _calc_hf_prop(
        self, hcore: np.ndarray, eri: np.ndarray, vhf: np.ndarray
    ) -> GenFockCls:
        """
        this function calculates the hartree-fock generalized fock matrix
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

        # initialize generalized Fock matrix
        hf_gen_fock = np.empty(
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
        hf_gen_fock[: self.full_nocc] = (
            2 * (self.inact_fock[:, : self.full_nocc] + act_fock_pi).transpose()
        )

        # calculate occupied-active generalized Fock matrix elements
        hf_gen_fock[self.full_nocc :] = (
            np.einsum("u,pu->up", self.occup, self.inact_fock[:, self.full_nocc :])
            + np.einsum("u,v,puvv->up", self.occup, self.occup, self.eri_gaaa)
            - (
                np.einsum("u,v,pvvu->up", occup_a, occup_a, self.eri_gaaa)
                + np.einsum("u,v,pvvu->up", occup_b, occup_b, self.eri_gaaa)
            )
        )

        return GenFockCls(hf_energy, hf_gen_fock)

    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
    ) -> Tuple[GenFockCls, np.ndarray]:
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

    def _sum(
        self, inc: List[GenFockArrayCls], hashes: List[np.ndarray], tup: np.ndarray
    ) -> GenFockCls:
        """
        this function performs a recursive summation and returns the final increment
        associated with a given tuple
        """
        # init res
        res = self._zero_target_arr(self.order - self.min_order)

        # occupied and virtual subspaces of tuple
        tup_occ = tup[tup < self.nocc]
        tup_virt = tup[self.nocc <= tup]

        # compute contributions from lower-order increments
        for k in range(self.order - 1, self.min_order - 1, -1):

            # loop over subtuples
            for tup_sub in tuples(
                tup_occ,
                tup_virt,
                self.ref_nelec,
                self.ref_nhole,
                self.vanish_exc,
                k,
            ):

                # compute index
                idx = hash_lookup(hashes[k - self.min_order], hash_1d(tup_sub))

                # sum up order increments
                if idx is not None:
                    res[k - self.min_order] += inc[k - self.min_order][idx]

        return GenFockCls(np.sum(res.energy), np.sum(res.gen_fock, axis=0))

    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
    ) -> GenFockCls:
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

        return GenFockCls(energy[-1], gen_fock) - self.hf_prop

    def _cc_kernel(
        self,
        method: str,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        higher_amp_extrap: bool,
    ) -> GenFockCls:
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

        return GenFockCls(e_cc, gen_fock) - self.hf_prop

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
    def _write_target_file(order: Optional[int], prop: GenFockCls, string: str) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        if order is None:
            np.savez(
                os.path.join(RST, f"{string}"),
                energy=prop.energy,
                gen_fock=prop.gen_fock,
            )
        else:
            np.savez(
                os.path.join(RST, f"{string}_{order}"),
                energy=prop.energy,
                gen_fock=prop.gen_fock,
            )

    @staticmethod
    def _read_target_file(file: str) -> GenFockCls:
        """
        this function reads files of attributes with the target type
        """
        target_dict = np.load(os.path.join(RST, file))
        return GenFockCls(target_dict["energy"], target_dict["gen_fock"])

    def _init_target_inst(self, value: float, *args: int) -> GenFockCls:
        """
        this function initializes an instance of the target type
        """
        return GenFockCls(
            value,
            np.full(
                (self.full_nocc + self.norb, self.full_norb), value, dtype=np.float64
            ),
        )

    def _zero_target_arr(self, length: int):
        """
        this function initializes an array of the target type with value zero
        """
        return GenFockArrayCls(
            np.zeros(length, dtype=np.float64),
            np.zeros(
                (length, self.full_nocc + self.norb, self.full_norb), dtype=np.float64
            ),
        )

    @staticmethod
    def _mpi_reduce_target(
        comm: MPI.Comm, values: GenFockCls, op: MPI.Op
    ) -> GenFockCls:
        """
        this function performs a MPI reduce operation on values of the target type
        """
        return GenFockCls(
            mpi_reduce(
                comm, np.array(values.energy, dtype=np.float64), root=0, op=op
            ).item(),
            mpi_reduce(comm, values.gen_fock, root=0, op=op),
        )

    @staticmethod
    def _write_inc_file(order: Optional[int], inc: GenFockArrayCls) -> None:
        """
        this function defines how to write increment restart files
        """
        if order is None:
            np.savez(
                os.path.join(RST, "mbe_inc"), energy=inc.energy, gen_fock=inc.gen_fock
            )
        else:
            np.savez(
                os.path.join(RST, f"mbe_inc_{order}"),
                energy=inc.energy,
                gen_fock=inc.gen_fock,
            )

    @staticmethod
    def _read_inc_file(file: str) -> GenFockArrayCls:
        """
        this function defines reads the increment restart files
        """
        target_dict = np.load(os.path.join(RST, file))

        return GenFockArrayCls(target_dict["energy"], target_dict["gen_fock"])

    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm
    ) -> Tuple[MPI.Win, MPI.Win]:
        """
        this function allocates a shared increment window
        """
        return (
            MPI.Win.Allocate_shared(
                8 * size if allocate else 0,
                8,
                comm=comm,  # type: ignore
            ),
            MPI.Win.Allocate_shared(
                8 * size * (self.full_nocc + self.norb) * self.full_norb
                if allocate
                else 0,
                8,
                comm=comm,  # type: ignore
            ),
        )

    def _open_shared_inc(
        self, window: Tuple[MPI.Win, MPI.Win], n_tuples: int, *args: int
    ) -> GenFockArrayCls:
        """
        this function opens a shared increment window
        """
        return GenFockArrayCls(
            open_shared_win(window[0], np.float64, (n_tuples,)),
            open_shared_win(
                window[1],
                np.float64,
                (n_tuples, self.full_nocc + self.norb, self.full_norb),
            ),
        )

    @staticmethod
    def _mpi_bcast_inc(comm: MPI.Comm, inc: GenFockArrayCls) -> GenFockArrayCls:
        """
        this function bcasts the increments
        """
        return GenFockArrayCls(
            mpi_bcast(comm, inc.energy),
            mpi_bcast(comm, inc.gen_fock),
        )

    @staticmethod
    def _mpi_reduce_inc(
        comm: MPI.Comm, inc: GenFockArrayCls, op: MPI.Op
    ) -> GenFockArrayCls:
        """
        this function performs a MPI reduce operation on the increments
        """
        return GenFockArrayCls(
            mpi_reduce(comm, inc.energy, root=0, op=op),
            mpi_reduce(comm, inc.gen_fock, root=0, op=op),
        )

    @staticmethod
    def _mpi_allreduce_inc(
        comm: MPI.Comm, inc: GenFockArrayCls, op: MPI.Op
    ) -> GenFockArrayCls:
        """
        this function performs a MPI allreduce operation on the increments
        """
        return GenFockArrayCls(
            mpi_allreduce(comm, inc.energy, op=op),
            mpi_allreduce(comm, inc.gen_fock, op=op),
        )

    @staticmethod
    def _mpi_gatherv_inc(
        comm: MPI.Comm, send_inc: GenFockArrayCls, recv_inc: GenFockArrayCls
    ) -> GenFockArrayCls:
        """
        this function performs a MPI gatherv operation on the increments
        """
        # number of increments for every rank
        recv_counts = {
            "energy": np.array(comm.allgather(send_inc.energy.size)),
            "gen_fock": np.array(comm.allgather(send_inc.gen_fock.size)),
        }

        return GenFockArrayCls(
            mpi_gatherv(comm, send_inc.energy, recv_inc.energy, recv_counts["energy"]),
            mpi_gatherv(
                comm, send_inc.gen_fock, recv_inc.gen_fock, recv_counts["gen_fock"]
            ),
        )

    @staticmethod
    def _flatten_inc(inc_lst: List[GenFockArrayCls], order: int) -> GenFockArrayCls:
        """
        this function flattens the supplied increment arrays
        """
        return GenFockArrayCls(
            np.array([inc.energy for inc in inc_lst]).reshape(-1),
            np.array([inc.gen_fock for inc in inc_lst]).reshape(-1),
        )

    @staticmethod
    def _free_inc(inc_win: Tuple[MPI.Win, MPI.Win]) -> None:
        """
        this function frees the supplied increment windows
        """
        inc_win[0].Free()
        inc_win[1].Free()

    @staticmethod
    def _screen(
        inc_tup: GenFockCls, screen: np.ndarray, tup: np.ndarray, screen_func: str
    ) -> np.ndarray:
        """
        this function modifies the screening array
        """
        if screen_func == "sum":
            return screen[tup] + np.sum(np.abs(inc_tup.gen_fock))
        else:
            return np.maximum(screen[tup], np.max(np.abs(inc_tup.gen_fock)))

    def _update_inc_stats(
        self,
        inc_tup: GenFockCls,
        min_inc: GenFockCls,
        mean_inc: GenFockCls,
        max_inc: GenFockCls,
        cas_idx: np.ndarray,
    ) -> Tuple[GenFockCls, GenFockCls, GenFockCls]:
        """
        this function updates the increment statistics
        """
        # add to total rdm
        mean_inc += inc_tup

        return min_inc, mean_inc, max_inc

    @staticmethod
    def _total_inc(inc: GenFockArrayCls, mean_inc: GenFockCls) -> GenFockCls:
        """
        this function calculates the total increment at a certain order
        """
        return GenFockCls(np.sum(inc.energy), mean_inc.gen_fock.copy())

    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: GenFockCls,
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
            f"      energy increment for root {self.fci_state_root:d} "
            + f"= {inc_tup.energy:.4e}\n"
        )
        string += (
            f"      generalized Fock matrix increment for root {self.fci_state_root:d} "
            + "= "
            + np.array2string(inc_tup.gen_fock, max_line_width=59, precision=4)
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
        energy = (
            f"energy for root {self.fci_state_root} "
            + f"(total increment = {tot_inc.energy:.4e})"
        )
        gen_fock = (
            f"generalized Fock matrix for root {self.fci_state_root} "
            + f"(total increment norm = {np.linalg.norm(tot_inc.gen_fock):.4e})"
        )

        # set string
        string: str = FILL_OUTPUT + "\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{energy:^81}\n"
        string += DIVIDER_OUTPUT + "\n"
        string += f" RESULT-{order:d}:{gen_fock:^81}\n"

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
