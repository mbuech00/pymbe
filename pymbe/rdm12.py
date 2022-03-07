#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
1- and 2-particle reduced density matrix expansion module
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
from typing import TYPE_CHECKING, cast, Tuple

from pymbe.expansion import ExpCls
from pymbe.kernel import main_kernel
from pymbe.output import DIVIDER as DIVIDER_OUTPUT, FILL as FILL_OUTPUT, mbe_debug
from pymbe.tools import (
    RST,
    RDMCls,
    packedRDMCls,
    nelec,
    tuples,
    hash_1d,
    hash_lookup,
)
from pymbe.parallel import mpi_reduce, mpi_allreduce, mpi_bcast, mpi_gatherv

if TYPE_CHECKING:

    import matplotlib
    from typing import List, Optional, Union

    from pymbe.pymbe import MBE


# get logger
logger = logging.getLogger("pymbe_logger")


class RDMExpCls(ExpCls[RDMCls, packedRDMCls, Tuple[MPI.Win, MPI.Win]]):
    """
    this class contains the pymbe expansion attributes
    """

    def __init__(self, mbe: MBE) -> None:
        """
        init expansion attributes
        """
        super(RDMExpCls, self).__init__(
            mbe,
            RDMCls(*cast(tuple, mbe.hf_prop)),
            RDMCls(*cast(tuple, mbe.ref_prop)),
            RDMCls(*cast(tuple, mbe.base_prop)),
        )

    def tot_prop(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the total 1- and 2-particle reduced density matrices
        """
        tot_rdm12 = self.mbe_tot_prop[-1]
        tot_rdm12 += self.hf_prop
        tot_rdm12[self.ref_space] += self.ref_prop
        tot_rdm12 += self.base_prop

        return tot_rdm12.rdm1, tot_rdm12.rdm2

    def plot_results(self) -> matplotlib.figure.Figure:
        """
        this function plots the 1- and 2-particle reduced density matrices
        """
        raise NotImplementedError

    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> Tuple[RDMCls, Tuple[int, int]]:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """
        # n_elec
        n_elec = nelec(self.occup, cas_idx)

        # perform main calc
        res = main_kernel(
            self.method,
            self.cc_backend,
            self.fci_solver,
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
            n_elec,
            self.verbose,
        )

        res_full = RDMCls(res["rdm1"], res["rdm2"])

        # perform base calc
        if self.base_method is not None:
            res = main_kernel(
                self.base_method,
                self.cc_backend,
                self.fci_solver,
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
                n_elec,
                self.verbose,
            )

            res_base = RDMCls(res["rdm1"], res["rdm2"])

            res_full -= res_base

        ind = np.where(np.in1d(np.concatenate((self.ref_space, tup)), self.ref_space))[
            0
        ]

        res_full[ind] -= self.ref_prop

        return res_full, n_elec

    def _sum(
        self,
        inc: List[packedRDMCls],
        hashes: List[np.ndarray],
        min_occ: int,
        min_virt: int,
        tup: np.ndarray,
    ) -> RDMCls:
        """
        this function performs a recursive summation and returns the final increment
        associated with a given tuple
        """
        # init res
        res = RDMCls(
            np.zeros(
                [self.ref_space.size + self.order, self.ref_space.size + self.order],
                dtype=np.float64,
            ),
            np.zeros(
                [
                    self.ref_space.size + self.order,
                    self.ref_space.size + self.order,
                    self.ref_space.size + self.order,
                    self.ref_space.size + self.order,
                ],
                dtype=np.float64,
            ),
        )

        # occupied and virtual subspaces of tuple
        tup_occ = tup[tup < self.nocc]
        tup_virt = tup[self.nocc <= tup]

        # rank of reference space and occupied and virtual tuple orbitals
        rank = np.argsort(np.argsort(np.concatenate((self.ref_space, tup))))
        ind_ref = rank[: self.ref_space.size]
        ind_tup_occ = rank[self.ref_space.size : self.ref_space.size + tup_occ.size]
        ind_tup_virt = rank[self.ref_space.size + tup_occ.size :]

        # compute contributions from lower-order increments
        for k in range(self.order - 1, self.min_order - 1, -1):

            # rank of all orbitals in casci space
            ind_casci = np.empty(self.ref_space.size + k, dtype=np.int64)

            # loop over subtuples
            for tup_sub, ind_sub in zip(
                tuples(tup_occ, tup_virt, min_occ, min_virt, k),
                tuples(ind_tup_occ, ind_tup_virt, min_occ, min_virt, k),
            ):

                # compute index
                idx = hash_lookup(hashes[k - self.min_order], hash_1d(tup_sub))

                # sum up order increments
                if idx is not None:

                    # add rank of reference space orbitals
                    ind_casci[: self.ref_space.size] = ind_ref

                    # add rank of subtuple orbitals
                    ind_casci[self.ref_space.size :] = ind_sub

                    # sort indices for faster assignment
                    ind_casci.sort()

                    # add subtuple rdms
                    res[ind_casci] += inc[k - self.min_order][idx]

        return res

    @staticmethod
    def _write_target_file(order: Optional[int], prop: RDMCls, string: str) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """
        if order is None:
            np.savez(os.path.join(RST, f"{string}"), rdm1=prop.rdm1, rdm2=prop.rdm2)
        else:
            np.savez(
                os.path.join(RST, f"{string}_{order}"), rdm1=prop.rdm1, rdm2=prop.rdm2
            )

    @staticmethod
    def _read_target_file(file: str) -> RDMCls:
        """
        this function reads files of attributes with the target type
        """
        rdm_dict = np.load(os.path.join(RST, file))
        return RDMCls(rdm_dict["rdm1"], rdm_dict["rdm2"])

    @staticmethod
    def _init_target_inst(value: float, norb: int) -> RDMCls:
        """
        this function initializes an instance of the target type
        """
        return RDMCls(
            np.full(2 * (norb,), value, dtype=np.float64),
            np.full(4 * (norb,), value, dtype=np.float64),
        )

    @staticmethod
    def _mpi_reduce_target(comm: MPI.Comm, values: RDMCls, op: MPI.Op) -> RDMCls:
        """
        this function performs a MPI reduce operation on values of the target type
        """
        return RDMCls(
            mpi_reduce(comm, values.rdm1, root=0, op=op),
            mpi_reduce(comm, values.rdm2, root=0, op=op),
        )

    @staticmethod
    def _write_inc_file(order: Optional[int], inc: packedRDMCls) -> None:
        """
        this function defines how to write increment restart files
        """
        if order is None:
            np.savez(os.path.join(RST, "mbe_inc"), rdm1=inc.rdm1, rdm2=inc.rdm2)
        else:
            np.savez(
                os.path.join(RST, f"mbe_inc_{order}"), rdm1=inc.rdm1, rdm2=inc.rdm2
            )

    @staticmethod
    def _read_inc_file(file: str) -> packedRDMCls:
        """
        this function defines reads the increment restart files
        """
        rdm_dict = np.load(os.path.join(RST, file))

        return packedRDMCls(rdm_dict["rdm1"], rdm_dict["rdm2"])

    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm
    ) -> Tuple[MPI.Win, MPI.Win]:
        """
        this function allocates a shared increment window
        """
        # get current length of incs array
        idx = len(self.incs)

        # generate packing and unpacking indices if these have not been generated yet
        if len(packedRDMCls.rdm1_size) == idx:
            packedRDMCls.get_pack_idx(self.ref_space.size + self.min_order + idx)

        return (
            MPI.Win.Allocate_shared(
                8 * size * packedRDMCls.rdm1_size[idx] if allocate else 0,
                8,
                comm=comm,  # type: ignore
            ),
            MPI.Win.Allocate_shared(
                8 * size * packedRDMCls.rdm2_size[idx] if allocate else 0,
                8,
                comm=comm,  # type: ignore
            ),
        )

    @staticmethod
    def _open_shared_inc(
        window: Tuple[MPI.Win, MPI.Win],
        n_tuples: int,
        idx: int,
    ) -> packedRDMCls:
        """
        this function opens a shared increment window
        """
        return packedRDMCls.open_shared_RDM(window, n_tuples, idx)

    @staticmethod
    def _mpi_bcast_inc(comm: MPI.Comm, inc: packedRDMCls) -> packedRDMCls:
        """
        this function bcasts the increments
        """
        return packedRDMCls(
            mpi_bcast(comm, inc.rdm1),
            mpi_bcast(comm, inc.rdm2),
        )

    @staticmethod
    def _mpi_reduce_inc(comm: MPI.Comm, inc: packedRDMCls, op: MPI.Op) -> packedRDMCls:
        """
        this function performs a MPI reduce operation on the increments
        """
        return packedRDMCls(
            mpi_reduce(comm, inc.rdm1, root=0, op=op),
            mpi_reduce(comm, inc.rdm2, root=0, op=op),
        )

    @staticmethod
    def _mpi_allreduce_inc(
        comm: MPI.Comm, inc: packedRDMCls, op: MPI.Op
    ) -> packedRDMCls:
        """
        this function performs a MPI allreduce operation on the increments
        """
        return packedRDMCls(
            mpi_allreduce(comm, inc.rdm1, op=op),
            mpi_allreduce(comm, inc.rdm2, op=op),
        )

    @staticmethod
    def _mpi_gatherv_inc(
        comm: MPI.Comm, send_inc: packedRDMCls, recv_inc: packedRDMCls
    ) -> packedRDMCls:
        """
        this function performs a MPI gatherv operation on the increments
        """
        # number of increments for every rank
        recv_counts = {
            "rdm1": np.array(comm.allgather(send_inc.rdm1.size)),
            "rdm2": np.array(comm.allgather(send_inc.rdm2.size)),
        }

        return packedRDMCls(
            mpi_gatherv(comm, send_inc.rdm1, recv_inc.rdm1, recv_counts["rdm1"]),
            mpi_gatherv(comm, send_inc.rdm2, recv_inc.rdm2, recv_counts["rdm2"]),
        )

    @staticmethod
    def _flatten_inc(inc_lst: List[packedRDMCls], order: int) -> packedRDMCls:
        """
        this function flattens the supplied increment arrays
        """
        return packedRDMCls(
            np.array([inc.rdm1 for inc in inc_lst]).reshape(-1),
            np.array([inc.rdm2 for inc in inc_lst]).reshape(-1),
            order,
        )

    @staticmethod
    def _free_inc(inc_win: Tuple[MPI.Win, MPI.Win]) -> None:
        """
        this function frees the supplied increment windows
        """
        inc_win[0].Free()
        inc_win[1].Free()

    @staticmethod
    def _screen(inc_tup: RDMCls, screen: np.ndarray, tup: np.ndarray) -> np.ndarray:
        """
        this function modifies the screening array
        """
        return np.maximum(
            screen[tup],
            np.maximum(np.max(np.abs(inc_tup.rdm1)), np.max(np.abs(inc_tup.rdm2))),
        )

    def _update_inc_stats(
        self,
        inc_tup: RDMCls,
        min_inc: RDMCls,
        mean_inc: RDMCls,
        max_inc: RDMCls,
        tup: np.ndarray,
    ) -> Tuple[RDMCls, RDMCls, RDMCls]:
        """
        this function updates the increment statistics
        """
        # get indices in full rdm
        idx_full = np.concatenate((self.ref_space, tup))
        idx_full.sort()

        # add to total rdm
        mean_inc[idx_full] += inc_tup

        return min_inc, mean_inc, max_inc

    @staticmethod
    def _total_inc(inc: packedRDMCls, mean_inc: RDMCls) -> RDMCls:
        """
        this function calculates the total increment at a certain order
        """
        return mean_inc.copy()

    def _mbe_debug(
        self,
        nelec_tup: Tuple[int, int],
        inc_tup: RDMCls,
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
            f"      rdm1 increment for root {self.fci_state_root:d} = "
            + np.array2string(inc_tup.rdm1, max_line_width=59, precision=4)
            + "\n"
        )
        string += (
            f"      rdm2 increment for root {self.fci_state_root:d} = "
            + np.array2string(inc_tup.rdm2, max_line_width=59, precision=4)
            + "\n"
        )

        return string

    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for an rdm12 calculation
        """
        # set string
        string = FILL_OUTPUT + "\n"
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
