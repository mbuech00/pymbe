#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
expansion module
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
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, cast, TypeVar, Generic, Tuple, List, Union

from pymbe.output import (
    main_header,
    mbe_header,
    mbe_status,
    mbe_end,
    screen_results,
    purge_header,
    purge_results,
    purge_end,
)
from pymbe.tools import (
    RST,
    RDMCls,
    packedRDMCls,
    GenFockCls,
    packedGenFockCls,
    pi_space,
    natural_keys,
    n_tuples,
    is_file,
    read_file,
    write_file,
    pi_prune,
    tuples,
    tuples_with_nocc,
    get_nelec,
    get_nhole,
    get_nexc,
    start_idx,
    core_cas,
    idx_tril,
    hash_1d,
    hash_lookup,
    get_occup,
    get_vhf,
    e_core_h1e,
)
from pymbe.parallel import (
    mpi_reduce,
    mpi_allreduce,
    mpi_bcast,
    mpi_gatherv,
    open_shared_win,
)
from pymbe.results import timings_prt

if TYPE_CHECKING:
    from pyscf import gto
    from typing import Dict, Optional

    from pymbe.pymbe import MBE
    from pymbe.parallel import MPICls


# define variable type for target properties
TargetType = TypeVar("TargetType", float, np.ndarray, RDMCls, GenFockCls)

# define variable type for increment arrays
IncType = TypeVar("IncType", np.ndarray, packedRDMCls, packedGenFockCls)

# define variable type for MPI windows of increment arrays
MPIWinType = TypeVar(
    "MPIWinType", MPI.Win, Tuple[MPI.Win, MPI.Win], Tuple[MPI.Win, MPI.Win, MPI.Win]
)

# define variable type for integers describing electronic states
StateIntType = TypeVar("StateIntType", int, List[int])

# define variable type for numpy arrays describing electronic states
StateArrayType = TypeVar("StateArrayType", np.ndarray, List[np.ndarray])


# get logger
logger = logging.getLogger("pymbe_logger")


SCREEN = 1000.0  # random, non-sensical number


MAX_MEM = 1e10
CONV_TOL = 1.0e-10
SPIN_TOL = 1.0e-05


class ExpCls(
    Generic[TargetType, IncType, MPIWinType, StateIntType, StateArrayType],
    metaclass=ABCMeta,
):
    """
    this class contains the pymbe expansion attributes
    """

    def __init__(self, mbe: MBE, base_prop: TargetType) -> None:
        """
        init expansion attributes
        """
        # expansion model
        self.method: str = mbe.method
        self.cc_backend: str = mbe.cc_backend
        self.hf_guess: bool = mbe.hf_guess

        # target property
        self.target: str = mbe.target

        # system
        self.norb: int = cast(int, mbe.norb)
        self.nelec: StateArrayType = cast(StateArrayType, mbe.nelec)
        self.point_group: str = cast(str, mbe.point_group)
        self.orbsym: np.ndarray = cast(np.ndarray, mbe.orbsym)
        self.fci_state_sym: StateIntType = cast(StateIntType, mbe.fci_state_sym)
        self.fci_state_root: StateIntType = cast(StateIntType, mbe.fci_state_root)
        self.fci_state_weights: np.ndarray = cast(np.ndarray, mbe.fci_state_weights)
        self._state_occup()

        # integrals
        (
            self.hcore,
            self.eri,
            self.vhf,
            self.hcore_win,
            self.eri_win,
            self.vhf_win,
        ) = self._int_wins(mbe.hcore, mbe.eri, mbe.mpi)

        # orbital representation
        self.orb_type: str = mbe.orb_type

        # reference and expansion spaces
        self.ref_space: np.ndarray = mbe.ref_space
        self.ref_occ = self.ref_space[self.ref_space < self.nocc]
        self.ref_virt = self.ref_space[self.nocc <= self.ref_space]
        self.ref_nelec = get_nelec(self.occup, self.ref_space)
        self.ref_nhole = get_nhole(self.ref_nelec, self.ref_space)
        self.exp_space: List[np.ndarray] = [cast(np.ndarray, mbe.exp_space)]

        # base model
        self.base_method: Optional[str] = mbe.base_method
        self.base_prop: TargetType = base_prop

        # total mbe property
        self.mbe_tot_prop: List[TargetType] = []

        # increment windows
        self.incs: List[List[MPIWinType]] = []

        # hash windows
        self.hashes: List[List[MPI.Win]] = []

        # timings
        self.time: Dict[str, List[float]] = {"mbe": [], "purge": []}

        # statistics
        self.min_inc: List[TargetType] = []
        self.mean_inc: List[TargetType] = []
        self.max_inc: List[TargetType] = []

        # number of tuples
        self.n_tuples: Dict[str, List[List[int]]] = {"theo": [], "calc": [], "inc": []}

        # screening
        self.screen_start: int = mbe.screen_start
        self.screen_perc: float = mbe.screen_perc
        self.screen_func: str = mbe.screen_func
        self.screen = np.zeros(self.norb, dtype=np.float64)
        self.screen_orbs = np.array([], dtype=np.int64)

        # restart
        self.rst: bool = mbe.rst
        self.rst_freq: int = mbe.rst_freq
        self.restarted: bool = mbe.restarted

        # order
        self.order: int = 0
        self.min_order: int = 1

        if mbe.max_order is not None:
            max_order = min(self.exp_space[0].size, mbe.max_order)
        else:
            max_order = self.exp_space[0].size
        self.max_order: int = max_order

        self.final_order: int = 0

        # number of vanishing excitations for current model
        self.vanish_exc: int = 0
        if self.base_method is None:
            self.vanish_exc = 1
        elif self.base_method in ["ccsd", "ccsd(t)"]:
            self.vanish_exc = 2
        elif self.base_method == "ccsdt":
            self.vanish_exc = 3
        elif self.base_method == "ccsdtq":
            self.vanish_exc = 4

        # verbose
        self.verbose: int = mbe.verbose

        # pi pruning
        self.pi_prune: bool = mbe.pi_prune
        if self.pi_prune:
            self.orbsym_linear = cast(np.ndarray, mbe.orbsym_linear)
            pi_orbs, pi_hashes = pi_space(
                "Dooh" if self.point_group == "D2h" else "Coov",
                self.orbsym_linear,
                self.exp_space[0],
            )
            self.pi_orbs: np.ndarray = pi_orbs
            self.pi_hashes: np.ndarray = pi_hashes

        # hartree fock property
        self.hf_prop: TargetType

        # reference space property
        self.ref_prop: TargetType

    def driver_master(self, mpi: MPICls) -> None:
        """
        this function is the main pymbe master function
        """
        # print expansion headers
        logger.info(main_header(mpi=mpi, method=self.method))

        # print output from restarted calculation
        if self.restarted:
            for i in range(self.min_order, self.start_order):
                # print mbe header
                logger.info(
                    mbe_header(
                        i,
                        sum(self.n_tuples["calc"][i - self.min_order]),
                        1.0 if i < self.screen_start else self.screen_perc,
                    )
                )

                # print mbe end
                logger.info(mbe_end(i, self.time["mbe"][i - self.min_order]))

                # print mbe results
                logger.info(self._mbe_results(i))

                # print screening results
                self.screen_orbs = np.setdiff1d(
                    self.exp_space[i - self.min_order],
                    self.exp_space[i - self.min_order + 1],
                )
                if 0 < self.screen_orbs.size:
                    logger.info(screen_results(i, self.screen_orbs, self.exp_space))

        # begin or resume mbe expansion depending
        for self.order in range(self.start_order, self.max_order + 1):
            # theoretical and actual number of tuples at current order
            if len(self.n_tuples["inc"]) == self.order - self.min_order:
                self.n_tuples["theo"].append([])
                self.n_tuples["calc"].append([])
                self.n_tuples["inc"].append([])
                for tup_nocc in range(self.order + 1):
                    self.n_tuples["theo"][-1].append(
                        n_tuples(
                            self.exp_space[0][self.exp_space[0] < self.nocc],
                            self.exp_space[0][self.nocc <= self.exp_space[0]],
                            self.ref_nelec,
                            self.ref_nhole,
                            -1,
                            self.order,
                            tup_nocc,
                        )
                    )
                    self.n_tuples["calc"][-1].append(
                        n_tuples(
                            self.exp_space[-1][self.exp_space[-1] < self.nocc],
                            self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                            self.ref_nelec,
                            self.ref_nhole,
                            self.vanish_exc,
                            self.order,
                            tup_nocc,
                        )
                    )
                    self.n_tuples["inc"][-1].append(self.n_tuples["calc"][-1][-1])
                    if self.rst:
                        write_file(
                            np.asarray(self.n_tuples["theo"][-1][-1]),
                            "mbe_n_tuples_theo",
                            order=self.order,
                            nocc=tup_nocc,
                        )
                        write_file(
                            np.asarray(self.n_tuples["calc"][-1][-1]),
                            "mbe_n_tuples_calc",
                            order=self.order,
                            nocc=tup_nocc,
                        )
                        write_file(
                            np.asarray(self.n_tuples["inc"][-1][-1]),
                            "mbe_n_tuples_inc",
                            order=self.order,
                            nocc=tup_nocc,
                        )

            # print mbe header
            logger.info(
                mbe_header(
                    self.order,
                    sum(self.n_tuples["calc"][-1]),
                    1.0 if self.order < self.screen_start else self.screen_perc,
                )
            )

            # main mbe function
            self._mbe(mpi)

            # print mbe end
            logger.info(mbe_end(self.order, self.time["mbe"][-1]))

            # print mbe results
            logger.info(self._mbe_results(self.order))

            # update screen_orbs
            if self.order > self.min_order:
                self.screen_orbs = np.setdiff1d(self.exp_space[-2], self.exp_space[-1])

            # print screening results
            if 0 < self.screen_orbs.size:
                logger.info(
                    screen_results(self.order, self.screen_orbs, self.exp_space)
                )

            # print header
            logger.info(purge_header(self.order))

            # main purging function
            self._purge(mpi)

            # print purging results
            if self.order + 1 <= self.exp_space[-1].size:
                logger.info(purge_results(self.n_tuples, self.min_order, self.order))

            # print purge end
            logger.info(purge_end(self.order, self.time["purge"][-1]))

            # write restart files
            if self.rst:
                if self.screen_orbs.size > 0:
                    for order in range(1, self.order + 1):
                        k = order - 1
                        for tup_nocc in range(order + 1):
                            l = tup_nocc
                            hashes = open_shared_win(
                                self.hashes[k][l],
                                np.int64,
                                (self.n_tuples["inc"][k][l],),
                            )
                            write_file(hashes, "mbe_hashes", order=order, nocc=tup_nocc)
                            inc = self._open_shared_inc(
                                self.incs[k][l],
                                self.n_tuples["inc"][k][l],
                                order,
                                tup_nocc,
                            )
                            self._write_inc_file(inc, order, tup_nocc)
                            write_file(
                                np.asarray(self.n_tuples["inc"][k][l]),
                                "mbe_n_tuples_inc",
                                order=order,
                                nocc=tup_nocc,
                            )
                else:
                    for tup_nocc in range(self.order + 1):
                        l = tup_nocc
                        hashes = open_shared_win(
                            self.hashes[-1][l], np.int64, (self.n_tuples["inc"][-1][l],)
                        )
                        write_file(
                            hashes, "mbe_hashes", order=self.order, nocc=tup_nocc
                        )
                        inc = self._open_shared_inc(
                            self.incs[-1][l],
                            self.n_tuples["inc"][-1][l],
                            self.order,
                            tup_nocc,
                        )
                        self._write_inc_file(inc, self.order, l)
                        write_file(
                            np.asarray(self.n_tuples["inc"][-1][l]),
                            "mbe_n_tuples_inc",
                            order=self.order,
                            nocc=tup_nocc,
                        )
                self._write_target_file(
                    self.mbe_tot_prop[-1], "mbe_tot_prop", self.order
                )
                write_file(
                    np.asarray(self.time["mbe"][-1]), "mbe_time_mbe", order=self.order
                )
                write_file(
                    np.asarray(self.time["purge"][-1]),
                    "mbe_time_purge",
                    order=self.order,
                )

            # convergence check
            if self.exp_space[-1].size < self.order + 1 or self.order == self.max_order:
                # final order
                self.final_order = self.order

                # total timing
                self.time["total"] = [
                    mbe + purge
                    for mbe, purge in zip(self.time["mbe"], self.time["purge"])
                ]

                # final results
                logger.info("\n\n")

                break

        # wake up slaves
        mpi.global_comm.bcast({"task": "exit"}, root=0)

    def driver_slave(self, mpi: MPICls) -> None:
        """
        this function is the main pymbe slave function
        """
        # set loop/waiting logical
        slave = True

        # enter slave state
        while slave:
            # task id
            msg = mpi.global_comm.bcast(None, root=0)

            if msg["task"] == "mbe":
                # receive order
                self.order = msg["order"]

                # actual number of tuples at current order
                if len(self.n_tuples["inc"]) == self.order - self.min_order:
                    self.n_tuples["inc"].append([])
                    for tup_nocc in range(self.order + 1):
                        self.n_tuples["inc"][-1].append(
                            n_tuples(
                                self.exp_space[-1][self.exp_space[-1] < self.nocc],
                                self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                                self.ref_nelec,
                                self.ref_nhole,
                                self.vanish_exc,
                                self.order,
                                tup_nocc,
                            )
                        )

                # main mbe function
                self._mbe(
                    mpi,
                    rst_read=msg["rst_read"],
                    tup_idx=msg["tup_idx"],
                    tup=msg["tup"],
                )

                # update screen_orbs
                if self.order == self.min_order:
                    self.screen_orbs = np.array([], dtype=np.int64)
                else:
                    self.screen_orbs = np.setdiff1d(
                        self.exp_space[-2], self.exp_space[-1]
                    )

            elif msg["task"] == "purge":
                # receive order
                self.order = msg["order"]

                # main purging function
                self._purge(mpi)

            elif msg["task"] == "exit":
                slave = False

    def print_results(self, mol: Optional[gto.Mole], mpi: MPICls) -> str:
        """
        this function handles printing of results
        """
        # print header
        string = main_header() + "\n\n"

        # print timings
        string += timings_prt(self, self.method) + "\n\n"

        # print results
        string += self._results_prt()

        return string

    @abstractmethod
    def _state_occup(self) -> None:
        """
        this function initializes certain state attributes
        """
        self.nocc: int
        self.spin: StateIntType
        self.occup: np.ndarray

    def _int_wins(
        self,
        hcore_in: Optional[np.ndarray],
        eri_in: Optional[np.ndarray],
        mpi: MPICls,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MPI.Win, MPI.Win, MPI.Win]:
        """
        this function creates shared memory windows for integrals on every node
        """
        # allocate hcore in shared mem
        hcore_win = MPI.Win.Allocate_shared(
            8 * self.norb**2 if mpi.local_master else 0,
            8,
            comm=mpi.local_comm,  # type: ignore
        )
        hcore = open_shared_win(hcore_win, np.float64, 2 * (self.norb,))

        # set hcore on global master
        if mpi.global_master:
            hcore[:] = cast(np.ndarray, hcore_in)

        # mpi_bcast hcore
        if mpi.num_masters > 1 and mpi.local_master:
            hcore[:] = mpi_bcast(mpi.master_comm, hcore)

        # allocate eri in shared mem
        eri_win = MPI.Win.Allocate_shared(
            8 * (self.norb * (self.norb + 1) // 2) ** 2 if mpi.local_master else 0,
            8,
            comm=mpi.local_comm,  # type: ignore
        )
        eri = open_shared_win(
            eri_win, np.float64, 2 * (self.norb * (self.norb + 1) // 2,)
        )

        # set eri on global master
        if mpi.global_master:
            eri[:] = cast(np.ndarray, eri_in)

        # mpi_bcast eri
        if mpi.num_masters > 1 and mpi.local_master:
            eri[:] = mpi_bcast(mpi.master_comm, eri)

        # allocate vhf in shared mem
        vhf_win = MPI.Win.Allocate_shared(
            8 * self.nocc * self.norb**2 if mpi.local_master else 0,
            8,
            comm=mpi.local_comm,  # type: ignore
        )
        vhf = open_shared_win(vhf_win, np.float64, (self.nocc, self.norb, self.norb))

        # compute and set hartree-fock potential on global master
        if mpi.global_master:
            vhf[:] = get_vhf(eri, self.nocc, self.norb)

        # mpi_bcast vhf
        if mpi.num_masters > 1 and mpi.local_master:
            vhf[:] = mpi_bcast(mpi.master_comm, vhf)

        # mpi barrier
        mpi.global_comm.Barrier()

        return hcore, eri, vhf, hcore_win, eri_win, vhf_win

    def _init_dep_attrs(self, mbe: MBE) -> None:
        """
        this function inititializes attributes that depend on other attributes
        """
        # hartree fock property
        self.hf_prop = self._hf_prop(mbe.mpi)

        # reference space property
        self.ref_prop = self._init_target_inst(
            0.0, self.ref_space.size, self.ref_occ.size
        )
        if get_nexc(self.ref_nelec, self.ref_nhole) > self.vanish_exc:
            self.ref_prop = self._ref_prop(mbe.mpi)

        # attributes from restarted calculation
        if self.restarted:
            start_order = self._restart_main(mbe.mpi)
        else:
            start_order = self.min_order
        self.start_order: int = start_order

    def _hf_prop(self, mpi: MPICls) -> TargetType:
        """
        this function calculates and bcasts the hartree-fock property
        """
        # calculate reference space property on global master
        if mpi.global_master:
            # compute hartree-fock property
            hf_prop = self._calc_hf_prop(self.hcore, self.eri, self.vhf)

            # bcast ref_prop to slaves
            mpi.global_comm.bcast(hf_prop, root=0)

        else:
            # receive ref_prop from master
            hf_prop = mpi.global_comm.bcast(None, root=0)

        return hf_prop

    @abstractmethod
    def _calc_hf_prop(
        self, hcore: np.ndarray, eri: np.ndarray, vhf: np.ndarray
    ) -> TargetType:
        """
        this function calculates the hartree-fock property
        """

    def _ref_prop(self, mpi: MPICls) -> TargetType:
        """
        this function returns reference space properties
        """
        # calculate reference space property on global master
        if mpi.global_master:
            # core_idx and cas_idx
            core_idx, cas_idx = core_cas(
                self.nocc, self.ref_space, np.array([], dtype=np.int64)
            )

            # get cas_space h2e
            cas_idx_tril = idx_tril(cas_idx)
            h2e_cas = self.eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(self.hcore, self.vhf, core_idx, cas_idx)

            # compute reference space property
            ref_prop, _ = self._inc(e_core, h1e_cas, h2e_cas, core_idx, cas_idx)

            # bcast ref_prop to slaves
            mpi.global_comm.bcast(ref_prop, root=0)

        else:
            # receive ref_prop from master
            ref_prop = mpi.global_comm.bcast(None, root=0)

        return ref_prop

    def _restart_main(self, mpi: MPICls) -> int:
        """
        this function reads in all expansion restart files and returns the start order
        """
        # list sorted filenames in files list
        if mpi.global_master:
            files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]
            files.sort(key=natural_keys)

        # distribute filenames
        if mpi.global_master:
            mpi.global_comm.bcast(files, root=0)
        else:
            files = mpi.global_comm.bcast(None, root=0)

        # loop over n_tuples files
        if mpi.global_master:
            for i in range(len(files)):
                if "mbe_n_tuples" in files[i]:
                    file_split = files[i].split("_")
                    n_tuples_type = file_split[3]
                    order = int(file_split[4])
                    if len(self.n_tuples[n_tuples_type]) < order - self.min_order + 1:
                        self.n_tuples[n_tuples_type].append(
                            [np.load(os.path.join(RST, files[i])).tolist()]
                        )
                    else:
                        self.n_tuples[n_tuples_type][-1].append(
                            np.load(os.path.join(RST, files[i])).tolist()
                        )
            mpi.global_comm.bcast(self.n_tuples, root=0)
        else:
            self.n_tuples = mpi.global_comm.bcast(None, root=0)

        # loop over all other files
        for i in range(len(files)):
            # read hashes
            if "mbe_hashes" in files[i]:
                read_order = int(files[i].split("_")[2])
                order = len(self.hashes)
                if order < read_order:
                    self.hashes.append([])
                    order += 1
                k = order - 1
                tup_nocc = len(self.hashes[-1])
                n_tuples = self.n_tuples["inc"][k][tup_nocc]
                hashes_mpi_win = MPI.Win.Allocate_shared(
                    8 * n_tuples if mpi.local_master else 0,
                    8,
                    comm=mpi.local_comm,  # type: ignore
                )
                self.hashes[-1].append(hashes_mpi_win)
                hashes = open_shared_win(self.hashes[-1][-1], np.int64, (n_tuples,))
                if mpi.global_master:
                    hashes[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    hashes[:] = mpi_bcast(mpi.master_comm, hashes)
                mpi.local_comm.Barrier()

            # read increments
            elif "mbe_inc" in files[i]:
                read_order = int(files[i].split("_")[2])
                order = len(self.incs)
                if order < read_order:
                    self.incs.append([])
                    order += 1
                k = order - 1
                tup_nocc = len(self.incs[-1])
                n_tuples = self.n_tuples["inc"][k][tup_nocc]
                inc_mpi_win = self._allocate_shared_inc(
                    n_tuples, mpi.local_master, mpi.local_comm, order, tup_nocc
                )
                self.incs[-1].append(inc_mpi_win)
                inc = self._open_shared_inc(
                    self.incs[-1][-1], n_tuples, order, tup_nocc
                )
                if mpi.global_master:
                    inc[:] = self._read_inc_file(files[i])
                if mpi.num_masters > 1 and mpi.local_master:
                    inc[:] = self._mpi_bcast_inc(mpi.master_comm, inc)
                mpi.local_comm.Barrier()

            if mpi.global_master:
                # read expansion spaces
                if "exp_space" in files[i]:
                    self.exp_space.append(np.load(os.path.join(RST, files[i])))

                # read screening array
                elif "mbe_screen" in files[i]:
                    self.screen = np.load(os.path.join(RST, files[i]))

                # read total properties
                elif "mbe_tot_prop" in files[i]:
                    self.mbe_tot_prop.append(self._read_target_file(files[i]))

                # read minimum increment
                elif "mbe_min_inc" in files[i]:
                    self.min_inc.append(self._read_target_file(files[i]))

                # read mean increment
                elif "mbe_mean_inc" in files[i]:
                    self.mean_inc.append(self._read_target_file(files[i]))

                # read max increment
                elif "mbe_max_inc" in files[i]:
                    self.max_inc.append(self._read_target_file(files[i]))

                # read timings
                elif "mbe_time_mbe" in files[i]:
                    self.time["mbe"].append(
                        np.load(os.path.join(RST, files[i])).tolist()
                    )
                elif "mbe_time_purge" in files[i]:
                    self.time["purge"].append(
                        np.load(os.path.join(RST, files[i])).tolist()
                    )

        # bcast exp_space and screen
        if mpi.global_master:
            mpi.global_comm.bcast(self.exp_space, root=0)
            mpi.global_comm.bcast(self.screen, root=0)
        else:
            self.exp_space = mpi.global_comm.bcast(None, root=0)
            self.screen = mpi.global_comm.bcast(None, root=0)

        # mpi barrier
        mpi.global_comm.Barrier()

        return self.min_order + len(self.mbe_tot_prop)

    def _mbe(
        self,
        mpi: MPICls,
        rst_read: bool = False,
        tup_idx: int = 0,
        tup: Optional[np.ndarray] = None,
    ) -> None:
        """
        this function is the mbe main function
        """
        if mpi.global_master:
            # read restart files
            rst_read = is_file("mbe_idx", self.order) and is_file("mbe_tup", self.order)
            # start indices
            tup_idx = read_file("mbe_idx", self.order).item() if rst_read else 0
            # start tuples
            tup = read_file("mbe_tup", self.order) if rst_read else None
            # wake up slaves
            msg = {
                "task": "mbe",
                "order": self.order,
                "rst_read": rst_read,
                "tup_idx": tup_idx,
                "tup": tup,
            }
            mpi.global_comm.bcast(msg, root=0)

        # load and initialize hashes
        hashes, hashes_win = self._load_hashes(
            mpi.global_master, mpi.local_master, mpi.local_comm, rst_read
        )

        # load and initialize increments
        inc, inc_win = self._load_inc(
            mpi.global_master, mpi.local_master, mpi.local_comm, rst_read
        )

        # init time
        if mpi.global_master:
            if not rst_read:
                self.time["mbe"].append(0.0)
            time = MPI.Wtime()

        # init increment statistics
        if mpi.global_master and rst_read:
            min_inc = self.min_inc[-1]
            mean_inc = self.mean_inc[-1]
            max_inc = self.max_inc[-1]
        else:
            min_inc = self._init_target_inst(1.0e12, self.norb, self.nocc)
            mean_inc = self._init_target_inst(0.0, self.norb, self.nocc)
            max_inc = self._init_target_inst(0.0, self.norb, self.nocc)

        # mpi barrier
        mpi.global_comm.Barrier()

        # occupied and virtual expansion spaces
        exp_occ = self.exp_space[-1][self.exp_space[-1] < self.nocc]
        exp_virt = self.exp_space[-1][self.nocc <= self.exp_space[-1]]

        # init screen array
        screen = np.zeros(self.norb, dtype=np.float64)
        if rst_read:
            if mpi.global_master:
                screen = self.screen

        # get total number of tuples for this order
        n_tuples = sum(self.n_tuples["inc"][-1])

        # set rst_write
        rst_write = self.rst and mpi.global_size < self.rst_freq < n_tuples

        # start tuples
        tup_occ: Optional[np.ndarray]
        tup_virt: Optional[np.ndarray]
        if tup is not None:
            tup_occ = tup[tup < self.nocc]
            tup_virt = tup[self.nocc <= tup]
            if tup_occ.size == 0:
                tup_occ = None
            if tup_virt.size == 0:
                tup_virt = None
        else:
            tup_occ = tup_virt = None
        order_start, occ_start, virt_start = start_idx(
            exp_occ, exp_virt, tup_occ, tup_virt
        )

        # loop until no tuples left
        for tup_idx, tup in enumerate(
            tuples(
                exp_occ,
                exp_virt,
                self.ref_nelec,
                self.ref_nhole,
                self.vanish_exc,
                self.order,
                order_start,
                occ_start,
                virt_start,
            ),
            tup_idx,
        ):
            # distribute tuples
            if tup_idx % mpi.global_size != mpi.global_rank:
                continue

            # write restart files and re-init time
            if rst_write and tup_idx % self.rst_freq < mpi.global_size:
                # mpi barrier
                mpi.local_comm.Barrier()

                # reduce hashes & increments onto global master
                if mpi.num_masters > 1 and mpi.local_master:
                    for tup_nocc in range(self.order + 1):
                        hashes[-1][tup_nocc][:] = mpi_reduce(
                            mpi.master_comm, hashes[-1][tup_nocc], root=0, op=MPI.SUM
                        )
                        if not mpi.global_master:
                            hashes[-1][tup_nocc][:].fill(0)
                        inc[-1][tup_nocc][:] = self._mpi_reduce_inc(
                            mpi.master_comm, inc[-1][tup_nocc], MPI.SUM
                        )
                        if not mpi.global_master:
                            inc[-1][tup_nocc][:].fill(0.0)

                # reduce increment statistics onto global master
                min_inc = self._mpi_reduce_target(mpi.global_comm, min_inc, MPI.MIN)
                mean_inc = self._mpi_reduce_target(mpi.global_comm, mean_inc, MPI.SUM)
                max_inc = self._mpi_reduce_target(mpi.global_comm, max_inc, MPI.MAX)
                if not mpi.global_master:
                    min_inc = self._init_target_inst(1.0e12, self.norb, self.nocc)
                    mean_inc = self._init_target_inst(0.0, self.norb, self.nocc)
                    max_inc = self._init_target_inst(0.0, self.norb, self.nocc)

                # reduce screen onto global master
                screen = mpi_reduce(mpi.global_comm, screen, root=0, op=MPI.MAX)

                # reduce mbe_idx onto global master
                mbe_idx = mpi.global_comm.allreduce(tup_idx, op=MPI.MIN)
                # send tup corresponding to mbe_idx to master
                if mpi.global_master:
                    if tup_idx == mbe_idx:
                        mbe_tup = tup
                    else:
                        mbe_tup = np.empty(self.order, dtype=np.int64)
                        mpi.global_comm.Recv(mbe_tup, source=MPI.ANY_SOURCE, tag=101)
                elif tup_idx == mbe_idx:
                    mpi.global_comm.Send(tup, dest=0, tag=101)
                # update rst_write
                rst_write = mbe_idx + self.rst_freq < n_tuples - mpi.global_size

                if mpi.global_master:
                    # write restart files
                    self._write_target_file(min_inc, "mbe_min_inc", self.order)
                    self._write_target_file(mean_inc, "mbe_mean_inc", self.order)
                    self._write_target_file(max_inc, "mbe_max_inc", self.order)
                    write_file(screen, "mbe_screen", order=self.order)
                    write_file(np.asarray(mbe_idx), "mbe_idx", order=self.order)
                    write_file(mbe_tup, "mbe_tup", order=self.order)
                    for tup_nocc in range(self.order + 1):
                        write_file(
                            hashes[-1][tup_nocc],
                            "mbe_hashes",
                            order=self.order,
                            nocc=tup_nocc,
                        )
                        self._write_inc_file(inc[-1][tup_nocc], self.order, tup_nocc)
                    self.time["mbe"][-1] += MPI.Wtime() - time
                    write_file(
                        np.asarray(self.time["mbe"][-1]),
                        "mbe_time_mbe",
                        order=self.order,
                    )
                    # re-init time
                    time = MPI.Wtime()
                    # print status
                    logger.info(mbe_status(self.order, mbe_idx / n_tuples))

            # pi-pruning
            if self.pi_prune:
                if not pi_prune(self.pi_orbs, self.pi_hashes, tup):
                    screen[tup] = SCREEN
                    continue

            # get core and cas indices
            core_idx, cas_idx = core_cas(self.nocc, self.ref_space, tup)

            # get h2e indices
            cas_idx_tril = idx_tril(cas_idx)

            # get h2e_cas
            h2e_cas = self.eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(self.hcore, self.vhf, core_idx, cas_idx)

            # calculate increment
            inc_tup, nelec_tup = self._inc(e_core, h1e_cas, h2e_cas, core_idx, cas_idx)

            # calculate increment
            if self.order > self.min_order:
                inc_tup -= self._sum(inc, hashes, tup)

            # get number of occupied orbitals in tuple
            nocc_tup = max(nelec_tup - self.ref_nelec)

            # get index in hash and increment arrays
            idx = tup_idx - sum(self.n_tuples["inc"][-1][:nocc_tup])

            # add hash and increment
            hashes[-1][nocc_tup][idx] = hash_1d(tup)
            inc[-1][nocc_tup][idx] = inc_tup

            # screening procedure
            screen[tup] = self._screen(inc_tup, screen, tup, self.screen_func)

            # debug print
            logger.debug(self._mbe_debug(nelec_tup, inc_tup, cas_idx, tup))

            # update increment statistics
            min_inc, mean_inc, max_inc = self._update_inc_stats(
                inc_tup, min_inc, mean_inc, max_inc, cas_idx
            )

        # mpi barrier
        mpi.global_comm.Barrier()

        # print final status
        if mpi.global_master:
            logger.info(mbe_status(self.order, 1.0))

        if mpi.local_master:
            for tup_nocc in range(self.order + 1):
                # allreduce hashes & increments among local masters
                hashes[-1][tup_nocc][:] = mpi_allreduce(
                    mpi.master_comm, hashes[-1][tup_nocc], op=MPI.SUM
                )
                inc[-1][tup_nocc][:] = self._mpi_allreduce_inc(
                    mpi.master_comm, inc[-1][tup_nocc], op=MPI.SUM
                )

                # sort hashes and increments
                inc[-1][tup_nocc][:] = inc[-1][tup_nocc][
                    np.argsort(hashes[-1][tup_nocc])
                ]
                hashes[-1][tup_nocc][:].sort()

        # increment statistics
        min_inc = self._mpi_reduce_target(mpi.global_comm, min_inc, MPI.MIN)
        mean_inc = self._mpi_reduce_target(mpi.global_comm, mean_inc, MPI.SUM)
        max_inc = self._mpi_reduce_target(mpi.global_comm, max_inc, MPI.MAX)
        if mpi.global_master:
            # total current-order increment
            tot = self._total_inc(inc[-1], mean_inc)
            if n_tuples != 0:
                mean_inc /= n_tuples

        # write restart files & save timings
        if mpi.global_master:
            if self.rst:
                self._write_target_file(min_inc, "mbe_min_inc", self.order)
                self._write_target_file(mean_inc, "mbe_mean_inc", self.order)
                self._write_target_file(max_inc, "mbe_max_inc", self.order)
                write_file(np.asarray(n_tuples), "mbe_idx", order=self.order)
                for tup_nocc in range(self.order + 1):
                    write_file(
                        hashes[-1][tup_nocc],
                        "mbe_hashes",
                        order=self.order,
                        nocc=tup_nocc,
                    )
                    self._write_inc_file(inc[-1][tup_nocc], self.order, tup_nocc)
            self.time["mbe"][-1] += MPI.Wtime() - time

        # allreduce screen
        tot_screen = mpi_allreduce(mpi.global_comm, screen, op=MPI.MAX)

        # update expansion space wrt screened orbitals
        tot_screen = tot_screen[self.exp_space[-1]]
        thres = 1.0 if self.order < self.screen_start else self.screen_perc
        screen_idx = int(thres * self.exp_space[-1].size)
        if self.screen_func == "rnd":
            rng = np.random.default_rng()
            self.exp_space.append(
                rng.choice(self.exp_space[-1], size=screen_idx, replace=False)
            )
        else:
            self.exp_space.append(
                self.exp_space[-1][np.sort(np.argsort(tot_screen)[::-1][:screen_idx])]
            )

        # write restart files
        if mpi.global_master:
            if self.rst:
                write_file(tot_screen, "mbe_screen", order=self.order)
                write_file(self.exp_space[-1], "exp_space", order=self.order + 1)

        # mpi barrier
        mpi.local_comm.Barrier()

        # append window to hashes
        if len(self.hashes) == len(self.n_tuples["inc"]):
            self.hashes[-1] = hashes_win
        else:
            self.hashes.append(hashes_win)

        # append window to increments
        if len(self.incs) == len(self.n_tuples["inc"]):
            self.incs[-1] = inc_win
        else:
            self.incs.append(inc_win)

        if mpi.global_master:
            # append total property
            self.mbe_tot_prop.append(tot)
            if self.order > self.min_order:
                self.mbe_tot_prop[-1] += self.mbe_tot_prop[-2]

            # append increment statistics
            if len(self.mean_inc) > self.order - self.min_order:
                self.mean_inc[-1] = mean_inc
                self.min_inc[-1] = min_inc
                self.max_inc[-1] = max_inc
            else:
                self.mean_inc.append(mean_inc)
                self.min_inc.append(min_inc)
                self.max_inc.append(max_inc)

        return

    def _purge(self, mpi: MPICls) -> None:
        """
        this function purges the lower-order hashes & increments
        """
        # wake up slaves
        if mpi.global_master:
            msg = {"task": "purge", "order": self.order}
            mpi.global_comm.bcast(msg, root=0)

        # do not purge at min_order or in case of no screened orbs
        if (
            self.order == self.min_order
            or self.screen_orbs.size == 0
            or self.exp_space[-1].size < self.order + 1
        ):
            self.time["purge"].append(0.0)
            return

        # init time
        if mpi.global_master:
            time = MPI.Wtime()

        # occupied and virtual expansion spaces
        exp_occ = self.exp_space[-1][self.exp_space[-1] < self.nocc]
        exp_virt = self.exp_space[-1][self.nocc <= self.exp_space[-1]]

        # loop over previous orders
        for order in range(1, self.order + 1):
            k = order - 1
            # loop over number of occupied orbitals
            for tup_nocc in range(order + 1):
                l = tup_nocc
                # load k-th order hashes and increments
                hashes = open_shared_win(
                    self.hashes[k][l],
                    np.int64,
                    (self.n_tuples["inc"][k][l],),
                )
                inc = self._open_shared_inc(
                    self.incs[k][l], self.n_tuples["inc"][k][l], order, tup_nocc
                )

                # check if hashes and increments are available
                if hashes.size == 0:
                    continue

                # mpi barrier
                mpi.local_comm.barrier()

                # init list for storing hashes at order k
                hashes_lst: List[int] = []

                # init list for storing increments at order k
                inc_lst: List[IncType] = []

                # loop until no tuples left
                for tup_idx, tup in enumerate(
                    tuples_with_nocc(exp_occ, exp_virt, order, tup_nocc)
                ):
                    # distribute tuples
                    if tup_idx % mpi.global_size != mpi.global_rank:
                        continue

                    # compute index
                    idx = hash_lookup(hashes, hash_1d(tup))

                    # add inc_tup and its hash to lists of increments/hashes
                    if idx is not None:
                        inc_lst.append(inc[idx])
                        hashes_lst.append(hash_1d(tup))

                # recast hashes_lst and inc_lst as np.array and TargetType
                hashes_arr = np.array(hashes_lst, dtype=np.int64)
                inc_arr = self._flatten_inc(inc_lst, k)

                # deallocate k-th order hashes and increments
                self.hashes[k][l].Free()
                self._free_inc(self.incs[k][l])

                # number of hashes for every rank
                recv_counts = np.array(mpi.global_comm.allgather(hashes_arr.size))

                # update n_tuples
                self.n_tuples["inc"][k][l] = int(np.sum(recv_counts))

                # init hashes for present order
                hashes_win = MPI.Win.Allocate_shared(
                    8 * self.n_tuples["inc"][k][l] if mpi.local_master else 0,
                    8,
                    comm=mpi.local_comm,  # type: ignore
                )
                self.hashes[k][l] = hashes_win
                hashes = open_shared_win(
                    hashes_win, np.int64, (self.n_tuples["inc"][k][l],)
                )

                # gatherv hashes on global master
                hashes[:] = mpi_gatherv(
                    mpi.global_comm, hashes_arr, hashes, recv_counts
                )

                # bcast hashes among local masters
                if mpi.local_master:
                    hashes[:] = mpi_bcast(mpi.master_comm, hashes)

                # init increments for present order
                self.incs[k][l] = self._allocate_shared_inc(
                    self.n_tuples["inc"][k][l],
                    mpi.local_master,
                    mpi.local_comm,
                    order,
                    tup_nocc,
                )
                inc = self._open_shared_inc(
                    self.incs[k][l],
                    self.n_tuples["inc"][k][l],
                    order,
                    tup_nocc,
                )

                # gatherv increments on global master
                inc[:] = self._mpi_gatherv_inc(mpi.global_comm, inc_arr, inc)

                # bcast increments among local masters
                if mpi.local_master:
                    inc[:] = self._mpi_bcast_inc(mpi.master_comm, inc)

                # sort hashes and increments
                if mpi.local_master:
                    inc[:] = inc[np.argsort(hashes)]
                    hashes[:].sort()

        # mpi barrier
        mpi.global_comm.barrier()

        # save timing
        if mpi.global_master:
            self.time["purge"].append(MPI.Wtime() - time)

        return

    def free_ints(self) -> None:
        """
        this function deallocates integrals in shared memory after the calculation is
        done
        """
        # free integrals
        self.hcore_win.Free()
        self.eri_win.Free()
        self.vhf_win.Free()

        return

    @abstractmethod
    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
    ) -> Tuple[TargetType, np.ndarray]:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """

    def _kernel(
        self,
        method: str,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
    ) -> TargetType:
        """
        this function return the result property from a given method
        """
        if method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"]:
            res = self._cc_kernel(method, core_idx, cas_idx, nelec, h1e, h2e, False)

        elif method == "fci":
            res = self._fci_kernel(e_core, h1e, h2e, core_idx, cas_idx, nelec)

        return res

    @abstractmethod
    def _fci_kernel(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
    ) -> TargetType:
        """
        this function returns the results of a fci calculation
        """

    @abstractmethod
    def _cc_kernel(
        self,
        method: str,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        h1e: np.ndarray,
        h2e: np.ndarray,
        higher_amp_extrap: bool,
    ) -> TargetType:
        """
        this function returns the results of a cc calculation
        """

    @abstractmethod
    def _sum(
        self, inc: List[List[IncType]], hashes: List[List[np.ndarray]], tup: np.ndarray
    ) -> TargetType:
        """
        this function performs a recursive summation and returns the final increment
        associated with a given tuple
        """

    @staticmethod
    @abstractmethod
    def _write_target_file(prop: TargetType, string: str, order: int) -> None:
        """
        this function defines how to write restart files for instances of the target
        type
        """

    @staticmethod
    @abstractmethod
    def _read_target_file(file: str) -> TargetType:
        """
        this function reads files of attributes with the target type
        """

    @abstractmethod
    def _init_target_inst(
        self, value: float, tup_norb: int, tup_nocc: int
    ) -> TargetType:
        """
        this function initializes an instance of the target type
        """

    @staticmethod
    @abstractmethod
    def _mpi_reduce_target(
        comm: MPI.Comm, values: TargetType, op: MPI.Op
    ) -> TargetType:
        """
        this function performs a MPI reduce operation on values of the target type
        """

    def _load_hashes(
        self,
        global_master: bool,
        local_master: bool,
        local_comm: MPI.Comm,
        rst_read: bool,
    ) -> Tuple[List[List[np.ndarray]], List[MPI.Win]]:
        """
        this function loads all previous-order hashes and initializes the current-order
        hashes
        """
        # load hashes for previous orders
        hashes: List[List[np.ndarray]] = []
        for order in range(1, self.order):
            k = order - 1
            hashes.append([])
            for tup_nocc in range(order + 1):
                l = tup_nocc
                hashes[k].append(
                    open_shared_win(
                        self.hashes[k][l], np.int64, (self.n_tuples["inc"][k][l],)
                    )
                )

        # init hashes for present order
        hashes_win: List[MPI.Win] = []
        hashes.append([])
        for tup_nocc in range(self.order + 1):
            l = tup_nocc
            if rst_read:
                hashes_win.append(self.hashes[-1][l])
            else:
                hashes_win.append(
                    MPI.Win.Allocate_shared(
                        8 * self.n_tuples["inc"][-1][l] if local_master else 0,
                        8,
                        comm=local_comm,  # type: ignore
                    )
                )
            hashes[-1].append(
                open_shared_win(
                    hashes_win[-1], np.int64, (self.n_tuples["inc"][-1][l],)
                )
            )
            if local_master and not global_master:
                hashes[-1][l][:].fill(0)

        return hashes, hashes_win

    @staticmethod
    @abstractmethod
    def _write_inc_file(inc: IncType, order: int, nocc: int) -> None:
        """
        this function defines how to write increment restart files
        """

    @staticmethod
    @abstractmethod
    def _read_inc_file(file: str) -> IncType:
        """
        this function defines reads the increment restart files
        """

    def _load_inc(
        self,
        global_master: bool,
        local_master: bool,
        local_comm: MPI.Comm,
        rst_read: bool,
    ) -> Tuple[List[List[IncType]], List[MPIWinType]]:
        """
        this function loads all previous-order increments and initializes the
        current-order increments
        """
        # load increments for previous orders
        inc: List[List[IncType]] = []
        for order in range(1, self.order):
            k = order - 1
            inc.append([])
            for tup_nocc in range(order + 1):
                l = tup_nocc
                inc[k].append(
                    self._open_shared_inc(
                        self.incs[k][l], self.n_tuples["inc"][k][l], order, tup_nocc
                    )
                )

        # init increments for present order
        inc_win: List[MPIWinType] = []
        inc.append([])
        for tup_nocc in range(self.order + 1):
            l = tup_nocc
            if rst_read:
                inc_win.append(self.incs[-1][l])
            else:
                inc_win.append(
                    self._allocate_shared_inc(
                        self.n_tuples["inc"][-1][l],
                        local_master,
                        local_comm,
                        self.order,
                        tup_nocc,
                    )
                )
            inc[-1].append(
                self._open_shared_inc(
                    inc_win[-1],
                    self.n_tuples["inc"][-1][l],
                    self.order,
                    tup_nocc,
                )
            )
            if (local_master and not global_master) or (global_master and not rst_read):
                inc[-1][l][:].fill(0)

        return inc, inc_win

    @abstractmethod
    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm, tup_norb: int, tup_nocc
    ) -> MPIWinType:
        """
        this function allocates a shared increment window
        """

    @abstractmethod
    def _open_shared_inc(
        self, window: MPIWinType, n_tuples: int, tup_orb: int, tup_nocc: int
    ) -> IncType:
        """
        this function opens a shared increment window
        """

    @staticmethod
    @abstractmethod
    def _mpi_bcast_inc(comm: MPI.Comm, inc: IncType) -> IncType:
        """
        this function bcasts the increments
        """

    @staticmethod
    @abstractmethod
    def _mpi_reduce_inc(comm: MPI.Comm, inc: IncType, op: MPI.Op) -> IncType:
        """
        this function performs a MPI reduce operation on the increments
        """

    @staticmethod
    @abstractmethod
    def _mpi_allreduce_inc(comm: MPI.Comm, inc: IncType, op: MPI.Op) -> IncType:
        """
        this function performs a MPI allreduce operation on the increments
        """

    @staticmethod
    @abstractmethod
    def _mpi_gatherv_inc(
        comm: MPI.Comm, send_inc: IncType, recv_inc: IncType
    ) -> IncType:
        """
        this function performs a MPI gatherv operation on the increments
        """

    @staticmethod
    @abstractmethod
    def _flatten_inc(inc_lst: List[IncType], order: int) -> IncType:
        """
        this function flattens the supplied increment arrays
        """

    @staticmethod
    @abstractmethod
    def _free_inc(inc_win: MPIWinType) -> None:
        """
        this function frees the supplied increment windows
        """

    @staticmethod
    @abstractmethod
    def _screen(
        inc_tup: TargetType, screen: np.ndarray, tup: np.ndarray, screen_func: str
    ) -> np.ndarray:
        """
        this function modifies the screening array
        """

    @abstractmethod
    def _update_inc_stats(
        self,
        inc_tup: TargetType,
        min_inc: TargetType,
        mean_inc: TargetType,
        max_inc: TargetType,
        cas_idx: np.ndarray,
    ) -> Tuple[TargetType, TargetType, TargetType]:
        """
        this function updates the increment statistics
        """

    @staticmethod
    @abstractmethod
    def _total_inc(inc: List[IncType], mean_inc: TargetType) -> TargetType:
        """
        this function calculates the total current-order increment
        """

    @abstractmethod
    def _mbe_debug(
        self,
        nelec_tup: np.ndarray,
        inc_tup: TargetType,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> str:
        """
        this function prints mbe debug information
        """

    @abstractmethod
    def _mbe_results(self, order: int) -> str:
        """
        this function prints mbe results statistics for an energy or excitation energy
        calculation
        """

    @abstractmethod
    def _prop_summ(
        self,
    ) -> Tuple[
        Union[float, np.floating], Union[float, np.floating], Union[float, np.floating]
    ]:
        """
        this function returns the hf, base and total property as a float
        """

    @abstractmethod
    def _results_prt(self) -> str:
        """
        this function prints the target property table
        """


# define variable type for single target properties
SingleTargetType = TypeVar("SingleTargetType", float, np.ndarray)


class SingleTargetExpCls(
    ExpCls[SingleTargetType, np.ndarray, MPI.Win, int, np.ndarray], metaclass=ABCMeta
):
    """
    this class holds all function definitions for single-target expansions irrespective
    of whether the target is a scalar or an array type
    """

    def _state_occup(self) -> None:
        """
        this function initializes certain state attributes for a single state
        """
        self.nocc = np.max(self.nelec)
        self.spin = abs(self.nelec[0] - self.nelec[1])
        self.occup = get_occup(self.norb, self.nelec)

    def _sum(
        self,
        inc: List[List[np.ndarray]],
        hashes: List[List[np.ndarray]],
        tup: np.ndarray,
    ) -> SingleTargetType:
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
            # loop over number of occupied orbitals
            for l in range(k + 1):
                # check if hashes are available
                if hashes[k - self.min_order][l].size > 0:
                    # loop over subtuples
                    for tup_sub in tuples_with_nocc(tup_occ, tup_virt, k, l):
                        # compute index
                        idx = hash_lookup(
                            hashes[k - self.min_order][l], hash_1d(tup_sub)
                        )

                        # sum up order increments
                        if idx is not None:
                            res[k - self.min_order] += inc[k - self.min_order][l][idx]

        return np.sum(res, axis=0)

    @abstractmethod
    def _zero_target_arr(self, length: int):
        """
        this function initializes an array of the target type with value zero
        """

    @abstractmethod
    def _init_target_inst(self, value: float, *args: int) -> SingleTargetType:
        """
        this function initializes an instance of the target type
        """

    @staticmethod
    def _write_inc_file(inc: np.ndarray, order: int, nocc: int) -> None:
        """
        this function defines writes the increment restart files
        """
        write_file(inc, "mbe_inc", order=order, nocc=nocc)

    @staticmethod
    def _read_inc_file(file: str) -> np.ndarray:
        """
        this function defines reads the increment restart files
        """
        return np.load(os.path.join(RST, file))

    @abstractmethod
    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm, *args: int
    ) -> MPI.Win:
        """
        this function allocates a shared increment window
        """

    @abstractmethod
    def _open_shared_inc(
        self, window: MPI.Win, n_tuples: int, *args: int
    ) -> np.ndarray:
        """
        this function opens a shared increment window
        """

    @staticmethod
    def _mpi_bcast_inc(comm: MPI.Comm, inc: np.ndarray) -> np.ndarray:
        """
        this function bcasts the increments
        """
        return mpi_bcast(comm, inc)

    @staticmethod
    def _mpi_reduce_inc(comm: MPI.Comm, inc: np.ndarray, op: MPI.Op) -> np.ndarray:
        """
        this function performs a MPI reduce operation on the increments
        """
        return mpi_reduce(comm, inc, root=0, op=op)

    @staticmethod
    def _mpi_allreduce_inc(comm: MPI.Comm, inc: np.ndarray, op: MPI.Op) -> np.ndarray:
        """
        this function performs a MPI allreduce operation on the increments
        """
        return mpi_allreduce(comm, inc, op=op)

    @staticmethod
    def _mpi_gatherv_inc(
        comm: MPI.Comm, send_inc: np.ndarray, recv_inc: np.ndarray
    ) -> np.ndarray:
        """
        this function performs a MPI gatherv operation on the increments
        """
        # number of increments for every rank
        recv_counts = np.array(comm.allgather(send_inc.size))

        return mpi_gatherv(comm, send_inc, recv_inc, recv_counts)

    @staticmethod
    def _free_inc(inc_win: MPI.Win) -> None:
        """
        this function frees the supplied increment windows
        """
        inc_win.Free()

    def _update_inc_stats(
        self,
        inc_tup: SingleTargetType,
        min_inc: SingleTargetType,
        mean_inc: SingleTargetType,
        max_inc: SingleTargetType,
        cas_idx: np.ndarray,
    ) -> Tuple[SingleTargetType, SingleTargetType, SingleTargetType]:
        """
        this function updates the increment statistics
        """
        min_inc = np.minimum(min_inc, np.abs(inc_tup))
        mean_inc += inc_tup
        max_inc = np.maximum(max_inc, np.abs(inc_tup))

        return min_inc, mean_inc, max_inc
