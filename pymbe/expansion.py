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
from typing import TYPE_CHECKING, cast, TypeVar, Generic, Tuple, Union

from pymbe.kernel import e_core_h1e
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
    pi_space,
    natural_keys,
    n_tuples,
    is_file,
    read_file,
    write_file,
    pi_prune,
    tuples,
    nelecs,
    nholes,
    start_idx,
    core_cas,
    idx_tril,
    hash_1d,
    hash_lookup,
)
from pymbe.parallel import (
    mpi_reduce,
    mpi_allreduce,
    mpi_bcast,
    mpi_gatherv,
    open_shared_win,
)
from pymbe.results import atom_prt, summary_prt, timings_prt

if TYPE_CHECKING:

    from pyscf import gto
    from typing import List, Dict, Optional

    from pymbe.pymbe import MBE
    from pymbe.parallel import MPICls


# define variable type for target properties
TargetType = TypeVar("TargetType", float, np.ndarray, RDMCls)

# define variable type for increment arrays
IncType = TypeVar("IncType", np.ndarray, packedRDMCls)

# define variable type for MPI windows of increment arrays
MPIWinType = TypeVar("MPIWinType", MPI.Win, Tuple[MPI.Win, MPI.Win])


# get logger
logger = logging.getLogger("pymbe_logger")


SCREEN = 1000.0  # random, non-sensical number


class ExpCls(Generic[TargetType, IncType, MPIWinType], metaclass=ABCMeta):
    """
    this class contains the pymbe expansion attributes
    """

    def __init__(
        self, mbe: MBE, hf_prop: TargetType, ref_prop: TargetType, base_prop: TargetType
    ) -> None:
        """
        init expansion attributes
        """
        # expansion model
        self.method: str = mbe.method
        self.fci_solver: str = mbe.fci_solver
        self.cc_backend: str = mbe.cc_backend
        self.hf_guess: bool = mbe.hf_guess

        # target property
        self.target: str = mbe.target

        # system
        self.nuc_energy = cast(float, mbe.nuc_energy)
        self.ncore: int = mbe.ncore
        self.nocc = cast(int, mbe.nocc)
        self.norb = cast(int, mbe.norb)
        self.spin = cast(int, mbe.spin)
        self.point_group = cast(str, mbe.point_group)
        self.orbsym = cast(np.ndarray, mbe.orbsym)
        self.fci_state_sym = cast(int, mbe.fci_state_sym)
        self.fci_state_root = cast(int, mbe.fci_state_root)

        # hf calculation
        self.hf_prop: TargetType = hf_prop
        self.occup = cast(np.ndarray, mbe.occup)

        # integrals
        hcore, eri, vhf = self._int_wins(
            mbe.hcore, mbe.eri, mbe.vhf, mbe.mpi, self.norb, self.nocc
        )
        self.hcore: MPI.Win = hcore
        self.vhf: MPI.Win = vhf
        self.eri: MPI.Win = eri

        # orbital representation
        self.orb_type: str = mbe.orb_type

        # reference space
        self.ref_space: np.ndarray = mbe.ref_space
        self.ref_n_elecs = nelecs(self.occup, self.ref_space)
        self.ref_n_holes = nholes(self.ref_n_elecs, self.ref_space)
        self.ref_prop: TargetType = ref_prop

        # expansion space
        self.exp_space: List[np.ndarray] = [
            np.array(
                [i for i in range(self.ncore, self.norb) if i not in self.ref_space],
                dtype=np.int64,
            )
        ]

        # base model
        self.base_method: Optional[str] = mbe.base_method
        self.base_prop: TargetType = base_prop

        # total mbe property
        self.mbe_tot_prop: List[TargetType] = []

        # increment windows
        self.incs: List[MPIWinType] = []

        # hash windows
        self.hashes: List[MPI.Win] = []

        # timings
        self.time: Dict[str, List[float]] = {"mbe": [], "purge": []}

        # statistics
        self.min_inc: List[TargetType] = []
        self.mean_inc: List[TargetType] = []
        self.max_inc: List[TargetType] = []

        # number of tuples
        self.n_tuples: Dict[str, List[int]] = {"theo": [], "calc": [], "inc": []}

        # screening
        self.screen_start: int = mbe.screen_start
        self.screen_perc: float = mbe.screen_perc
        self.screen = np.zeros(self.norb, dtype=np.float64)
        self.screen_orbs = np.array([], dtype=np.int64)

        # restart
        self.rst: bool = mbe.rst
        self.rst_freq: int = mbe.rst_freq
        self.restarted: bool = mbe.restarted

        # order
        self.order: int = 0
        self.min_order: int = 1

        if self.restarted:
            start_order = self._restart_main(mbe.mpi)
        else:
            start_order = self.min_order
        self.start_order: int = start_order

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
                        self.n_tuples["calc"][i - self.min_order],
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
                self.n_tuples["theo"].append(
                    n_tuples(
                        self.exp_space[0][self.exp_space[0] < self.nocc],
                        self.exp_space[0][self.nocc <= self.exp_space[0]],
                        self.ref_n_elecs,
                        self.ref_n_holes,
                        -1,
                        self.order,
                    )
                )
                self.n_tuples["calc"].append(
                    n_tuples(
                        self.exp_space[-1][self.exp_space[-1] < self.nocc],
                        self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                        self.ref_n_elecs,
                        self.ref_n_holes,
                        self.vanish_exc,
                        self.order,
                    )
                )
                self.n_tuples["inc"].append(self.n_tuples["calc"][-1])
                if self.rst:
                    write_file(
                        self.order,
                        np.asarray(self.n_tuples["theo"][-1]),
                        "mbe_n_tuples_theo",
                    )
                    write_file(
                        self.order,
                        np.asarray(self.n_tuples["calc"][-1]),
                        "mbe_n_tuples_calc",
                    )
                    write_file(
                        self.order,
                        np.asarray(self.n_tuples["inc"][-1]),
                        "mbe_n_tuples_inc",
                    )

            # print mbe header
            logger.info(
                mbe_header(
                    self.order,
                    self.n_tuples["calc"][-1],
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
                    for k in range(self.order - self.min_order + 1):
                        hashes = open_shared_win(
                            self.hashes[k], np.int64, (self.n_tuples["inc"][k],)
                        )
                        write_file(k + self.min_order, hashes, "mbe_hashes")
                        inc = self._open_shared_inc(
                            self.incs[k], self.n_tuples["inc"][k], k
                        )
                        self._write_inc_file(k + self.min_order, inc)
                        write_file(
                            k + self.min_order,
                            np.asarray(self.n_tuples["inc"][k]),
                            "mbe_n_tuples_inc",
                        )
                else:
                    hashes = open_shared_win(
                        self.hashes[-1], np.int64, (self.n_tuples["inc"][-1],)
                    )
                    write_file(self.order, hashes, "mbe_hashes")
                    inc = self._open_shared_inc(
                        self.incs[-1],
                        self.n_tuples["inc"][-1],
                        self.order - self.min_order,
                    )
                    self._write_inc_file(self.order, inc)
                    write_file(
                        self.order,
                        np.asarray(self.n_tuples["inc"][-1]),
                        "mbe_n_tuples_inc",
                    )
                self._write_target_file(
                    self.order, self.mbe_tot_prop[-1], "mbe_tot_prop"
                )
                write_file(self.order, np.asarray(self.time["mbe"][-1]), "mbe_time_mbe")
                write_file(
                    self.order, np.asarray(self.time["purge"][-1]), "mbe_time_purge"
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
                    self.n_tuples["inc"].append(
                        n_tuples(
                            self.exp_space[-1][self.exp_space[-1] < self.nocc],
                            self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                            self.ref_n_elecs,
                            self.ref_n_holes,
                            self.vanish_exc,
                            self.order,
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

        # print atom info
        if mol and mol.atom:
            string += atom_prt(mol) + "\n\n"

        # print summary
        string += summary_prt(mpi, self, *self._prop_summ()) + "\n\n"

        # print timings
        string += timings_prt(self, self.method) + "\n\n"

        # print results
        string += self._results_prt()

        return string

    def _int_wins(
        self,
        hcore_in: Optional[np.ndarray],
        eri_in: Optional[np.ndarray],
        vhf_in: Optional[np.ndarray],
        mpi: MPICls,
        norb: int,
        nocc: int,
    ) -> Tuple[MPI.Win, MPI.Win, MPI.Win]:
        """
        this function creates shared memory windows for integrals on every node
        """
        # allocate hcore in shared mem
        hcore_win = MPI.Win.Allocate_shared(
            8 * norb**2 if mpi.local_master else 0,
            8,
            comm=mpi.local_comm,  # type: ignore
        )
        hcore = open_shared_win(hcore_win, np.float64, 2 * (norb,))

        # set hcore on global master
        if mpi.global_master:
            hcore[:] = cast(np.ndarray, hcore_in)

        # mpi_bcast hcore
        if mpi.num_masters > 1 and mpi.local_master:
            hcore[:] = mpi_bcast(mpi.master_comm, hcore)

        # allocate eri in shared mem
        eri_win = MPI.Win.Allocate_shared(
            8 * (norb * (norb + 1) // 2) ** 2 if mpi.local_master else 0,
            8,
            comm=mpi.local_comm,  # type: ignore
        )
        eri = open_shared_win(eri_win, np.float64, 2 * (norb * (norb + 1) // 2,))

        # set eri on global master
        if mpi.global_master:
            eri[:] = cast(np.ndarray, eri_in)

        # mpi_bcast eri
        if mpi.num_masters > 1 and mpi.local_master:
            eri[:] = mpi_bcast(mpi.master_comm, eri)

        # allocate vhf in shared mem
        vhf_win = MPI.Win.Allocate_shared(
            8 * nocc * norb**2 if mpi.local_master else 0,
            8,
            comm=mpi.local_comm,  # type: ignore
        )
        vhf = open_shared_win(vhf_win, np.float64, (nocc, norb, norb))

        # set vhf on global master
        if mpi.global_master:
            vhf[:] = cast(np.ndarray, vhf_in)

        # mpi_bcast vhf
        if mpi.num_masters > 1 and mpi.local_master:
            vhf[:] = mpi_bcast(mpi.master_comm, vhf)

        # mpi barrier
        mpi.global_comm.Barrier()

        return hcore_win, eri_win, vhf_win

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
                    if "theo" in files[i]:
                        self.n_tuples["theo"].append(
                            np.load(os.path.join(RST, files[i])).tolist()
                        )
                    if "inc" in files[i]:
                        self.n_tuples["inc"].append(
                            np.load(os.path.join(RST, files[i])).tolist()
                        )
                    if "calc" in files[i]:
                        self.n_tuples["calc"].append(
                            np.load(os.path.join(RST, files[i])).tolist()
                        )
            mpi.global_comm.bcast(self.n_tuples, root=0)
        else:
            self.n_tuples = mpi.global_comm.bcast(None, root=0)

        # loop over all other files
        for i in range(len(files)):

            # read hashes
            if "mbe_hashes" in files[i]:
                n_tuples = self.n_tuples["inc"][len(self.hashes)]
                self.hashes.append(
                    MPI.Win.Allocate_shared(
                        8 * n_tuples if mpi.local_master else 0,
                        8,
                        comm=mpi.local_comm,  # type: ignore
                    )
                )
                hashes = open_shared_win(self.hashes[-1], np.int64, (n_tuples,))
                if mpi.global_master:
                    hashes[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    hashes[:] = mpi_bcast(mpi.master_comm, hashes)
                mpi.local_comm.Barrier()

            # read increments
            elif "mbe_inc" in files[i]:
                n_tuples = self.n_tuples["inc"][len(self.incs)]
                self.incs.append(
                    self._allocate_shared_inc(
                        n_tuples, mpi.local_master, mpi.local_comm
                    )
                )
                inc = self._open_shared_inc(self.incs[-1], n_tuples, len(self.incs) - 1)
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
            rst_read = is_file(self.order, "mbe_idx") and is_file(self.order, "mbe_tup")
            # start indices
            tup_idx = read_file(self.order, "mbe_idx").item() if rst_read else 0
            # start tuples
            tup = read_file(self.order, "mbe_tup") if rst_read else None
            # wake up slaves
            msg = {
                "task": "mbe",
                "order": self.order,
                "rst_read": rst_read,
                "tup_idx": tup_idx,
                "tup": tup,
            }
            mpi.global_comm.bcast(msg, root=0)

        # load hcore
        hcore = open_shared_win(self.hcore, np.float64, 2 * (self.norb,))

        # load eri
        eri = open_shared_win(
            self.eri, np.float64, 2 * (self.norb * (self.norb + 1) // 2,)
        )

        # load vhf
        vhf = open_shared_win(self.vhf, np.float64, (self.nocc, self.norb, self.norb))

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
            min_inc = self._init_target_inst(1.0e12, self.norb)
            mean_inc = self._init_target_inst(0.0, self.norb)
            max_inc = self._init_target_inst(0.0, self.norb)

        # init pair_corr statistics
        if (
            self.ref_space.size == 0
            and self.order == 2
            and self.base_method is None
            and self.target != "rdm12"
        ):
            pair_corr: Optional[List[np.ndarray]] = [
                np.zeros(
                    self.n_tuples["inc"][self.order - self.min_order], dtype=np.float64
                ),
                np.zeros(
                    [self.n_tuples["inc"][self.order - self.min_order], 2],
                    dtype=np.int32,
                ),
            ]
        else:
            pair_corr = None

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

        # set rst_write
        rst_write = (
            self.rst and mpi.global_size < self.rst_freq < self.n_tuples["inc"][-1]
        )

        # start tuples
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
                self.ref_n_elecs,
                self.ref_n_holes,
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
                    hashes[-1][:] = mpi_reduce(
                        mpi.master_comm, hashes[-1], root=0, op=MPI.SUM
                    )
                    if not mpi.global_master:
                        hashes[-1][:].fill(0)
                    inc[-1][:] = self._mpi_reduce_inc(mpi.master_comm, inc[-1], MPI.SUM)
                    if not mpi.global_master:
                        inc[-1][:].fill(0.0)

                # reduce increment statistics onto global master
                min_inc = self._mpi_reduce_target(mpi.global_comm, min_inc, MPI.MIN)
                mean_inc = self._mpi_reduce_target(mpi.global_comm, mean_inc, MPI.SUM)
                max_inc = self._mpi_reduce_target(mpi.global_comm, max_inc, MPI.MAX)
                if not mpi.global_master:
                    min_inc = self._init_target_inst(1.0e12, self.norb)
                    mean_inc = self._init_target_inst(0.0, self.norb)
                    max_inc = self._init_target_inst(0.0, self.norb)

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
                rst_write = (
                    mbe_idx + self.rst_freq < self.n_tuples["inc"][-1] - mpi.global_size
                )

                if mpi.global_master:
                    # write restart files
                    self._write_target_file(self.order, min_inc, "mbe_min_inc")
                    self._write_target_file(self.order, mean_inc, "mbe_mean_inc")
                    self._write_target_file(self.order, max_inc, "mbe_max_inc")
                    write_file(self.order, screen, "mbe_screen")
                    write_file(self.order, np.asarray(mbe_idx), "mbe_idx")
                    write_file(self.order, mbe_tup, "mbe_tup")
                    write_file(self.order, hashes[-1], "mbe_hashes")
                    self._write_inc_file(self.order, inc[-1])
                    self.time["mbe"][-1] += MPI.Wtime() - time
                    write_file(
                        self.order, np.asarray(self.time["mbe"][-1]), "mbe_time_mbe"
                    )
                    # re-init time
                    time = MPI.Wtime()
                    # print status
                    logger.info(
                        mbe_status(self.order, mbe_idx / self.n_tuples["inc"][-1])
                    )

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
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(self.nuc_energy, hcore, vhf, core_idx, cas_idx)

            # calculate increment
            inc_tup, n_elecs_tup = self._inc(
                e_core, h1e_cas, h2e_cas, core_idx, cas_idx, tup
            )

            # calculate increment
            if self.order > self.min_order:
                inc_tup -= self._sum(inc, hashes, tup)

            # add hash and increment
            hashes[-1][tup_idx] = hash_1d(tup)
            inc[-1][tup_idx] = inc_tup

            # screening procedure
            screen[tup] = self._screen(inc_tup, screen, tup)

            # debug print
            logger.debug(self._mbe_debug(n_elecs_tup, inc_tup, cas_idx, tup))

            # update increment statistics
            min_inc, mean_inc, max_inc = self._update_inc_stats(
                inc_tup, min_inc, mean_inc, max_inc, tup
            )

            # update pair_corr statistics
            if pair_corr is not None:
                inc_arr = np.asarray(inc_tup)
                if self.target in ["energy", "excitation"]:
                    pair_corr[0][tup_idx] = inc_arr
                elif self.target in ["dipole", "trans"]:
                    pair_corr[0][tup_idx] = inc_arr[np.argmax(np.abs(inc_arr))]
                pair_corr[1][tup_idx] = tup

        # mpi barrier
        mpi.global_comm.Barrier()

        # print final status
        if mpi.global_master:
            logger.info(mbe_status(self.order, 1.0))

        # allreduce hashes & increments among local masters
        if mpi.local_master:
            hashes[-1][:] = mpi_allreduce(mpi.master_comm, hashes[-1], op=MPI.SUM)
            inc[-1][:] = self._mpi_allreduce_inc(mpi.master_comm, inc[-1], op=MPI.SUM)

        # sort hashes and increments
        if mpi.local_master:
            inc[-1][:] = inc[-1][np.argsort(hashes[-1])]
            hashes[-1][:].sort()

        # increment statistics
        min_inc = self._mpi_reduce_target(mpi.global_comm, min_inc, MPI.MIN)
        mean_inc = self._mpi_reduce_target(mpi.global_comm, mean_inc, MPI.SUM)
        max_inc = self._mpi_reduce_target(mpi.global_comm, max_inc, MPI.MAX)
        if mpi.global_master:
            # total current-order increment
            tot = self._total_inc(inc[-1], mean_inc)
            if self.n_tuples["inc"][-1] != 0:
                mean_inc /= self.n_tuples["inc"][-1]

        # pair_corr statistics
        if pair_corr is not None:
            pair_corr = [
                mpi_reduce(mpi.global_comm, pair_corr[0], root=0, op=MPI.SUM),
                mpi_reduce(mpi.global_comm, pair_corr[1], root=0, op=MPI.SUM),
            ]

        # write restart files & save timings
        if mpi.global_master:
            if self.rst:
                self._write_target_file(self.order, min_inc, "mbe_min_inc")
                self._write_target_file(self.order, mean_inc, "mbe_mean_inc")
                self._write_target_file(self.order, max_inc, "mbe_max_inc")
                write_file(self.order, np.asarray(self.n_tuples["inc"][-1]), "mbe_idx")
                write_file(self.order, hashes[-1], "mbe_hashes")
                self._write_inc_file(self.order, inc[-1])
            self.time["mbe"][-1] += MPI.Wtime() - time

        # allreduce screen
        tot_screen = mpi_allreduce(mpi.global_comm, screen, op=MPI.MAX)

        # update expansion space wrt screened orbitals
        tot_screen = tot_screen[self.exp_space[-1]]
        thres = 1.0 if self.order < self.screen_start else self.screen_perc
        screen_idx = int(thres * self.exp_space[-1].size)
        self.exp_space.append(
            self.exp_space[-1][np.sort(np.argsort(tot_screen)[::-1][:screen_idx])]
        )

        # write restart files
        if mpi.global_master:
            if self.rst:
                write_file(self.order, tot_screen, "mbe_screen")
                write_file(self.order + 1, self.exp_space[-1], "exp_space")

        # mpi barrier
        mpi.local_comm.Barrier()

        if mpi.global_master and pair_corr is not None:
            pair_corr[1] = pair_corr[1][np.argsort(np.abs(pair_corr[0]))[::-1]]
            pair_corr[0] = pair_corr[0][np.argsort(np.abs(pair_corr[0]))[::-1]]
            logger.debug("\n " + "-" * 74)
            logger.debug(f'{"pair correlation information":^75s}')
            logger.debug(" " + "-" * 74)
            logger.debug(
                " orbital tuple  |  absolute corr.  |  relative corr.  |  "
                "cumulative corr."
            )
            logger.debug(" " + "-" * 74)
            for i in range(10):
                logger.debug(
                    f"   [{pair_corr[1][i][0]:3d},{pair_corr[1][i][1]:3d}]    |"
                    f"    {pair_corr[0][i]:.3e}    |"
                    f"        {pair_corr[0][i] / pair_corr[0][0]:.2f}      |"
                    f"        {np.sum(pair_corr[0][:i+1]) / np.sum(pair_corr[0]):.2f}"
                )
            logger.debug(" " + "-" * 74 + "\n")

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
        for k in range(self.min_order, self.order + 1):

            # load k-th order hashes and increments
            hashes = open_shared_win(
                self.hashes[k - self.min_order],
                np.int64,
                (self.n_tuples["inc"][k - self.min_order],),
            )
            inc = self._open_shared_inc(
                self.incs[k - self.min_order],
                self.n_tuples["inc"][k - self.min_order],
                k - self.min_order,
            )

            # mpi barrier
            mpi.local_comm.barrier()

            # init list for storing hashes at order k
            hashes_lst: List[int] = []

            # init list for storing increments at order k
            inc_lst: List[IncType] = []

            # loop until no tuples left
            for tup_idx, tup in enumerate(
                tuples(
                    exp_occ,
                    exp_virt,
                    self.ref_n_elecs,
                    self.ref_n_holes,
                    self.vanish_exc,
                    k,
                )
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
            inc_arr = self._flatten_inc(inc_lst, k - self.min_order)

            # deallocate k-th order hashes and increments
            self.hashes[k - self.min_order].Free()
            self._free_inc(self.incs[k - self.min_order])

            # number of hashes for every rank
            recv_counts = np.array(mpi.global_comm.allgather(hashes_arr.size))

            # update n_tuples
            self.n_tuples["inc"][k - self.min_order] = int(np.sum(recv_counts))

            # init hashes for present order
            hashes_win = MPI.Win.Allocate_shared(
                8 * self.n_tuples["inc"][k - self.min_order] if mpi.local_master else 0,
                8,
                comm=mpi.local_comm,  # type: ignore
            )
            self.hashes[k - self.min_order] = hashes_win
            hashes = open_shared_win(
                hashes_win, np.int64, (self.n_tuples["inc"][k - self.min_order],)
            )

            # gatherv hashes on global master
            hashes[:] = mpi_gatherv(mpi.global_comm, hashes_arr, hashes, recv_counts)

            # bcast hashes among local masters
            if mpi.local_master:
                hashes[:] = mpi_bcast(mpi.master_comm, hashes)

            # init increments for present order
            self.incs[k - self.min_order] = self._allocate_shared_inc(
                self.n_tuples["inc"][k - self.min_order],
                mpi.local_master,
                mpi.local_comm,
            )
            inc = self._open_shared_inc(
                self.incs[k - self.min_order],
                self.n_tuples["inc"][k - self.min_order],
                k - self.min_order,
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

    @abstractmethod
    def _inc(
        self,
        e_core: float,
        h1e_cas: np.ndarray,
        h2e_cas: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        tup: np.ndarray,
    ) -> Tuple[TargetType, np.ndarray]:
        """
        this function calculates the current-order contribution to the increment
        associated with a given tuple
        """

    @abstractmethod
    def _sum(
        self,
        inc: List[IncType],
        hashes: List[np.ndarray],
        tup: np.ndarray,
    ) -> TargetType:
        """
        this function performs a recursive summation and returns the final increment
        associated with a given tuple
        """

    @staticmethod
    @abstractmethod
    def _write_target_file(order: Optional[int], prop: TargetType, string: str) -> None:
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

    @staticmethod
    @abstractmethod
    def _init_target_inst(value: float, norb: int) -> TargetType:
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
    ) -> Tuple[List[np.ndarray], MPI.Win]:
        """
        this function loads all previous-order hashes and initializes the current-order
        hashes
        """
        # load hashes for previous orders
        hashes: List[np.ndarray] = []
        for k in range(self.order - self.min_order):
            hashes.append(
                open_shared_win(self.hashes[k], np.int64, (self.n_tuples["inc"][k],))
            )

        # init hashes for present order
        if rst_read:
            hashes_win = self.hashes[-1]
        else:
            hashes_win = MPI.Win.Allocate_shared(
                8 * self.n_tuples["inc"][-1] if local_master else 0,
                8,
                comm=local_comm,  # type: ignore
            )
        hashes.append(
            open_shared_win(hashes_win, np.int64, (self.n_tuples["inc"][-1],))
        )
        if local_master and not global_master:
            hashes[-1][:].fill(0)

        return hashes, hashes_win

    @staticmethod
    @abstractmethod
    def _write_inc_file(order: Optional[int], inc: IncType) -> None:
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
    ) -> Tuple[List[IncType], MPIWinType]:
        """
        this function loads all previous-order increments and initializes the
        current-order increments
        """
        inc: List[IncType] = []

        # load increments for previous orders
        for k in range(self.order - self.min_order):
            inc.append(self._open_shared_inc(self.incs[k], self.n_tuples["inc"][k], k))

        # init increments for present order
        if rst_read:
            inc_win = self.incs[-1]
        else:
            inc_win = self._allocate_shared_inc(
                self.n_tuples["inc"][-1], local_master, local_comm
            )
        inc.append(
            self._open_shared_inc(
                inc_win, self.n_tuples["inc"][-1], self.order - self.min_order
            )
        )

        if (local_master and not global_master) or (global_master and not rst_read):
            inc[-1][:].fill(0.0)

        return inc, inc_win

    @abstractmethod
    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm
    ) -> MPIWinType:
        """
        this function allocates a shared increment window
        """

    @staticmethod
    @abstractmethod
    def _open_shared_inc(
        window: MPIWinType,
        n_tuples: int,
        idx: int,
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
    def _screen(inc_tup: TargetType, screen: np.ndarray, tup: np.ndarray) -> np.ndarray:
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
        tup: np.ndarray,
    ) -> Tuple[TargetType, TargetType, TargetType]:
        """
        this function updates the increment statistics
        """

    @staticmethod
    @abstractmethod
    def _total_inc(inc: IncType, mean_inc: TargetType) -> TargetType:
        """
        this function calculates the total current-order increment
        """

    @abstractmethod
    def _mbe_debug(
        self,
        n_elecs_tup: np.ndarray,
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
    ExpCls[SingleTargetType, np.ndarray, MPI.Win], metaclass=ABCMeta
):
    """
    this class holds all function definitions for single-target expansions irrespective
    of whether the target is a scalar or an array type
    """

    def _sum(
        self,
        inc: List[np.ndarray],
        hashes: List[np.ndarray],
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

            # loop over subtuples
            for tup_sub in tuples(
                tup_occ,
                tup_virt,
                self.ref_n_elecs,
                self.ref_n_holes,
                self.vanish_exc,
                k,
            ):

                # compute index
                idx = hash_lookup(hashes[k - self.min_order], hash_1d(tup_sub))

                # sum up order increments
                if idx is not None:
                    res[k - self.min_order] += inc[k - self.min_order][idx]

        return np.sum(res, axis=0)

    @staticmethod
    def _zero_target_arr(length: int):
        """
        this function initializes an array of the target type with value zero
        """

    @staticmethod
    def _write_inc_file(order: Optional[int], inc: np.ndarray) -> None:
        """
        this function defines writes the increment restart files
        """
        write_file(order, inc, "mbe_inc")

    @staticmethod
    def _read_inc_file(file: str) -> np.ndarray:
        """
        this function defines reads the increment restart files
        """
        return np.load(os.path.join(RST, file))

    @abstractmethod
    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm
    ) -> MPI.Win:
        """
        this function allocates a shared increment window
        """

    @staticmethod
    @abstractmethod
    def _open_shared_inc(
        window: MPI.Win,
        n_tuples: int,
        idx: Optional[int] = None,
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
        tup: np.ndarray,
    ) -> Tuple[SingleTargetType, SingleTargetType, SingleTargetType]:
        """
        this function updates the increment statistics
        """
        min_inc = np.minimum(min_inc, np.abs(inc_tup))
        mean_inc += inc_tup
        max_inc = np.maximum(max_inc, np.abs(inc_tup))

        return min_inc, mean_inc, max_inc
