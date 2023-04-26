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
import numpy as np
from mpi4py import MPI
from abc import ABCMeta, abstractmethod
from numpy.polynomial.polynomial import Polynomial
from typing import TYPE_CHECKING, cast, TypeVar, Generic, Tuple, List, Union

from pymbe.logger import logger
from pymbe.output import (
    main_header,
    mbe_header,
    mbe_status,
    mbe_end,
    redundant_results,
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
    GenFockArrayCls,
    pi_space,
    natural_keys,
    n_tuples,
    orb_n_tuples,
    is_file,
    read_file,
    write_file,
    write_file_mult,
    pi_prune,
    tuples,
    orb_tuples,
    get_nelec,
    get_nhole,
    get_nexc,
    start_idx,
    core_cas,
    cas,
    idx_tril,
    hash_1d,
    hash_lookup,
    get_occup,
    get_vhf,
    e_core_h1e,
    symm_eqv_tup,
    symm_eqv_inc,
    get_lex_cas,
    is_lex_tup,
    apply_symm_op,
    reduce_symm_eqv_orbs,
    get_eqv_inc_orbs,
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
    from typing import Dict, Optional

    from pymbe.pymbe import MBE
    from pymbe.parallel import MPICls


# define variable type for target properties
TargetType = TypeVar("TargetType", float, np.ndarray, RDMCls, GenFockCls)

# define variable type for input target properties
InputTargetType = TypeVar(
    "InputTargetType",
    float,
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[float, np.ndarray],
)

# define variable type for increment arrays
IncType = TypeVar("IncType", np.ndarray, packedRDMCls, GenFockArrayCls)

# define variable type for MPI windows of increment arrays
MPIWinType = TypeVar("MPIWinType", MPI.Win, Tuple[MPI.Win, MPI.Win])

# define variable type for integers describing electronic states
StateIntType = TypeVar("StateIntType", int, List[int])

# define variable type for numpy arrays describing electronic states
StateArrayType = TypeVar("StateArrayType", np.ndarray, List[np.ndarray])


SCREEN = 1000.0  # random, non-sensical number


MAX_MEM = 1e10
CONV_TOL = 1.0e-10
SPIN_TOL = 1.0e-05


class ExpCls(
    Generic[
        TargetType,
        InputTargetType,
        IncType,
        MPIWinType,
        StateIntType,
        StateArrayType,
    ],
    metaclass=ABCMeta,
):
    """
    this class contains the pymbe expansion attributes
    """

    def __init__(self, mbe: MBE) -> None:
        """
        init expansion attributes
        """
        # expansion model
        self.method = mbe.method
        self.cc_backend = mbe.cc_backend
        self.hf_guess = mbe.hf_guess

        # target property
        self.target = mbe.target

        # system
        self.norb = mbe.norb
        self.nelec: StateArrayType = cast(StateArrayType, mbe.nelec)
        self.point_group = mbe.point_group
        if isinstance(mbe.orbsym, np.ndarray):
            self.orbsym = mbe.orbsym
        else:
            self.orbsym = np.zeros(self.norb, dtype=np.int64)
        self.fci_state_sym: StateIntType = cast(StateIntType, mbe.fci_state_sym)
        self.fci_state_root: StateIntType = cast(StateIntType, mbe.fci_state_root)
        if hasattr(mbe, "fci_state_weights"):
            self.fci_state_weights = mbe.fci_state_weights
        self._state_occup()

        # optional system parameters for generalized Fock matrix
        if hasattr(mbe, "full_norb"):
            self.full_norb = mbe.full_norb
        if hasattr(mbe, "full_nocc"):
            self.full_nocc = mbe.full_nocc
        if hasattr(self, "full_norb") and hasattr(self, "full_nocc"):
            self.full_nvirt = self.full_norb - self.norb - self.full_nocc

        # integrals
        self.hcore, self.eri, self.vhf = self._int_wins(mbe.hcore, mbe.eri, mbe.mpi)

        # optional integrals for (transition) dipole moment
        if hasattr(mbe, "dipole_ints"):
            self.dipole_ints = mbe.dipole_ints

        # optional integrals for generalized Fock matrix
        if hasattr(mbe, "inact_fock"):
            self.inact_fock = mbe.inact_fock
        if hasattr(mbe, "eri_goaa"):
            self.eri_goaa = mbe.eri_goaa
        if hasattr(mbe, "eri_gaao"):
            self.eri_gaao = mbe.eri_gaao
        if hasattr(mbe, "eri_gaaa"):
            self.eri_gaaa = mbe.eri_gaaa

        # orbital representation
        self.orb_type = mbe.orb_type

        # reference and expansion spaces
        self.ref_space = mbe.ref_space
        self.ref_nelec = get_nelec(self.occup, self.ref_space)
        self.ref_nhole = get_nhole(self.ref_nelec, self.ref_space)
        self.exp_space = [mbe.exp_space]

        # base model
        self.base_method = mbe.base_method
        self.base_prop: TargetType = self._convert_to_target(
            cast(InputTargetType, mbe.base_prop)
        )

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
        self.n_tuples: Dict[str, List[int]] = {
            "theo": [],
            "screen": [],
            "van": [],
            "calc": [],
            "inc": [],
        }

        # screening
        self.screen_type = mbe.screen_type
        self.screen_start = mbe.screen_start
        self.screen_perc = mbe.screen_perc
        self.screen_thres = mbe.screen_thres
        self.screen_func = mbe.screen_func
        self.screen: List[Dict[str, np.ndarray]] = []
        self.screen_orbs = np.array([], dtype=np.int64)
        self.mbe_tot_error: List[float] = []

        # restart
        self.rst = mbe.rst
        self.rst_freq = mbe.rst_freq
        self.restarted = mbe.restarted

        # order
        self.order = 0
        self.start_order = 0
        self.min_order = 1
        self.max_order = mbe.max_order
        self.final_order = 0

        # number of vanishing excitations for current model
        if self.base_method is None:
            self.vanish_exc = 1
        elif self.base_method in ["ccsd", "ccsd(t)"]:
            self.vanish_exc = 2
        elif self.base_method == "ccsdt":
            self.vanish_exc = 3
        elif self.base_method == "ccsdtq":
            self.vanish_exc = 4

        # verbose
        self.verbose = mbe.verbose

        # pi-pruning
        self.pi_prune = mbe.pi_prune
        if self.pi_prune:
            self.orbsym_linear = mbe.orbsym_linear
            self.pi_orbs, self.pi_hashes = pi_space(
                "Dooh" if self.point_group == "D2h" else "Coov",
                self.orbsym_linear,
                self.exp_space[0],
            )

        # exclude single excitations
        self.no_singles = mbe.no_singles

        # localized orbital symmetry
        self.nsymm = 0
        self.symm_eqv_orbs: Optional[List[List[Dict[int, Tuple[int, ...]]]]] = None
        self.symm_inv_ref_space = False
        self.eqv_inc_orbs: Optional[List[np.ndarray]] = None
        if (
            mbe.orb_type == "local"
            and isinstance(mbe.orbsym, list)
            and all([isinstance(symm_op, dict) for symm_op in mbe.orbsym])
        ):
            if self.target not in ["rdm12", "genfock"]:
                self.nsymm = len(mbe.orbsym)
                self.symm_eqv_orbs = [
                    reduce_symm_eqv_orbs(
                        mbe.orbsym, cas(self.ref_space, self.exp_space[0])
                    )
                ]
                self.symm_inv_ref_space = True
                for symm_op in self.symm_eqv_orbs[0]:
                    perm_ref_space = apply_symm_op(symm_op, self.ref_space)
                    if perm_ref_space is None or not np.array_equal(
                        np.sort(mbe.ref_space), np.sort(perm_ref_space)
                    ):
                        self.symm_inv_ref_space = False
                        break
                if self.symm_inv_ref_space:
                    self.symm_eqv_orbs[0] = reduce_symm_eqv_orbs(
                        mbe.orbsym, self.exp_space[0]
                    )
                self.eqv_inc_orbs = [
                    get_eqv_inc_orbs(self.symm_eqv_orbs[0], self.nsymm, self.norb)
                ]

        # hartree fock property
        self.hf_prop: TargetType = self._hf_prop(mbe.mpi)

        # reference space property
        self.ref_prop: TargetType = self._init_target_inst(0.0, self.ref_space.size)
        if get_nexc(self.ref_nelec, self.ref_nhole) > self.vanish_exc:
            self.ref_prop = self._ref_prop(mbe.mpi)

        # attributes from restarted calculation
        if self.restarted:
            self._restart_main(mbe.mpi)

    def driver_master(self, mpi: MPICls) -> None:
        """
        this function is the main pymbe master function
        """
        # print expansion headers
        logger.info(main_header(mpi=mpi, method=self.method))

        # begin mbe expansion depending
        for self.order in range(self.min_order, self.max_order + 1):
            # theoretical and actual number of tuples at current order
            if not self.restarted or self.order > self.start_order:
                self._ntuples(mpi)

            # print mbe header
            logger.info(
                mbe_header(
                    self.order,
                    self.n_tuples["calc"][self.order - self.min_order],
                    self.screen_type,
                    1.0 if self.order < self.screen_start else self.screen_perc,
                    self.screen_thres,
                )
            )

            # main mbe function
            if not self.restarted or (
                self.order > self.start_order and len(self.mbe_tot_prop) < self.order
            ):
                self._mbe(mpi)
                self._mbe_restart()
            else:
                logger.info(mbe_status(self.order, 1.0))

            # print mbe end
            logger.info(
                mbe_end(self.order, self.time["mbe"][self.order - self.min_order])
            )

            # print mbe results
            logger.info(self._mbe_results(self.order))

            # print redundant increment results
            logger.info(
                redundant_results(
                    self.order,
                    self.n_tuples["screen"][self.order - self.min_order],
                    self.n_tuples["van"][self.order - self.min_order],
                    self.n_tuples["calc"][self.order - self.min_order],
                    self.pi_prune or self.symm_eqv_orbs is not None,
                )
            )

            # main screening function
            if not self.restarted or (
                self.order > self.start_order and len(self.exp_space) == self.order
            ):
                self._screen(mpi)
                self._screen_restart()

            # update screen_orbs
            if self.order > self.min_order:
                self.screen_orbs = np.setdiff1d(
                    self.exp_space[self.order - self.min_order],
                    self.exp_space[self.order - self.min_order + 1],
                )

            # print screening results
            if self.screen_orbs.size > 0:
                logger.info(
                    screen_results(
                        self.order,
                        self.screen_orbs,
                        self.exp_space,
                        self.screen_type,
                        self.mbe_tot_error[-1]
                        if self.screen_type == "adaptive"
                        else 0.0,
                    )
                )

            # purge only if orbitals were screened away and if there is another order
            if (
                self.screen_orbs.size > 0
                and self.exp_space[self.order - self.min_order + 1].size > self.order
            ):
                # print header
                logger.info(purge_header(self.order))

                # main purging function
                if not self.restarted or self.order > self.start_order:
                    self._purge(mpi)

                # print purging results
                logger.info(purge_results(self.n_tuples, self.min_order, self.order))

                # print purge end
                logger.info(
                    purge_end(
                        self.order, self.time["purge"][self.order - self.min_order]
                    )
                )

            # write restart files for this order
            if not self.restarted or self.order > self.start_order:
                self._purge_restart()

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

            if msg["task"] == "ntuples":
                # receive order
                self.order = msg["order"]

                # number of tuples
                self._ntuples(mpi)

            elif msg["task"] == "mbe":
                # receive order
                self.order = msg["order"]

                # main mbe function
                self._mbe(
                    mpi,
                    rst_read=msg["rst_read"],
                    tup_idx=msg["tup_idx"],
                    tup=msg["tup"],
                )

            elif msg["task"] == "screen":
                # receive order
                self.order = msg["order"]

                # main screening function
                self._screen(mpi)

            elif msg["task"] == "purge":
                # receive order
                self.order = msg["order"]

                # main purging function
                self._purge(mpi)

            elif msg["task"] == "exit":
                slave = False

    def print_results(self, mpi: MPICls) -> str:
        """
        this function handles printing of results
        """
        # print header
        string = main_header(mpi, self.method) + "\n\n"

        # print timings
        string += (
            timings_prt(
                self.method, self.min_order, self.final_order, self.n_tuples, self.time
            )
            + "\n\n"
        )

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
        self, hcore_in: np.ndarray, eri_in: np.ndarray, mpi: MPICls
    ) -> Tuple[MPI.Win, MPI.Win, MPI.Win]:
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
            hcore[:] = hcore_in

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
            eri[:] = eri_in

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

        return hcore_win, eri_win, vhf_win

    @staticmethod
    @abstractmethod
    def _convert_to_target(prop: InputTargetType) -> TargetType:
        """
        this function converts the input target type into the used target type
        """

    def _hf_prop(self, mpi: MPICls) -> TargetType:
        """
        this function calculates and bcasts the hartree-fock property
        """
        # calculate reference space property on global master
        if mpi.global_master:
            # load hcore
            hcore = open_shared_win(self.hcore, np.float64, 2 * (self.norb,))

            # load eri
            eri = open_shared_win(
                self.eri, np.float64, 2 * (self.norb * (self.norb + 1) // 2,)
            )

            # load vhf
            vhf = open_shared_win(
                self.vhf, np.float64, (self.nocc, self.norb, self.norb)
            )

            # compute hartree-fock property
            hf_prop = self._calc_hf_prop(hcore, eri, vhf)

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
            # load hcore
            hcore = open_shared_win(self.hcore, np.float64, 2 * (self.norb,))

            # load eri
            eri = open_shared_win(
                self.eri, np.float64, 2 * (self.norb * (self.norb + 1) // 2,)
            )

            # load vhf
            vhf = open_shared_win(
                self.vhf, np.float64, (self.nocc, self.norb, self.norb)
            )

            # core_idx and cas_idx
            core_idx, cas_idx = core_cas(
                self.nocc, self.ref_space, np.array([], dtype=np.int64)
            )

            # get cas_space h2e
            cas_idx_tril = idx_tril(cas_idx)
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(hcore, vhf, core_idx, cas_idx)

            # compute reference space property
            ref_prop, _ = self._inc(e_core, h1e_cas, h2e_cas, core_idx, cas_idx)

            # bcast ref_prop to slaves
            mpi.global_comm.bcast(ref_prop, root=0)

        else:
            # receive ref_prop from master
            ref_prop = mpi.global_comm.bcast(None, root=0)

        return ref_prop

    def _restart_main(self, mpi: MPICls) -> None:
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
                    for key in self.n_tuples.keys():
                        if key in files[i]:
                            self.n_tuples[key].append(
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
                    self.screen.append(dict(np.load(os.path.join(RST, files[i]))))

                # read total properties
                elif "mbe_tot_prop" in files[i]:
                    self.mbe_tot_prop.append(self._read_target_file(files[i]))

                # read total error
                elif "mbe_tot_error" in files[i]:
                    self.mbe_tot_error.append(
                        np.load(os.path.join(RST, files[i])).item()
                    )

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

                # read start order
                elif "mbe_start_order" in files[i]:
                    self.start_order = np.load(os.path.join(RST, files[i])).item()

        # bcast exp_space
        if mpi.global_master:
            mpi.global_comm.bcast(self.exp_space, root=0)
        else:
            self.exp_space = mpi.global_comm.bcast(None, root=0)

        # update symmetry-equivalent orbitals wrt screened orbitals
        if self.symm_eqv_orbs is not None and self.eqv_inc_orbs is not None:
            for exp_space in self.exp_space:
                self.symm_eqv_orbs.append(
                    reduce_symm_eqv_orbs(
                        self.symm_eqv_orbs[-1],
                        exp_space
                        if self.symm_inv_ref_space
                        else cas(self.ref_space, exp_space),
                    )
                )
                self.eqv_inc_orbs.append(
                    get_eqv_inc_orbs(self.symm_eqv_orbs[-1], self.nsymm, self.norb)
                )

        # mpi barrier
        mpi.global_comm.Barrier()

        return

    def _ntuples(self, mpi: MPICls) -> None:
        """
        this function determines the theoretical and actual number of tuples
        """
        if mpi.global_master:
            # determine theoretical number of tuples
            if len(self.n_tuples["theo"]) == self.order - self.min_order:
                self.n_tuples["theo"].append(
                    n_tuples(
                        self.exp_space[0][self.exp_space[0] < self.nocc],
                        self.exp_space[0][self.nocc <= self.exp_space[0]],
                        self.ref_nelec,
                        self.ref_nhole,
                        -1,
                        self.order,
                    )
                )

            # determine screened number of tuples
            if len(self.n_tuples["screen"]) == self.order - self.min_order:
                self.n_tuples["screen"].append(
                    n_tuples(
                        self.exp_space[-1][self.exp_space[-1] < self.nocc],
                        self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                        self.ref_nelec,
                        self.ref_nhole,
                        -1,
                        self.order,
                    )
                )

            # determine vanishing number of tuples
            if len(self.n_tuples["van"]) == self.order - self.min_order:
                self.n_tuples["van"].append(
                    n_tuples(
                        self.exp_space[-1][self.exp_space[-1] < self.nocc],
                        self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                        self.ref_nelec,
                        self.ref_nhole,
                        self.vanish_exc,
                        self.order,
                    )
                )

        # determine number of increments
        if len(self.n_tuples["inc"]) == self.order - self.min_order:
            # wake up slaves
            if mpi.global_master:
                msg = {"task": "ntuples", "order": self.order}
                mpi.global_comm.bcast(msg, root=0)

            # determine number of non-redundant increments
            if self.pi_prune or (
                self.symm_eqv_orbs is not None and self.eqv_inc_orbs is not None
            ):
                # occupied and virtual expansion spaces
                exp_occ = self.exp_space[-1][self.exp_space[-1] < self.nocc]
                exp_virt = self.exp_space[-1][self.nocc <= self.exp_space[-1]]

                # initialize number of tuples
                ntuples = 0

                # loop until no tuples left
                for tup_idx, tup in enumerate(
                    tuples(
                        exp_occ,
                        exp_virt,
                        self.ref_nelec,
                        self.ref_nhole,
                        self.vanish_exc,
                        self.order,
                    )
                ):
                    # distribute tuples
                    if tup_idx % mpi.global_size != mpi.global_rank:
                        continue

                    # pi-pruning
                    if self.pi_prune and pi_prune(self.pi_orbs, self.pi_hashes, tup):
                        ntuples += 1

                    # symmetry-pruning
                    elif self.eqv_inc_orbs is not None:
                        # add reference space if it is not symmetry-invariant
                        if self.symm_inv_ref_space:
                            cas_idx = tup
                            ref_space = None
                        else:
                            cas_idx = cas(self.ref_space, tup)
                            ref_space = self.ref_space

                        if is_lex_tup(cas_idx, self.eqv_inc_orbs[-1], ref_space):
                            ntuples += 1

                # get total number of non-redundant increments
                self.n_tuples["inc"].append(
                    mpi.global_comm.allreduce(ntuples, op=MPI.SUM)
                )

            else:
                if len(self.n_tuples["inc"]) == self.order - self.min_order:
                    self.n_tuples["inc"].append(
                        n_tuples(
                            self.exp_space[-1][self.exp_space[-1] < self.nocc],
                            self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                            self.ref_nelec,
                            self.ref_nhole,
                            self.vanish_exc,
                            self.order,
                        )
                    )

        if mpi.global_master:
            # determine number of calculations
            if len(self.n_tuples["calc"]) == self.order - self.min_order:
                self.n_tuples["calc"].append(self.n_tuples["inc"][-1])

            # write restart files
            if self.rst:
                for key in self.n_tuples.keys():
                    write_file(
                        np.asarray(self.n_tuples[key][-1]),
                        "mbe_n_tuples_" + key,
                        self.order,
                    )

        return

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
            and self.target not in ["rdm12", "genfock"]
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

        # init screen arrays
        screen = self._init_screen()
        if rst_read:
            if mpi.global_master:
                screen = self.screen[-1]
        else:
            self.screen.append(screen)

        # set screening function for mpi
        screen_mpi_func = {"sum_abs": MPI.SUM, "sum": MPI.SUM, "max": MPI.MAX}

        # set rst_write
        rst_write = (
            self.rst and mpi.global_size < self.rst_freq < self.n_tuples["inc"][-1]
        )

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

        # initialize number of increments from previous calculation
        n_prev_tup_idx = 0

        # loop until no tuples left
        for tup in tuples(
            exp_occ,
            exp_virt,
            self.ref_nelec,
            self.ref_nhole,
            self.vanish_exc,
            self.order,
            order_start,
            occ_start,
            virt_start,
        ):
            # write restart files and re-init time
            if rst_write and tup_idx % self.rst_freq < n_prev_tup_idx:
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
                if screen_mpi_func is not None:
                    for screen_func, screen_array in screen.items():
                        screen[screen_func] = mpi_reduce(
                            mpi.global_comm,
                            screen_array,
                            root=0,
                            op=screen_mpi_func[screen_func],
                        )
                    if not mpi.global_master:
                        screen = self._init_screen()

                # update rst_write
                rst_write = (
                    tup_idx + self.rst_freq < self.n_tuples["inc"][-1] - mpi.global_size
                )

                if mpi.global_master:
                    # write restart files
                    self._write_target_file(min_inc, "mbe_min_inc", self.order)
                    self._write_target_file(mean_inc, "mbe_mean_inc", self.order)
                    self._write_target_file(max_inc, "mbe_max_inc", self.order)
                    write_file_mult(screen, "mbe_screen", self.order)
                    write_file(np.asarray(tup_idx), "mbe_idx", self.order)
                    write_file(tup, "mbe_tup", self.order)
                    write_file(hashes[-1], "mbe_hashes", self.order)
                    self._write_inc_file(inc[-1], self.order)
                    self.time["mbe"][-1] += MPI.Wtime() - time
                    write_file(
                        np.asarray(self.time["mbe"][-1]), "mbe_time_mbe", self.order
                    )
                    # re-init time
                    time = MPI.Wtime()
                    # print status
                    logger.info(
                        mbe_status(self.order, tup_idx / self.n_tuples["inc"][-1])
                    )

            # pi-pruning
            if self.pi_prune and not pi_prune(self.pi_orbs, self.pi_hashes, tup):
                for screen_func in screen.keys():
                    screen[screen_func][tup] = SCREEN
                continue

            # symmetry-pruning
            if self.symm_eqv_orbs is not None and self.eqv_inc_orbs is not None:
                # add reference space if it is not symmetry-invariant
                if self.symm_inv_ref_space:
                    cas_idx = tup
                    ref_space = None
                else:
                    cas_idx = cas(self.ref_space, tup)
                    ref_space = self.ref_space

                # check if tuple is last symmetrically equivalent tuple
                eqv_tup_set = symm_eqv_tup(cas_idx, self.symm_eqv_orbs[-1], ref_space)

                # skip calculation if symmetrically equivalent tuple will come later
                if eqv_tup_set is None:
                    n_prev_tup_idx = 0
                    continue

                # check for symmetrically equivalent increments
                eqv_inc_lex_tup, eqv_inc_set = symm_eqv_inc(
                    self.eqv_inc_orbs[-1], eqv_tup_set, ref_space
                )

                # save number of different increments for calculation
                n_prev_tup_idx = len(eqv_inc_lex_tup)

            else:
                # every tuple is unique without symmetry pruning
                eqv_inc_lex_tup = [tup]
                eqv_inc_set = [[tup]]
                n_prev_tup_idx = 1

            # distribute tuples
            if tup_idx % mpi.global_size != mpi.global_rank:
                tup_idx += n_prev_tup_idx
                continue

            # get core and cas indices
            core_idx, cas_idx = core_cas(self.nocc, self.ref_space, tup)

            # get h2e indices
            cas_idx_tril = idx_tril(cas_idx)

            # get h2e_cas
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(hcore, vhf, core_idx, cas_idx)

            # calculate CASCI property
            target_tup, nelec_tup = self._inc(
                e_core, h1e_cas, h2e_cas, core_idx, cas_idx
            )

            # loop over equivalent increment sets
            for tup, eqv_set in zip(eqv_inc_lex_tup, eqv_inc_set):
                # calculate increment
                if self.order > self.min_order:
                    inc_tup = target_tup - self._sum(inc, hashes, tup)
                else:
                    inc_tup = target_tup

                # add hash and increment
                hashes[-1][tup_idx] = hash_1d(tup)
                inc[-1][tup_idx] = inc_tup

                # screening procedure
                if screen_mpi_func is not None:
                    for eqv_tup in eqv_set:
                        for screen_func in screen.keys():
                            screen[screen_func][eqv_tup] = self._add_screen(
                                inc_tup, screen[screen_func], eqv_tup, screen_func
                            )

                # debug print
                logger.debug(self._mbe_debug(nelec_tup, inc_tup, cas_idx, tup))

                # update increment statistics
                min_inc, mean_inc, max_inc = self._update_inc_stats(
                    inc_tup, min_inc, mean_inc, max_inc, cas_idx, len(eqv_set)
                )

                # update pair_corr statistics
                if pair_corr is not None:
                    inc_arr = np.asarray(inc_tup)
                    if self.target in ["energy", "excitation"]:
                        pair_corr[0][tup_idx] = inc_arr
                    elif self.target in ["dipole", "trans"]:
                        pair_corr[0][tup_idx] = inc_arr[np.argmax(np.abs(inc_arr))]
                    pair_corr[1][tup_idx] = tup

                # increment tuple counter
                tup_idx += 1

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
            if self.n_tuples["van"][-1] != 0:
                mean_inc /= self.n_tuples["van"][-1]

        # pair_corr statistics
        if pair_corr is not None:
            pair_corr = [
                mpi_reduce(mpi.global_comm, pair_corr[0], root=0, op=MPI.SUM),
                mpi_reduce(mpi.global_comm, pair_corr[1], root=0, op=MPI.SUM),
            ]

        # reduce screen
        if screen_mpi_func is not None:
            for screen_func, screen_array in screen.items():
                self.screen[-1][screen_func] = mpi_reduce(
                    mpi.global_comm,
                    screen_array,
                    op=screen_mpi_func[screen_func],
                )

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
            for i in range(pair_corr[0].size):
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

        # save statistics & timings
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

            self.time["mbe"][-1] += MPI.Wtime() - time

        return

    def _mbe_restart(self) -> None:
        """
        this function writes restart files for one mbe order
        """
        if self.rst:
            hashes = open_shared_win(
                self.hashes[-1], np.int64, (self.n_tuples["inc"][-1],)
            )
            write_file(hashes, "mbe_hashes", self.order)
            inc = self._open_shared_inc(
                self.incs[-1], self.n_tuples["inc"][-1], self.order - self.min_order
            )
            self._write_inc_file(inc, self.order)
            self._write_target_file(self.min_inc[-1], "mbe_min_inc", self.order)
            self._write_target_file(self.mean_inc[-1], "mbe_mean_inc", self.order)
            self._write_target_file(self.max_inc[-1], "mbe_max_inc", self.order)
            write_file_mult(self.screen[-1], "mbe_screen", self.order)
            write_file(np.asarray(self.n_tuples["inc"][-1]), "mbe_idx", self.order)
            write_file(np.asarray(self.time["mbe"][-1]), "mbe_time_mbe", self.order)
            self._write_target_file(self.mbe_tot_prop[-1], "mbe_tot_prop", self.order)

        return

    def _screen(self, mpi: MPICls) -> None:
        """
        this function decides what orbitals will be screened away
        """
        # wake up slaves
        if mpi.global_master:
            msg = {"task": "screen", "order": self.order}
            mpi.global_comm.bcast(msg, root=0)

        # fixed screening procedure
        if self.screen_type == "fixed":
            # start screening
            if mpi.global_master:
                thres = 1.0 if self.order < self.screen_start else self.screen_perc
                nkeep = int(thres * self.exp_space[-1].size)
                if self.screen_func == "rnd":
                    rng = np.random.default_rng()
                    # update expansion space wrt screened orbitals
                    self.exp_space.append(
                        rng.choice(self.exp_space[-1], size=nkeep, replace=False)
                    )
                else:
                    orb_screen = np.abs(
                        self.screen[-1][self.screen_func][self.exp_space[-1]]
                    )
                    orb_significance = np.argsort(orb_screen)[::-1]
                    # update expansion space wrt screened orbitals
                    self.exp_space.append(
                        self.exp_space[-1][np.sort(orb_significance[:nkeep])]
                    )

                # bcast updated expansion space
                mpi.global_comm.bcast(self.exp_space[-1], root=0)

            else:
                # receive updated expansion space
                self.exp_space.append(mpi.global_comm.bcast(None, root=0))

        # adaptive screening procedure
        elif self.screen_type == "adaptive":
            # add previous expansion space for current order
            self.exp_space.append(self.exp_space[-1])

            # occupied and virtual expansion spaces
            exp_occ = self.exp_space[-1][self.exp_space[-1] < self.nocc]
            exp_virt = self.exp_space[-1][self.nocc <= self.exp_space[-1]]

            # start screening
            if mpi.global_master:
                # initialize error for current order
                self.mbe_tot_error.append(
                    self.mbe_tot_error[-1] if self.order > self.min_order else 0.0
                )

                # initialize minimum orbital contribution
                min_orb_contrib = 0.0

                # get number of tuples per orbital
                ntup_occ, ntup_virt = orb_n_tuples(
                    self.exp_space[-1][self.exp_space[-1] < self.nocc],
                    self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                    self.ref_nelec,
                    self.ref_nhole,
                    self.vanish_exc,
                    self.order,
                )

                # calculate relative factor
                self.screen[-1]["rel_factor"] = np.divide(
                    np.abs(self.screen[-1]["sum"]),
                    self.screen[-1]["sum_abs"],
                    out=np.zeros(self.norb, dtype=np.float64),
                    where=self.screen[-1]["sum_abs"] != 0.0,
                )

                # calculate mean absolute increment
                self.screen[-1]["mean_abs_inc"] = np.empty(self.norb, dtype=np.float64)
                self.screen[-1]["mean_abs_inc"][: self.nocc] = (
                    (self.screen[-1]["sum_abs"][: self.nocc] / ntup_occ)
                    if ntup_occ > 0
                    else 0.0
                )
                self.screen[-1]["mean_abs_inc"][self.nocc :] = (
                    (self.screen[-1]["sum_abs"][self.nocc :] / ntup_virt)
                    if ntup_virt > 0
                    else 0.0
                )

                # initialize boolean to keep screening
                keep_screening = True

                # remove orbitals until minimum orbital contribution is larger than
                # threshold
                while keep_screening:
                    # define maximum possible order
                    max_order = self.exp_space[-1].size

                    # check if expansion has ended
                    if self.order >= max_order:
                        keep_screening = False
                        break

                    # define error allowed per orbital
                    error_thresh = self.screen_thres - self.mbe_tot_error[-1]

                    # initialize array for error estimate
                    error_estimate = np.zeros(
                        (self.exp_space[-1].size, max_order - self.order),
                        dtype=np.float64,
                    )

                    # get number of tuples for remaining orders
                    ntup_order_occ = np.empty(max_order - self.order, dtype=np.int64)
                    ntup_order_virt = np.empty(max_order - self.order, dtype=np.int64)
                    for order in range(self.order + 1, max_order + 1):
                        (
                            ntup_order_occ[order - (self.order + 1)],
                            ntup_order_virt[order - (self.order + 1)],
                        ) = orb_n_tuples(
                            self.exp_space[-1][self.exp_space[-1] < self.nocc],
                            self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                            self.ref_nelec,
                            self.ref_nhole,
                            self.vanish_exc,
                            order,
                        )
                    ntup_order_tot = [
                        n_tuples(
                            self.exp_space[-1][self.exp_space[-1] < self.nocc],
                            self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                            self.ref_nelec,
                            self.ref_nhole,
                            self.vanish_exc,
                            order,
                        )
                        for order in range(self.order + 1, max_order + 1)
                    ]

                    # check if expansion has ended
                    if np.sum(ntup_order_tot) == 0:
                        keep_screening = False
                        break

                    # get maximum relative factor from last two orders
                    rel_factor = np.max(
                        [screen["rel_factor"] for screen in self.screen[-2:]]
                    )

                    # initialize array for
                    good_fit = np.ones(self.exp_space[-1].size, dtype=bool)

                    # loop over orbitals
                    for orb_idx, orb in enumerate(self.exp_space[-1]):
                        # get mean absolute increments for orbital
                        mean_abs_inc = np.array(
                            [screen["mean_abs_inc"][orb] for screen in self.screen],
                            dtype=np.float64,
                        )

                        # log transform mean absolute increments
                        log_mean_abs_inc = np.log(mean_abs_inc[mean_abs_inc > 0.0])

                        # get orders for fit
                        orders = self.min_order + np.argwhere(
                            mean_abs_inc > 0.0
                        ).reshape(-1)

                        # require at least 3 points to fit
                        if orders.size > 2:
                            # get pearson correlation coefficient
                            r2_value = (
                                np.corrcoef(
                                    orders,
                                    log_mean_abs_inc,
                                )[0, 1]
                                ** 2
                            )

                            # check if correlation is good enough
                            if r2_value > 0.9:
                                # fit log-transformed mean absolute increments
                                fit = Polynomial.fit(orders, log_mean_abs_inc, 1)

                            else:
                                # fit is not good
                                good_fit[orb_idx] = False

                                # assume mean absolute increment does not decrease
                                fit = Polynomial([log_mean_abs_inc[-1], 0.0])

                        else:
                            keep_screening = False
                            break

                        # get estimates for remaining orders
                        error_estimate[orb_idx] = rel_factor * np.exp(
                            fit(np.arange(self.order + 1, max_order + 1))
                        )
                        error_estimate[orb_idx] *= (
                            ntup_order_occ if orb < self.nocc else ntup_order_virt
                        )

                    # check if orbitals will be screened away at this order
                    if not keep_screening:
                        break

                    # calculate total error
                    tot_error = np.sum(error_estimate, axis=1)

                    # calculate difference to allowed error
                    error_diff = np.zeros_like(tot_error)
                    error_diff[: self.nocc] = (
                        np.sum(ntup_order_occ) / np.sum(ntup_order_tot)
                    ) * error_thresh - tot_error[: self.nocc]
                    error_diff[self.nocc :] = (
                        np.sum(ntup_order_virt) / np.sum(ntup_order_tot)
                    ) * error_thresh - tot_error[self.nocc :]

                    # get index in expansion space for minimum orbital contribution
                    min_idx = np.argmax(error_diff)

                    # get minimum orbital contribution
                    min_orb_contrib = tot_error[min_idx]

                    # screen orbital away if contribution is smaller than threshold
                    if error_diff[min_idx] > 0.0:
                        # log screening
                        logger.info2(
                            f" Orbital {self.exp_space[-1][min_idx]} is screened away "
                            f"(Error = {tot_error[min_idx]:>10.4e}, Factor = "
                            f"{rel_factor:>10.4e})"
                        )
                        if not good_fit[self.exp_space[-1][min_idx]]:
                            logger.info2(" Screened orbital R^2 value is < 0.9")

                        # add screened orbital contribution to error
                        self.mbe_tot_error[-1] += min_orb_contrib

                        # signal other processes to continue screening
                        mpi.global_comm.bcast(keep_screening, root=0)

                        # bcast orbital to screen away
                        mpi.global_comm.bcast(min_idx, root=0)

                        # remove orbital contributions
                        self._screen_remove_orb_contrib(
                            mpi, exp_occ, exp_virt, self.exp_space[-1][min_idx]
                        )

                        # remove orbital from expansion space
                        self.exp_space[-1] = np.delete(self.exp_space[-1], min_idx)

                        # occupied and virtual expansion spaces
                        exp_occ = self.exp_space[-1][self.exp_space[-1] < self.nocc]
                        exp_virt = self.exp_space[-1][self.nocc <= self.exp_space[-1]]

                        # loop over all orders
                        for k in range(self.min_order, self.order + 1):
                            # get number of tuples per orbital
                            ntup_occ, ntup_virt = orb_n_tuples(
                                exp_occ,
                                exp_virt,
                                self.ref_nelec,
                                self.ref_nhole,
                                self.vanish_exc,
                                k,
                            )

                            # calculate relative factor
                            self.screen[k - self.min_order]["rel_factor"] = np.divide(
                                np.abs(self.screen[k - self.min_order]["sum"]),
                                self.screen[k - self.min_order]["sum_abs"],
                                out=np.zeros(self.norb, dtype=np.float64),
                                where=self.screen[k - self.min_order]["sum_abs"] != 0.0,
                            )

                            # calculate mean absolute increment
                            self.screen[k - self.min_order]["mean_abs_inc"][
                                : self.nocc
                            ] = (
                                (
                                    self.screen[k - self.min_order]["sum_abs"][
                                        : self.nocc
                                    ]
                                    / ntup_occ
                                )
                                if ntup_occ > 0
                                else 0.0
                            )
                            self.screen[k - self.min_order]["mean_abs_inc"][
                                self.nocc :
                            ] = (
                                (
                                    self.screen[k - self.min_order]["sum_abs"][
                                        self.nocc :
                                    ]
                                    / ntup_virt
                                )
                                if ntup_virt > 0
                                else 0.0
                            )

                    # stop screening if no other orbitals contribute above threshold
                    else:
                        keep_screening = False

                # signal other processes to stop screening
                mpi.global_comm.bcast(keep_screening, root=0)

                # log screening
                if np.array_equal(self.exp_space[-1], self.exp_space[-2]):
                    logger.info2(
                        f" No orbitals were screened away (Factor = "
                        f"{rel_factor:>10.4e})"
                    )

            else:
                # initialize boolean to keep screening
                keep_screening = True

                # remove orbitals until minimum orbital contribution is larger than
                # threshold
                while keep_screening:
                    # determine if still screening
                    keep_screening = mpi.global_comm.bcast(None, root=0)

                    if keep_screening:
                        # get minimum orbital
                        min_idx = mpi.global_comm.bcast(None, root=0)

                        # remove orbital contributions
                        self._screen_remove_orb_contrib(
                            mpi, exp_occ, exp_virt, self.exp_space[-1][min_idx]
                        )

                        # remove orbital from expansion space
                        self.exp_space[-1] = np.delete(self.exp_space[-1], min_idx)

                        # occupied and virtual expansion spaces
                        exp_occ = self.exp_space[-1][self.exp_space[-1] < self.nocc]
                        exp_virt = self.exp_space[-1][self.nocc <= self.exp_space[-1]]

        # update symmetry-equivalent orbitals wrt screened orbitals
        if self.symm_eqv_orbs is not None and self.eqv_inc_orbs is not None:
            self.symm_eqv_orbs.append(
                reduce_symm_eqv_orbs(
                    self.symm_eqv_orbs[-1],
                    self.exp_space[-1]
                    if self.symm_inv_ref_space
                    else cas(self.ref_space, self.exp_space[-1]),
                )
            )
            self.eqv_inc_orbs.append(
                get_eqv_inc_orbs(self.symm_eqv_orbs[-1], self.nsymm, self.norb)
            )

        return

    def _screen_restart(self) -> None:
        """
        this function writes restart files after screening
        """
        # write restart file
        if self.rst:
            if self.screen_type == "adaptive":
                for k in range(self.order - self.min_order + 1):
                    write_file_mult(self.screen[k], "mbe_screen", k + self.min_order)
                write_file(
                    np.asarray(self.mbe_tot_error[-1]), "mbe_tot_error", self.order
                )
            write_file(self.exp_space[-1], "exp_space", self.order + 1)

        return

    def _purge(self, mpi: MPICls) -> None:
        """
        this function purges the lower-order hashes & increments
        """
        # wake up slaves
        if mpi.global_master:
            msg = {"task": "purge", "order": self.order}
            mpi.global_comm.bcast(msg, root=0)

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

            # init list for storing indices for increments at order k
            idx_lst: List[int] = []

            # loop until no tuples left
            for tup_idx, tup in enumerate(
                tuples(
                    exp_occ,
                    exp_virt,
                    self.ref_nelec,
                    self.ref_nhole,
                    self.vanish_exc,
                    k,
                )
            ):
                # distribute tuples
                if tup_idx % mpi.global_size != mpi.global_rank:
                    continue

                # pi-pruning
                if self.pi_prune and not pi_prune(self.pi_orbs, self.pi_hashes, tup):
                    continue

                # symmetry-pruning
                if self.symm_eqv_orbs is not None and self.eqv_inc_orbs is not None:
                    # add reference space if it is not symmetry-invariant
                    if self.symm_inv_ref_space:
                        cas_idx = tup
                        ref_space = None
                    else:
                        cas_idx = cas(self.ref_space, tup)
                        ref_space = self.ref_space

                    # check if tuple is last symmetrically equivalent tuple
                    lex_tup = is_lex_tup(cas_idx, self.eqv_inc_orbs[-1], ref_space)

                    # skip tuple if symmetrically equivalent tuple will come later
                    if not lex_tup:
                        continue

                    # get lexicographically greatest tuple using last order symmetry
                    lex_cas = get_lex_cas(cas_idx, self.eqv_inc_orbs[-2], ref_space)

                    # remove reference space if it is not symmetry-invariant
                    if self.symm_inv_ref_space:
                        tup_last = lex_cas
                    else:
                        tup_last = np.setdiff1d(
                            lex_cas, self.ref_space, assume_unique=True
                        )

                else:
                    # no redundant tuples
                    tup_last = tup

                # compute index
                idx = hash_lookup(hashes, hash_1d(tup_last))

                # add inc_tup and its hash to lists of increments/hashes
                if idx is not None:
                    idx_lst.append(idx.item())
                    hashes_lst.append(hash_1d(tup))
                else:
                    raise RuntimeError("Last order tuple not found:", tup_last)

            # recast hashes_lst and inc_lst as np.array and TargetType
            hashes_arr = np.array(hashes_lst, dtype=np.int64)
            inc_arr = inc[idx_lst]

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

    def _purge_restart(self) -> None:
        """
        this function writes restart files after finishing an order
        """
        # write restart files
        if (
            self.screen_orbs.size > 0
            and self.exp_space[self.order - self.min_order + 1].size > self.order
        ):
            if self.rst:
                for k in range(self.order - self.min_order + 1):
                    hashes = open_shared_win(
                        self.hashes[k], np.int64, (self.n_tuples["inc"][k],)
                    )
                    write_file(hashes, "mbe_hashes", k + self.min_order)
                    inc = self._open_shared_inc(
                        self.incs[k], self.n_tuples["inc"][k], k
                    )
                    self._write_inc_file(inc, k + self.min_order)
                    write_file(
                        np.asarray(self.n_tuples["inc"][k]),
                        "mbe_n_tuples_inc",
                        k + self.min_order,
                    )
        else:
            self.time["purge"].append(0.0)
            if self.rst:
                hashes = open_shared_win(
                    self.hashes[-1], np.int64, (self.n_tuples["inc"][-1],)
                )
                write_file(hashes, "mbe_hashes", self.order)
                inc = self._open_shared_inc(
                    self.incs[-1], self.n_tuples["inc"][-1], self.order - self.min_order
                )
                self._write_inc_file(inc, self.order)
                write_file(
                    np.asarray(self.n_tuples["inc"][-1]), "mbe_n_tuples_inc", self.order
                )
        if self.rst:
            write_file(np.asarray(self.time["purge"][-1]), "mbe_time_purge", self.order)
            write_file(np.asarray(self.order), "mbe_start_order")

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
        self, inc: List[IncType], hashes: List[np.ndarray], tup: np.ndarray
    ) -> TargetType:
        """
        this function performs a recursive summation and returns the final increment
        associated with a given tuple
        """

    @staticmethod
    @abstractmethod
    def _write_target_file(prop: TargetType, string: str, order: Optional[int]) -> None:
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
    def _init_target_inst(self, value: float, norb: int) -> TargetType:
        """
        this function initializes an instance of the target type
        """

    def _init_screen(self) -> Dict[str, np.ndarray]:
        """
        this function initializes the screening arrays
        """
        return {
            "sum_abs": np.zeros(self.norb, dtype=np.float64),
            "max": np.zeros(self.norb, dtype=np.float64),
        }

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
        if local_master and (not global_master or not rst_read):
            hashes[-1][:].fill(0)

        return hashes, hashes_win

    @staticmethod
    @abstractmethod
    def _write_inc_file(inc: IncType, order: Optional[int]) -> None:
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

        if local_master and (not global_master or not rst_read):
            inc[-1][:].fill(0.0)

        return inc, inc_win

    @abstractmethod
    def _allocate_shared_inc(
        self, size: int, allocate: bool, comm: MPI.Comm
    ) -> MPIWinType:
        """
        this function allocates a shared increment window
        """

    @abstractmethod
    def _open_shared_inc(self, window: MPIWinType, n_tuples: int, idx: int) -> IncType:
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
    def _free_inc(inc_win: MPIWinType) -> None:
        """
        this function frees the supplied increment windows
        """

    @staticmethod
    @abstractmethod
    def _add_screen(
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
        n_eqv_tups: int,
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

    def _screen_remove_orb_contrib(
        self, mpi: MPICls, exp_occ: np.ndarray, exp_virt: np.ndarray, orb: int
    ) -> None:
        """
        this function removes orbital contributions to the screening arrays
        """
        # loop over all orders
        for k in range(self.min_order, self.order + 1):
            # initialize arrays for contributions to be removed
            remove_sum_abs = np.zeros(self.norb, dtype=np.float64)
            remove_sum = np.zeros(self.norb, dtype=np.float64)

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

            for tup_idx, tup in enumerate(
                orb_tuples(
                    exp_occ,
                    exp_virt,
                    self.ref_nelec,
                    self.ref_nhole,
                    self.vanish_exc,
                    k,
                    orb,
                )
            ):
                # distribute tuples
                if tup_idx % mpi.global_size != mpi.global_rank:
                    continue

                # pi-pruning
                if self.pi_prune and not pi_prune(self.pi_orbs, self.pi_hashes, tup):
                    continue

                # symmetry-pruning
                if self.symm_eqv_orbs is not None and self.eqv_inc_orbs is not None:
                    # add reference space if it is not symmetry-invariant
                    if self.symm_inv_ref_space:
                        cas_idx = tup
                        ref_space = None
                    else:
                        cas_idx = cas(self.ref_space, tup)
                        ref_space = self.ref_space

                    # get lexicographically greatest tuple
                    lex_cas = get_lex_cas(cas_idx, self.eqv_inc_orbs[-1], ref_space)

                    # remove reference space if it is not symmetry-invariant
                    if self.symm_inv_ref_space:
                        tup_last = lex_cas
                    else:
                        tup_last = np.setdiff1d(
                            lex_cas,
                            self.ref_space,
                            assume_unique=True,
                        )

                else:
                    # no redundant tuples
                    tup_last = tup

                # compute index
                idx = hash_lookup(hashes, hash_1d(tup_last))

                # remove inc_tup from the screen arrays
                if idx is not None:
                    remove_sum_abs[tup] = self._add_screen(
                        inc[idx.item()], remove_sum_abs, tup, "sum_abs"
                    )
                    remove_sum[tup] = self._add_screen(
                        inc[idx.item()], remove_sum, tup, "sum"
                    )
                else:
                    raise RuntimeError("Last order tuple not found:", tup_last)

            # reduce contributions
            remove_sum_abs = mpi_reduce(mpi.global_comm, remove_sum_abs, op=MPI.SUM)
            remove_sum = mpi_reduce(mpi.global_comm, remove_sum, op=MPI.SUM)

            # remove contributions to screening functions
            if mpi.global_master:
                self.screen[k - self.min_order]["sum_abs"] -= remove_sum_abs
                self.screen[k - self.min_order]["sum"] -= remove_sum

        return

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
    ExpCls[SingleTargetType, SingleTargetType, np.ndarray, MPI.Win, int, np.ndarray],
    metaclass=ABCMeta,
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

    @staticmethod
    def _convert_to_target(prop: SingleTargetType) -> SingleTargetType:
        """
        this function converts the input target type into the used target type
        """
        return prop

    def _sum(
        self, inc: List[np.ndarray], hashes: List[np.ndarray], tup: np.ndarray
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
                self.ref_nelec,
                self.ref_nhole,
                self.vanish_exc,
                k,
            ):
                # pi-pruning
                if self.pi_prune and not pi_prune(
                    self.pi_orbs, self.pi_hashes, tup_sub
                ):
                    continue

                # symmetry-pruning
                if self.eqv_inc_orbs is not None:
                    # add reference space if it is not symmetry-invariant
                    if self.symm_inv_ref_space:
                        ref_space = None
                        cas_idx = tup_sub
                    else:
                        ref_space = self.ref_space
                        cas_idx = cas(self.ref_space, tup_sub)

                    # get lexicographically greatest tuple
                    lex_cas = get_lex_cas(cas_idx, self.eqv_inc_orbs[-1], ref_space)

                    # remove reference space if it is not symmetry-invariant
                    if self.symm_inv_ref_space:
                        tup_sub = lex_cas
                    else:
                        tup_sub = np.setdiff1d(
                            lex_cas, self.ref_space, assume_unique=True
                        )

                # compute index
                idx = hash_lookup(hashes[k - self.min_order], hash_1d(tup_sub))

                # sum up order increments
                if idx is not None:
                    res[k - self.min_order] += inc[k - self.min_order][idx.item()]
                else:
                    raise RuntimeError("Subtuple not found:", tup_sub)

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
    def _write_inc_file(inc: np.ndarray, order: Optional[int]) -> None:
        """
        this function defines writes the increment restart files
        """
        write_file(inc, "mbe_inc", order)

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

        return mpi_gatherv(comm, send_inc.ravel(), recv_inc, recv_counts)

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
        neqv_tups: int,
    ) -> Tuple[SingleTargetType, SingleTargetType, SingleTargetType]:
        """
        this function updates the increment statistics
        """
        min_inc = np.minimum(min_inc, np.abs(inc_tup))
        mean_inc += neqv_tups * inc_tup
        max_inc = np.maximum(max_inc, np.abs(inc_tup))

        return min_inc, mean_inc, max_inc
