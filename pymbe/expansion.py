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
from pickle import load, dump
from scipy import optimize
from mpi4py import MPI
from abc import ABCMeta, abstractmethod
from numpy.polynomial.polynomial import Polynomial
from pyscf import gto, scf, cc, fci
from typing import TYPE_CHECKING, cast, TypeVar, Generic, Tuple, List, Union, Dict

from pymbe.logger import logger
from pymbe.output import (
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
    init_wfn,
    hop_no_singles,
    get_subspace_det_addr,
)
from pymbe.parallel import mpi_reduce, mpi_allreduce, mpi_bcast, open_shared_win
from pymbe.results import timings_prt

if TYPE_CHECKING:
    from typing import Optional, Callable

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
        self.tup_sq_overlaps: Dict[float, np.ndarray] = {}
        self._state_occup()

        # optional system parameters for generalized Fock matrix
        if hasattr(mbe, "full_norb"):
            self.full_norb = mbe.full_norb
        if hasattr(mbe, "full_nocc"):
            self.full_nocc = mbe.full_nocc
        if hasattr(self, "full_norb") and hasattr(self, "full_nocc"):
            self.full_nvirt = self.full_norb - self.norb - self.full_nocc

        # integrals
        self.hcore, self.eri, self.vhf = self._int_wins(mbe, mbe.mpi)

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
        self.ref_thres = mbe.ref_thres

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
            "prev": [],
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

        # individual orbital contributions
        self.orb_contrib: List[np.ndarray] = []

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

        # dryrun
        self.dryrun = mbe.dryrun

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
                    perm_ref_space: Optional[np.ndarray]
                    if self.ref_space.size == 0:
                        perm_ref_space = np.array([], dtype=np.int64)
                    else:
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
        self.ref_prop: TargetType
        self.ref_civec: np.ndarray
        if get_nexc(self.ref_nelec, self.ref_nhole) > self.vanish_exc:
            self.ref_prop, self.ref_civec = self._ref_prop(mbe.mpi)
        else:
            self.ref_prop = self._init_target_inst(0.0, self.ref_space.size)
            self.ref_civec = init_wfn(self.ref_space.size, self.ref_nelec, 1)
            self.ref_civec[0, 0, 0] = 1.0

        # attributes from restarted calculation
        if self.restarted:
            self._restart_main(mbe.mpi)

    def driver_master(self, mpi: MPICls) -> bool:
        """
        this function is the main pymbe master function
        """
        # initialize convergence boolean
        converged = False

        # begin mbe expansion depending
        for self.order in range(self.min_order, self.max_order + 1):
            # theoretical and actual number of tuples at current order
            if not self.restarted or self.order > self.start_order:
                self._ntuples(mpi)

            # print mbe header
            logger.info(
                mbe_header(
                    self.order,
                    self.n_tuples["inc"][self.order - self.min_order],
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
                    self.n_tuples["inc"][self.order - self.min_order],
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
                        np.setdiff1d(
                            self.exp_space[0],
                            self.exp_space[self.order - self.min_order + 1],
                        ).size,
                        self.screen_type,
                        self.mbe_tot_error[self.order - self.min_order]
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

            # check if overlap is larger than threshold for any tuple
            if len(self.tup_sq_overlaps) != 0:
                break

            # convergence check
            if self.exp_space[-1].size < self.order + 1 or self.order == self.max_order:
                # convergence boolean
                converged = True

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

        # bcast convergence boolean
        mpi.global_comm.bcast(converged, root=0)

        return converged

    def driver_slave(self, mpi: MPICls) -> bool:
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
                # receive convergence boolean
                converged = mpi.global_comm.bcast(None, root=0)

                # leave loop
                slave = False

        return converged

    def print_results(self, mpi: MPICls) -> str:
        """
        this function handles printing of results
        """
        # print timings
        string = (
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

    def _int_wins(self, mbe: MBE, mpi: MPICls) -> Tuple[MPI.Win, MPI.Win, MPI.Win]:
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
            hcore[:] = mbe.hcore

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
            eri[:] = mbe.eri

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

        # mpi barrier (ensure integrals are broadcasted to all nodes before continuing)
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

    def _ref_prop(self, mpi: MPICls) -> Tuple[TargetType, np.ndarray]:
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

            # perform main calc
            if self.method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"]:
                ref_prop = self._cc_kernel(
                    self.method,
                    core_idx,
                    cas_idx,
                    self.ref_nelec,
                    h1e_cas,
                    h2e_cas,
                    False,
                )
                ref_civec = init_wfn(cas_idx.size, self.ref_nelec, 1)
                ref_civec[0, 0, 0] = 1.0
            elif self.method == "fci":
                ref_prop, civec = self._fci_kernel(
                    e_core,
                    h1e_cas,
                    h2e_cas,
                    core_idx,
                    cas_idx,
                    self.ref_nelec,
                    False,
                )
                ref_civec = np.stack(civec)

            # perform base calc
            if self.base_method is not None:
                ref_prop -= self._cc_kernel(
                    self.base_method,
                    core_idx,
                    cas_idx,
                    self.ref_nelec,
                    h1e_cas,
                    h2e_cas,
                    False,
                )

            # log results
            logger.info(self._ref_results(ref_prop))

            # bcast ref_prop and ref_civec to slaves
            mpi.global_comm.bcast(ref_prop, root=0)
            mpi.global_comm.bcast(ref_civec, root=0)

        else:
            # receive ref_prop and ref_civec from master
            ref_prop = mpi.global_comm.bcast(None, root=0)
            ref_civec = mpi.global_comm.bcast(None, root=0)

        return ref_prop, ref_civec

    @abstractmethod
    def _ref_results(self, ref_prop: TargetType) -> str:
        """
        this function prints reference space results for a target calculation
        """

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
            split_file_name = os.path.splitext(files[i])[0].split("_")
            if len(split_file_name) == 3:
                # read hashes
                if split_file_name[1] == "hashes" and split_file_name[2].isdigit():
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
                elif split_file_name[1] == "inc" and split_file_name[2].isdigit():
                    n_tuples = self.n_tuples["inc"][len(self.incs)]
                    self.incs.append(
                        self._allocate_shared_inc(
                            n_tuples, mpi.local_master, mpi.local_comm
                        )
                    )
                    inc = self._open_shared_inc(
                        self.incs[-1], n_tuples, len(self.incs) - 1
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

                # read orbital contributions
                elif "mbe_orb_contrib" in files[i]:
                    self.orb_contrib.append(np.load(os.path.join(RST, files[i])))

                # read squared overlaps and respective tuples
                elif "tup_sq_overlaps" in files[i]:
                    with open(os.path.join(RST, files[i]), "rb") as f:
                        self.tup_sq_overlaps = load(f)

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
            # initialize number of calculations
            if len(self.n_tuples["calc"]) == self.order - self.min_order:
                self.n_tuples["calc"].append(0)

            # determine number of increments before purging
            if len(self.n_tuples["prev"]) == self.order - self.min_order:
                self.n_tuples["prev"].append(self.n_tuples["inc"][-1])

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
            rst_read = is_file(self.order, "mbe_tup_idx") and is_file(
                self.order, "mbe_tup"
            )
            # start tuple indices
            tup_idx = read_file("mbe_tup_idx", self.order).item() if rst_read else 0
            # start increment array indices
            inc_idx = read_file("mbe_inc_idx", self.order).item() if rst_read else 0
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
            mpi.local_master, mpi.local_comm, rst_read
        )

        # load and initialize increments
        inc, inc_win = self._load_inc(mpi.local_master, mpi.local_comm, rst_read)

        # mpi barrier (ensures hashes and inc arrays are zeroed on the local masters
        # before slaves start writing)
        mpi.local_comm.Barrier()

        # init time
        if mpi.global_master:
            if not rst_read:
                self.time["mbe"].append(0.0)
            time = MPI.Wtime()

        # init number of global master
        if mpi.global_master and rst_read:
            n_calc = self.n_tuples["calc"][-1]
        else:
            n_calc = 0

        # init increment statistics
        if mpi.global_master and rst_read:
            min_inc = self.min_inc[-1]
            mean_inc = self.mean_inc[-1]
            max_inc = self.max_inc[-1]
        else:
            min_inc = self._init_target_inst(1.0e12, self.norb)
            mean_inc = self._init_target_inst(0.0, self.norb)
            max_inc = self._init_target_inst(0.0, self.norb)

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
            self.rst and mpi.global_size < self.rst_freq < self.n_tuples["van"][-1]
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

        # initialize list for hashes and increments
        hashes_lst: List[int] = []
        inc_lst: List[TargetType] = []

        # perform calculation if not dryrun
        if not self.dryrun:
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
                start=tup_idx,
            ):
                # write restart files and re-init time
                if rst_write and tup_idx % self.rst_freq == 0:
                    # convert hashes and increments to array
                    hashes_arr = np.array(hashes_lst, dtype=np.int64)
                    inc_arr = self._init_inc_arr_from_lst(inc_lst)

                    # re-initialize hash and increment lists
                    hashes_lst = []
                    inc_lst = []

                    # number of tuples for every rank
                    recv_counts = np.array(mpi.global_comm.gather(hashes_arr.size))

                    # buffer to store hashes and increments
                    if mpi.global_master:
                        hash_buf: Optional[np.ndarray] = hashes[-1][
                            inc_idx : inc_idx + np.sum(recv_counts)
                        ]
                        inc_buf: Optional[IncType] = inc[-1][
                            inc_idx : inc_idx + np.sum(recv_counts)
                        ]
                    else:
                        hash_buf = inc_buf = None

                    # gatherv hashes and increments onto global master
                    mpi.global_comm.Gatherv(hashes_arr, (hash_buf, recv_counts), root=0)
                    self._mpi_gatherv_inc(mpi.global_comm, inc_arr, inc_buf)

                    # increment restart index
                    if mpi.global_master:
                        inc_idx += np.sum(recv_counts)

                    # reduce number of calculations onto global master
                    n_calc = mpi_reduce(
                        mpi.global_comm,
                        np.array(n_calc, dtype=np.int64),
                        root=0,
                        op=MPI.SUM,
                    ).item()
                    if not mpi.global_master:
                        n_calc = 0

                    # reduce increment statistics onto global master
                    min_inc = self._mpi_reduce_target(mpi.global_comm, min_inc, MPI.MIN)
                    mean_inc = self._mpi_reduce_target(
                        mpi.global_comm, mean_inc, MPI.SUM
                    )
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

                    # gather maximum squared overlaps on global master
                    if mpi.global_master:
                        sq_overlaps = cast(
                            List[Dict[float, np.ndarray]],
                            mpi.global_comm.gather(self.tup_sq_overlaps, root=0),
                        )
                        self.tup_sq_overlaps = {
                            sq_overlap: tup
                            for d in sq_overlaps
                            for sq_overlap, tup in d.items()
                        }
                    else:
                        mpi.global_comm.gather(self.tup_sq_overlaps, root=0)
                        self.tup_sq_overlaps = {}

                    # update rst_write
                    rst_write = (
                        tup_idx + self.rst_freq
                        < self.n_tuples["van"][-1] - mpi.global_size
                    )

                    if mpi.global_master:
                        # write restart files
                        self._write_target_file(min_inc, "mbe_min_inc", self.order)
                        self._write_target_file(mean_inc, "mbe_mean_inc", self.order)
                        self._write_target_file(max_inc, "mbe_max_inc", self.order)
                        write_file_mult(screen, "mbe_screen", self.order)
                        write_file(np.asarray(tup_idx), "mbe_tup_idx", self.order)
                        write_file(np.asarray(inc_idx), "mbe_inc_idx", self.order)
                        write_file(np.asarray(n_calc), "mbe_n_tuples_calc", self.order)
                        write_file(tup, "mbe_tup", self.order)
                        write_file(hashes[-1], "mbe_hashes", self.order)
                        self._write_inc_file(inc[-1], self.order)
                        with open(os.path.join(RST, "tup_sq_overlaps.pkl"), "wb") as f:
                            dump(self.tup_sq_overlaps, f)
                        self.time["mbe"][-1] += MPI.Wtime() - time
                        write_file(
                            np.asarray(self.time["mbe"][-1]), "mbe_time_mbe", self.order
                        )
                        # re-init time
                        time = MPI.Wtime()
                        # print status
                        logger.info(
                            mbe_status(self.order, inc_idx / self.n_tuples["inc"][-1])
                        )

                # distribute tuples
                if tup_idx % mpi.global_size != mpi.global_rank:
                    continue

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
                    eqv_tup_set = symm_eqv_tup(
                        cas_idx, self.symm_eqv_orbs[-1], ref_space
                    )

                    # skip calculation if symmetrically equivalent tuple will come later
                    if eqv_tup_set is None:
                        continue

                    # check for symmetrically equivalent increments
                    eqv_inc_lex_tup, eqv_inc_set = symm_eqv_inc(
                        self.eqv_inc_orbs[-1], eqv_tup_set, ref_space
                    )

                else:
                    # every tuple is unique without symmetry pruning
                    eqv_inc_lex_tup = [tup]
                    eqv_inc_set = [[tup]]

                # get core and cas indices
                core_idx, cas_idx = core_cas(self.nocc, self.ref_space, tup)

                # get h2e indices
                cas_idx_tril = idx_tril(cas_idx)

                # get h2e_cas
                h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

                # compute e_core and h1e_cas
                e_core, h1e_cas = e_core_h1e(hcore, vhf, core_idx, cas_idx)

                # get nelec_tup
                nelec_tup = get_nelec(self.occup, cas_idx)

                # calculate CASCI property
                target_tup = self._inc(
                    e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec_tup
                )

                # increment calculation counter
                n_calc += 1

                # loop over equivalent increment sets
                for tup, eqv_set in zip(eqv_inc_lex_tup, eqv_inc_set):
                    # calculate increment
                    inc_tup = target_tup - self._sum(inc, hashes, tup)

                    # add hash and increment
                    hashes_lst.append(hash_1d(tup))
                    inc_lst.append(inc_tup)

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

        # mpi barrier (ensures all slaves are done writing to hashes and inc arrays
        # before these are reduced and zeroed)
        mpi.global_comm.Barrier()

        # print final status
        if mpi.global_master:
            logger.info(mbe_status(self.order, 1.0))

        # convert hashes and increments to array
        hashes_arr = np.array(hashes_lst, dtype=np.int64)
        inc_arr = self._init_inc_arr_from_lst(inc_lst)

        # delete hash and increment lists
        del hashes_lst
        del inc_lst

        # number of tuples for every rank
        recv_counts = np.array(mpi.global_comm.gather(hashes_arr.size))

        # buffer to store hashes and increments
        if mpi.global_master:
            hash_buf = hashes[-1][-np.sum(recv_counts) :]
            inc_buf = inc[-1][-np.sum(recv_counts) :]
        else:
            hash_buf = inc_buf = None

        # gatherv hashes and increments onto global master
        mpi.global_comm.Gatherv(hashes_arr, (hash_buf, recv_counts), root=0)
        self._mpi_gatherv_inc(mpi.global_comm, inc_arr, inc_buf)

        # bcast hashes among local masters
        if mpi.local_master:
            hashes[-1][:] = mpi_bcast(mpi.master_comm, hashes[-1])
            inc[-1][:] = self._mpi_bcast_inc(mpi.master_comm, inc[-1])

        # sort hashes and increments
        if mpi.local_master:
            inc[-1][:] = inc[-1][np.argsort(hashes[-1])]
            hashes[-1][:].sort()

        # number of actual calculations
        n_calc = mpi_reduce(
            mpi.global_comm, np.array(n_calc, dtype=np.int64), root=0, op=MPI.SUM
        ).item()

        # increment statistics
        min_inc = self._mpi_reduce_target(mpi.global_comm, min_inc, MPI.MIN)
        mean_inc = self._mpi_reduce_target(mpi.global_comm, mean_inc, MPI.SUM)
        max_inc = self._mpi_reduce_target(mpi.global_comm, max_inc, MPI.MAX)
        if mpi.global_master:
            # total current-order increment
            tot = self._total_inc(inc[-1], mean_inc)
            if self.n_tuples["van"][-1] != 0:
                mean_inc /= self.n_tuples["van"][-1]

        # reduce screen
        if screen_mpi_func is not None:
            for screen_func, screen_array in screen.items():
                screen[screen_func] = mpi_reduce(
                    mpi.global_comm,
                    screen_array,
                    op=screen_mpi_func[screen_func],
                )

        # gather maximum squared overlaps on global master
        if mpi.global_master:
            sq_overlaps = cast(
                List[Dict[float, np.ndarray]],
                mpi.global_comm.gather(self.tup_sq_overlaps, root=0),
            )
            self.tup_sq_overlaps = {
                sq_overlap: tup for d in sq_overlaps for sq_overlap, tup in d.items()
            }
        else:
            mpi.global_comm.gather(self.tup_sq_overlaps, root=0)

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

            # append total number of calculations
            if len(self.n_tuples["calc"]) > self.order - self.min_order:
                self.n_tuples["calc"][-1] = n_calc
            else:
                self.n_tuples["calc"].append(n_calc)

            # append increment statistics
            if len(self.mean_inc) > self.order - self.min_order:
                self.mean_inc[-1] = mean_inc
                self.min_inc[-1] = min_inc
                self.max_inc[-1] = max_inc
            else:
                self.mean_inc.append(mean_inc)
                self.min_inc.append(min_inc)
                self.max_inc.append(max_inc)

            # append screening arrays
            if len(self.screen) > self.order - self.min_order:
                self.screen[-1] = screen
            else:
                self.screen.append(screen)

            # append orb_contrib statistics
            if "sum" in screen.keys():
                orb_contrib = screen["sum"] / self.order
            else:
                orb_contrib = np.zeros(self.norb, dtype=np.float64)
            if len(self.orb_contrib) > self.order - self.min_order:
                self.orb_contrib[-1] = orb_contrib
            else:
                self.orb_contrib.append(orb_contrib)

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
            with open(os.path.join(RST, "tup_sq_overlaps.pkl"), "wb") as f:
                dump(self.tup_sq_overlaps, f)
            self._write_target_file(self.min_inc[-1], "mbe_min_inc", self.order)
            self._write_target_file(self.mean_inc[-1], "mbe_mean_inc", self.order)
            self._write_target_file(self.max_inc[-1], "mbe_max_inc", self.order)
            write_file_mult(self.screen[-1], "mbe_screen", self.order)
            write_file(self.orb_contrib[-1], "mbe_orb_contrib", self.order)
            write_file(np.asarray(self.n_tuples["inc"][-1]), "mbe_tup_idx", self.order)
            write_file(
                np.asarray(self.n_tuples["calc"][-1]), "mbe_n_tuples_calc", self.order
            )
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

                # get number of tuples per orbital
                ntup_occ = orb_n_tuples(
                    self.exp_space[-1][self.exp_space[-1] < self.nocc],
                    self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                    self.ref_nelec,
                    self.ref_nhole,
                    self.vanish_exc,
                    self.order,
                    "occ",
                )
                ntup_virt = orb_n_tuples(
                    self.exp_space[-1][self.exp_space[-1] < self.nocc],
                    self.exp_space[-1][self.nocc <= self.exp_space[-1]],
                    self.ref_nelec,
                    self.ref_nhole,
                    self.vanish_exc,
                    self.order,
                    "virt",
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

                # log individual orbital contributions
                logger.info2(" ----------------------------------------------")
                logger.info2("  Orbital | Mean absolute increment |  Factor  ")
                logger.info2(" ----------------------------------------------")
                for orb in self.exp_space[-1]:
                    logger.info2(
                        f"     {orb:3}  | "
                        f"       {self.screen[-1]['mean_abs_inc'][orb]:>10.4e}       | "
                        f"{self.screen[-1]['rel_factor'][orb]:>8.2e}     "
                    )
                logger.info2(" ----------------------------------------------")

                # occupied and virtual expansion spaces
                exp_occ = self.exp_space[-1][self.exp_space[-1] < self.nocc]
                exp_virt = self.exp_space[-1][self.nocc <= self.exp_space[-1]]

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

                    # define allowed error
                    error_thresh = self.screen_thres - self.mbe_tot_error[-1]

                    # get index in expansion space for minimum mean absolute increment
                    min_idx = np.argmin(self.screen[-1]["mean_abs_inc"])

                    # check if minimum mean absolute increment comes close to
                    # convergence threshold
                    if 0.0 < self.screen[-1]["mean_abs_inc"][min_idx] < 1e1 * CONV_TOL:
                        # log screening
                        logger.info2(
                            f" Orbital {self.exp_space[-1][min_idx]} is screened away "
                            "due to the majority of increments getting close to "
                            "convergence\n criterium"
                        )

                    else:
                        # initialize array for estimated quantities
                        est_error = np.zeros(
                            (self.exp_space[-1].size, max_order - self.order),
                            dtype=np.float64,
                        )
                        est_rel_factor = np.zeros_like(est_error)
                        est_mean_abs_inc = np.zeros_like(est_error)

                        # initialize array for orbital contribution errors
                        tot_error = np.zeros(self.exp_space[-1].size, dtype=np.float64)

                        # initialize array for error difference to threshold
                        error_diff = np.zeros_like(tot_error)

                        # loop over orbitals
                        for orb_idx, orb in enumerate(self.exp_space[-1]):
                            # get mean absolute increments and relative factor for
                            # orbital
                            mean_abs_inc = np.array(
                                [screen["mean_abs_inc"][orb] for screen in self.screen],
                                dtype=np.float64,
                            )
                            rel_factor = np.array(
                                [screen["rel_factor"][orb] for screen in self.screen],
                                dtype=np.float64,
                            )

                            # log transform mean absolute increments
                            log_mean_abs_inc = np.log(mean_abs_inc[mean_abs_inc > 0.0])

                            # get corresponding relative factors
                            rel_factor = rel_factor[mean_abs_inc > 0.0]

                            # get orders for fit
                            orders = self.min_order + np.argwhere(
                                mean_abs_inc > 0.0
                            ).reshape(-1)

                            # require at least 3 points to fit
                            if orders.size > 2:
                                # fit logarithmic mean absolute increment
                                (opt_slope, opt_zero), cov = np.polyfit(
                                    orders, log_mean_abs_inc, 1, cov=True
                                )
                                err_slope, err_zero = np.sqrt(np.diag(cov))
                                opt_slope += 2 * err_slope
                                opt_zero += 2 * err_zero

                                # assume mean absolute increment does not decrease
                                mean_abs_inc_fit = Polynomial([opt_zero, opt_slope])

                                # define fitting function for relative factor
                                def rel_factor_fit(x, half, slope):
                                    return 1.0 / (
                                        1.0 + ((x - orders[0]) / half) ** slope
                                    )

                                # fit relative factor
                                if np.count_nonzero(rel_factor < 0.5) > 2:
                                    (opt_half, opt_slope), cov = optimize.curve_fit(
                                        rel_factor_fit,
                                        orders,
                                        rel_factor,
                                        bounds=([0.5, 1.0], [max_order + 1, np.inf]),
                                        maxfev=1000000,
                                    )
                                    err_half, err_slope = np.sqrt(np.diag(cov))

                                    opt_half = min(
                                        opt_half + 2 * err_half, max_order + 1
                                    )
                                    opt_slope = max(opt_slope - 2 * err_slope, 1.0)
                                else:
                                    opt_half = opt_slope = 0.0
                                    rel_factor_fit = lambda *args: 1.0

                                # initialize number of tuples for orbital for remaining
                                # orders
                                ntup_all_orb = 0

                                # initialize number of total tuples for remaining orders
                                ntup_all_tot = 0

                                # loop over remaining orders
                                for order_idx, order in enumerate(
                                    range(self.order + 1, max_order + 1)
                                ):
                                    ntup_order_orb = orb_n_tuples(
                                        exp_occ,
                                        exp_virt,
                                        self.ref_nelec,
                                        self.ref_nhole,
                                        self.vanish_exc,
                                        order,
                                        "occ" if orb < self.nocc else "virt",
                                    )
                                    ntup_all_orb += ntup_order_orb
                                    ntup_all_tot += n_tuples(
                                        exp_occ,
                                        exp_virt,
                                        self.ref_nelec,
                                        self.ref_nhole,
                                        self.vanish_exc,
                                        order,
                                    )

                                    est_rel_factor[orb_idx, order_idx] = rel_factor_fit(
                                        order, opt_half, opt_slope
                                    )

                                    est_mean_abs_inc[orb_idx, order_idx] = np.exp(
                                        mean_abs_inc_fit(order)
                                    )

                                    # calculate the error for this order
                                    est_error[orb_idx, order_idx] = (
                                        est_rel_factor[orb_idx, order_idx]
                                        * ntup_order_orb
                                        * est_mean_abs_inc[orb_idx, order_idx]
                                    )

                                    # add to total error
                                    tot_error[orb_idx] += est_error[orb_idx, order_idx]

                                    # stop if order contributes less than 1%
                                    if (
                                        est_error[orb_idx, order_idx]
                                        / tot_error[orb_idx]
                                        < 0.01
                                    ):
                                        break

                                # calculate difference to allowed error
                                error_diff[orb_idx] = (
                                    ntup_all_orb / ntup_all_tot
                                ) * error_thresh - tot_error[orb_idx]

                            # expansion is too short
                            else:
                                keep_screening = False
                                break

                        # stop screening if expansion is too short
                        if not keep_screening:
                            break

                        # get index in expansion space for minimum orbital contribution
                        min_idx = np.argmax(error_diff)

                        # screen orbital away if contribution is smaller than threshold
                        if error_diff[min_idx] > 0.0:
                            # log screening
                            logger.info2(
                                f" Orbital {self.exp_space[-1][min_idx]} is screened "
                                f"away (Error = {tot_error[min_idx]:>10.4e})"
                            )
                            logger.info2(" " + 70 * "-")
                            logger.info2(
                                "  Order | Est. relative factor | Est. mean abs. "
                                "increment | Est. error"
                            )
                            logger.info2(" " + 70 * "-")
                            for order, factor, mean_abs_inc, error in zip(
                                range(self.order + 1, max_order + 1),
                                est_rel_factor[min_idx],
                                est_mean_abs_inc[min_idx],
                                est_error[min_idx],
                            ):
                                if error == 0.0:
                                    break
                                logger.info2(
                                    f"  {order:5} |      {factor:>10.4e}      |        "
                                    f"{mean_abs_inc:>10.4e}        | {error:>10.4e}"
                                )
                            logger.info2(" " + 70 * "-" + "\n")

                            # add screened orbital contribution to error
                            self.mbe_tot_error[-1] += tot_error[min_idx]

                        # orbital with minimum contribution is not screened away
                        else:
                            keep_screening = False

                    if keep_screening:
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
                        for order_idx, order in enumerate(
                            range(self.min_order, self.order + 1)
                        ):
                            # get number of tuples per orbital
                            ntup_occ = orb_n_tuples(
                                exp_occ,
                                exp_virt,
                                self.ref_nelec,
                                self.ref_nhole,
                                self.vanish_exc,
                                order,
                                "occ",
                            )
                            ntup_virt = orb_n_tuples(
                                exp_occ,
                                exp_virt,
                                self.ref_nelec,
                                self.ref_nhole,
                                self.vanish_exc,
                                order,
                                "virt",
                            )

                            # calculate relative factor
                            self.screen[order_idx]["rel_factor"] = np.divide(
                                np.abs(self.screen[order_idx]["sum"]),
                                self.screen[order_idx]["sum_abs"],
                                out=np.zeros(self.norb, dtype=np.float64),
                                where=self.screen[order_idx]["sum_abs"] != 0.0,
                            )

                            # calculate mean absolute increment
                            self.screen[order_idx]["mean_abs_inc"][: self.nocc] = (
                                (
                                    self.screen[order_idx]["sum_abs"][: self.nocc]
                                    / ntup_occ
                                )
                                if ntup_occ > 0
                                else 0.0
                            )
                            self.screen[order_idx]["mean_abs_inc"][self.nocc :] = (
                                (
                                    self.screen[order_idx]["sum_abs"][self.nocc :]
                                    / ntup_virt
                                )
                                if ntup_virt > 0
                                else 0.0
                            )

                            # remove remaining orbitals if expansion space no longer
                            # produces valid increments
                            if np.all(self.screen[-1]["mean_abs_inc"] == 0.0):
                                self.exp_space[-1] = np.array([])

                    # stop screening if no other orbitals contribute above threshold
                    else:
                        keep_screening = False

                # signal other processes to stop screening
                mpi.global_comm.bcast(keep_screening, root=0)

                # log screening
                if np.array_equal(self.exp_space[-1], self.exp_space[-2]):
                    logger.info2(f" No orbitals were screened away.")

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
            mpi.global_comm.Gatherv(hashes_arr, (hashes, recv_counts), root=0)

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
            self._mpi_gatherv_inc(mpi.global_comm, inc_arr, inc)

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
        nelec: np.ndarray,
    ) -> TargetType:
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
        ref_guess: bool = True,
    ) -> TargetType:
        """
        this function return the result property from a given method
        """
        if method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"]:
            res = self._cc_kernel(method, core_idx, cas_idx, nelec, h1e, h2e, False)

        elif method == "fci":
            res, _ = self._fci_kernel(
                e_core, h1e, h2e, core_idx, cas_idx, nelec, ref_guess
            )

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
        ref_guess: bool,
    ) -> Tuple[TargetType, List[np.ndarray]]:
        """
        this function returns the results of a fci calculation
        """

    def _fci_driver(
        self,
        e_core: float,
        h1e: np.ndarray,
        h2e: np.ndarray,
        cas_idx: np.ndarray,
        nelec: np.ndarray,
        spin: int,
        wfnsym: int,
        roots: List[int],
        ref_guess: bool,
        conv_tol: float = CONV_TOL,
    ) -> Tuple[
        List[float],
        List[np.ndarray],
        Union[fci.direct_spin0_symm.FCI, fci.direct_spin0_symm.FCI],
    ]:
        """
        this function is the general fci driver function for all calculations
        """
        # init fci solver
        solver: Union[fci.direct_spin0_symm.FCI, fci.direct_spin0_symm.FCI]
        if spin == 0:
            solver = fci.direct_spin0_symm.FCI()
        else:
            solver = fci.direct_spin1_symm.FCI()

        # settings
        solver.conv_tol = conv_tol
        solver.lindep = min(1.0e-14, solver.conv_tol * 1.0e-1)
        solver.max_memory = MAX_MEM
        solver.max_cycle = 5000
        solver.max_space = 25
        solver.davidson_only = True
        solver.pspace_size = 0
        if self.verbose >= 4:
            solver.verbose = 10
        solver.wfnsym = wfnsym
        solver.orbsym = self.orbsym[cas_idx]
        solver.nroots = max(roots) + 1

        # create special function for hamiltonian operation when singles are omitted
        hop: Optional[Callable[[np.ndarray], np.ndarray]] = None
        if self.no_singles:
            hop = hop_no_singles(solver, cas_idx.size, nelec, spin, h1e, h2e)

        # starting guess
        ci0: Union[List[np.ndarray], None]
        ref_space_addr: Tuple[np.ndarray, np.ndarray]
        if ref_guess:
            # get addresses of reference space determinants in wavefunction
            ref_space_addr = get_subspace_det_addr(
                cas_idx, nelec, self.ref_space, self.ref_nelec
            )

            # reference space starting guess
            ci0 = []
            for civec in self.ref_civec:
                ci0.append(init_wfn(cas_idx.size, nelec, 1).squeeze(axis=0))
                ci0[-1][ref_space_addr[0].reshape(-1, 1), ref_space_addr[1]] = civec
        else:
            # hf starting guess
            if self.hf_guess:
                ci0 = [init_wfn(cas_idx.size, nelec, 1).squeeze(axis=0)]
                ci0[0][0, 0] = 1.0
            else:
                ci0 = None

        # interface
        def _fci_interface(
            roots: List[int],
        ) -> Tuple[List[float], List[np.ndarray], List[bool]]:
            """
            this function provides an interface to solver.kernel
            """
            # perform calc
            e, c = solver.kernel(
                h1e, h2e, cas_idx.size, nelec, ecore=e_core, ci0=ci0, hop=hop
            )

            # collect results
            if solver.nroots == 1:
                e, c, converged = [e], [c], [solver.converged]
            else:
                converged = solver.converged

            # check if reference space determinants should be used to choose states
            if ref_guess:
                # get coefficients of reference space determinants for every state
                inc_ref_civecs = np.stack(c)[
                    :, ref_space_addr[0].reshape(-1, 1), ref_space_addr[1]
                ]

                # determine squared overlaps and incremental norm of coefficients of
                # the states
                sq_overlaps = (
                    np.einsum("Iij,Jij->IJ", self.ref_civec.conj(), inc_ref_civecs) ** 2
                )
                norm_states = np.sum(sq_overlaps, axis=1)

                # determine maximum squared overlap and respective root index for every
                # state
                max_sq_overlaps = np.empty(self.ref_civec.shape[0], dtype=np.float64)
                root_idx = np.empty(self.ref_civec.shape[0], dtype=np.int64)
                for _ in range(self.ref_civec.shape[0]):
                    idx = np.unravel_index(np.argmax(sq_overlaps), sq_overlaps.shape)
                    max_sq_overlaps[idx[0]] = sq_overlaps[idx]
                    root_idx[idx[0]] = idx[1]
                    sq_overlaps[idx[0], :] = 0.0
                    sq_overlaps[:, idx[1]] = 0.0

                # define how long the number of roots is to be increased: either until
                # the root with the maximum squared overlap has been found for every
                # state or until all roots have been exhausted
                if isinstance(solver, fci.direct_spin0_symm.FCI):
                    # get total number of singlet states
                    strsa = fci.cistring.gen_strings4orblist(
                        range(cas_idx.size), nelec[0]
                    )
                    airreps = fci.direct_spin1_symm._gen_strs_irrep(
                        strsa, self.orbsym[cas_idx]
                    )
                    sym_allowed = (airreps[:, None] ^ airreps) == wfnsym
                    max_singlet_roots = (
                        np.count_nonzero(sym_allowed)
                        + np.count_nonzero(sym_allowed.diagonal())
                    ) // 2

                    # norm_states will not tend to 1 because higher spin states can also
                    # overlap with reference state: check whether the maximum
                    # number of singlet states has been found for this irrep
                    def find_roots():
                        return (
                            np.any(max_sq_overlaps <= 1 - norm_states)
                            and solver.nroots < max_singlet_roots
                        )

                else:
                    # norm_states will tend to 1 as all states are calculated
                    def find_roots():
                        return np.any(max_sq_overlaps <= 1 - norm_states)

                while find_roots():
                    # calculate additional root
                    solver.nroots += 1

                    # perform calc
                    e, c = solver.kernel(
                        h1e, h2e, cas_idx.size, nelec, ecore=e_core, ci0=c
                    )
                    converged = solver.converged

                    # get coefficients of reference space determinants
                    inc_ref_civec = c[-1][
                        ref_space_addr[0].reshape(-1, 1), ref_space_addr[1]
                    ]

                    # get squared overlap with reference space wavefunctions
                    sq_overlaps = (
                        np.einsum("Iij,ij->I", self.ref_civec.conj(), inc_ref_civec)
                        ** 2
                    )
                    norm_states += sq_overlaps

                    # check whether squared overlap is larger than current maximum
                    # squared overlap for any state
                    larger_sq_overlap = np.where(sq_overlaps > max_sq_overlaps)[0]

                    # save state with the highest squared overlap
                    if larger_sq_overlap.size > 0:
                        idx = larger_sq_overlap[
                            np.argmax(sq_overlaps[larger_sq_overlap])
                        ]
                        max_sq_overlaps[idx] = sq_overlaps[idx]
                        root_idx[idx] = solver.nroots - 1

                # check whether any maximum squared overlap is below threshold
                min_sq_overlap = np.min(max_sq_overlaps).astype(float)
                if (
                    isinstance(self.ref_thres, int)
                    and (
                        self.ref_space.size < self.ref_thres
                        or np.any(min_sq_overlap < 0.9)
                    )
                ) or (
                    isinstance(self.ref_thres, float)
                    and np.any(min_sq_overlap < self.ref_thres)
                ):
                    self.tup_sq_overlaps[min_sq_overlap] = np.setdiff1d(
                        cas_idx, self.ref_space
                    )

                # get root indices
                roots = root_idx.tolist()

            # collect results
            return (
                [e[root] for root in roots],
                [c[root] for root in roots],
                [converged[root] for root in roots],
            )

        # perform calc
        energies, civecs, converged = _fci_interface(roots)

        # multiplicity check
        for root in range(len(civecs)):
            _, mult = solver.spin_square(civecs[root], cas_idx.size, nelec)

            if np.abs((spin + 1) - mult) > SPIN_TOL:
                # fix spin by applying level shift
                solver.nroots = max(roots) + 1
                sz = np.abs(nelec[0] - nelec[1]) * 0.5
                solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.0))

                # perform calc
                energies, civecs, converged = _fci_interface(roots)

                # verify correct spin
                for root in range(len(civecs)):
                    s, mult = solver.spin_square(civecs[root], cas_idx.size, nelec)
                    if np.abs((spin + 1) - mult) > SPIN_TOL:
                        raise RuntimeError(
                            f"spin contamination for root entry = {root}\n"
                            f"2*S + 1 = {mult:.6f}\n"
                            f"cas_idx = {cas_idx}\n"
                            f"cas_sym = {self.orbsym[cas_idx]}"
                        )

        # convergence check
        for root in roots:
            if not converged[root]:
                raise RuntimeError(
                    f"state {root} not converged\n"
                    f"cas_idx = {cas_idx}\n"
                    f"cas_sym = {self.orbsym[cas_idx]}",
                )

        return energies, civecs, solver

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

    def _ccsd_driver_pyscf(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        core_idx: np.ndarray,
        cas_idx: np.ndarray,
        spin: int,
        converge_amps: bool = False,
    ) -> Tuple[
        float,
        Union[cc.ccsd.CCSD, cc.uccsd.UCCSD],
        Union[cc.ccsd._ChemistsERIs, cc.uccsd._ChemistsERIs],
    ]:
        """
        this function is the general cc driver function for all calculations
        """
        # init ccsd solver
        mol_tmp = gto.Mole(verbose=0)
        mol_tmp._built = True
        mol_tmp.max_memory = MAX_MEM
        mol_tmp.incore_anyway = True

        if spin == 0:
            hf = scf.RHF(mol_tmp)
        else:
            hf = scf.UHF(mol_tmp)

        hf.get_hcore = lambda *args: h1e
        hf._eri = h2e

        if spin == 0:
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
        if converge_amps:
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
        if not ccsd.converged:
            raise RuntimeError(
                f"CCSD error: no convergence, core_idx = {core_idx}, "
                f"cas_idx = {cas_idx}"
            )

        # energy
        energy = ccsd.e_corr

        return energy, ccsd, eris

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
        self, local_master: bool, local_comm: MPI.Comm, rst_read: bool
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
        self, local_master: bool, local_comm: MPI.Comm, rst_read: bool
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

    @abstractmethod
    def _init_inc_arr_from_lst(self, inc_lst: List[TargetType]) -> IncType:
        """
        this function creates an increment array from a list of increments
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
        comm: MPI.Comm, send_inc: IncType, recv_inc: Optional[IncType]
    ) -> None:
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
        this function prints mbe results statistics for a target calculation
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

    def _init_inc_arr_from_lst(self, inc_lst: List[SingleTargetType]) -> np.ndarray:
        """
        this function creates an increment array from a list of increments
        """
        return np.array(inc_lst, dtype=np.float64)

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
        comm: MPI.Comm, send_inc: np.ndarray, recv_inc: Optional[np.ndarray]
    ) -> None:
        """
        this function performs a MPI gatherv operation on the increments
        """
        # size of arrays on every rank
        recv_counts = np.array(comm.gather(send_inc.size))

        comm.Gatherv(send_inc, (recv_inc, recv_counts), root=0)

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
