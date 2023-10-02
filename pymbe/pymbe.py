#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main pymbe module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import os
import shutil
from pyscf import ao2mo, symm
import numpy as np
from typing import TYPE_CHECKING, cast

from pymbe.logger import logger
from pymbe.parallel import MPICls
from pymbe.setup import (
    restart_read_kw,
    restart_read_system,
    general_setup,
    calc_setup,
    ref_space_update,
)
from pymbe.energy import EnergyExpCls
from pymbe.excitation import ExcExpCls
from pymbe.dipole import DipoleExpCls
from pymbe.trans import TransExpCls
from pymbe.rdm12 import ssRDMExpCls, saRDMExpCls
from pymbe.genfock import ssGenFockExpCls, saGenFockExpCls
from pymbe.tools import RST, ground_state_sym

if TYPE_CHECKING:
    from typing import Union, Optional, Tuple, List, Dict
    from pyscf import gto
    from matplotlib import figure


class MBE:
    def __init__(
        self,
        method: str = "fci",
        cc_backend: str = "pyscf",
        hf_guess: bool = True,
        target: str = "energy",
        mol: Optional[gto.Mole] = None,
        norb: Optional[int] = None,
        nelec: Optional[
            Union[
                int,
                Tuple[int, int],
                np.ndarray,
                List[Tuple[int, int]],
                List[np.ndarray],
            ]
        ] = None,
        point_group: Optional[str] = None,
        orbsym: Optional[Union[np.ndarray, List[Dict[int, Tuple[int, ...]]]]] = None,
        fci_state_sym: Optional[Union[str, int, List[str], List[int]]] = None,
        fci_state_root: Optional[Union[int, List[int]]] = None,
        fci_state_weights: Optional[List[float]] = None,
        orb_type: str = "can",
        hcore: Optional[np.ndarray] = None,
        eri: Optional[np.ndarray] = None,
        ref_space: np.ndarray = np.array([], dtype=np.int64),
        exp_space: Optional[np.ndarray] = None,
        ref_thres: float = 0.0,
        base_method: Optional[str] = None,
        base_prop: Optional[
            Union[float, np.ndarray, Tuple[Union[float, np.ndarray], np.ndarray]]
        ] = None,
        screen_type: str = "fixed",
        screen_start: int = 4,
        screen_perc: float = 0.9,
        screen_thres: float = 1.0e-2,
        screen_func: str = "max",
        max_order: Optional[int] = None,
        rst: bool = True,
        rst_freq: int = int(1e6),
        verbose: int = 0,
        dryrun: bool = False,
        pi_prune: bool = False,
        orbsym_linear: Optional[np.ndarray] = None,
        no_singles: bool = False,
        dipole_ints: Optional[np.ndarray] = None,
        full_norb: Optional[int] = None,
        full_nocc: Optional[int] = None,
        inact_fock: Optional[np.ndarray] = None,
        eri_goaa: Optional[np.ndarray] = None,
        eri_gaao: Optional[np.ndarray] = None,
        eri_gaaa: Optional[np.ndarray] = None,
    ):
        # exp object
        self.exp: Union[
            EnergyExpCls,
            ExcExpCls,
            DipoleExpCls,
            TransExpCls,
            ssRDMExpCls,
            saRDMExpCls,
            ssGenFockExpCls,
            saGenFockExpCls,
        ]

        # mpi object
        self.mpi = MPICls()

        # input handling
        if self.mpi.global_master:
            # check for restart folder
            if not os.path.isdir(RST):
                # expansion model
                self.method = method
                self.cc_backend = cc_backend
                self.hf_guess = hf_guess

                # target property
                self.target = target

                # system
                if norb is not None:
                    self.norb = norb
                elif mol:
                    # copy number of orbitals from mol object
                    if isinstance(mol.nao, int):
                        self.norb = mol.nao
                    else:
                        self.norb = mol.nao.item()
                # convert number of electrons to numpy array
                self.nelec: Union[np.ndarray, List[np.ndarray]]
                if isinstance(nelec, int):
                    self.nelec = np.asarray((nelec // 2, nelec // 2), dtype=np.int64)
                elif isinstance(nelec, tuple):
                    self.nelec = np.asarray(nelec, dtype=np.int64)
                elif isinstance(nelec, list):
                    self.nelec = [
                        np.asarray(state, dtype=np.int64)
                        if isinstance(state, tuple)
                        else state
                        for state in nelec
                    ]
                elif nelec is not None:
                    self.nelec = nelec
                elif nelec is None and mol:
                    # copy number of electrons from mol object
                    self.nelec = np.asarray(mol.nelec, dtype=np.int64)
                if point_group is not None:
                    self.point_group = cast(str, symm.addons.std_symb(point_group))
                elif mol and orbsym is not None:
                    # copy point group from mol object
                    if orb_type == "local":
                        self.point_group = mol.topgroup
                    else:
                        self.point_group = mol.groupname
                else:
                    # set default value for point group
                    self.point_group = "C1"
                if orbsym is not None:
                    self.orbsym = orbsym
                else:
                    # set default value for orbital symmetry
                    if self.point_group == "C1" and hasattr(self, "norb"):
                        self.orbsym = np.zeros(self.norb, dtype=np.int64)
                if fci_state_sym is not None:
                    self.fci_state_sym = fci_state_sym
                else:
                    # set default value for fci wavefunction state symmetry
                    if (
                        hasattr(self, "orbsym")
                        and isinstance(self.orbsym, np.ndarray)
                        and isinstance(self.nelec, np.ndarray)
                    ):
                        self.fci_state_sym = ground_state_sym(
                            self.orbsym, self.nelec, self.point_group
                        )
                    elif (
                        hasattr(self, "orbsym")
                        and isinstance(self.orbsym, np.ndarray)
                        and isinstance(self.nelec, list)
                    ):
                        self.fci_state_sym = [
                            ground_state_sym(self.orbsym, state, self.point_group)
                            for state in self.nelec
                        ]
                    else:
                        self.fci_state_sym = 0
                if fci_state_root is not None:
                    self.fci_state_root = fci_state_root
                else:
                    # set default value for fci wavefunction state root
                    if self.target in ["excitation", "trans"]:
                        self.fci_state_root = 1
                    else:
                        self.fci_state_root = 0
                if fci_state_weights is not None:
                    self.fci_state_weights = fci_state_weights
                # set default values for state-averaged rdm12 and genfock calculations
                if self.target in ["rdm12", "genfock"]:
                    state_vars = [
                        "nelec",
                        "fci_state_sym",
                        "fci_state_root",
                        "fci_state_weights",
                    ]
                    for state_var in state_vars:
                        if hasattr(self, state_var) and isinstance(
                            getattr(self, state_var), list
                        ):
                            nstates = len(getattr(self, state_var))
                            self.hf_guess = False
                            if hasattr(self, "nelec") and isinstance(
                                self.nelec, np.ndarray
                            ):
                                self.nelec = [self.nelec for _ in range(nstates)]
                            if hasattr(self, "fci_state_sym") and isinstance(
                                self.fci_state_sym, int
                            ):
                                self.fci_state_sym = [
                                    self.fci_state_sym for _ in range(nstates)
                                ]
                            if hasattr(self, "fci_state_root") and isinstance(
                                self.fci_state_root, int
                            ):
                                self.fci_state_root = [
                                    self.fci_state_root for _ in range(nstates)
                                ]
                            if not hasattr(self, "fci_state_weights"):
                                self.fci_state_weights = [
                                    1 / nstates for _ in range(nstates)
                                ]
                            break

                # optional system parameters for generalized Fock matrix
                if full_norb is not None:
                    self.full_norb = full_norb
                if full_nocc is not None:
                    self.full_nocc = full_nocc

                # orbital representation
                self.orb_type = orb_type

                # integrals
                if hcore is not None:
                    self.hcore = hcore
                if eri is not None and hasattr(self, "norb"):
                    # reorder electron repulsion integrals
                    self.eri = ao2mo.restore(4, eri, self.norb)

                # optional integrals for (transition) dipole moment
                if dipole_ints is not None:
                    self.dipole_ints = dipole_ints

                # optional integrals for generalized Fock matrix
                if inact_fock is not None:
                    self.inact_fock = inact_fock
                if eri_goaa is not None:
                    self.eri_goaa = eri_goaa
                if eri_gaao is not None:
                    self.eri_gaao = eri_gaao
                if eri_gaaa is not None:
                    self.eri_gaaa = eri_gaaa

                # reference space
                self.ref_space = ref_space
                if exp_space is not None:
                    self.exp_space = exp_space
                elif hasattr(self, "norb"):
                    # set default value for expansion space
                    self.exp_space = np.array(
                        [i for i in range(self.norb) if i not in self.ref_space],
                        dtype=np.int64,
                    )
                self.ref_thres = ref_thres

                # base model
                self.base_method = base_method
                if base_prop is not None:
                    self.base_prop = base_prop
                elif base_method is None:
                    # set default value for base model property
                    if self.target in ["energy", "excitation"]:
                        self.base_prop = 0.0
                    elif self.target in ["dipole", "trans"]:
                        self.base_prop = np.zeros(3, dtype=np.float64)
                    elif self.target == "rdm12" and hasattr(self, "norb"):
                        self.base_prop = (
                            np.zeros(2 * (self.norb,), dtype=np.float64),
                            np.zeros(4 * (self.norb,), dtype=np.float64),
                        )
                    elif (
                        self.target == "genfock"
                        and hasattr(self, "full_nocc")
                        and hasattr(self, "norb")
                        and hasattr(self, "full_norb")
                    ):
                        self.base_prop = (
                            0.0,
                            np.zeros(
                                (self.full_nocc + self.norb, self.full_norb),
                                dtype=np.float64,
                            ),
                        )

                # screening
                self.screen_type = screen_type
                self.screen_start = screen_start
                self.screen_perc = screen_perc
                self.screen_thres = screen_thres
                self.screen_func = screen_func
                if max_order is not None and hasattr(self, "exp_space"):
                    self.max_order = min(self.exp_space.size, max_order)
                elif max_order is not None:
                    self.max_order = max_order
                elif hasattr(self, "exp_space"):
                    # set default value for maximum expansion order
                    self.max_order = self.exp_space.size

                # restart
                self.rst = rst
                self.rst_freq = rst_freq
                self.restarted = False

                # verbose
                self.verbose = verbose

                # dryrun
                self.dryrun = dryrun

                # pi-pruning
                self.pi_prune = pi_prune
                if orbsym_linear is not None:
                    self.orbsym_linear = orbsym_linear

                # exclude single excitations
                self.no_singles = no_singles

            else:
                # read keywords
                self = restart_read_kw(self)

                # read system quantities
                self = restart_read_system(self)

                # restart logical
                self.restarted = True

    def kernel(
        self,
    ) -> Optional[
        Union[float, np.ndarray, Tuple[Union[float, np.ndarray], np.ndarray]]
    ]:
        """
        this function is the main pymbe kernel
        """
        # general settings
        if self.mpi.global_master:
            general_setup(self)

        # initialize convergence boolean
        converged = False

        # start loop over reference spaces
        while not converged:
            # calculation setup
            self = calc_setup(self)

            # initialize exp object
            if self.target == "energy":
                self.exp = EnergyExpCls(self)
            elif self.target == "excitation":
                self.exp = ExcExpCls(self)
            elif self.target == "dipole":
                self.exp = DipoleExpCls(self)
            elif self.target == "trans":
                self.exp = TransExpCls(self)
            elif (
                self.target == "rdm12"
                and isinstance(self.nelec, np.ndarray)
                and isinstance(self.fci_state_sym, int)
                and isinstance(self.fci_state_root, int)
            ):
                self.exp = ssRDMExpCls(self)
            elif (
                self.target == "rdm12"
                and isinstance(self.nelec, list)
                and isinstance(self.fci_state_sym, list)
                and isinstance(self.fci_state_root, list)
            ):
                self.exp = saRDMExpCls(self)
            elif (
                self.target == "genfock"
                and isinstance(self.nelec, np.ndarray)
                and isinstance(self.fci_state_sym, int)
                and isinstance(self.fci_state_root, int)
            ):
                self.exp = ssGenFockExpCls(self)
            elif (
                self.target == "genfock"
                and isinstance(self.nelec, list)
                and isinstance(self.fci_state_sym, list)
                and isinstance(self.fci_state_root, list)
            ):
                self.exp = saGenFockExpCls(self)

            if self.mpi.global_master:
                # main master driver
                converged = self.exp.driver_master(self.mpi)

                # delete restart folder
                if self.rst:
                    shutil.rmtree(RST)

                if not converged:
                    # update reference space
                    self.ref_space, self.exp_space = ref_space_update(
                        self.exp.tup_sq_overlaps, self.ref_space, self.exp_space
                    )

            else:
                # main slave driver
                converged = self.exp.driver_slave(self.mpi)

        # calculate total electronic property
        prop = self.final_prop(prop_type="electronic")

        return prop

    def results(self) -> str:
        """
        this function returns pymbe results as a string
        """
        if self.mpi.global_master:
            output_str = self.exp.print_results(self.mpi)
        else:
            output_str = ""

        return output_str

    def final_prop(
        self,
        prop_type: str = "total",
        nuc_prop: Optional[Union[float, np.ndarray]] = None,
    ) -> Optional[
        Union[float, np.ndarray, Tuple[Union[float, np.ndarray], np.ndarray]]
    ]:
        """
        this function returns the total property
        """
        if self.mpi.global_master:
            if not isinstance(prop_type, str):
                raise TypeError(
                    "final_prop: property type (prop_type keyword argument) must be a "
                    "string"
                )
            if prop_type not in ["correlation", "electronic", "total"]:
                raise ValueError(
                    "final_prop: valid property types (prop_type keyword argument) "
                    "are: total, correlation and electronic"
                )
            if prop_type in ["correlation", "electronic"]:
                prop = self.exp.prop(prop_type)
            elif prop_type == "total":
                if isinstance(self.exp, EnergyExpCls):
                    if not isinstance(nuc_prop, float):
                        raise TypeError(
                            "final_prop: nuclear repulsion energy (nuc_prop keyword "
                            "argument) must be a float"
                        )
                    prop = self.exp.prop(prop_type, nuc_prop=nuc_prop)
                elif isinstance(self.exp, DipoleExpCls):
                    if not isinstance(nuc_prop, np.ndarray):
                        raise TypeError(
                            "final_prop: nuclear dipole moment (nuc_prop keyword "
                            "argument) must be a np.ndarray"
                        )
                    prop = self.exp.prop(prop_type, nuc_prop=nuc_prop)
                else:
                    prop = self.exp.prop(prop_type)

        else:
            prop = None

        return prop

    def plot(
        self,
        y_axis: str = "correlation",
        nuc_prop: Optional[Union[float, np.ndarray]] = None,
    ) -> Optional[figure.Figure]:
        """
        this function plots pymbe results
        """
        if self.mpi.global_master:
            if not isinstance(y_axis, str):
                raise TypeError(
                    "plot: y-axis property (y_axis keyword argument) must be a string"
                )
            if y_axis not in ["correlation", "electronic", "total"]:
                raise ValueError(
                    "plot: valid y-axis properties (y_axis keyword argument) are: "
                    "correlation, electronic and total",
                )
            if y_axis in ["correlation", "electronic"]:
                fig = self.exp.plot_results(y_axis)
            elif y_axis == "total":
                if isinstance(self.exp, EnergyExpCls):
                    if not isinstance(nuc_prop, float):
                        raise TypeError(
                            "plot: nuclear repulsion energy (nuc_prop keyword argument) "
                            "must be a float"
                        )
                    fig = self.exp.plot_results(y_axis, nuc_prop=nuc_prop)
                elif isinstance(self.exp, DipoleExpCls):
                    if not isinstance(nuc_prop, np.ndarray):
                        raise TypeError(
                            "plot: nuclear dipole moment (nuc_prop keyword argument) must "
                            "be a np.ndarray"
                        )
                    fig = self.exp.plot_results(y_axis, nuc_prop=nuc_prop)
                else:
                    fig = self.exp.plot_results(y_axis)

        else:
            fig = None

        return fig
