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

import shutil
import logging
from dataclasses import dataclass, field
from pyscf import gto
import numpy as np
from typing import TYPE_CHECKING, cast

from pymbe.setup import settings, main as setup_main
from pymbe.output import DIVIDER
from pymbe.energy import EnergyExpCls
from pymbe.excitation import ExcExpCls
from pymbe.dipole import DipoleExpCls
from pymbe.trans import TransExpCls
from pymbe.rdm12 import RDMExpCls
from pymbe.genfock import GenFockExpCls
from pymbe.tools import RST, assertion

if TYPE_CHECKING:

    from typing import Union, Optional, Tuple
    from matplotlib import figure

    from pymbe.parallel import MPICls


@dataclass
class MBE:

    # expansion model
    method: str = "fci"
    cc_backend: str = "pyscf"
    hf_guess: bool = True

    # target property
    target: str = "energy"

    # system
    mol: Optional[gto.Mole] = None
    norb: Optional[int] = None
    nelec: Optional[Union[int, Tuple[int, int], np.ndarray]] = None
    point_group: Optional[str] = None
    orbsym: Optional[np.ndarray] = None
    fci_state_sym: Optional[Union[str, int]] = None
    fci_state_root: Optional[int] = None

    # orbital representation
    orb_type: str = "can"

    # integrals
    hcore: Optional[np.ndarray] = None
    eri: Optional[np.ndarray] = None

    # reference space
    ref_space: np.ndarray = np.array([], dtype=np.int64)
    exp_space: Optional[np.ndarray] = None

    # base model
    base_method: Optional[str] = None
    base_prop: Optional[
        Union[float, np.ndarray, Tuple[Union[float, np.ndarray], np.ndarray]]
    ] = None

    # screening
    screen_start: int = 4
    screen_perc: float = 0.9
    screen_func: str = "max"
    max_order: Optional[int] = None

    # restart
    rst: bool = True
    rst_freq: int = int(1e6)
    restarted: bool = field(init=False)

    # verbose
    verbose: int = 0

    # pi-pruning
    pi_prune: bool = False
    orbsym_linear: Optional[np.ndarray] = None

    # optional integrals for (transition) dipole moment
    dipole_ints: Optional[np.ndarray] = None

    # optional system parameters and integrals for generalized Fock matrix
    full_norb: Optional[int] = None
    full_nocc: Optional[int] = None
    inact_fock: Optional[np.ndarray] = None
    eri_goaa: Optional[np.ndarray] = None
    eri_gaao: Optional[np.ndarray] = None
    eri_gaaa: Optional[np.ndarray] = None
    no_singles: bool = False

    # mpi object
    mpi: MPICls = field(init=False)

    # exp object
    exp: Union[
        EnergyExpCls, ExcExpCls, DipoleExpCls, TransExpCls, RDMExpCls, GenFockExpCls
    ] = field(init=False)

    def kernel(
        self,
    ) -> Optional[
        Union[float, np.ndarray, Tuple[Union[float, np.ndarray], np.ndarray]]
    ]:
        """
        this function is the main pymbe kernel
        """
        # general settings
        settings()

        # calculation setup
        self = setup_main(self)

        # initialize exp object
        if self.target == "energy":
            self.exp = EnergyExpCls(self)
        elif self.target == "excitation":
            self.exp = ExcExpCls(self)
        elif self.target == "dipole":
            self.exp = DipoleExpCls(self)
        elif self.target == "trans":
            self.exp = TransExpCls(self)
        elif self.target == "rdm12":
            self.exp = RDMExpCls(self)
        elif self.target == "genfock":
            self.exp = GenFockExpCls(self)

        # dump flags
        if self.mpi.global_master:
            self.dump_flags()

        if self.mpi.global_master:

            # main master driver
            self.exp.driver_master(self.mpi)

            # delete restart file
            if self.rst:
                shutil.rmtree(RST)

        else:

            # main slave driver
            self.exp.driver_slave(self.mpi)

        # calculate total electronic property
        prop = self.final_prop(prop_type="electronic")

        return prop

    def dump_flags(self) -> None:
        """
        this function dumps all input flags
        """
        # get logger
        logger = logging.getLogger("pymbe_logger")

        # dump flags
        logger.info("\n" + DIVIDER + "\n")
        for key, value in vars(self).items():
            if key in [
                "mol",
                "hcore",
                "eri",
                "mpi",
                "exp",
                "dipole_ints",
                "inact_fock",
                "eri_goaa",
                "eri_gaao",
                "eri_gaaa",
            ]:
                logger.debug(" " + key + " = " + str(value))
            else:
                logger.info(" " + key + " = " + str(value))
        logger.debug("")
        for key, value in vars(self.mpi).items():
            logger.debug(" " + key + " = " + str(value))
        logger.info("\n" + DIVIDER)

    def results(self) -> str:
        """
        this function returns pymbe results as a string
        """
        if self.mpi.global_master:

            output_str = self.exp.print_results(self.mol, self.mpi)

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

            assertion(
                isinstance(prop_type, str),
                "final_prop: property type (prop_type keyword argument) must be a "
                "string",
            )
            assertion(
                prop_type in ["correlation", "electronic", "total"],
                "final_prop: valid property types (prop_type keyword argument) are: "
                "total, correlation and electronic",
            )
            if prop_type in ["correlation", "electronic"]:
                prop = self.exp.prop(prop_type)
            elif prop_type == "total":
                if isinstance(self.exp, EnergyExpCls):
                    assertion(
                        isinstance(nuc_prop, float),
                        "final_prop: nuclear repulsion energy (nuc_prop keyword "
                        "argument) must be a float",
                    )
                    prop = self.exp.prop(prop_type, nuc_prop=cast(float, nuc_prop))
                elif isinstance(self.exp, DipoleExpCls):
                    assertion(
                        isinstance(nuc_prop, np.ndarray),
                        "final_prop: nuclear dipole moment (nuc_prop keyword argument) "
                        "must be a np.ndarray",
                    )
                    prop = self.exp.prop(prop_type, nuc_prop=cast(np.ndarray, nuc_prop))
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

            assertion(
                isinstance(y_axis, str),
                "plot: y-axis property (y_axis keyword argument) must be a string",
            )
            assertion(
                y_axis in ["correlation", "electronic", "total"],
                "plot: valid y-axis properties (y_axis keyword argument) are: "
                "correlation, electronic and total",
            )
            if y_axis in ["correlation", "electronic"]:
                fig = self.exp.plot_results(y_axis)
            elif y_axis == "total":
                if isinstance(self.exp, EnergyExpCls):
                    assertion(
                        isinstance(nuc_prop, float),
                        "plot: nuclear repulsion energy (nuc_prop keyword argument) "
                        "must be a float",
                    )
                    fig = self.exp.plot_results(y_axis, nuc_prop=cast(float, nuc_prop))
                elif isinstance(self.exp, DipoleExpCls):
                    assertion(
                        isinstance(nuc_prop, np.ndarray),
                        "plot: nuclear dipole moment (nuc_prop keyword argument) must "
                        "be a np.ndarray",
                    )
                    fig = self.exp.plot_results(
                        y_axis, nuc_prop=cast(np.ndarray, nuc_prop)
                    )
                else:
                    fig = self.exp.plot_results(y_axis)

        else:

            fig = None

        return fig
