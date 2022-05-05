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
from dataclasses import dataclass, field
from pyscf import gto
import numpy as np
from typing import TYPE_CHECKING

from pymbe.setup import settings, main as setup_main
from pymbe.energy import EnergyExpCls
from pymbe.excitation import ExcExpCls
from pymbe.dipole import DipoleExpCls
from pymbe.trans import TransExpCls
from pymbe.rdm12 import RDMExpCls
from pymbe.tools import RST

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
    nuc_energy: Optional[float] = None
    nuc_dipole: Optional[np.ndarray] = None
    ncore: int = 0
    norb: Optional[int] = None
    nelec: Optional[Union[int, Tuple[int, int], np.ndarray]] = None
    point_group: Optional[str] = None
    orbsym: Optional[np.ndarray] = None
    fci_state_sym: Optional[Union[str, int]] = None
    fci_state_root: Optional[int] = None

    # hf calculation
    hf_prop: Optional[Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None

    # orbital representation
    orb_type: str = "can"

    # integrals
    hcore: Optional[np.ndarray] = None
    eri: Optional[np.ndarray] = None
    vhf: np.ndarray = field(init=False)
    dipole_ints: Optional[np.ndarray] = None

    # reference space
    ref_space: np.ndarray = np.array([], dtype=np.int64)

    # base model
    base_method: Optional[str] = None
    base_prop: Optional[Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None

    # screening
    screen_start: int = 4
    screen_perc: float = 0.9
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

    # mpi object
    mpi: MPICls = field(init=False)

    # exp object
    exp: Union[EnergyExpCls, ExcExpCls, DipoleExpCls, TransExpCls, RDMExpCls] = field(
        init=False
    )

    def kernel(
        self,
    ) -> Optional[Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
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

        if self.mpi.global_master:

            # main master driver
            self.exp.driver_master(self.mpi)

            # delete restart file
            if self.rst:
                shutil.rmtree(RST)

            # calculate total property
            prop = self.exp.tot_prop()

        else:

            # main slave driver
            self.exp.driver_slave(self.mpi)

            # calculate total property
            prop = None

        return prop

    def results(self) -> str:
        """
        this function returns pymbe results as a string
        """
        output_str = self.exp.print_results(self.mol, self.mpi)

        return output_str

    def plot(self) -> figure.Figure:
        """
        this function plots pymbe results
        """
        fig = self.exp.plot_results()

        return fig
