#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main pymbe module
"""

from __future__ import annotations

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import shutil
from dataclasses import dataclass, field
from pyscf import gto
import numpy as np
from typing import TYPE_CHECKING

from pymbe.setup import settings, main as setup_main
from pymbe.driver import master as driver_master, slave as driver_slave
from pymbe.tools import RST
from pymbe.results import print_results, plot_results

if TYPE_CHECKING:

    from typing import Union, Optional
    from matplotlib import figure
    
    from pymbe.parallel import MPICls
    from pymbe.expansion import ExpCls


@dataclass
class MBE():

        # expansion model
        method: str = 'fci'
        fci_solver: str = 'pyscf_spin0'
        cc_backend: str = 'pyscf'
        hf_guess: bool = True

        # target property
        target: str = 'energy'

        # system
        mol: Optional[gto.Mole] = None
        nuc_energy: Optional[float] = None
        nuc_dipole: Optional[np.ndarray] = None
        ncore: int = 0
        nocc: Optional[int] = None
        norb: Optional[int] = None
        spin: Optional[int] = None
        point_group: Optional[str] = None
        orbsym: Optional[np.ndarray] = None
        fci_state_sym: Optional[Union[str, int]] = None
        fci_state_root: Optional[int] = None

        # hf calculation
        hf_prop: Optional[Union[float, np.ndarray]] = None
        occup: Optional[np.ndarray] = None

        # orbital representation
        orb_type: str = 'can'

        # integrals
        hcore: Optional[np.ndarray] = None
        vhf: Optional[np.ndarray] = None
        eri: Optional[np.ndarray] = None
        dipole_ints: Optional[np.ndarray] = None

        # reference space
        ref_space: np.ndarray = np.array([], dtype=np.int64)
        ref_prop: Optional[Union[float, np.ndarray]] = None

        # base model
        base_method: Optional[str] = None
        base_prop: Optional[Union[float, np.ndarray]] = None

        # screening
        screen_start: int = 4
        screen_perc: float = .9
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
        exp: ExpCls = field(init=False)

        def kernel(self) -> None:
                """
                this function is the main pymbe kernel
                """
                # general settings
                settings()

                # calculation setup
                self = setup_main(self)

                if self.mpi.global_master:

                    # main master driver
                    driver_master(self.mpi, self.exp)

                    # delete restart file
                    if self.rst:
                        shutil.rmtree(RST)

                else:

                    # main slave driver
                    driver_slave(self.mpi, self.exp)


        def results(self) -> str:
                """
                this function returns pymbe results as a string
                """
                output_str = print_results(self.mol, self.mpi, self.exp)

                return output_str


        def plot(self) -> figure.Figure:
                """
                this function plots pymbe results
                """
                fig = plot_results(self.exp)

                return fig
