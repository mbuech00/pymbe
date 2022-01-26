#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
expansion module
"""

from __future__ import annotations

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import os
import numpy as np
from mpi4py import MPI
from typing import TYPE_CHECKING, cast

from pymbe.tools import RST, pi_space, natural_keys, inc_shape, inc_dim
from pymbe.parallel import mpi_bcast

if TYPE_CHECKING:

    from typing import List, Dict, Tuple, Optional

    from pymbe.pymbe import MBE
    from pymbe.parallel import MPICls


class ExpCls:
        """
        this class contains the pymbe expansion attributes
        """
        def __init__(self, mbe: MBE) -> None:
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
                if self.target == 'dipole':
                    self.nuc_dipole = cast(np.ndarray, mbe.nuc_dipole)
                self.ncore: int = mbe.ncore
                self.nocc = cast(int, mbe.nocc)
                self.norb = cast(int, mbe.norb)
                self.spin = cast(int, mbe.spin)
                self.point_group = cast(str, mbe.point_group)
                self.orbsym = cast(np.ndarray, mbe.orbsym)
                self.fci_state_sym = cast(int, mbe.fci_state_sym)
                self.fci_state_root = cast(int, mbe.fci_state_root)
                
                # hf calculation
                self.hf_prop = np.asarray(mbe.hf_prop, dtype=np.float64)
                self.occup = cast(np.ndarray, mbe.occup)

                # integrals
                hcore, vhf, eri = int_wins(mbe.hcore, mbe.vhf, mbe.eri, \
                                           mbe.mpi, self.norb, self.nocc)
                self.hcore: MPI.Win = hcore
                self.vhf: MPI.Win = vhf
                self.eri: MPI.Win = eri
                self.dipole_ints: Optional[np.ndarray] = mbe.dipole_ints
                
                # orbital representation
                self.orb_type: str = mbe.orb_type

                # reference space
                self.ref_space: np.ndarray = mbe.ref_space
                self.ref_prop = np.asarray(mbe.ref_prop, dtype=np.float64)

                # expansion space
                self.exp_space: List[np.ndarray] = [np.array([i for i in range(self.ncore, self.norb) if i not in self.ref_space], dtype=np.int64)]
                
                # base model
                self.base_method: Optional[str] = mbe.base_method
                self.base_prop = np.asarray(mbe.base_prop, dtype=np.float64)

                # total mbe property
                self.mbe_tot_prop: List[np.ndarray] = []

                # increment windows
                self.incs: List[MPI.Win] = []

                # hash windows
                self.hashes: List[MPI.Win] = []

                # timings
                self.time: Dict[str, List[float]] = {'mbe': [], 'purge': []}

                # statistics
                self.min_inc: List[np.ndarray] = []
                self.mean_inc: List[np.ndarray] = []
                self.max_inc: List[np.ndarray] = []

                # number of tuples
                self.n_tuples: Dict[str, List[int]] = {'theo': [], 'calc': [], \
                                                       'inc': []}  

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

                if self.base_method in ['ccsd', 'ccsd(t)', 'ccsdt']:
                    min_order = 4 - min(2, self.ref_space[self.ref_space < self.nocc].size) \
                                - min(2, self.ref_space[self.nocc <= self.ref_space].size)
                elif self.base_method == 'ccsdtq':
                    min_order = 6 - min(3, self.ref_space[self.ref_space < self.nocc].size) \
                                - min(3, self.ref_space[self.nocc <= self.ref_space].size)
                else:
                    min_order = 2 - self.ref_space.size
                self.min_order: int = max(1, min_order)

                if self.restarted:
                    start_order = restart_main(mbe.mpi, self)
                else:
                    start_order = self.min_order
                self.start_order: int = start_order

                if mbe.max_order is not None:
                    max_order = min(self.exp_space[0].size, mbe.max_order)
                else:
                    max_order = self.exp_space[0].size
                self.max_order: int = max_order

                self.final_order: int = 0

                # verbose
                self.verbose: int = mbe.verbose
                
                # pi pruning
                self.pi_prune: bool = mbe.pi_prune

                if self.pi_prune:

                    self.orbsym_linear = cast(np.ndarray, mbe.orbsym_linear)

                    pi_orbs, pi_hashes = pi_space('Dooh' if self.point_group == 'D2h' else 'Coov', \
                                                  self.orbsym_linear, self.exp_space[0])
                    self.pi_orbs: np.ndarray = pi_orbs
                    self.pi_hashes: np.ndarray = pi_hashes


def int_wins(hcore_in: Optional[np.ndarray], vhf_in: Optional[np.ndarray], \
             eri_in: Optional[np.ndarray], mpi: MPICls, norb: int, \
             nocc: int) -> Tuple[MPI.Win, MPI.Win, MPI.Win]:
        """
        this function created shared memory windows for integrals on every node
        """
        # allocate hcore in shared mem
        hcore_win = MPI.Win.Allocate_shared(8 * norb ** 2 if mpi.local_master else 0, 8, comm=mpi.local_comm) # type: ignore
        buf = hcore_win.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(norb,) * 2) # type: ignore

        # set hcore on global master
        if mpi.global_master:
            hcore[:] = cast(np.ndarray, hcore_in)

        # mpi_bcast hcore
        if mpi.num_masters > 1 and mpi.local_master:
            hcore[:] = mpi_bcast(mpi.master_comm, hcore)

        # allocate vhf in shared mem
        vhf_win = MPI.Win.Allocate_shared(8 * nocc * norb ** 2 if mpi.local_master else 0, 8, comm=mpi.local_comm) # type: ignore
        buf = vhf_win.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(nocc, norb, norb)) # type: ignore
        
        # set vhf on global master
        if mpi.global_master:
            vhf[:] = cast(np.ndarray, vhf_in)

        # mpi_bcast vhf
        if mpi.num_masters > 1 and mpi.local_master:
            vhf[:] = mpi_bcast(mpi.master_comm, vhf)

        # allocate eri in shared mem
        eri_win = MPI.Win.Allocate_shared(8 * (norb * (norb + 1) // 2) ** 2 if mpi.local_master else 0, 8, comm=mpi.local_comm) # type: ignore
        buf = eri_win.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(norb * (norb + 1) // 2,) * 2) # type: ignore

        # set eri on global master
        if mpi.global_master:
            eri[:] = cast(np.ndarray, eri_in)

        # mpi_bcast eri
        if mpi.num_masters > 1 and mpi.local_master:
            eri[:] = mpi_bcast(mpi.master_comm, eri)

        # mpi barrier
        mpi.global_comm.Barrier()

        return hcore_win, vhf_win, eri_win


def restart_main(mpi: MPICls, exp: ExpCls) -> int:
        """
        this function reads in all expansion restart files and returns the start order
        """
        # list sorted filenames in files list
        if mpi.global_master:
            files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]
            files.sort(key = natural_keys)

        # distribute filenames
        if mpi.global_master:
            mpi.global_comm.bcast(files, root=0)
        else:
            files = mpi.global_comm.bcast(None, root=0)

        # loop over n_tuples files
        if mpi.global_master:
            for i in range(len(files)):
                if 'mbe_n_tuples' in files[i]:
                    if 'theo' in files[i]:
                        exp.n_tuples['theo'].append(np.load(os.path.join(RST, files[i])).tolist())
                    if 'inc' in files[i]:
                        exp.n_tuples['inc'].append(np.load(os.path.join(RST, files[i])).tolist())
                    if 'calc' in files[i]:
                        exp.n_tuples['calc'].append(np.load(os.path.join(RST, files[i])).tolist())
            mpi.global_comm.bcast(exp.n_tuples, root=0)
        else:
            exp.n_tuples = mpi.global_comm.bcast(None, root=0)

        # loop over all other files
        for i in range(len(files)):

            # read hashes
            if 'mbe_hashes' in files[i]:
                n_tuples = exp.n_tuples['inc'][len(exp.hashes)]
                exp.hashes.append(MPI.Win.Allocate_shared(8 * n_tuples if mpi.local_master else 0, 8, comm=mpi.local_comm)) # type: ignore
                buf = exp.hashes[-1].Shared_query(0)[0]
                hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(n_tuples,)) # type: ignore
                if mpi.global_master:
                    hashes[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    hashes[:] = mpi_bcast(mpi.master_comm, hashes)
                mpi.local_comm.Barrier()

            # read increments
            elif 'mbe_inc' in files[i]:
                n_tuples = exp.n_tuples['inc'][len(exp.incs)]
                exp.incs.append(MPI.Win.Allocate_shared(8 * n_tuples * inc_dim(exp.target) if mpi.local_master else 0, 8, comm=mpi.local_comm)) # type: ignore
                buf = exp.incs[-1].Shared_query(0)[0]
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=inc_shape(n_tuples, inc_dim(exp.target))) # type: ignore
                if mpi.global_master:
                    inc[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    inc[:] = mpi_bcast(mpi.master_comm, inc)
                mpi.local_comm.Barrier()

            if mpi.global_master:

                # read expansion spaces
                if 'exp_space' in files[i]:
                    exp.exp_space.append(np.load(os.path.join(RST, files[i])))

                # read total properties
                elif 'mbe_screen' in files[i]:
                    exp.screen = np.load(os.path.join(RST, files[i]))

                # read total properties
                elif 'mbe_tot_prop' in files[i]:
                    exp.mbe_tot_prop.append(np.load(os.path.join(RST, files[i])))

                # read inc statistics
                elif 'mbe_stats_inc' in files[i]:
                    inc_stats = np.load(os.path.join(RST, files[i]))
                    exp.min_inc.append(inc_stats[0])
                    exp.mean_inc.append(inc_stats[1])
                    exp.max_inc.append(inc_stats[2])

                # read timings
                elif 'mbe_time_mbe' in files[i]:
                    exp.time['mbe'].append(np.load(os.path.join(RST, files[i])).tolist())
                elif 'mbe_time_purge' in files[i]:
                    exp.time['purge'].append(np.load(os.path.join(RST, files[i])).tolist())

        # bcast exp_space and screen
        if mpi.global_master:
            mpi.global_comm.bcast(exp.exp_space, root=0)
            mpi.global_comm.bcast(exp.screen, root=0)
        else:
            exp.exp_space = mpi.global_comm.bcast(None, root=0)
            exp.screen = mpi.global_comm.bcast(None, root=0)

        # mpi barrier
        mpi.global_comm.Barrier()

        return exp.min_order + len(exp.mbe_tot_prop)
