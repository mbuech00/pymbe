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
from typing import TYPE_CHECKING

from pymbe.tools import RST, pi_space, natural_keys, cast_away_optional, \
                        assume_int
from pymbe.parallel import mpi_bcast

if TYPE_CHECKING:

    from typing import List, Dict, Union, Tuple, Optional

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
                self.nuc_energy: float = cast_away_optional(mbe.nuc_energy)
                if self.target == 'dipole':
                    self.nuc_dipole: np.ndarray = cast_away_optional(mbe.nuc_dipole)
                self.ncore: int = mbe.ncore
                self.nocc: int = cast_away_optional(mbe.nocc)
                self.norb: int = cast_away_optional(mbe.norb)
                self.spin: int = cast_away_optional(mbe.spin)
                self.point_group: str = cast_away_optional(mbe.point_group)
                self.orbsym: np.ndarray = cast_away_optional(mbe.orbsym)
                self.fci_state_sym: int = assume_int(mbe.fci_state_sym)
                self.fci_state_root: int = cast_away_optional(mbe.fci_state_root)
                
                # hf calculation
                self.hf_prop: Union[float, np.ndarray] = cast_away_optional(mbe.hf_prop)
                self.occup: np.ndarray = cast_away_optional(mbe.occup)

                # integrals
                hcore, vhf, eri = int_wins(cast_away_optional(mbe.hcore), \
                                           cast_away_optional(mbe.vhf), \
                                           cast_away_optional(mbe.eri), \
                                           mbe.mpi, self.norb, self.nocc)
                self.hcore: MPI.Win = hcore
                self.vhf: MPI.Win = vhf
                self.eri: MPI.Win = eri
                self.dipole_ints: Optional[np.ndarray] = mbe.dipole_ints
                
                # orbital representation
                self.orb_type: str = mbe.orb_type

                # reference space
                self.ref_space: np.ndarray = mbe.ref_space
                self.ref_prop: Union[float, np.ndarray] = cast_away_optional(mbe.ref_prop)

                # expansion space
                self.exp_space: List[np.ndarray] = [np.array([i for i in range(self.ncore, self.norb) if i not in self.ref_space], dtype=np.int64)]
                
                # base model
                self.base_method: Optional[str] = mbe.base_method
                self.base_prop: Union[float, np.ndarray] = cast_away_optional(mbe.base_prop)

                # property list dict
                self.prop: Dict[str, Dict[str, Union[List[float], \
                                MPI.Win]]] = {str(self.target): {'inc': [], \
                                                                 'tot': [], \
                                                                 'hashes': []}}

                # timings
                self.time: Dict[str, Union[List[float], \
                                np.ndarray]] = {'mbe': [], 'purge': []}

                # statistics
                self.mean_inc: Union[List[float], np.ndarray] = []
                self.min_inc: Union[List[float], np.ndarray] = []
                self.max_inc: Union[List[float], np.ndarray] = []
                self.mean_ndets: Union[List[int], np.ndarray] = []
                self.min_ndets: Union[List[int], np.ndarray] = []
                self.max_ndets: Union[List[int], np.ndarray] = []

                # number of tuples
                self.n_tuples: Dict[str, List[int]] = {'theo': [], 'calc': [], \
                                                       'inc': []}  

                # screening
                self.screen_start: int = mbe.screen_start
                self.screen_perc: float = mbe.screen_perc
                self.screen: np.ndarray = None
                self.screen_orbs: np.ndarray = None

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
                    max_order = min(self.exp_space[0].size, cast_away_optional(mbe.max_order))
                else:
                    max_order = self.exp_space[0].size
                self.max_order: int = max_order

                self.final_order: int = 0

                # debug
                self.debug: int = mbe.debug
                
                # pi pruning
                self.pi_prune: bool = mbe.pi_prune

                if self.pi_prune:

                    self.orbsym_linear: np.ndarray = cast_away_optional(mbe.orbsym_linear)

                    pi_orbs, pi_hashes = pi_space('Dooh' if self.point_group == 'D2h' else 'Coov', \
                                                  self.orbsym_linear, self.exp_space[0])
                    self.pi_orbs: np.ndarray = pi_orbs
                    self.pi_hashes: np.ndarray = pi_hashes


def int_wins(hcore_in: np.ndarray, vhf_in: np.ndarray, eri_in: np.ndarray, \
             mpi: MPICls, norb: int, \
             nocc: int) -> Tuple[MPI.Win, MPI.Win, MPI.Win]:
        """
        this function created shared memory windows for integrals on every node
        """
        # allocate hcore in shared mem
        if mpi.local_master:
            hcore_win = MPI.Win.Allocate_shared(8 * norb**2, 8, comm=mpi.local_comm)
        else:
            hcore_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
        buf = hcore_win.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(norb,) * 2)

        # set hcore on global master
        if mpi.global_master:
            hcore[:] = hcore_in

        # mpi_bcast hcore
        if mpi.num_masters > 1 and mpi.local_master:
            hcore[:] = mpi_bcast(mpi.master_comm, hcore)

        # allocate vhf in shared mem
        if mpi.local_master:
            vhf_win = MPI.Win.Allocate_shared(8 * nocc*norb**2, 8, comm=mpi.local_comm)
        else:
            vhf_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
        buf = vhf_win.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(nocc, norb, norb))
        
        # set vhf on global master
        if mpi.global_master:
            vhf[:] = vhf_in

        # mpi_bcast vhf
        if mpi.num_masters > 1 and mpi.local_master:
            vhf[:] = mpi_bcast(mpi.master_comm, vhf)

        # allocate eri in shared mem
        if mpi.local_master:
            eri_win = MPI.Win.Allocate_shared(8 * (norb * (norb + 1) // 2) ** 2, 8, comm=mpi.local_comm)
        else:
            eri_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
        buf = eri_win.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(norb * (norb + 1) // 2,) * 2)

        # set eri on global master
        if mpi.global_master:
            eri[:] = eri_in

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
                n_tuples = exp.n_tuples['inc'][len(exp.prop[exp.target]['hashes'])]
                exp.prop[exp.target]['hashes'].append(MPI.Win.Allocate_shared(8 * n_tuples if mpi.local_master else 0, \
                                                                              8, comm=mpi.local_comm))
                buf = exp.prop[exp.target]['hashes'][-1].Shared_query(0)[0] # type: ignore
                hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(n_tuples,))
                if mpi.global_master:
                    hashes[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    hashes[:] = mpi_bcast(mpi.master_comm, hashes)
                mpi.local_comm.Barrier()

            # read increments
            elif 'mbe_inc' in files[i]:
                n_tuples = exp.n_tuples['inc'][len(exp.prop[exp.target]['inc'])]
                if mpi.local_master:
                    if exp.target in ['energy', 'excitation']:
                        exp.prop[exp.target]['inc'].append(MPI.Win.Allocate_shared(8 * n_tuples, 8, comm=mpi.local_comm))
                    elif exp.target in ['dipole', 'trans']:
                        exp.prop[exp.target]['inc'].append(MPI.Win.Allocate_shared(8 * n_tuples * 3, 8, comm=mpi.local_comm))
                else:
                    exp.prop[exp.target]['inc'].append(MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm))
                buf = exp.prop[exp.target]['inc'][-1].Shared_query(0)[0] # type: ignore
                if exp.target in ['energy', 'excitation']:
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tuples,))
                elif exp.target in ['dipole', 'trans']:
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tuples, 3))
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
                elif 'mbe_tot' in files[i]:
                    exp.prop[exp.target]['tot'].append(np.load(os.path.join(RST, files[i])))

                # read ndets statistics
                elif 'mbe_mean_ndets' in files[i]:
                    exp.mean_ndets.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_min_ndets' in files[i]:
                    exp.min_ndets.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_max_ndets' in files[i]:
                    exp.max_ndets.append(np.load(os.path.join(RST, files[i])))

                # read inc statistics
                elif 'mbe_mean_inc' in files[i]:
                    exp.mean_inc.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_min_inc' in files[i]:
                    exp.min_inc.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_max_inc' in files[i]:
                    exp.max_inc.append(np.load(os.path.join(RST, files[i])))

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

        return exp.min_order + len(exp.prop[exp.target]['tot'])
