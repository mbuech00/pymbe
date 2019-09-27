#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
expansion module containing all expansion attributes
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import functools
import copy
from typing import List, Dict, Union, Any

import parallel
import system
import calculation
import tools


class ExpCls:
        """
        this class contains the pymbe expansion attributes
        """
        def __init__(self, mol: system.MolCls, calc: calculation.CalcCls) -> None:
                """
                init expansion attributes
                """
                # set expansion model dict
                self.model = copy.deepcopy(calc.model)

                # init prop dict
                self.prop: Dict[str, Dict[str, List[np.ndarray]]] = {str(calc.target): {'inc': [], 'tot': []}}

                # set max_order
                if calc.misc['order'] is not None:
                    self.max_order = min(calc.exp_space['tot'].size, calc.misc['order'])
                else:
                    self.max_order = calc.exp_space['tot'].size

                # init timings and and statistics lists
                self.time: Dict[str, List[float]] = {'mbe': [], 'screen': []}
                self.mean_inc: List[float] = []
                self.min_inc: List[float] = []
                self.max_inc: List[float] = []
                self.mean_ndets: List[float] = []
                self.min_ndets: List[float] = []
                self.max_ndets: List[float] = []

                # init order
                self.order: int = 0

                # init attributes
                self.hashes: List[MPI.Win] = [None]
                self.tuples: MPI.Win = None
                self.n_tasks: List[int] = [0]
                self.min_order: int = 0
                self.start_order: int = 0
                self.final_order: int = 0


def init_tup(mpi: parallel.MPICls, mol: system.MolCls, \
                calc: calculation.CalcCls) -> Union[List[MPI.Win], MPI.Win, List[int], int]:
        """
        this function initializes tuples and hashes
        """
        # init tuples
        if calc.ref_space.size > 0:

            if np.all(calc.occup[calc.ref_space] == 0.0):
                tuples_tmp = np.array([[i] for i in calc.exp_space['occ']], dtype=np.int32)
            elif np.all(calc.occup[calc.ref_space] > 0.0):
                tuples_tmp = np.array([[a] for a in calc.exp_space['virt']], dtype=np.int32)
            else:
                tuples_tmp = np.array([[p] for p in calc.exp_space['tot']], dtype=np.int32)

        else:

            tuples_tmp = np.array([[i, a] for i in calc.exp_space['occ'] for a in calc.exp_space['virt']], dtype=np.int32)

        # pi-orbital pruning
        if calc.extra['pi_prune']:
            tuples_tmp = np.array([tup for tup in tuples_tmp if tools.pi_prune(calc.exp_space['pi_orbs'], \
                                calc.exp_space['pi_hashes'], tup)], dtype=np.int32)

        # min_order
        min_order = tuples_tmp.shape[1]

        # init tuples and hashes
        if mpi.local_master:

            # allocate tuples
            tuples_win = MPI.Win.Allocate_shared(4 * tuples_tmp.size, 4, comm=mpi.local_comm)
            buf = tuples_win.Shared_query(0)[0]
            tuples = np.ndarray(buffer=buf, dtype=np.int32, shape=tuples_tmp.shape)

            # place tuples in shared memory space
            tuples[:] = tuples_tmp

            # allocate hashes
            hashes_win = MPI.Win.Allocate_shared(8 * tuples.shape[0], 8, comm=mpi.local_comm)
            buf = hashes_win.Shared_query(0)[0]
            hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(tuples.shape[0],))

            # compute hashes
            hashes[:] = tools.hash_2d(tuples)

            # sort hashes
            hashes.sort()

            # mpi barrier
            mpi.local_comm.Barrier()

            return [hashes_win], tuples_win, [tuples_tmp.shape[0]], min_order

        else:

            # get handle to tuples window
            tuples_win = MPI.Win.Allocate_shared(0, 4, comm=mpi.local_comm)

            # get handle to hashes window
            hashes_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)

            # mpi barrier
            mpi.local_comm.Barrier()

            return [hashes_win], tuples_win, [tuples_tmp.shape[0]], min_order


