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
from typing import List, Dict, Tuple, Any

import parallel
import calculation
import tools


class ExpCls:
        """
        this class contains the pymbe expansion attributes
        """
        def __init__(self, calc: calculation.CalcCls) -> None:
                """
                init expansion attributes
                """
                # set expansion model dict
                self.model = copy.deepcopy(calc.model)

                # init prop dict
                self.prop: Dict[str, Dict[str, List[np.ndarray]]] = {str(calc.target_mbe): {'inc': [], 'tot': []}}

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
                self.min_ndets: List[int] = []
                self.max_ndets: List[int] = []

                # init order
                self.order: int = 0

                # init attributes
                self.hashes: List[MPI.Win] = [None]
                self.tuples: MPI.Win = None
                self.n_tasks: List[int] = [0]
                self.min_order: int = 0
                self.start_order: int = 0
                self.final_order: int = 0


def init_tup(occup: np.ndarray, ref_space: np.ndarray, \
                exp_space_occ: np.ndarray, exp_space_virt: np.ndarray, \
                exp_space_tot: np.ndarray, local_master: bool, \
                local_comm: MPI.Comm, pi_prune: bool, pi_orbs: np.ndarray, \
                pi_hashes: np.ndarray) -> Tuple[List[MPI.Win], MPI.Win, List[int], int]:
        """
        this function initializes tuples and hashes

        example:
        >>> occup = np.array([2.] * 4 + [0.] * 6)
        >>> ref_space = np.arange(2)
        >>> exp_space_occ = np.arange(2, 4)
        >>> exp_space_virt = np.arange(4, 10)
        >>> exp_space_tot = np.arange(2, 10)
        >>> init_tup(occup, ref_space, exp_space_occ, exp_space_virt, exp_space_tot,
        ...          MPI.COMM_WORLD.Get_rank() == 0, MPI.COMM_WORLD, False, None, None) # doctest: +ELLIPSIS
        ([<mpi4py.MPI.Win object at 0x...>], <mpi4py.MPI.Win object at 0x...>, [6], 1)
        >>> ref_space = np.array([])
        >>> exp_space_occ = np.arange(4)
        >>> exp_space_tot = np.arange(10)
        >>> init_tup(occup, ref_space, exp_space_occ, exp_space_virt, exp_space_tot,
        ...          MPI.COMM_WORLD.Get_rank() == 0, MPI.COMM_WORLD, False, None, None) # doctest: +ELLIPSIS
        ([<mpi4py.MPI.Win object at 0x...>], <mpi4py.MPI.Win object at 0x...>, [24], 2)
        >>> ref_space = np.arange(3)
        >>> exp_space_occ = np.arange(3, 4)
        >>> exp_space_tot = np.arange(3, 10)
        >>> pi_orbs = np.array([1, 2, 4, 5], dtype=np.int16)
        >>> pi_hashes = np.array([-2163557957507198923, 1937934232745943291])
        >>> init_tup(occup, ref_space, exp_space_occ, exp_space_virt, exp_space_tot,
        ...          MPI.COMM_WORLD.Get_rank() == 0, MPI.COMM_WORLD,
        ...          True, pi_orbs, pi_hashes) # doctest: +ELLIPSIS
        ([<mpi4py.MPI.Win object at 0x...>], <mpi4py.MPI.Win object at 0x...>, [4], 1)
        """
        # init tuples
        if ref_space.size > 0:

            if np.all(occup[ref_space] == 0.0):
                tuples_tmp = np.array([[i] for i in exp_space_occ], dtype=np.int16)
            elif np.all(occup[ref_space] > 0.0):
                tuples_tmp = np.array([[a] for a in exp_space_virt], dtype=np.int16)
            else:
                tuples_tmp = np.array([[p] for p in exp_space_tot], dtype=np.int16)

        else:

            tuples_tmp = np.array([[i, a] for i in exp_space_occ for a in exp_space_virt], dtype=np.int16)

        # pi-orbital pruning
        if pi_prune:
            tuples_tmp = np.array([tup for tup in tuples_tmp if tools.pi_prune(pi_orbs, \
                                pi_hashes, tup)], dtype=np.int16)

        # min_order
        min_order = tuples_tmp.shape[1]

        # init tuples and hashes
        if local_master:

            # allocate tuples
            tuples_win = MPI.Win.Allocate_shared(2 * tuples_tmp.size, 2, comm=local_comm)
            buf = tuples_win.Shared_query(0)[0]
            tuples = np.ndarray(buffer=buf, dtype=np.int16, shape=tuples_tmp.shape)

            # place tuples in shared memory space
            tuples[:] = tuples_tmp

            # make tuples read-only
            tuples.flags.writeable = False

            # allocate hashes
            hashes_win = MPI.Win.Allocate_shared(8 * tuples.shape[0], 8, comm=local_comm)
            buf = hashes_win.Shared_query(0)[0]
            hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(tuples.shape[0],))

            # compute hashes
            hashes[:] = tools.hash_2d(tuples)

            # sort hashes
            hashes.sort()

            # mpi barrier
            local_comm.Barrier()

        else:

            # get handle to tuples window
            tuples_win = MPI.Win.Allocate_shared(0, 2, comm=local_comm)

            # get handle to hashes window
            hashes_win = MPI.Win.Allocate_shared(0, 8, comm=local_comm)

            # mpi barrier
            local_comm.Barrier()

        return [hashes_win], tuples_win, [tuples_tmp.shape[0]], min_order


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)


