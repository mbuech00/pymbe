#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
purging module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import itertools
from typing import Tuple, List, Dict, Union, Any

import parallel
import system
import calculation
import expansion
import tools


def main(mpi: parallel.MPICls, mol: system.MolCls, calc: calculation.CalcCls, \
            exp: expansion.ExpCls) -> Tuple[Dict[str, Union[List[float], MPI.Win]], Dict[str, List[int]]]:
        """
        this function purges the lower-order hashes & increments
        """
        # wake up slaves
        if mpi.global_master:
            msg = {'task': 'purge', 'order': exp.order}
            mpi.global_comm.bcast(msg, root=0)

        # do not purge at min_order or in case of no screened orbs
        if exp.order == exp.min_order or exp.screen_orbs.size == 0 or exp.exp_space[-1].size < exp.order + 1:
            exp.time['purge'].append(0.)
            return exp.prop[calc.target_mbe], exp.n_tuples

        # increment dimensions
        dim = tools.inc_dim(calc.target_mbe)

        # init time
        if mpi.global_master:
            time = MPI.Wtime()

        # occupied and virtual expansion spaces
        exp_occ = exp.exp_space[-1][exp.exp_space[-1] < mol.nocc]
        exp_virt = exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]]

        # allow for tuples with only virtual or occupied MOs
        ref_occ = tools.occ_prune(calc.occup, calc.ref_space)
        ref_virt = tools.virt_prune(calc.occup, calc.ref_space)

        # loop over previous orders
        for k in range(exp.min_order, exp.order):

            # load k-th order hashes and increments
            buf = exp.prop[calc.target_mbe]['hashes'][k-exp.min_order].Shared_query(0)[0] # type: ignore
            hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['inc'][k-exp.min_order],))

            buf = exp.prop[calc.target_mbe]['inc'][k-exp.min_order].Shared_query(0)[0] # type: ignore
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape=tools.inc_shape(exp.n_tuples['inc'][k-exp.min_order], dim))

            # mpi barrier
            mpi.local_comm.barrier()

            # init list for storing hashes at order k
            hashes_tmp: Any = []
            # init list for storing increments at order k
            inc_tmp: Any = []

            # loop until no tuples left
            for tup_idx, tup in enumerate(tools.tuples(exp_occ, exp_virt, ref_occ, ref_virt, k)):

                # distribute tuples
                if tup_idx % mpi.global_size != mpi.global_rank:
                    continue

                # compute index
                idx = tools.hash_lookup(hashes, tools.hash_1d(tup))

                # add inc_tup and its hash to lists of increments/hashes
                if idx is not None:
                    inc_tmp.append(inc[idx])
                    hashes_tmp.append(tools.hash_1d(tup))

            # deallocate k-th order hashes
            exp.prop[calc.target_mbe]['hashes'][k-exp.min_order].Free() # type: ignore
            # deallocate k-th order increments
            exp.prop[calc.target_mbe]['inc'][k-exp.min_order].Free() # type: ignore

            # recast hashes_tmp as np.array
            hashes_tmp = np.asarray(hashes_tmp, dtype=np.int64)

            # number of hashes
            recv_counts = np.array(mpi.global_comm.allgather(hashes_tmp.size))

            # update n_tuples
            exp.n_tuples['inc'][k-exp.min_order] = int(np.sum(recv_counts))

            # init hashes for present order
            hashes_win = MPI.Win.Allocate_shared(8 * np.sum(recv_counts) if mpi.local_master else 0, \
                                                 8, comm=mpi.local_comm)
            exp.prop[calc.target_mbe]['hashes'][k-exp.min_order] = hashes_win
            buf = hashes_win.Shared_query(0)[0] # type: ignore
            hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['inc'][k-exp.min_order],))

            # gatherv hashes on global master
            hashes[:] = parallel.gatherv(mpi.global_comm, hashes_tmp, hashes, recv_counts)

            # bcast hashes among local masters
            if mpi.local_master:
                hashes[:] = parallel.bcast(mpi.master_comm, hashes)

            # recast inc_tmp as np.array
            inc_tmp = np.asarray(inc_tmp, dtype=np.float64).reshape(-1,)

            # number of increments
            recv_counts = np.array(mpi.global_comm.allgather(inc_tmp.size))

            # init increments for present order
            inc_win = MPI.Win.Allocate_shared(8 * np.sum(recv_counts) if mpi.local_master else 0, \
                                              8, comm=mpi.local_comm)
            exp.prop[calc.target_mbe]['inc'][k-exp.min_order] = inc_win
            buf = inc_win.Shared_query(0)[0] # type: ignore
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape=tools.inc_shape(exp.n_tuples['inc'][k-exp.min_order], dim))

            # gatherv increments on global master
            inc[:] = parallel.gatherv(mpi.global_comm, inc_tmp, inc, recv_counts)

            # bcast increments among local masters
            if mpi.local_master:
                inc[:] = parallel.bcast(mpi.master_comm, inc)

            # sort hashes and increments
            if mpi.local_master:
                inc[:] = inc[np.argsort(hashes)]
                hashes[:].sort()

        # mpi barrier
        mpi.global_comm.barrier()

        # save timing
        if mpi.global_master:
            exp.time['purge'].append(MPI.Wtime() - time)

        return exp.prop[calc.target_mbe], exp.n_tuples


if __name__ == "__main__":
    import doctest
    doctest.testmod()

