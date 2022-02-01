#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
purging module
"""

from __future__ import annotations

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from typing import TYPE_CHECKING

from pymbe.parallel import mpi_gatherv, mpi_bcast
from pymbe.tools import inc_dim, inc_shape, min_orbs, tuples, hash_1d, \
                        hash_lookup

if TYPE_CHECKING:

    from typing import Tuple, List, Dict, Any

    from pymbe.parallel import MPICls
    from pymbe.expansion import ExpCls


def main(mpi: MPICls, exp: ExpCls) -> Tuple[List[MPI.Win], List[MPI.Win], \
                                            Dict[str, List[int]]]:
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
            return exp.incs, exp.hashes, exp.n_tuples

        # increment dimensions
        dim = inc_dim(exp.target)

        # init time
        if mpi.global_master:
            time = MPI.Wtime()

        # occupied and virtual expansion spaces
        exp_occ = exp.exp_space[-1][exp.exp_space[-1] < exp.nocc]
        exp_virt = exp.exp_space[-1][exp.nocc <= exp.exp_space[-1]]

        # get minimum number of virtual or occupied MOs in tuple
        min_occ, min_virt = min_orbs(exp.occup, exp.ref_space, exp.vanish_exc)

        # loop over previous orders
        for k in range(exp.min_order, exp.order+1):

            # load k-th order hashes and increments
            buf = exp.hashes[k-exp.min_order].Shared_query(0)[0]
            hashes = np.ndarray(buffer=buf, dtype=np.int64, shape = (exp.n_tuples['inc'][k-exp.min_order],)) # type: ignore

            buf = exp.incs[k-exp.min_order].Shared_query(0)[0]
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape = inc_shape(exp.n_tuples['inc'][k-exp.min_order], dim)) # type: ignore

            # mpi barrier
            mpi.local_comm.barrier()

            # init list for storing hashes at order k
            hashes_tmp: Any = []
            # init list for storing increments at order k
            inc_tmp: Any = []

            # loop until no tuples left
            for tup_idx, tup in enumerate(tuples(exp_occ, exp_virt, min_occ, min_virt, k)):

                # distribute tuples
                if tup_idx % mpi.global_size != mpi.global_rank:
                    continue

                # compute index
                idx = hash_lookup(hashes, hash_1d(tup))

                # add inc_tup and its hash to lists of increments/hashes
                if idx is not None:
                    inc_tmp.append(inc[idx])
                    hashes_tmp.append(hash_1d(tup))

            # recast hashes_tmp and inc_tmp as np.array
            hashes_tmp = np.asarray(hashes_tmp, dtype=np.int64)
            inc_tmp = np.asarray(inc_tmp, dtype=np.float64).reshape(-1,)

            # deallocate k-th order hashes and increments
            exp.hashes[k-exp.min_order].Free()
            exp.incs[k-exp.min_order].Free()

            # number of hashes
            recv_counts = np.array(mpi.global_comm.allgather(hashes_tmp.size))

            # update n_tuples
            exp.n_tuples['inc'][k-exp.min_order] = int(np.sum(recv_counts))

            # init hashes for present order
            hashes_win = MPI.Win.Allocate_shared(8 * np.sum(recv_counts) if mpi.local_master else 0, 8, comm=mpi.local_comm) # type: ignore
            exp.hashes[k-exp.min_order] = hashes_win
            buf = hashes_win.Shared_query(0)[0]
            hashes = np.ndarray(buffer=buf, dtype=np.int64, shape = (exp.n_tuples['inc'][k-exp.min_order],)) # type: ignore

            # gatherv hashes on global master
            hashes[:] = mpi_gatherv(mpi.global_comm, hashes_tmp, hashes, recv_counts)

            # bcast hashes among local masters
            if mpi.local_master:
                hashes[:] = mpi_bcast(mpi.master_comm, hashes)

            # number of increments
            recv_counts = np.array(mpi.global_comm.allgather(inc_tmp.size))

            # init increments for present order
            inc_win = MPI.Win.Allocate_shared(8 * np.sum(recv_counts) if mpi.local_master else 0, 8, comm=mpi.local_comm) # type: ignore
            exp.incs[k-exp.min_order] = inc_win
            buf = inc_win.Shared_query(0)[0]
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape = inc_shape(exp.n_tuples['inc'][k-exp.min_order], dim)) # type: ignore

            # gatherv increments on global master
            inc[:] = mpi_gatherv(mpi.global_comm, inc_tmp, inc, recv_counts)

            # bcast increments among local masters
            if mpi.local_master:
                inc[:] = mpi_bcast(mpi.master_comm, inc)

            # sort hashes and increments
            if mpi.local_master:
                inc[:] = inc[np.argsort(hashes)]
                hashes[:].sort()

        # mpi barrier
        mpi.global_comm.barrier()

        # save timing
        if mpi.global_master:
            exp.time['purge'].append(MPI.Wtime() - time)

        return exp.incs, exp.hashes, exp.n_tuples
