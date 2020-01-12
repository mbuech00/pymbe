#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
screening module containing all input generation in pymbe
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import itertools
from typing import List

import parallel
import system
import calculation
import expansion
import tools


def main(mpi: parallel.MPICls, mol: system.MolCls, calc: calculation.CalcCls, exp: expansion.ExpCls) -> np.ndarray:
        """
        this function returns the orbitals to be screened away
        """
        # wake up slaves
        if mpi.global_master:
            msg = {'task': 'screen', 'order': exp.order}
            mpi.global_comm.bcast(msg, root=0)

        # load increments for current order
        buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0] # type: ignore
        if calc.target_mbe in ['energy', 'excitation']:
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[-1],))
        elif calc.target_mbe in ['dipole', 'trans']:
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[-1], 3))

        # load hashes for current order
        buf = exp.hashes[-1].Shared_query(0)[0]
        hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples[-1],))

        # mpi barrier
        mpi.local_comm.barrier()

        # occupied and virtual expansion spaces
        occ_space = calc.exp_space[calc.exp_space < mol.nocc]
        virt_space = calc.exp_space[mol.nocc <= calc.exp_space]

        # allow for tuples with only occupied or virtual MOs
        occ_only = tools.virt_prune(calc.occup, calc.ref_space)
        virt_only = tools.occ_prune(calc.occup, calc.ref_space)

        # init list of screened orbitals
        screen_orbs: List[int] = []

        # loop over orbitals
        for mo_idx, mo in enumerate(calc.exp_space):

            # distribute orbitals
            if mo_idx % mpi.global_size != mpi.global_rank:
                continue

            # init screen
            screen = True

            # generate tuples
            for tup in tools.tuples(occ_space, virt_space, occ_only, virt_only, exp.order, restrict=mo):

                # convert to hash
                tup_hash: np.ndarray = tools.hash_1d(np.asarray(tup, dtype=np.int64))

                # get index of tuple
                inc_idx: np.ndarray = tools.hash_compare(hashes, tup_hash)

                # screening procedure
                if inc.ndim == 1:
                    screen &= np.all(np.abs(inc[inc_idx]) < calc.thres['inc'])
                else:
                    screen &= np.all(np.abs(inc[inc_idx, :]) < calc.thres['inc'], axis=1)

                # no screening
                if not screen:
                    break

            # add orbital to list of screened orbitals
            if screen:
                screen_orbs.append(mo)

        # allgather number of screened orbitals
        recv_counts = np.array(mpi.global_comm.allgather(len(screen_orbs)))

        # allocate total array of screened orbitals
        tot_screen_orbs = np.empty(np.sum(recv_counts), dtype=np.int64)

        # gatherv all screened orbitals onto global master
        if mpi.global_master:
            tot_screen_orbs = parallel.gatherv(mpi.global_comm, np.asarray(screen_orbs, dtype=np.int64), recv_counts)
        else:
            screen_orbs = parallel.gatherv(mpi.global_comm, np.asarray(screen_orbs, dtype=np.int64), recv_counts)

        # bcast total array of screened orbitals
        tot_screen_orbs = parallel.bcast(mpi.global_comm, tot_screen_orbs)

        return tot_screen_orbs


if __name__ == "__main__":
    import doctest
    doctest.testmod()

