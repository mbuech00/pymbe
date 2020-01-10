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
import functools
import itertools
from typing import Tuple, List, Dict, Union, Any

import parallel
import system
import calculation
import expansion
import tools


def main(mpi: parallel.MPICls, mol: system.MolCls, calc: calculation.CalcCls, exp: expansion.ExpCls) -> np.ndarray:
        """
        this function returns the number of tuples at the following order
        """
        # wake up slaves
        msg = {'task': 'screen', 'order': exp.order}
        mpi.global_comm.bcast(msg, root=0)
#        # send availability to master
#        if mpi.global_rank <= ???:
#            mpi.global_comm.send(None, dest=0, tag=TAGS.ready)
        return np.array([], dtype=np.int)

        # load increments for current order
        buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0] # type: ignore
        if calc.target_mbe in ['energy', 'excitation']:
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[-1],))
        elif calc.target_mbe in ['dipole', 'trans']:
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[-1], 3))

        # load hashes for current order
        buf = exp.hashes[-1].Shared_query(0)[0]
        hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples[-1],))

#        # mpi barrier
#        mpi.local_comm.barrier()

        # occupied and virtual expansion spaces
        occ_space = calc.exp_space[calc.exp_space < mol.nocc]
        virt_space = calc.exp_space[mol.nocc <= calc.exp_space]

        # allow for tuples with only occupied or virtual MOs
        occ_only = tools.virt_prune(calc.occup, calc.ref_space)
        virt_only = tools.occ_prune(calc.occup, calc.ref_space)

        # init list of screened orbitals
        screen_orbs: List[int] = []

        # loop over orbitals
        for i in calc.exp_space:

            # init screen bool
            screen = True

            # generate tuples
            for tup in tools.tuples(occ_space, virt_space, occ_only, virt_only, exp.order, restrict=i):

                # get idx
                idx: np.ndarray = tools.hash_compare(hashes, tools.hash_1d(tup))

                # screening procedure
                if inc.ndim == 1:

                    screen &= np.abs(inc[idx]) < calc.thres

                else:

                    screen &= np.all(np.abs(inc[idx]) < calc.thres)

                # if any increment is large enough, then quit screening
                if not screen:
                    break

            # add orb i to list of screened orbitals
            if screen:
                screen_orbs.append(i)

#        # mpi barrier
#        mpi.global_comm.barrier()

        return np.asarray(screen_orbs, dtype=np.int)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

