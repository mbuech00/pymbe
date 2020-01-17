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
from typing import List, Union

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

        # do not screen at order k = 1
        if exp.order == 1:
            return np.array([], dtype=np.int64)

        # load increments for current order
        buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0] # type: ignore
        if calc.target_mbe in ['energy', 'excitation']:
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[-1],))
        elif calc.target_mbe in ['dipole', 'trans']:
            inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[-1], 3))

        # mpi barrier
        mpi.local_comm.barrier()

        # occupied and virtual expansion spaces
        exp_occ = tuple(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc])
        exp_virt = tuple(exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]])

        # allow for tuples with only virtual or occupied MOs
        ref_occ = tools.occ_prune(calc.occup, calc.ref_space)
        ref_virt = tools.virt_prune(calc.occup, calc.ref_space)

        # init screen array
        screen = np.ones(exp.exp_space[-1].size, dtype=bool)

        # init distributions
        if exp.exp_space[-1].size <= mpi.global_size:
            dist = np.array_split(np.arange(mpi.global_size), exp.exp_space[-1].size)
        else:
            dist = [np.array([mo % mpi.global_size]) for mo in range(exp.exp_space[-1].size)]

        # loop over orbitals
        for mo_idx, mo in enumerate(exp.exp_space[-1]):

            # distribute orbitals
            if mpi.global_rank not in dist[mo_idx]:
                continue

            # generate all possible tuples that include mo
            for task_idx, tup_idx in enumerate(tools.include_idx(exp_occ, exp_virt, ref_occ, ref_virt, exp.order, mo)):

                # distribute tuples
                if dist[mo_idx][task_idx % dist[mo_idx].size] != mpi.global_rank:
                    continue

                # screening procedure
                if inc.ndim == 1:
                    screen[mo_idx] &= np.abs(inc[tup_idx]) < calc.thres['inc']
                else:
                    screen[mo_idx] &= np.all(np.abs(inc[tup_idx, :]) < calc.thres['inc'])

        # allreduce screened orbitals
        tot_screen = parallel.allreduce(mpi.global_comm, screen, op=MPI.LAND)

        return np.array([mo for mo in exp.exp_space[-1][tot_screen]], dtype=np.int64)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

