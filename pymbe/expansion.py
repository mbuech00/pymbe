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
import copy

import tools


class ExpCls(object):
        """
        this class contains the pymbe expansion attributes
        """
        def __init__(self, mol, calc):
                """
                init expansion attributes

                :param mol: pymbe mol object
                :param calc: pymbe calc object
                """
                # set expansion model dict
                self.model = copy.deepcopy(calc.model)

                # init prop dict
                self.prop = {calc.target: {'inc': [], 'tot': []}}

                # set max_order
                if calc.misc['order'] is not None:
                    self.max_order = min(calc.exp_space['tot'].size, calc.misc['order'])
                else:
                    self.max_order = calc.exp_space['tot'].size

                # init timings and and statistics lists
                self.time = {'mbe': [], 'screen': []}
                self.mean_inc = []
                self.min_inc = []
                self.max_inc = []
                self.mean_ndets = []
                self.min_ndets = []
                self.max_ndets = []

                # init order
                self.order = 0


def init_tup(mpi, mol, calc):
        """
        this function initializes tuples and hashes

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: list with MPI window handle to hashes: numpy array of shape (n_tuples,),
                 list with integer [n_tasks],
                 numpy array of shape (n_tuples, min_order) [tuples]
        """
        # init tuples
        if calc.ref_space.size > 0:
            if np.all(calc.occup[calc.ref_space] == 0.0):
                tuples = np.array([[i] for i in calc.exp_space['occ']], dtype=np.int32)
            elif np.all(calc.occup[calc.ref_space] > 0.0):
                tuples = np.array([[a] for a in calc.exp_space['virt']], dtype=np.int32)
            else:
                tuples = np.array([[p] for p in calc.exp_space['tot']], dtype=np.int32)
        else:
            tuples = np.array([[i, a] for i in calc.exp_space['occ'] for a in calc.exp_space['virt']], dtype=np.int32)

        # pi-orbital pruning
        if calc.extra['pi_prune']:
            tuples = np.array([tup for tup in tuples if tools.pi_prune(calc.exp_space['pi_orbs'], \
                                calc.exp_space['pi_hashes'], tup)], dtype=np.int32)

        # init hashes
        if mpi.master:

            hashes_win = MPI.Win.Allocate_shared(8 * tuples.shape[0], 8, comm=mpi.comm)
            buf = hashes_win.Shared_query(0)[0]
            hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(tuples.shape[0],))

            # compute hashes
            hashes[:] = tools.hash_2d(tuples)

            # sort wrt hashes
            tuples = tuples[hashes.argsort()]
            hashes.sort()

        else:

            hashes_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.comm)
            buf = hashes_win.Shared_query(0)[0]
            hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(tuples.shape[0],))

        # mpi barrier
        mpi.comm.Barrier()

        return [hashes_win], [tuples.shape[0]], tuples


