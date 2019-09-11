#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
mbe module containing all functions related to MBEs in pymbe
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
import sys
import itertools
import scipy.misc

import restart
import kernel
import output
import expansion
import driver
import parallel
import tools


# tags
TAGS = tools.enum('ready', 'task', 'rst', 'exit')


def master(mpi, mol, calc, exp):
        """
        this master function returns two arrays of (i) number of determinants and (ii) mbe increments

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: MPI window handle to increments [inc_win],
                 float with total energy correction [tot],
                 three floats or numpy array of shapes (n_tuples, 3) depending on target [mean_inc, min_inc, max_inc]
        """
        # wake up slaves
        msg = {'task': 'mbe', 'order': exp.order}
        mpi.comm.bcast(msg, root=0)

        # number of slaves
        n_slaves = mpi.size - 1

        # load tuples
        buf = exp.tuples.Shared_query(0)[0]
        tuples = np.ndarray(buffer=buf, dtype=np.int32, shape=(exp.n_tasks[-1], exp.order))

        # init increments
        if len(exp.prop[calc.target]['inc']) == len(exp.hashes):

            # load restart increments
            inc_win = exp.prop[calc.target]['inc'][-1]
            buf = inc_win.Shared_query(0)[0]
            if calc.target in ['energy', 'excitation']:
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1],))
            elif calc.target in ['dipole', 'trans']:
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1], 3))

        else:

            # new increments
            if calc.target in ['energy', 'excitation']:
                inc_win = MPI.Win.Allocate_shared(8 * exp.n_tasks[-1], 8, comm=mpi.comm)
            elif calc.target in ['dipole', 'trans']:
                inc_win = MPI.Win.Allocate_shared(8 * exp.n_tasks[-1] * 3, 8, comm=mpi.comm)
            buf = inc_win.Shared_query(0)[0]
            if calc.target in ['energy', 'excitation']:
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1],))
            elif calc.target in ['dipole', 'trans']:
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1], 3))

            # save increments
            if calc.misc['rst']:
                restart.write_gen(exp.order, inc, 'mbe_inc')

        # mpi barrier
        mpi.comm.Barrier()

        # start index
        if calc.target in ['energy', 'excitation']:
            task_start = np.count_nonzero(inc)
        elif calc.target in ['dipole', 'trans']:
            task_start = np.count_nonzero(np.count_nonzero(inc, axis=1))

        # loop until no tasks left
        for tup_idx in range(task_start, exp.n_tasks[-1]):

            # get slave
            parallel.probe(mpi, TAGS.ready)

            # send tup_idx to slave
            mpi.comm.send(tup_idx, dest=mpi.stat.source, tag=TAGS.task)

            # write restart file
            if calc.misc['rst'] and tup_idx % calc.misc['rst_interval'] == 0:

                # send rst signal to all slaves
                for slave_idx in range(n_slaves):

                    # get slave
                    mpi.comm.recv(None, source=slave_idx+1, tag=TAGS.ready)

                    # send rst signal to slave
                    mpi.comm.send(None, dest=slave_idx+1, tag=TAGS.rst)

                # mpi barrier
                mpi.comm.Barrier()

                # save increments
                restart.write_gen(exp.order, inc, 'mbe_inc')

                # print status
                print(output.mbe_status(tup_idx / exp.n_tasks[-1]))

        # done with all tasks
        while n_slaves > 0:

            # get slave
            parallel.probe(mpi, TAGS.ready)

            # send exit signal to slave
            mpi.comm.send(None, dest=mpi.stat.source, tag=TAGS.exit)

            # remove slave
            n_slaves -= 1

        # print final status
        print(output.mbe_status(1.0))

        # mpi barrier
        mpi.comm.Barrier()

        # save increments
        restart.write_gen(exp.order, inc, 'mbe_inc')

        # total property
        tot = tools.fsum(inc)

        # statistics
        if calc.target in ['energy', 'excitation']:

            # increments
            if inc.any():
                mean_inc = np.mean(inc[np.nonzero(inc)])
                min_inc = np.min(np.abs(inc[np.nonzero(inc)]))
                max_inc = np.max(np.abs(inc[np.nonzero(inc)]))
            else:
                mean_inc = min_inc = max_inc = 0.0

        elif calc.target in ['dipole', 'trans']:

            # init result arrays
            mean_inc = np.empty(3, dtype=np.float64)
            min_inc = np.empty(3, dtype=np.float64)
            max_inc = np.empty(3, dtype=np.float64)

            # loop over x, y, and z
            for k in range(3):

                # increments
                if inc[:, k].any():
                    mean_inc[k] = np.mean(inc[:, k][np.nonzero(inc[:, k])])
                    min_inc[k] = np.min(np.abs(inc[:, k][np.nonzero(inc[:, k])]))
                    max_inc[k] = np.max(np.abs(inc[:, k][np.nonzero(inc[:, k])]))
                else:
                    mean_inc[k] = min_inc[k] = max_inc[k] = 0.0

        return inc_win, tot, mean_inc, min_inc, max_inc


def slave(mpi, mol, calc, exp):
        """
        this slave function returns an array of mbe increments

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: MPI window handle to increments
        """
        # load eri
        buf = mol.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb*(mol.norb + 1) // 2,) * 2)

        # load hcore
        buf = mol.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # load vhf
        buf = mol.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # load tuples
        buf = exp.tuples.Shared_query(0)[0]
        tuples = np.ndarray(buffer=buf, dtype=np.int32, shape=(exp.n_tasks[-1], exp.order))

        # load increments for previous orders
        inc = []
        for k in range(exp.order-exp.min_order):
            buf = exp.prop[calc.target]['inc'][k].Shared_query(0)[0]
            if calc.target in ['energy', 'excitation']:
                inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[k],)))
            elif calc.target in ['dipole', 'trans']:
                inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[k], 3)))

        # init increments for present order
        if len(exp.prop[calc.target]['inc']) == len(exp.hashes):
            inc_win = exp.prop[calc.target]['inc'][-1]
            buf = inc_win.Shared_query(0)[0]
        else:
            inc_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.comm)
            buf = inc_win.Shared_query(0)[0]
        if calc.target in ['energy', 'excitation']:
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1],)))
        elif calc.target in ['dipole', 'trans']:
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1], 3)))

        # load hashes for current and previous orders
        hashes = []
        for k in range(exp.order-exp.min_order+1):
            buf = exp.hashes[k].Shared_query(0)[0]
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tasks[k],)))

        # mpi barrier
        mpi.comm.Barrier()

        # send availability to master
        mpi.comm.send(None, dest=0, tag=TAGS.ready)

        # receive work from master
        while True:

            # probe for available task
            mpi.comm.Probe(source=0, tag=MPI.ANY_TAG, status=mpi.stat)

            # do task
            if mpi.stat.tag == TAGS.task:

                # receive tup_idx
                tup_idx = mpi.comm.recv(source=0, tag=TAGS.task)

                # recover tup
                tup = tuples[tup_idx]

                # get core and cas indices
                core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, tup)

                # get h2e indices
                cas_idx_tril = tools.cas_idx_tril(cas_idx)

                # get h2e_cas
                h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

                # compute e_core and h1e_cas
                e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

                # get inc_idx
                inc_idx = tools.hash_compare(hashes[-1], tools.hash_1d(tup))

                # calculate increment
                inc[-1][inc_idx] = _inc(mol, calc, exp.min_order, exp.order, e_core, h1e_cas, h2e_cas, \
                                        inc, hashes, tup, core_idx, cas_idx)

                # send availability to master
                mpi.comm.send(None, dest=0, tag=TAGS.ready)

            elif mpi.stat.tag == TAGS.rst:

                # receive rst signal
                mpi.comm.recv(None, source=0, tag=TAGS.rst)

                # mpi barrier
                mpi.comm.Barrier()

                # send availability to master
                mpi.comm.send(None, dest=0, tag=TAGS.ready)

            elif mpi.stat.tag == TAGS.exit:

                # receive exit signal
                mpi.comm.recv(None, source=0, tag=TAGS.exit)

                break

        # mpi barrier
        mpi.comm.Barrier()

        return inc_win


def _inc(mol, calc, min_order, order, e_core, h1e_cas, h2e_cas, inc, hashes, tup, core_idx, cas_idx):
        """
        this function calculates the increment associated with a given tuple

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param min_order: minimum (start) order. integer
        :param order: current order. integer
        :param e_core: core energy. float
        :param h1e_cas: cas space 1-e Hamiltonian. numpy array of shape (n_cas, n_cas)
        :param h2e_cas: cas space 2-e Hamiltonian. numpy array of shape (n_cas*(n_cas + 1) // 2, n_cas*(n_cas + 1) // 2)
        :param inc: property increments to all order. list of numpy arrays of shapes (n_tuples,) or (n_tuples, 3) depending on target
        :param hashes: hashes to all order. list of numpy arrays of shapes (n_tuples,)
        :param tup: given tuple of orbitals. numpy array of shape (order,)
        :param core_idx: core space indices. numpy array of shape (n_core,)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :return: float or numpy array of shape (3,) depending on target
        """
        # nelec
        nelec = tools.nelec(calc.occup, cas_idx)

        # perform main calc
        inc_tup = kernel.main(mol, calc, calc.model['method'], e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec)

        # perform base calc
        if calc.base['method'] is not None:
            inc_tup -= kernel.main(mol, calc, calc.base['method'], e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec)

        # subtract reference space property
        inc_tup -= calc.prop['ref'][calc.target]

        # calculate increment
        if order > min_order:
            if np.any(inc_tup != 0.0):
                inc_tup -= _sum(calc.occup, calc.ref_space, calc.exp_space, \
                                calc.target, min_order, order, \
                                inc, hashes, tup, pi_prune=calc.extra['pi_prune'])

        # debug print
        if mol.debug >= 1:
            print(output.mbe_debug(mol.atom, mol.symmetry, calc.orbsym, calc.state['root'], \
                                    tools.ndets(calc.occup, cas_idx, n_elec=nelec), \
                                    nelec, inc_tup, order, cas_idx, tup))

        return inc_tup


def _sum(occup, ref_space, exp_space, target, min_order, order, inc, hashes, tup, pi_prune=False):
        """
        this function performs a recursive summation

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param exp_space: dictionary of expansion spaces. dict
        :param target: calculation target. string
        :param min_order: minimum (start) order. integer
        :param order: current order. integer
        :param inc: property increments to all order. list of numpy arrays of shapes (n_tuples,) or (n_tuples, 3) depending on target
        :param hashes: hashes to all order. list of numpy arrays of shapes (n_tuples,)
        :param tup: given tuple of orbitals. numpy array of shape (order,)
        :return: float or numpy array of shape (3,) depending on target
        """
        # init res
        if target in ['energy', 'excitation']:
            res = 0.0
        else:
            res = np.zeros(3, dtype=np.float64)

        # compute contributions from lower-order increments
        for k in range(order-1, min_order-1, -1):

            # generate array with all subsets of particular tuple
            combs = np.array([comb for comb in itertools.combinations(tup, k)], dtype=np.int32)

            # prune combinations without a mix of occupied and virtual orbitals
            if min_order == 2:
                combs = combs[np.fromiter(map(functools.partial(tools.corr_prune, occup), combs), \
                                              dtype=bool, count=combs.shape[0])]

            # prune combinations with non-degenerate pairs of pi-orbitals
            if pi_prune:
                combs = combs[np.fromiter(map(functools.partial(tools.pi_prune, \
                                              exp_space['pi_orbs'], exp_space['pi_hashes']), combs), \
                                              dtype=bool, count=combs.shape[0])]

            if combs.size == 0:
                continue

            # convert to sorted hashes
            combs_hash = tools.hash_2d(combs)
            combs_hash.sort()

            # get indices of combinations
            idx = tools.hash_compare(hashes[k-min_order], combs_hash)

            # assertion
            tools.assertion(idx is not None, 'error in recursive increment calculation:\n'
                                             'k = {:}\ntup:\n{:}\ncombs:\n{:}'. \
                                             format(k, tup, combs))

            # add up lower-order increments
            res += tools.fsum(inc[k-min_order][idx])

        return res


