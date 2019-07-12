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
import functools
from mpi4py import MPI
import sys
import itertools
import scipy.misc

import kernel
import output
import expansion
import driver
import parallel
import tools


# tags
TAGS = tools.enum('ready', 'tup', 'h2e', 'exit')


def master(mpi, mol, calc, exp):
        """
        this master function returns two arrays of (i) number of determinants and (ii) mbe increments

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: two numpy arrays of shapes (n_tuples,) [ndets] and (n_tuples,) or (n_tuples, 3) [inc] depending on target
        """
        # wake up slaves
        msg = {'task': 'mbe', 'order': exp.order}
        mpi.comm.bcast(msg, root=0)

        # number of tasks
        n_tasks = exp.hashes[-1].size
        # number of available slaves
        slaves_avail = min(mpi.size - 1, n_tasks)

        # init requests
        if mpi.non_block:
            req_tup = MPI.Request()
            req_h2e = MPI.Request()

        # compute number of determinants in the individual casci calculations (ignoring symmetry)
        ndets = np.fromiter(map(functools.partial(tools.ndets, calc.occup, ref_space=calc.ref_space), \
                                exp.tuples[-1]), dtype=np.float64, count=n_tasks)

        # order tasks wrt number of determinants (from most electrons to fewest electrons)
        tasks = np.argsort(ndets)[::-1]

        # loop until no tasks left
        for task in tasks:

            # get tup
            tup = exp.tuples[-1][task]

            # get slave
            parallel.probe(mpi, TAGS.ready)

            # send tup
            if mpi.non_block:
                req_tup.Wait()
                req_tup = mpi.comm.Isend([tup, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup)
            else:
                mpi.comm.Send([tup, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup)

            # get cas indices
            cas_idx = tools.cas(calc.ref_space, tup)

            # get h2e indices
            cas_idx_tril = tools.cas_idx_tril(cas_idx)

            # get h2e_cas
            h2e_cas = mol.eri[cas_idx_tril[:, None], cas_idx_tril]

            # send h2e_cas
            if mpi.non_block:
                req_h2e.Wait()
                req_h2e = mpi.comm.Isend([h2e_cas, MPI.DOUBLE], dest=mpi.stat.source, tag=TAGS.h2e)
            else:
                mpi.comm.Send([h2e_cas, MPI.DOUBLE], dest=mpi.stat.source, tag=TAGS.h2e)

        # done with all tasks
        while slaves_avail > 0:

            # get slave
            parallel.probe(mpi, TAGS.ready)

            # send exit signal to slave
            mpi.comm.send(None, dest=mpi.stat.source, tag=TAGS.exit)

            # remove slave
            slaves_avail -= 1

        # wait for all data communication to be finished
        if mpi.non_block:
            MPI.Request.Waitall([req_tup, req_h2e])

        # init increments
        inc = _init_inc(n_tasks, calc.target)

        # allreduce increments
        inc = parallel.allreduce(mpi, inc)

        return ndets, inc


def slave(mpi, mol, calc, exp):
        """
        this slave function returns an array of mbe increments

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: numpy array of shape (n_tuples,)
        """
        # number of tasks
        n_tasks = exp.hashes[-1].size
        # number of needed slaves
        slaves_needed = min(mpi.size - 1, n_tasks)

        # init tup
        tup = np.empty(exp.order, dtype=np.int32)

        # init increments
        inc = _init_inc(n_tasks, calc.target)

        # init h2e_cas
        h2e_cas = _init_h2e(calc.ref_space, exp.order)

        # send availability to master
        if mpi.rank <= slaves_needed:
            mpi.comm.send(None, dest=0, tag=TAGS.ready)

        # receive work from master
        while True:

            # early exit in case of large proc count
            if mpi.rank > slaves_needed:
                break

            # probe for available task
            mpi.comm.Probe(source=0, tag=MPI.ANY_TAG, status=mpi.stat)

            # do task
            if mpi.stat.tag == TAGS.tup:

                # receive tup
                if mpi.non_block:
                    req_tup = mpi.comm.Irecv([tup, MPI.INT], source=0, tag=TAGS.tup)
                else:
                    mpi.comm.Recv([tup, MPI.INT], source=0, tag=TAGS.tup)

                # receive h2e_cas
                if mpi.non_block:
                    req_h2e = mpi.comm.Irecv([h2e_cas, MPI.DOUBLE], source=0, tag=TAGS.h2e)
                else:
                    mpi.comm.Recv([h2e_cas, MPI.DOUBLE], source=0, tag=TAGS.h2e)

                # get core and cas indices
                if mpi.non_block:
                    req_tup.Wait()
                core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, tup)

                # compute e_core and h1e_cas
                e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, mol.hcore, mol.vhf, core_idx, cas_idx)

                # get task_idx
                task_idx = tools.hash_compare(exp.hashes[-1], tools.hash_1d(tup))

                # calculate increment
                if mpi.non_block:
                    req_h2e.Wait()
                inc[task_idx] = _inc(mol, calc, exp, e_core, h1e_cas, h2e_cas, tup, core_idx, cas_idx)

                # send availability to master
                mpi.comm.send(None, dest=0, tag=TAGS.ready)

            elif mpi.stat.tag == TAGS.exit:

                # exit
                mpi.comm.recv(None, source=0, tag=TAGS.exit)
                break

        # allreduce increments
        return parallel.allreduce(mpi, inc)


def _inc(mol, calc, exp, e_core, h1e_cas, h2e_cas, tup, core_idx, cas_idx):
        """
        this function calculates the increment associated with a given tuple

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :param e_core: core energy. scalar
        :param h1e_cas: cas space 1-e Hamiltonian. numpy array of shape (n_cas, n_cas)
        :param h2e_cas: cas space 2-e Hamiltonian. numpy array of shape (n_cas*(n_cas + 1) // 2, n_cas*(n_cas + 1) // 2)
        :param tup: given tuple of orbitals. numpy array of shape (order,)
        :param core_idx: core space indices. numpy array of shape (n_core,)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :return: scalar or numpy array of shape (3,) depending on target
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
        if exp.order > exp.min_order:
            if np.any(inc_tup != 0.0):
                inc_tup -= _sum(calc.occup, calc.mo_energy, calc.ref_space, calc.exp_space, \
                                calc.target, exp.min_order, exp.order, \
                                exp.prop[calc.target]['inc'], exp.hashes, \
                                tup, pi_prune=calc.extra['pi_prune'])

        # debug print
        if mol.debug >= 1:
            print(output.mbe_debug(mol.atom, mol.symmetry, calc.orbsym, calc.state['root'], \
                                    tools.ndets(calc.occup, cas_idx, n_elec=nelec), \
                                    nelec, inc_tup, exp.order, cas_idx, tup))

        return inc_tup


def _sum(occup, mo_energy, ref_space, exp_space, target, min_order, order, prop, hashes, tup, pi_prune=False):
        """
        this function performs a recursive summation

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param mo_energy: orbital energies. numpy array of shape (n_orb,)
        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param exp_space: dictionary of expansion spaces. dict
        :param target: calculation target. string
        :param min_order: minimum (start) order. integer
        :param order: current order. integer
        :param prop: property increments to all order. list of numpy arrays of shapes (n_tuples,) or (n_tuples, 3) depending on target
        :param hashes: hashes to all order. list of numpy arrays of shapes (n_tuples,)
        :param tup: given tuple of orbitals. numpy array of shape (order,)
        :return: scalar or numpy array of shape (3,) depending on target
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
            res += tools.fsum(prop[k-min_order][idx])

        return res


def _init_inc(n_tasks, target):
        """
        this function initializes the current order increments array

        :param n_tasks: number of tasks (tuples). integer
        :param target: calculation target. string
        :return: numpy array of shape (n_tasks,) or (n_tasks, 3) depending on target
        """
        if target in ['energy', 'excitation']:
            return np.zeros(n_tasks, dtype=np.float64)
        else:
            return np.zeros([n_tasks, 3], dtype=np.float64)


def _init_h2e(ref_space, order):
        """
        this function initializes the cas space 2-e Hamiltonian array

        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param order: current order. integer
        :return: numpy array of shape (n_cas*(n_cas + 1) // 2, n_cas*(n_cas + 1) // 2)
        """
        n_orb = ref_space.size + order
        return np.empty([(n_orb * (n_orb + 1)) // 2, (n_orb * (n_orb + 1)) // 2], dtype=np.float64)


