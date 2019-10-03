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
from typing import Tuple, List, Dict, Union

import parallel
import calculation
import expansion
import tools


# tags
class TAGS:
    ready, tup, tup_pi, tup_seed, tup_seed_pi, exit = range(6)


def master(mpi: parallel.MPICls, calc: calculation.CalcCls, \
            exp: expansion.ExpCls) -> Tuple[MPI.Win, MPI.Win, int]:
        """
        this master function returns two arrays of (i) child tuple hashes and (ii) the actual child tuples
        """
        # set number of available (needed) slaves and various tuples
        slaves_avail, tuples, tuples_pi, \
            tuples_seed, tuples_seed_pi = _set_screen(calc.occup, calc.ref_space, calc.exp_space, \
                                                        exp.n_tasks[-1], exp.min_order, exp.order, \
                                                        calc.extra['pi_prune'], mpi.global_size, exp.tuples)

        # wake up slaves
        msg = {'task': 'screen', 'order': exp.order, 'slaves_needed': slaves_avail}
        mpi.global_comm.bcast(msg, root=0)

        # mpi barrier
        mpi.local_comm.barrier()

        # loop until no tuples left
        for tup_idx in range(exp.n_tasks[-1]):

            # probe for available slaves
            mpi.global_comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

            # receive slave status
            mpi.global_comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)

            # send tups to available slave
            mpi.global_comm.send(tup_idx, dest=mpi.stat.source, tag=TAGS.tup)

        # pi-pruning
        if tuples_pi is not None:

            # loop until no tuples left
            for tup in tuples_pi:

                # probe for available slaves
                mpi.global_comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

                # receive slave status
                mpi.global_comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)

                # send tup to available slave
                mpi.global_comm.Send([tup, MPI.SHORT], dest=mpi.stat.source, tag=TAGS.tup_pi)

        # seed
        if tuples_seed is not None:

            # loop until no tuples left
            for tup in tuples_seed:

                # probe for available slaves
                mpi.global_comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

                # receive slave status
                mpi.global_comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)

                # send tup to available slave
                mpi.global_comm.Send([tup, MPI.SHORT], dest=mpi.stat.source, tag=TAGS.tup_seed)

        # seed w/ pi-pruning
        if tuples_seed_pi is not None:

            # loop until no tuples left
            for tup in tuples_seed_pi:

                # probe for available slaves
                mpi.global_comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

                # receive slave status
                mpi.global_comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)

                # send tup to available slave
                mpi.global_comm.Send([tup, MPI.SHORT], dest=mpi.stat.source, tag=TAGS.tup_seed_pi)

        # done with all tasks
        while slaves_avail > 0:

            # probe for available slaves
            mpi.global_comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

            # receive slave status
            mpi.global_comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)

            # send exit signal to slave
            mpi.global_comm.send(None, dest=mpi.stat.source, tag=TAGS.exit)

            # remove slave
            slaves_avail -= 1

        # init child tuples array
        if calc.extra['pi_prune'] and exp.order == 1:
            child_tup = tools.pi_pairs_deg(calc.exp_space['pi_orbs'], calc.exp_space['tot'])
        else:
            child_tup = np.array([], dtype=np.int16)

        # free parent_tuples window
        exp.tuples.Free()

        # free other parent tuples
        if tuples_pi is not None:
            del tuples_pi
        if tuples_seed is not None:
            del tuples_seed
        if tuples_seed_pi is not None:
            del tuples_seed_pi

        # allgather number of child tuples
        recv_counts = np.array(mpi.global_comm.allgather(child_tup.size))

        # no child tuples - expansion is converged
        if np.sum(recv_counts) == 0:
            return None, None, 0

        # allocate tuples
        tuples_win = MPI.Win.Allocate_shared(2 * np.sum(recv_counts), 2, comm=mpi.local_comm)
        buf = tuples_win.Shared_query(0)[0]
        tuples_new = np.ndarray(buffer=buf, dtype=np.int16, shape=(np.sum(recv_counts),))

        # gatherv all child tuples onto global master
        tuples_new[:] = parallel.gatherv(mpi.global_comm, child_tup, recv_counts)

        # reshape tuples_new
        tuples_new = tuples_new.reshape(-1, exp.order + 1)

        # bcast tuples
        if mpi.num_masters > 1:
            tuples_new[:] = parallel.bcast(mpi.master_comm, tuples_new)

        # mpi barrier
        mpi.local_comm.barrier()

        # n_tasks
        n_tasks = tuples_new.shape[0]

        # allocate hashes
        hashes_win = MPI.Win.Allocate_shared(8 * n_tasks, 8, comm=mpi.local_comm)
        buf = hashes_win.Shared_query(0)[0]
        hashes_new = np.ndarray(buffer=buf, dtype=np.int64, shape=(n_tasks,))

        # compute hashes
        hashes_new[:] = tools.hash_2d(tuples_new)

        # sort hashes
        hashes_new.sort()

        # save restart files
        if calc.misc['rst']:
            tools.write_file(None, tuples_new, 'mbe_tup')
            tools.write_file(exp.order+1, hashes_new, 'mbe_hash')

        # mpi barrier
        mpi.global_comm.barrier()

        return hashes_win, tuples_win, n_tasks


def slave(mpi: parallel.MPICls, calc: calculation.CalcCls, \
            exp: expansion.ExpCls, slaves_needed: int) -> Tuple[MPI.Win, MPI.Win, int]:
        """
        this slave function returns an array of child tuple hashes
        """
        # init list of child tuples
        child_tup: Union[List[int], np.ndarray] = []

        # send availability to master
        if mpi.global_rank <= slaves_needed:
            mpi.global_comm.send(None, dest=0, tag=TAGS.ready)

        # load tuples as main task tuples
        buf = exp.tuples.Shared_query(0)[0]
        tuples = np.ndarray(buffer=buf, dtype=np.int16, shape=(exp.n_tasks[-1], exp.order))

        # load increments for current and previous orders
        inc = []
        for k in range(exp.order-exp.min_order+1):
            buf = exp.prop[calc.target_mbe]['inc'][k].Shared_query(0)[0] # type: ignore
            if calc.target_mbe in ['energy', 'excitation']:
                inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[k],)))
            elif calc.target_mbe in ['dipole', 'trans']:
                inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[k], 3)))

        # load hashes for current and previous orders
        hashes = []
        for k in range(exp.order-exp.min_order+1):
            buf = exp.hashes[k].Shared_query(0)[0]
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tasks[k],)))

        # init tup_seed and tup_pi
        tup_seed = np.empty(exp.order, dtype=np.int16)
        tup_pi = np.empty(exp.order-1, dtype=np.int16)

        # mpi barrier
        mpi.local_comm.barrier()

        # receive work from master
        while True:

            # early exit in case of large proc count
            if mpi.global_rank > slaves_needed:
                break

            # probe for task
            mpi.global_comm.Probe(source=0, tag=MPI.ANY_TAG, status=mpi.stat)

            # do task
            if mpi.stat.tag in [TAGS.tup, TAGS.tup_pi, TAGS.tup_seed, TAGS.tup_seed_pi]:

                if mpi.stat.tag == TAGS.tup:

                    # receive tup_idx
                    tup_idx = mpi.global_comm.recv(source=0, tag=mpi.stat.tag)
                    tup = tuples[tup_idx]
                    tup_order = exp.order

                else:

                    if mpi.stat.tag == TAGS.tup_seed:

                        # receive tup_seed
                        mpi.global_comm.Recv([tup_seed, MPI.SHORT], source=0, tag=mpi.stat.tag)
                        tup = tup_seed
                        tup_order = exp.order

                    else:

                        # receive tup_pi or tup_seed_pi
                        mpi.global_comm.Recv([tup_pi, MPI.SHORT], source=0, tag=mpi.stat.tag)
                        tup = tup_pi
                        tup_order = exp.order - 1

                # spawn child tuples from parent tuples at exp.order
                orbs = _orbs(calc.occup, calc.mo_energy, calc.orbsym, calc.prot, \
                                calc.thres, calc.ref_space, calc.exp_space, exp.min_order, \
                                tup_order, hashes[-1], inc[-1], \
                                tup, pi_prune=calc.extra['pi_prune'], \
                                pi_gen=mpi.stat.tag in [TAGS.tup_pi, TAGS.tup_seed_pi])

                # deep pruning
                if calc.extra['pi_prune'] and exp.min_order < tup_order:
                    orbs = _deep_pruning(calc.occup, calc.mo_energy, calc.orbsym, calc.prot, \
                                            calc.thres, calc.ref_space, calc.exp_space, exp.min_order, \
                                            tup_order, hashes, inc, \
                                            tup, orbs, pi_gen=mpi.stat.tag in [TAGS.tup_pi, TAGS.tup_seed_pi])

                # recast parent tuple as list
                tup = tup.tolist()

                # reshape orbs in pairs of pi-orbitals
                if mpi.stat.tag in [TAGS.tup_pi, TAGS.tup_seed_pi]:
                    orbs = orbs.reshape(-1, 2)

                # loop over orbitals and add to list of child tuples
                for orb in orbs:
                    if mpi.stat.tag in [TAGS.tup_pi, TAGS.tup_seed_pi]:
                        child_tup += tup + orb.tolist()
                    else:
                        child_tup += tup + [orb]

                # send availability to master
                mpi.global_comm.send(None, dest=0, tag=TAGS.ready)

            elif mpi.stat.tag == TAGS.exit:

                # exit
                mpi.global_comm.recv(None, source=0, tag=TAGS.exit)
                break

        # recast child tuples as array
        child_tup = np.array(child_tup, dtype=np.int16)

        # reshape child tuples
        child_tup = child_tup.reshape(-1, exp.order + 1)

        # free parent_tuples window
        exp.tuples.Free()

        # allgather number of child tuples
        recv_counts = np.array(mpi.global_comm.allgather(child_tup.size))

        # no child tuples - expansion is converged
        if np.sum(recv_counts) == 0:
            return None, None, 0

        # get handle to tuples
        if mpi.local_master:
            tuples_win = MPI.Win.Allocate_shared(2 * np.sum(recv_counts), 2, comm=mpi.local_comm)
            buf = tuples_win.Shared_query(0)[0]
            tuples_new = np.ndarray(buffer=buf, dtype=np.int16, \
                                    shape=(np.sum(recv_counts) // (exp.order + 1), exp.order + 1))
        else:
            tuples_win = MPI.Win.Allocate_shared(0, 2, comm=mpi.local_comm)

        # gatherv all child tuples
        child_tup = parallel.gatherv(mpi.global_comm, child_tup, recv_counts)

        # bcast tuples
        if mpi.num_masters > 1 and mpi.local_master:
            tuples_new[:] = parallel.bcast(mpi.master_comm, tuples_new)

        # mpi barrier
        mpi.local_comm.barrier()

        # get handle to hashes window
        if mpi.local_master:
            hashes_win = MPI.Win.Allocate_shared(8 * tuples_new.shape[0], 8, comm=mpi.local_comm)
            buf = hashes_win.Shared_query(0)[0]
            hashes_new = np.ndarray(buffer=buf, dtype=np.int64, shape=(tuples_new.shape[0],))
        else:
            hashes_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)

        if mpi.local_master:

            # compute hashes
            hashes_new[:] = tools.hash_2d(tuples_new)

            # sort hashes
            hashes_new.sort()

        # mpi barrier
        mpi.global_comm.Barrier()

        return hashes_win, tuples_win, int(np.sum(recv_counts)) // (exp.order + 1)


def _set_screen(occup: np.ndarray, ref_space: np.ndarray, exp_space: Dict[str, np.ndarray], \
                    n_tasks: int, min_order: int, order: int, pi_prune: bool, global_size: int, \
                    tuples_win: MPI.Win) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        this function returns number of available slaves and various tuples

        example:
        >>> occup = np.array([2.] * 4 + [0.] * 6)
        >>> ref_space = np.arange(4, dtype=np.int16)
        >>> exp_space = {'occ': np.array([], dtype=np.int16),
        ...              'virt': np.arange(4, 10, dtype=np.int16),
        ...              'tot': np.arange(4, 10, dtype=np.int16)}
        >>> min_order = order = 1
        >>> n_tasks = 6
        >>> tuples_win = MPI.Win.Allocate_shared(2 * n_tasks * order, 2, comm=MPI.COMM_WORLD)
        >>> buf = tuples_win.Shared_query(0)[0]
        >>> tuples = np.ndarray(buffer=buf, dtype=np.int16, shape=(n_tasks, order))
        >>> tuples[:] = np.array([[4], [5], [6], [7], [8], [9]], dtype=np.int16)
        >>> _set_screen(occup, ref_space, exp_space, n_tasks,
        ...             min_order, order, False, 1, tuples_win)
        (0, array([[4],
               [5],
               [6],
               [7],
               [8],
               [9]], dtype=int16), None, None, None)
        >>> ref_space = np.array([])
        >>> exp_space['occ'] = np.arange(4, dtype=np.int16)
        >>> exp_space['tot'] = np.arange(10, dtype=np.int16)
        >>> min_order = order = 2
        >>> n_tasks = 24
        >>> tuples_win = MPI.Win.Allocate_shared(2 * n_tasks * order, 2, comm=MPI.COMM_WORLD)
        >>> buf = tuples_win.Shared_query(0)[0]
        >>> tuples = np.ndarray(buffer=buf, dtype=np.int16, shape=(n_tasks, order))
        >>> tuples[:] = np.array([[0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
        ...                       [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
        ...                       [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
        ...                       [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9]], dtype=np.int16)
        >>> _set_screen(occup, ref_space, exp_space, n_tasks,
        ...             min_order, order, False, 1, tuples_win)
        (0, array([[0, 4],
               [0, 5],
               [0, 6],
               [0, 7],
               [0, 8],
               [0, 9],
               [1, 4],
               [1, 5],
               [1, 6],
               [1, 7],
               [1, 8],
               [1, 9],
               [2, 4],
               [2, 5],
               [2, 6],
               [2, 7],
               [2, 8],
               [2, 9],
               [3, 4],
               [3, 5],
               [3, 6],
               [3, 7],
               [3, 8],
               [3, 9]], dtype=int16), None, array([[0, 1],
               [0, 2],
               [0, 3],
               [1, 2],
               [1, 3],
               [2, 3]], dtype=int16), None)
        >>> exp_space['pi_orbs'] = np.array([1, 2, 4, 5], dtype=np.int16)
        >>> exp_space['pi_hashes'] = np.array([-2163557957507198923, 1937934232745943291])
        >>> min_order, order = 2, 3
        >>> n_tasks = 22
        >>> tuples_win = MPI.Win.Allocate_shared(2 * n_tasks * order, 2, comm=MPI.COMM_WORLD)
        >>> buf = tuples_win.Shared_query(0)[0]
        >>> tuples = np.ndarray(buffer=buf, dtype=np.int16, shape=(n_tasks, order))
        >>> tuples[:] = np.array([[0, 4, 5], [0, 6, 7], [0, 6, 8], [0, 6, 9], [0, 7, 8], [0, 7, 9], [0, 8, 9],
        ...                       [0, 3, 6], [0, 3, 7], [0, 3, 8], [0, 3, 9],
        ...                       [1, 2, 6], [1, 2, 7], [1, 2, 8], [1, 2, 9],
        ...                       [3, 4, 5], [3, 6, 7], [3, 6, 8], [3, 6, 9], [3, 7, 8], [3, 7, 9], [3, 8, 9]] , dtype=np.int16)
        >>> _set_screen(occup, ref_space, exp_space, n_tasks,
        ...             min_order, order, True, 1, tuples_win)
        (0, array([[0, 4, 5],
               [0, 6, 7],
               [0, 6, 8],
               [0, 6, 9],
               [0, 7, 8],
               [0, 7, 9],
               [0, 8, 9],
               [0, 3, 6],
               [0, 3, 7],
               [0, 3, 8],
               [0, 3, 9],
               [1, 2, 6],
               [1, 2, 7],
               [1, 2, 8],
               [1, 2, 9],
               [3, 4, 5],
               [3, 6, 7],
               [3, 6, 8],
               [3, 6, 9],
               [3, 7, 8],
               [3, 7, 9],
               [3, 8, 9]], dtype=int16), array([[0, 6],
               [0, 7],
               [0, 8],
               [3, 6],
               [3, 7],
               [3, 8]], dtype=int16), array([[0, 1, 2],
               [1, 2, 3]], dtype=int16), array([[0, 3],
               [1, 2]], dtype=int16))
        """
        # load tuples as main task tuples
        buf = tuples_win.Shared_query(0)[0]
        tuples = np.ndarray(buffer=buf, dtype=np.int16, shape=(n_tasks, order))

        # option to treat pi-orbitals independently
        tuples_pi = None; n_tasks_pi = 0

        if pi_prune and min_order < order:

            # set tuples_pi
            tuples_pi = np.unique(tuples[:, :-1], axis=0)

            # prune combinations without a mix of occupied and virtual orbitals
            tuples_pi = tuples_pi[np.fromiter(map(functools.partial(tools.corr_prune, occup), tuples_pi), \
                                              dtype=bool, count=tuples_pi.shape[0])]

            # prune combinations that contain non-degenerate pairs of pi-orbitals
            tuples_pi = tuples_pi[np.fromiter(map(functools.partial(tools.pi_prune, \
                                              exp_space['pi_orbs'], \
                                              exp_space['pi_hashes']), tuples_pi), \
                                              dtype=bool, count=tuples_pi.shape[0])]

            # number of tasks
            n_tasks_pi = tuples_pi.shape[0]

        # potential seed for vacuum reference spaces
        tuples_seed = None; n_tasks_seed = 0
        tuples_seed_pi = None; n_tasks_seed_pi = 0

        if ref_space.size == 0 and order <= exp_space['occ'].size:

            # set tuples_seed
            tuples_seed = np.array([tup for tup in itertools.combinations(exp_space['occ'], order)], dtype=np.int16)

            # prune combinations that contain non-degenerate pairs of pi-orbitals
            if pi_prune:
                tuples_seed = tuples_seed[np.fromiter(map(functools.partial(tools.pi_prune, \
                                                        exp_space['pi_orbs'], \
                                                        exp_space['pi_hashes']), tuples_seed), \
                                                        dtype=bool, count=tuples_seed.shape[0])]

            # number of tasks
            n_tasks_seed = tuples_seed.shape[0]

            # option to treat pi-orbitals independently
            if pi_prune:

                # set tuples_seed_pi
                tuples_seed_pi = np.array([tup for tup in itertools.combinations(exp_space['occ'], order-1)], dtype=np.int16)

                # prune combinations that contain non-degenerate pairs of pi-orbitals
                tuples_seed_pi = tuples_seed_pi[np.fromiter(map(functools.partial(tools.pi_prune, \
                                                              exp_space['pi_orbs'], \
                                                              exp_space['pi_hashes']), tuples_seed_pi), \
                                                              dtype=bool, count=tuples_seed_pi.shape[0])]

                # number of tasks
                n_tasks_seed_pi = tuples_seed_pi.shape[0]

        # number of available slaves
        slaves_avail = min(global_size - 1, n_tasks + n_tasks_pi + n_tasks_seed + n_tasks_seed_pi)

        return slaves_avail, tuples, tuples_pi, tuples_seed, tuples_seed_pi


def _orbs(occup, mo_energy, orbsym, prot, thres, ref_space, exp_space, \
            min_order, order, hashes, prop, tup, pi_prune=False, pi_gen=False):
        """
        this function returns an array of child tuple orbitals subject to a given screening protocol

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param mo_energy: orbital energies. numpy array of shape (n_orb,)
        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param prot: screening protocol scheme. dict
        :param thres: threshold settings. dict
        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param exp_space: dictionary of expansion spaces. dict
        :param min_order: minimum (start) order. integer
        :param order: current order. integer
        :param hashes: current order hashes. numpy array of shape (n_tuples,)
        :param prop: current order property increments. numpy array of shape (n_tuples,)
        :param tup: current orbital tuple. numpy array of shape (order,)
        :param pi_prune: pi-orbital pruning logical. bool
        :param pi_gen: pi-orbital generation logical. bool
        :return: numpy array of shape (n_child_orbs,)
        """
        # truncate expansion space
        if min_order == 1:
            exp_space_trunc = exp_space['tot'][tup[-1] < exp_space['tot']]
        elif min_order == 2:
            exp_space_trunc = exp_space['virt'][tup[-1] < exp_space['virt']]

        if pi_gen:
            # consider only pairs of degenerate pi-orbitals in truncated expansion space
            exp_space_trunc = tools.pi_pairs_deg(exp_space['pi_orbs'], exp_space_trunc)
        else:
            if pi_prune:
                # consider only non-degenerate orbitals in truncated expansion space
                exp_space_trunc = tools.non_deg_orbs(exp_space['pi_orbs'], exp_space_trunc)

        # at min_order, spawn all possible child tuples
        if order <= min_order:
            return exp_space_trunc.ravel()

        # generate array with all k-1 order subsets of particular tuple
        combs = np.array([comb for comb in itertools.combinations(tup, order-1)], dtype=np.int16)

        # prune combinations without seed orbitals
        if min_order == 2:
            combs = combs[np.fromiter(map(functools.partial(tools.seed_prune, occup), combs), \
                                          dtype=bool, count=combs.shape[0])]

        # prune combinations that contain non-degenerate pairs of pi-orbitals
        if pi_prune:
            combs = combs[np.fromiter(map(functools.partial(tools.pi_prune, \
                                          exp_space['pi_orbs'], \
                                          exp_space['pi_hashes']), combs), \
                                          dtype=bool, count=combs.shape[0])]

        if combs.size == 0:
            return exp_space_trunc.ravel()

        # init list of child orbitals
        child_orbs = []

        # init orb_arr
        orb_arr = np.empty([combs.shape[0], 2 if pi_gen else 1], dtype=np.int16)

        # loop over orbitals of truncated expansion space
        for orb in exp_space_trunc:

            # add orbital(s) to combinations
            orb_arr[:] = orb
            combs_orb = np.concatenate((combs, orb_arr), axis=1)

            # convert to sorted hashes and reorder combs_orb accordingly
            combs_orb_hash = tools.hash_2d(combs_orb)
            combs_orb = combs_orb[np.argsort(combs_orb_hash)]
            combs_orb_hash.sort()

            # get indices of combinations
            idx = tools.hash_compare(hashes, combs_orb_hash)

            # only continue if child orbital is valid
            if idx is not None:

                # compute screening thresholds
                screen_thres = np.fromiter(map(functools.partial(_thres, \
                                    occup, thres, ref_space, prot['scheme']), combs_orb), \
                                    dtype=np.float64, count=idx.size)

                # add orbital to list of child orbitals if allowed
                if not _prot_screen(prot['scheme'], screen_thres, prop[idx]) or np.sum(screen_thres) == 0.0:

                    if pi_gen:
                        child_orbs += orb.tolist()
                    else:
                        child_orbs += [orb]

        return np.array(child_orbs, dtype=np.int16)


def _deep_pruning(occup, mo_energy, orbsym, prot, thres, ref_space, exp_space, \
            min_order, order, hashes, prop, tup, orbs, pi_gen=False):
        """
        this function returns an updated array of child tuple orbitals upon deep pruning

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param mo_energy: orbital energies. numpy array of shape (n_orb,)
        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param prot: screening protocols. dict
        :param thres: threshold settings. dict
        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param exp_space: dictionary of expansion spaces. dict of three numpy arrays with shapes (n_exp_tot,); (n_exp_occ,); (n_exp_virt)
        :param min_order: minimum (start) order. integer
        :param order: current order. integer
        :param hashes: hashes to all orders. list of numpy arrays of shapes (n_tuples,)
        :param prop: property increments to all orders. list of numpy arrays of shapes (n_tuples,)
        :param tup: current orbital tuple. numpy array of shape (order,)
        :param orbs: initial array of child tuple orbitals. numpy array of shape (n_child_orbs_old,)
        :param pi_gen: pi-orbital generation logical. bool
        :return: numpy array of shape (n_child_orbs_new,)
        """
        # deep pruning by removing an increasing number of pi-orbital pairs
        for k in range(tools.n_pi_orbs(exp_space['pi_orbs'], tup) // 2):

            # next-highest order without k number of pi-orbital pairs
            deep_order = order - (2 * k + 1)

            # spawn child tuples from parent tuples at deep_order
            if pi_gen:
                orbs_deep = _orbs(occup, mo_energy, orbsym, prot, thres, ref_space, exp_space, \
                                     min_order, deep_order, hashes[(deep_order+1)-min_order], \
                                     prop[(deep_order+1)-min_order], tup, pi_prune=True, pi_gen=True)
            else:
                orbs_deep = _orbs(occup, mo_energy, orbsym, prot, thres, ref_space, exp_space, \
                                     min_order, deep_order, hashes[deep_order-min_order], \
                                     prop[deep_order-min_order], tup, pi_prune=True, pi_gen=False)

            # update orbs
            orbs = np.intersect1d(orbs, orbs_deep)

        return orbs


def _prot_screen(scheme, thres, prop):
        """
        this function extracts increments with non-zero thresholds and calls screening function

        :param scheme: protocol scheme. integer
        :param thres: screening thresholds corresponding to increments. numpy array of shape (n_inc,)
        :param prop: property increments corresponding to given tuple of orbitals. numpy array of shape (n_inc,)
        :return: bool
        """
        # extract increments with non-zero thresholds
        inc = prop[np.nonzero(thres)]

        # screening procedure
        if inc.ndim == 1:

            screen = _prot_scheme(scheme, thres[np.nonzero(thres)], inc)

        else:

            # init screening logical
            screen = True

            # loop over dimensions: (x,y,z) = (0,1,2)
            for dim in range(3):

                # only screen based on relevant dimensions
                if np.sum(inc[:, dim]) != 0.0:
                    screen = _prot_scheme(scheme, thres[np.nonzero(thres)], inc[:, dim])

                # if any increment is large enough, then quit screening
                if not screen:
                    break

        return screen


def _prot_scheme(scheme, thres, prop):
        """
        this function screens according to chosen protocol scheme

        :param scheme: protocol scheme. integer
        :param thres: screening thresholds corresponding to increments. numpy array of shape (n_inc,)
        :param prop: property increments corresponding to given tuple of orbitals. numpy array of shape (n_inc,)
        :return: bool
        """
        if scheme == 1:
            # are *any* increments below their given threshold
            return np.any(np.abs(prop) < thres)
        else:
            # are *all* increments below their given threshold
            return np.all(np.abs(prop) < thres)


def _thres(occup: np.ndarray, thres: Dict[str, float], \
            ref_space: np.ndarray, scheme: int, tup: np.ndarray) -> float:
        """
        this function computes the screening threshold for the given tuple of orbitals

        example:
        >>> occup = np.array([2.] * 4 + [0.] * 6)
        >>> thres = {'init': 1.e-10, 'relax': 5.}
        >>> tup = np.array([5, 7, 8], dtype=np.int16)
        >>> ref_space = np.arange(4, dtype=np.int16)
        >>> scheme = 2
        >>> _thres(occup, thres, ref_space, scheme, tup)
        1e-10
        >>> tup = np.array([5, 7, 8, 9], dtype=np.int16)
        >>> _thres(occup, thres, ref_space, scheme, tup)
        5e-10
        >>> ref_space = np.array([], dtype=np.int16)
        >>> tup = np.array([0, 1, 2, 3, 5, 7, 9], dtype=np.int16)
        >>> _thres(occup, thres, ref_space, scheme, tup)
        1e-10
        >>> scheme = 3
        >>> _thres(occup, thres, ref_space, scheme, tup)
        5e-10
        """
        # determine involved dimensions
        nocc = np.count_nonzero(occup[ref_space] > 0.0)
        nocc += np.count_nonzero(occup[tup] > 0.0)
        nvirt = np.count_nonzero(occup[ref_space] == 0.0)
        nvirt += np.count_nonzero(occup[tup] == 0.0)

        # init thres
        screen_thres = 0.0

        # update thres
        if nocc > 0 and nvirt > 0:

            if scheme < 3:
                if nvirt >= 3:
                    screen_thres = thres['init'] * thres['relax'] ** (nvirt - 3)
            else:
                if max(nocc, nvirt) >= 3:
                    screen_thres = thres['init'] * thres['relax'] ** (max(nocc, nvirt) - 3)

        return screen_thres


if __name__ == "__main__":
    import doctest
    doctest.testmod()#verbose=True)


