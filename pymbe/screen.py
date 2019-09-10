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

import parallel
import tools


# tags
TAGS = tools.enum('ready', 'tup', 'tup_pi', 'tup_seed', 'tup_seed_pi', 'exit')


def master(mpi, calc, exp):
        """
        this master function returns two arrays of (i) child tuple hashes and (ii) the actual child tuples

        :param mpi: pymbe mpi object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: MPI window handle to numpy array of shape (n_child_tup,) [hashes],
                 integer [n_tasks],
                 numpy array of shape (n_child_tuples, order+1) [tuples]
        """
        # set number of available (needed) slaves, various tuples, and various task arrays
        slaves_avail, tuples, tasks, tuples_pi, tasks_pi, \
            tuples_seed, tasks_seed, tuples_seed_pi, tasks_seed_pi = _set_screen(mpi, calc, exp)

        # wake up slaves
        msg = {'task': 'screen', 'order': exp.order, 'slaves_needed': slaves_avail}
        mpi.comm.bcast(msg, root=0)

        # loop until no tasks left
        for task in tasks:

            # set tups
            tups = tuples[task]

            # get slave
            parallel.probe(mpi, TAGS.ready)

            # send tups to available slave
            mpi.comm.Send([tups, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup)

        # pi-pruning
        if tasks_pi is not None:

            # loop until no tasks left
            for task in tasks_pi:

                # set tups
                tups = tuples_pi[task]

                # get slave
                parallel.probe(mpi, TAGS.ready)

                # send tups to available slave
                mpi.comm.Send([tups, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup_pi)

        # seed
        if tasks_seed is not None:

            # loop until no tasks left
            for task in tasks_seed:

                # set tups
                tups = tuples_seed[task]

                # get slave
                parallel.probe(mpi, TAGS.ready)

                # send tups to available slave
                mpi.comm.Send([tups, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup_seed)

        # seed w/ pi-pruning
        if tasks_seed_pi is not None:

            # loop until no tasks left
            for task in tasks_seed_pi:

                # set tups
                tups = tuples_seed_pi[task]

                # get slave
                parallel.probe(mpi, TAGS.ready)

                # send tups to available slave
                mpi.comm.Send([tups, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup_seed_pi)

        # done with all tasks
        while slaves_avail > 0:

            # get slave
            parallel.probe(mpi, TAGS.ready)

            # send exit signal to slave
            mpi.comm.send(None, dest=mpi.stat.source, tag=TAGS.exit)

            # remove slave
            slaves_avail -= 1

        # init child tuples array
        if calc.extra['pi_prune'] and exp.order == 1:
            child_tup = tools.pi_pairs_deg(calc.exp_space['pi_orbs'], calc.exp_space['tot'])
        else:
            child_tup = np.array([], dtype=np.int32)

        # allgather number of child tuples
        recv_counts = np.array(mpi.comm.allgather(child_tup.size))

        # no child tuples - expansion is converged
        if np.sum(recv_counts) == 0:
            return None, None, None

        # gatherv all child tuples
        tuples_new = parallel.gatherv(mpi, child_tup, recv_counts)

        # reshape tuples
        tuples_new = tuples_new.reshape(-1, exp.order+1)

        # allocate hashes
        hashes_win = MPI.Win.Allocate_shared(8 * tuples_new.shape[0], 8, comm=mpi.comm)
        buf = hashes_win.Shared_query(0)[0]
        hashes_new = np.ndarray(buffer=buf, dtype=np.int64, shape=(tuples_new.shape[0],))

        # compute hashes
        hashes_new[:] = tools.hash_2d(tuples_new)

        # sort tuples wrt hashes
        tuples_new = tuples_new[hashes_new.argsort()]

        # sort hashes
        hashes_new.sort()

        # mpi barrier
        mpi.comm.Barrier()

        return hashes_win, hashes_new.size, tuples_new


def slave(mpi, calc, exp, slaves_needed):
        """
        this slave function returns an array of child tuple hashes

        :param mpi: pymbe mpi object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :param slaves_needed: the maximum number of required slaves. integer
        :return: MPI window handle to numpy array of shape (n_child_tup,) [hashes],
                 integer [n_tasks]
        """
        # init list of child tuples
        child_tup = []

        # send availability to master
        if mpi.rank <= slaves_needed:
            mpi.comm.send(None, dest=0, tag=TAGS.ready)

        # load increments for current order
        buf = exp.prop[calc.target]['inc'][-1].Shared_query(0)[0]
        inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tasks[-1],))

        # load hashes for current and previous orders
        hashes = []
        for k in range(exp.order-exp.min_order+1):
            buf = exp.hashes[k].Shared_query(0)[0]
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tasks[k],)))

        # receive work from master
        while True:

            # early exit in case of large proc count
            if mpi.rank > slaves_needed:
                break

            # probe for task
            mpi.comm.Probe(source=0, tag=MPI.ANY_TAG, status=mpi.stat)

            # do task
            if mpi.stat.tag in [TAGS.tup, TAGS.tup_pi, TAGS.tup_seed, TAGS.tup_seed_pi]:

                # set tup_order
                tup_order = exp.order
                if mpi.stat.tag in [TAGS.tup_pi, TAGS.tup_seed_pi]:
                    tup_order -= 1

                # get number of elements in tups
                n_elms = mpi.stat.Get_elements(MPI.INT)

                # init tups
                tups = np.empty([n_elms // tup_order, tup_order], dtype=np.int32)

                # receive tups
                mpi.comm.Recv([tups, MPI.INT], source=0, tag=mpi.stat.tag)

                # loop over tups
                for tup in tups:

                    # spawn child tuples from parent tuples at exp.order
                    orbs = _orbs(calc.occup, calc.mo_energy, calc.orbsym, calc.prot, \
                                    calc.thres, calc.ref_space, calc.exp_space, exp.min_order, \
                                    tup_order, hashes[-1], inc, \
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
                mpi.comm.send(None, dest=0, tag=TAGS.ready)

            elif mpi.stat.tag == TAGS.exit:

                # exit
                mpi.comm.recv(None, source=0, tag=TAGS.exit)
                break

        # recast child tuples as array
        child_tup = np.array(child_tup, dtype=np.int32)

        # allgather number of child tuples
        recv_counts = np.array(mpi.comm.allgather(child_tup.size))

        # no child tuples - expansion is converged
        if np.sum(recv_counts) == 0:
            return None, None

        # gatherv all child tuples
        child_tup = parallel.gatherv(mpi, child_tup, recv_counts)

        # get handle to hashes
        hashes_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.comm)

        # mpi barrier
        mpi.comm.Barrier()

        return hashes_win, int(np.sum(recv_counts)) // (exp.order+1)


def _set_screen(mpi, calc, exp):
        """
        this function returns number of available slave, various tuples, and various task arrays

        :param mpi: pymbe mpi object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: integer [slaves_avail],
                 numpy array of shape (n_tuples, order) [tuples],
                 list of numpy arrays of various shapes [tasks],
                 numpy array of shape (n_tuples_pi, order-1) or None [tuples_pi] depending on pi-pruning,
                 list of numpy arrays of various shapes or None [tasks_pi] depending on pi-pruning,
                 numpy array of shape (n_tuples_seed, order) or None [tuples_seed] depending on ref_space,
                 list of numpy arrays of various shapes or None [tasks_seed] depending on ref_space,
                 numpy array of shape (n_tuples_seed_pi, order-1) or None [tuples_seed_pi] depending on ref_space and pi-pruning,
                 list of numpy arrays of various shapes or None [tasks_seed_pi] depending on ref_space and pi-pruning,
        """
        # set main task tuples
        tuples = exp.tuples

        # number of tasks
        n_tasks = tuples.shape[0]

        # option to treat pi-orbitals independently
        tuples_pi = tasks_pi = None; n_tasks_pi = 0

        if calc.extra['pi_prune'] and exp.min_order < exp.order:

            # set tuples_pi
            tuples_pi = np.unique(exp.tuples[:, :-1], axis=0)

            # prune combinations without a mix of occupied and virtual orbitals
            tuples_pi = tuples_pi[np.fromiter(map(functools.partial(tools.corr_prune, calc.occup), tuples_pi), \
                                          dtype=bool, count=tuples_pi.shape[0])]

            # prune combinations that contain non-degenerate pairs of pi-orbitals
            tuples_pi = tuples_pi[np.fromiter(map(functools.partial(tools.pi_prune, \
                                                calc.exp_space['pi_orbs'], \
                                                calc.exp_space['pi_hashes']), tuples_pi), \
                                                dtype=bool, count=tuples_pi.shape[0])]

            # number of tasks
            n_tasks_pi = tuples_pi.shape[0]

        # potential seed for vacuum reference spaces
        tuples_seed = tasks_seed = None; n_tasks_seed = 0
        tuples_seed_pi = tasks_seed_pi = None; n_tasks_seed_pi = 0

        if calc.ref_space.size == 0 and exp.order <= calc.exp_space['occ'].size:

            # set tuples_seed
            tuples_seed = np.array([tup for tup in itertools.combinations(calc.exp_space['occ'], exp.order)], \
                                    dtype=np.int32)

            # prune combinations that contain non-degenerate pairs of pi-orbitals
            if calc.extra['pi_prune']:
                tuples_seed = tuples_seed[np.fromiter(map(functools.partial(tools.pi_prune, \
                                                        calc.exp_space['pi_orbs'], \
                                                        calc.exp_space['pi_hashes']), tuples_seed), \
                                                        dtype=bool, count=tuples_seed.shape[0])]

            # number of tasks
            n_tasks_seed = tuples_seed.shape[0]

            # option to treat pi-orbitals independently
            if calc.extra['pi_prune']:

                # set tuples_seed_pi
                tuples_seed_pi = np.array([tup for tup in itertools.combinations(calc.exp_space['occ'], exp.order-1)], \
                                           dtype=np.int32)

                # prune combinations that contain non-degenerate pairs of pi-orbitals
                tuples_seed_pi = tuples_seed_pi[np.fromiter(map(functools.partial(tools.pi_prune, \
                                                              calc.exp_space['pi_orbs'], \
                                                              calc.exp_space['pi_hashes']), tuples_seed_pi), \
                                                              dtype=bool, count=tuples_seed_pi.shape[0])]

                # number of tasks
                n_tasks_seed_pi = tuples_seed_pi.shape[0]

        # number of available slaves
        slaves_avail = min(mpi.size - 1, n_tasks + n_tasks_pi + n_tasks_seed + n_tasks_seed_pi)

        # make arrays of individual tasks
        tasks = tools.tasks(n_tasks, slaves_avail, mpi.task_size)
        if n_tasks_pi > 0:
            tasks_pi = tools.tasks(n_tasks_pi, slaves_avail, mpi.task_size)
        if n_tasks_seed > 0:
            tasks_seed = tools.tasks(n_tasks_seed, slaves_avail, mpi.task_size)
        if n_tasks_seed_pi > 0:
            tasks_seed_pi = tools.tasks(n_tasks_seed_pi, slaves_avail, mpi.task_size)

        return slaves_avail, tuples, tasks, tuples_pi, tasks_pi, \
                tuples_seed, tasks_seed, tuples_seed_pi, tasks_seed_pi


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
        combs = np.array([comb for comb in itertools.combinations(tup, order-1)], dtype=np.int32)

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
        orb_arr = np.empty([combs.shape[0], 2 if pi_gen else 1], dtype=np.int32)

        # loop over orbitals of truncated expansion space
        for orb in exp_space_trunc:

            # add orbital(s) to combinations
            orb_arr[:] = orb
            combs_orb = np.concatenate((combs, orb_arr), axis=1)

            # convert to sorted hashes
            combs_orb_hash = tools.hash_2d(combs_orb)
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

        return np.array(child_orbs, dtype=np.int32)


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
        :param hashes: hashes to all orders. numpy array of shape (n_tuples,)
        :param prop: property increments to all orders. numpy array of shape (n_tuples,)
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


def _thres(occup, thres, ref_space, scheme, tup):
        """
        this function computes the screening threshold for the given tuple of orbitals

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param thres: threshold settings. dict
        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param scheme: protocol scheme. integer
        :param tup: current orbital tuple. numpy array of shape (order,)
        :return: scalar
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


