#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
screening module which handles all input generation for subsequent order
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.6'
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
TAGS = tools.enum('ready', 'tup', 'exit')


def master(mpi, exp_space, min_order, order, hashes, tuples):
		"""
		this master function returns two arrays of (i) child tuple hashes and (ii) the actual child tuples

		:param mpi: instance of the pymbe mpi class
		:param exp_space: dictionary of expansion spaces. dict of three numpy arrays with shapes (n_exp_tot,); (n_exp_occ,); (n_exp_virt)
		:param min_order: minimum (start) order. integer
		:param order: current order. integer
		:param hashes: current order hashes. numpy array of shape (n_tuples,)
		:param tuples: current order tuples. numpy array of shape (n_tuples, order)
		:return: two numpy arrays of shapes (n_child_tup,) [hashes] and (n_child_tuples, order+1) [tuples]
		"""

		# wake up slaves
		msg = {'task': 'screen', 'order': order}
		mpi.comm.bcast(msg, root=0)

		# number of tasks
		n_tasks = hashes.size
		# number of available slaves
		slaves_avail = min(mpi.size - 1, n_tasks)

		# make array of individual tasks
		tasks = tools.tasks(n_tasks, slaves_avail, mpi.task_size)

		# init child tuples array
		child_tup = np.array([], dtype=np.int32)

		# loop until no tasks left
		for task in tasks:

			# set tups
			tups = tuples[task]

			# probe for available slaves
			mpi.comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

			# receive slave status
			mpi.comm.irecv(None, source=mpi.stat.source, tag=TAGS.ready)

			# send tups
			mpi.comm.Send([tups, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup)

		# done with all tasks
		while slaves_avail > 0:

			# probe for available slaves
			mpi.comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

			# receive slave status
			mpi.comm.irecv(None, source=mpi.stat.source, tag=TAGS.ready)

			# send exit signal
			mpi.comm.isend(None, dest=mpi.stat.source, tag=TAGS.exit)

			# remove slave
			slaves_avail -= 1

		# allgather number of child tuples
		recv_counts = parallel.screen_1(mpi, child_tup.size)

		# potential seed of occupied tuples for vacuum reference spaces
		if 1 < min_order and np.sum(recv_counts) > 0:

			# generate array with all k order subsets of occupied expansion space
			tuples_occ = np.array([tup for tup in itertools.combinations(exp_space['occ'], order)], \
									dtype=np.int32)

			# recast child tuples array as list
			child_tup = []

			# loop over occupied tuples
			for tup in tuples_occ:

				# recast parent tuple as list
				tup = tup.tolist()

				# loop over valid orbitals in virtual expansion space
				for orb in exp_space['virt'][tup[-1] < exp_space['virt']]:
					child_tup += tup + [orb]

			# recast child_tup as array once again
			child_tup = np.array(child_tup, dtype=np.int32)

			# add number of child tuples to recv_counts
			recv_counts[0] = child_tup.size

		# allgatherv tuples and hashes
		return parallel.screen_2(mpi, child_tup, recv_counts, order)


def slave(mpi, occup, scheme, thres, ref_space, exp_space, min_order, order, hashes, prop):
		"""
		this slave function returns an array of child tuple hashes

		:param mpi: instance of the pymbe mpi class
		:param occup: orbital occupation. numpy array of shape (n_orbs,)
		:param scheme: protocol scheme. integer
		:param thres: threshold settings. dict 
		:param ref_space: reference space. numpy array of shape (n_ref_tot,)
		:param exp_space: dictionary of expansion spaces. dict of three numpy arrays with shapes (n_exp_tot,); (n_exp_occ,); (n_exp_virt)
		:param min_order: minimum (start) order. integer
		:param order: current order. integer
		:param hashes: current order hashes. numpy array of shape (n_tuples,)
		:param prop: current order property increments. numpy array of shape (n_tuples,)
		:return: numpy array of shape (n_child_tup,)
		"""

		# number of tasks
		n_tasks = hashes.size
		# number of needed slaves
		slaves_needed = min(mpi.size - 1, n_tasks)

		# init list of child tuples
		child_tup = []

		# send availability to master
		if mpi.rank <= slaves_needed:
			mpi.comm.isend(None, dest=0, tag=TAGS.ready)

		# receive work from master
		while True:

			# early exit in case of large proc count
			if mpi.rank > slaves_needed:
				break

			# probe for task
			mpi.comm.Probe(source=0, tag=MPI.ANY_TAG, status=mpi.stat)

			# do jobs
			if mpi.stat.tag == TAGS.tup:

				# get number of elements in tups
				n_elms = mpi.stat.Get_elements(MPI.INT)

				# init tups
				tups = np.empty([n_elms // order, order], dtype=np.int32)

				# receive tups
				mpi.comm.Recv([tups, MPI.INT], source=0, tag=TAGS.tup)

				# loop over tups
				for tup in tups:

					# spawn child tuples from parent tuples at order k-1
					orbs = _orbs(occup, scheme, thres, ref_space, exp_space, \
									min_order, order, hashes, prop, tup)

					# recast parent tuple as list
					tup = tup.tolist()

					# loop over orbitals and add to list of child tuples
					for orb in orbs:
						child_tup += tup + [orb]

				# send availability to master
				mpi.comm.isend(None, dest=0, tag=TAGS.ready)

			elif mpi.stat.tag == TAGS.exit:

				# exit
				mpi.comm.irecv(None, source=0, tag=TAGS.exit)
				break

		# recast child tuples as array
		child_tup = np.array(child_tup, dtype=np.int32)

		# allgather number of child tuples
		recv_counts = parallel.screen_1(mpi, child_tup.size)

		# allgatherv hashes
		return parallel.screen_2(mpi, child_tup, recv_counts, order)[0]


def _orbs(occup, scheme, thres, ref_space, exp_space, min_order, order, hashes, prop, tup):
		"""
		this function returns an array child tuple orbitals subject to a given screening protocol

		:param occup: orbital occupation. numpy array of shape (n_orbs,)
		:param scheme: protocol scheme. integer
		:param thres: threshold settings. dict 
		:param ref_space: reference space. numpy array of shape (n_ref_tot,)
		:param exp_space: dictionary of expansion spaces. dict of three numpy arrays with shapes (n_exp_tot,); (n_exp_occ,); (n_exp_virt)
		:param min_order: minimum (start) order. integer
		:param order: current order. integer
		:param hashes: current order hashes. numpy array of shape (n_tuples,)
		:param prop: current order property increments. numpy array of shape (n_tuples,)
		:param tup: current orbital tuple. numpy array of shape (order,)
		:return: numpy array of shape (n_child_orbs,)
		"""

		# set expansion space
		if min_order == 1:
			exp_space = exp_space['tot'][tup[-1] < exp_space['tot']] 
		elif min_order == 2:
			exp_space = exp_space['virt'][tup[-1] < exp_space['virt']] 

		# at min_order, spawn all possible child tuples
		if order == min_order:
			return np.array([orb for orb in exp_space], dtype=np.int32)

		# generate array with all k-1 order subsets of particular tuple
		combs = np.array([comb for comb in itertools.combinations(tup, order-1)], dtype=np.int32)

		# prune combinations that do not corrspond to a correlated cas spaces
		if np.any(occup[tup] == 0.0):
			combs = combs[np.fromiter(map(functools.partial(tools.cas_corr, \
											occup, ref_space), combs), \
											dtype=bool, count=combs.shape[0])]

		# init list of child orbitals
		child_orbs = []

		# loop over orbitals of expansion space
		for orb in exp_space:

			# add orbital to combinations
			orb_column = np.empty(combs.shape[0], dtype=np.int32)
			orb_column[:] = orb
			combs_orb = np.concatenate((combs, orb_column[:, None]), axis=1)

			# convert to sorted hashes
			combs_orb_hash = tools.hash_2d(combs_orb)
			combs_orb_hash.sort()

			# get indices of combinations
			idx = tools.hash_compare(hashes, combs_orb_hash)

			# only continue if child orbital is valid
			if idx is not None:

				# compute screening thresholds
				screen_thres = np.fromiter(map(functools.partial(_thres, \
									occup, thres, ref_space, scheme), combs_orb), \
									dtype=np.float64, count=idx.size)
	
				# add orbital to list of child orbitals if allowed
				if not _prot_screen(scheme, screen_thres, prop[idx]) or np.sum(screen_thres) == 0.0:
					child_orbs += [orb]

		return np.array(child_orbs, dtype=np.int32)


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


