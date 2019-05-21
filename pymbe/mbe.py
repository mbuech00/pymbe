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
		req_tup = MPI.Request()
		req_h2e = MPI.Request()

		# compute number of determinants in the individual casci calculations (ignoring symmetry)
		ndets = np.fromiter(map(functools.partial(tools.ndets, calc.occup), \
								exp.tuples[-1]), dtype=np.float64, count=n_tasks)

		# rank tuples wrt number of determinants (from most electrons to fewest electrons)
		exp.tuples[-1] = exp.tuples[-1][np.argsort(ndets)[::-1]]

		# loop until no tasks left
		for tup in exp.tuples[-1]:

			# get cas indices
			cas_idx = tools.cas(calc.ref_space, tup)

			# only consider tuples with occupied and virtual orbitals
			if np.any(calc.occup[cas_idx] < 2.0) and np.any(calc.occup[cas_idx] > 0.0):

				# probe for available slaves
				mpi.comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)

				# receive slave status
				mpi.comm.irecv(None, source=mpi.stat.source, tag=TAGS.ready)

				# send tup
				req_tup.Wait()
				req_tup = mpi.comm.Isend([tup, MPI.INT], dest=mpi.stat.source, tag=TAGS.tup)

				# get h2e indices
				cas_idx_tril = tools.cas_idx_tril(cas_idx)

				# get h2e_cas
				h2e_cas = mol.eri[cas_idx_tril[:, None], cas_idx_tril]

				# send h2e_cas
				req_h2e.Wait()
				req_h2e = mpi.comm.Isend([h2e_cas, MPI.DOUBLE], dest=mpi.stat.source, tag=TAGS.h2e)

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

		# wait for all data communication to be finished
		MPI.Request.Waitall([req_tup, req_h2e])

		# revert back to sorting of tuples wrt hashes
		exp.tuples[-1] = exp.tuples[-1][np.argsort(np.argsort(ndets)[::-1])]

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
				req_tup = mpi.comm.Irecv([tup, MPI.INT], source=0, tag=TAGS.tup)

				# receive h2e_cas
				req_h2e = mpi.comm.Irecv([h2e_cas, MPI.DOUBLE], source=0, tag=TAGS.h2e)

				# get core and cas indices
				req_tup.Wait()
				core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, tup)

				# compute e_core and h1e_cas
				e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, mol.hcore, mol.vhf, core_idx, cas_idx)

				# get task_idx
				task_idx = tools.hash_compare(exp.hashes[-1], tools.hash_1d(tup))

				# calculate increment
				req_h2e.Wait()
				inc[task_idx] = _inc(mol, calc, exp, e_core, h1e_cas, h2e_cas, tup, core_idx, cas_idx)

				# send availability to master
				mpi.comm.isend(None, dest=0, tag=TAGS.ready)

			elif mpi.stat.tag == TAGS.exit:

				# exit
				mpi.comm.irecv(None, source=0, tag=TAGS.exit)
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
				inc_tup -= _sum(calc.occup, calc.ref_space, calc.target, exp.min_order, exp.order, \
								exp.prop[calc.target]['inc'], exp.hashes, tup)

		# debug print
		if mol.debug >= 1:
			print(output.mbe_debug(mol.atom, mol.symmetry, calc.orbsym, calc.state['root'], \
									tools.ndets(occup, cas_idx, nelec), \
									nelec, inc_tup, exp.order, cas_idx, tup))

		return inc_tup


def _sum(occup, ref_space, target, min_order, order, prop, hashes, tup):
		"""
		this function performs a recursive summation

		:param occup: orbital occupation. numpy array of shape (n_orbs,)
		:param ref_space: reference space. numpy array of shape (n_ref_tot,)
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

			# prune combinations that do not corrspond to a correlated cas spaces
			combs = combs[np.fromiter(map(functools.partial(tools.cas_corr, occup, ref_space), combs), \
										dtype=bool, count=combs.shape[0])]

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


