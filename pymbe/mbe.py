#!/usr/bin/env python
# -*- coding: utf-8 -*

""" mbe.py: mbe module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
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


def main(mpi, mol, calc, exp):
		""" mbe phase """
		# master and slave functions
		if mpi.master:
			# start time
			time = MPI.Wtime()
			# master function
			ndets, inc = _master(mpi, mol, calc, exp)
			# collect time
			exp.time['mbe'].append(MPI.Wtime() - time)
			# count non-zero increments
			if calc.target in ['energy', 'excitation']:
				exp.count.append(np.count_nonzero(inc))
			elif calc.target in ['dipole', 'trans']:
				exp.count.append(np.count_nonzero(np.count_nonzero(inc, axis=1)))
			# sum up total property
			exp.prop[calc.target]['tot'].append(tools.fsum(inc))
			if exp.order > exp.min_order:
				exp.prop[calc.target]['tot'][-1] += exp.prop[calc.target]['tot'][-2]
		else:
			# slave function
			inc = _slave(mpi, mol, calc, exp)
		# append increments and ndets
		exp.prop[calc.target]['inc'].append(inc)
		if mpi.master:
			exp.ndets.append(ndets)


def _master(mpi, mol, calc, exp):
		""" master function """
		# wake up slaves
		msg = {'task': 'mbe', 'order': exp.order}
		mpi.comm.bcast(msg, root=0)
		# number of tuples
		n_tuples = exp.tuples[-1].shape[0]
		# number of available slaves
		slaves_avail = min(mpi.size - 1, n_tuples)
		# init requests
		req_tup = MPI.Request()
		req_h2e = MPI.Request()
		# rank tuples wrt number of determinants (from most electrons to fewest electrons)
		ndets = np.fromiter(map(functools.partial(tools.ndets, calc.occup), \
								exp.tuples[-1]), dtype=np.float64, count=n_tuples)
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
		# init increments
		inc = _init_inc(n_tuples, calc.target)
		# allreduce increments
		parallel.mbe(mpi, inc)
		# revert back to sorting of tuples wrt hashes
		exp.tuples[-1] = exp.tuples[-1][np.argsort(np.argsort(ndets)[::-1])]
		return ndets, inc


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# number of tuples
		n_tuples = exp.hashes[-1].size
		# number of needed slaves
		slaves_needed = min(mpi.size - 1, n_tuples)
		# init tup
		tup = np.empty(exp.order, dtype=np.int32)
		# init increments
		inc = _init_inc(n_tuples, calc.target)
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
			# do jobs
			if mpi.stat.tag == TAGS.tup:
				# receive tup
				req_tup = mpi.comm.Irecv([tup, MPI.INT], source=0, tag=TAGS.tup)
				# receive h2e_cas
				req_h2e = mpi.comm.Irecv([h2e_cas, MPI.DOUBLE], source=0, tag=TAGS.h2e)
				# get core and cas indices
				req_tup.Wait()
				core_idx, cas_idx = tools.core_cas(mol, calc.ref_space, tup)
				# compute e_core and h1e_cas
				e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, mol.hcore, mol.vhf, core_idx, cas_idx)
				# get task_idx
				task_idx = tools.hash_compare(exp.hashes[-1], tools.hash_1d(tup))
				# calculate increment
				req_h2e.Wait()
				inc[task_idx] = _inc(mol, calc, exp, tup, \
										e_core, h1e_cas, h2e_cas, core_idx, cas_idx)
				# send availability to master
				mpi.comm.isend(None, dest=0, tag=TAGS.ready)
			elif mpi.stat.tag == TAGS.exit:
				# exit
				mpi.comm.irecv(None, source=0, tag=TAGS.exit)
				break
		# allreduce increments
		parallel.mbe(mpi, inc)
		return inc


def _inc(mol, calc, exp, tup, e_core, h1e_cas, h2e_cas, core_idx, cas_idx):
		""" calculate increments corresponding to tup """
		# nelec
		nelec = tools.nelec(calc.occup, cas_idx)
		# perform main calc
		inc_tup = kernel.main(mol, calc, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec)
		# perform base calc
		if calc.base['method'] is not None:
			inc_tup -= kernel.main(mol, calc, e_core, h1e_cas, h2e_cas, \
									core_idx, cas_idx, nelec, base=True)
		# subtract reference space correlation energy
		inc_tup -= calc.prop['ref'][calc.target]
		# calculate increment
		if exp.order > exp.min_order:
			if np.any(inc_tup != 0.0):
				inc_tup -= _sum(calc, exp, tup)
		# debug print
		if mol.debug >= 1:
			# ndets
			ndets_tup = tools.ndets(calc.occup, cas_idx, nelec)
			print(output.mbe_debug(mol, calc, ndets_tup, nelec, inc_tup, exp.order, cas_idx, tup))
		return inc_tup


def _sum(calc, exp, tup):
		""" recursive summation """
		# init res
		if calc.target in ['energy', 'excitation']:
			res = 0.0
		else:
			res = np.zeros(3, dtype=np.float64)
		# compute contributions from lower-order increments
		for k in range(exp.order-1, exp.min_order-1, -1):
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, k)], dtype=np.int32)
			# prune combinations that will not result in cas spaces
			# with a mix of occupied and virtual orbitals
			combs = np.array([comb for comb in combs if tools.cas_allow(calc.occup, calc.ref_space, comb)], \
								dtype=np.int32)
			if combs.size > 0:
				# convert to sorted hashes
				combs_hash = tools.hash_2d(combs)
				combs_hash.sort()
				# get indices
				idx = tools.hash_compare(exp.hashes[k-exp.min_order], combs_hash)
				tools.assertion(idx is not None, 'error in recursive increment '
													'calculation\nk = {:}\ntup:\n{:}\ncombs:\n{:}'. \
													format(k, tup, combs))
				# add up lower-order increments
				res += tools.fsum(exp.prop[calc.target]['inc'][k-exp.min_order][idx])
		return res


def _init_inc(n_tuples, target):
		""" init increments array """
		if target in ['energy', 'excitation']:
			return np.zeros(n_tuples, dtype=np.float64)
		else:
			return np.zeros([n_tuples, 3], dtype=np.float64)


def _init_h2e(ref_space, order):
		""" init cas space h2e """
		n_orb = ref_space.size + order
		return np.empty([(n_orb * (n_orb + 1)) // 2, (n_orb * (n_orb + 1)) // 2], dtype=np.float64)


