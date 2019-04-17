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
TAGS = tools.enum('start', 'data', 'ready', 'exit')


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
			if exp.order > 1:
				exp.prop[calc.target]['tot'][-1] += exp.prop[calc.target]['tot'][-2]
		else:
			# slave function
			ndets, inc = _slave(mpi, mol, calc, exp)
		# append increments and ndets
		exp.prop[calc.target]['inc'].append(inc)
		exp.ndets.append(ndets)


def _master(mpi, mol, calc, exp):
		""" master function """
		# wake up slaves
		msg = {'task': 'mbe', 'order': exp.order}
		mpi.comm.bcast(msg, root=0)
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, exp.tuples[-1].shape[0])
		# number of tasks
		n_tasks = exp.tuples[-1].shape[0]
		# init request
		req = MPI.Request()
		# start index
		i = 0
		# loop until no tasks left
		while True:
			# probe for available slaves
			if mpi.comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive slave status
				req = mpi.comm.Irecv([None, MPI.INT], source=mpi.stat.source, tag=TAGS.ready)
				# any tasks left?
				if i < n_tasks:
					# send task idx
					mpi.comm.Isend([np.array([i], dtype=np.int32), MPI.INT], \
									dest=mpi.stat.source, tag=TAGS.start)
					# get core and cas indices
					core_idx, cas_idx = tools.core_cas(mol, calc.ref_space, exp.tuples[-1][i])
					# get h2e indices
					cas_idx_tril = tools.cas_idx_tril(cas_idx)
					# send h2e_cas 
					mpi.comm.Isend([mol.eri[cas_idx_tril[:, None], cas_idx_tril], MPI.DOUBLE], \
									dest=mpi.stat.source, tag=TAGS.data)
					# increment index
					i += 1
					# wait for slave status
					req.Wait()
				else:
					# send exit signal
					mpi.comm.Isend([None, MPI.INT], dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# wait for slave status
					req.Wait()
					# any slaves left?
					if slaves_avail == 0:
						# exit loop
						break
		# init increments and ndets
		inc = _init_inc(n_tasks, calc.target)
		ndets = _init_ndets(n_tasks)
		# allreduce increments
		parallel.mbe(mpi, inc, ndets)
		return ndets, inc


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# init task_idx
		task_idx = np.empty(1, dtype=np.int32)
		# number of task
		n_tasks = exp.tuples[-1].shape[0]
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, n_tasks)
		# init increments and ndets
		inc = _init_inc(n_tasks, calc.target)
		ndets = _init_ndets(n_tasks)
		# init h2e_cas
		h2e_cas = _init_h2e(calc.ref_space, exp.order)
		# send availability to master
		if mpi.rank <= num_slaves:
			mpi.comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
		# receive work from master
		while True:
			# early exit in case of large proc count
			if mpi.rank > num_slaves:
				break
			# receive task_idx
			mpi.comm.Recv([task_idx, MPI.INT], source=0, status=mpi.stat)
			# do jobs
			if mpi.stat.tag == TAGS.start:
				# receive h2e_cas
				req = mpi.comm.Irecv([h2e_cas, MPI.DOUBLE], source=0, tag=TAGS.data)
				# get core and cas indices
				core_idx, cas_idx = tools.core_cas(mol, calc.ref_space, exp.tuples[-1][task_idx[0]])
				# compute e_core and h1e_cas
				e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, mol.hcore, mol.vhf, core_idx, cas_idx)
				# wait for h2e
				req.Wait()
				# calculate increment
				if not calc.extra['pruning'] or tools.n_pi_orbs(calc.orbsym, cas_idx[-exp.order:]) % 2 == 0:
					ndets[task_idx[0]], inc[task_idx[0]] = _inc(mol, calc, exp, e_core, h1e_cas, h2e_cas, core_idx, cas_idx)
				# send availability to master
				mpi.comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
			elif mpi.stat.tag == TAGS.exit:
				# exit
				break
		# allreduce increments
		parallel.mbe(mpi, inc, ndets)
		return ndets, inc


def _inc(mol, calc, exp, e_core, h1e_cas, h2e_cas, core_idx, cas_idx):
		""" calculate increments corresponding to tup """
		# nelec
		nelec = np.asarray((np.count_nonzero(calc.occup[cas_idx] > 0.), \
							np.count_nonzero(calc.occup[cas_idx] > 1.)), dtype=np.int32)
		if np.all(calc.occup[cas_idx] == 2.) or np.all(calc.occup[cas_idx] == 0.):
			# return in case of no correlation (no occupied or no virtuals)
			if calc.target in ['energy', 'excitation']:
				return 0.0, 0.0
			else:
				return 0.0, np.zeros(3, dtype=np.float64)
		else:
			# ndets
			ndets_tup = tools.num_dets(cas_idx.size, nelec[0], nelec[1])
			# perform calc
			inc_tup = kernel.main(mol, calc, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec)
			if calc.base['method'] is not None:
				inc_tup -= kernel.main(mol, calc, e_core, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, base=True)
			inc_tup -= calc.prop['ref'][calc.target]
			if exp.order > 1:
				if np.any(inc_tup != 0.0):
					inc_tup -= _sum(calc, exp, cas_idx[-exp.order:])
			# debug print
			if mol.debug >= 1:
				print(output.mbe_debug(mol, calc, exp, ndets_tup, nelec, inc_tup, cas_idx))
			return ndets_tup, inc_tup


def _sum(calc, exp, tup):
		""" recursive summation """
		# init res
		if calc.target in ['energy', 'excitation']:
			res = 0.0
		else:
			res = np.zeros(3, dtype=np.float64)
		# compute contributions from lower-order increments
		for k in range(exp.order-1, 0, -1):
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, k)], dtype=np.int32)
			# pi-orbital pruning
			if calc.extra['pruning']:
				combs = combs[[tools.pi_orb_pruning(calc.mo_energy, calc.orbsym, combs[comb, :]) for comb in range(combs.shape[0])]]
			# convert to sorted hashes
			combs_hash = tools.hash_2d(combs)
			combs_hash.sort()
			# get indices
			indx = tools.hash_compare(exp.hashes[k-1], combs_hash)
			tools.assertion(indx is not None, 'error in recursive increment calculation (tuple not found)')
			# add up lower-order increments
			if calc.target in ['energy', 'excitation']:
				res += tools.fsum(exp.prop[calc.target]['inc'][k-1][indx])
			else:
				res += tools.fsum(exp.prop[calc.target]['inc'][k-1][indx, :])
		return res


def _init_inc(n_tuples, target):
		""" init increments array """
		if target in ['energy', 'excitation']:
			return np.zeros(n_tuples, dtype=np.float64)
		else:
			return np.zeros([n_tuples, 3], dtype=np.float64)


def _init_ndets(n_tuples):
		""" init ndets array """
		return np.zeros(n_tuples, dtype=np.float64)


def _init_h2e(ref_space, order):
		""" init cas space h2e """
		n_orb = ref_space.size + order
		return np.empty([(n_orb * (n_orb + 1)) // 2, (n_orb * (n_orb + 1)) // 2], dtype=np.float64)


