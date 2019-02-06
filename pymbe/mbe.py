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
from pyscf import symm
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
TAGS = tools.enum('start', 'ready', 'exit')


def main(mpi, mol, calc, exp):
		""" mbe phase """
		# master and slave functions
		if mpi.master:
			# start time
			time = MPI.Wtime()
			# master function
			inc = _master(mpi, mol, calc, exp)
			# collect time
			exp.time['mbe'].append(MPI.Wtime() - time)
			# count non-zero increments
			exp.count.append(np.count_nonzero(inc, axis=0 if calc.target in ['energy', 'excitation'] else 1))
			# sum up total property
			exp.prop[calc.target]['tot'].append(tools.fsum(inc))
			if exp.order > 1:
				exp.prop[calc.target]['tot'][-1] += exp.prop[calc.target]['tot'][-2]
		else:
			# slave function
			inc = _slave(mpi, mol, calc, exp)
		# append increments
		exp.prop[calc.target]['inc'].append(inc)


def _master(mpi, mol, calc, exp):
		""" master function """
		# wake up slaves
		msg = {'task': 'mbe', 'order': exp.order}
		mpi.comm.bcast(msg, root=0)
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, exp.tuples[-1].shape[0])
		# task list and number of tasks
		tasks = tools.tasks(exp.tuples[-1].shape[0], num_slaves, calc.mpi['task_size'])
		# init request
		req = MPI.Request()
		# start index
		i = 0
		# init increments
		inc = _init_inc(exp.tuples[-1].shape[0], calc.target)
		# loop until no tasks left
		while True:
			# probe for available slaves
			if mpi.comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive slave status
				req = mpi.comm.Irecv([None, MPI.INT], source=mpi.stat.source, tag=TAGS.ready)
				# any tasks left?
				if i < len(tasks):
					# send index
					mpi.comm.Isend([np.array([i], dtype=np.int32), MPI.INT], dest=mpi.stat.source, tag=TAGS.start)
					# increment index
					i += 1
					# wait for completion
					req.Wait()
				else:
					# send exit signal
					mpi.comm.Isend([None, MPI.INT], dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# wait for completion
					req.Wait()
					# any slaves left?
					if slaves_avail == 0:
						# exit loop
						break
		# allreduce increments
		parallel.mbe(mpi, inc)
		return inc


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# init idx
		idx = np.empty(1, dtype=np.int32)
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, exp.tuples[-1].shape[0])
		# task list
		tasks = tools.tasks(exp.tuples[-1].shape[0], num_slaves, calc.mpi['task_size'])
		# init increments
		inc = _init_inc(exp.tuples[-1].shape[0], calc.target)
		# send availability to master
		if mpi.rank <= num_slaves:
			mpi.comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
		# receive work from master
		while True:
			# early exit in case of large proc count
			if mpi.rank > num_slaves:
				break
			# receive index
			mpi.comm.Recv([idx, MPI.INT], source=0, status=mpi.stat)
			# do jobs
			if mpi.stat.tag == TAGS.start:
				# get task
				task = tasks[idx[0]]
				# loop over tasks
				for n, task_idx in enumerate(task):
					# send availability to master
					if n == task.size - 1:
						mpi.comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
					# calculate increments
					if not calc.extra['sigma'] or (calc.extra['sigma'] and tools.sigma_prune(calc.mo_energy, calc.orbsym, exp.tuples[-1][task_idx], mbe=True)):
						inc[task_idx] = _inc(mpi, mol, calc, exp, exp.tuples[-1][task_idx])
			elif mpi.stat.tag == TAGS.exit:
				# exit
				break
		# allreduce increments
		parallel.mbe(mpi, inc)
		return inc


def _inc(mpi, mol, calc, exp, tup):
		""" calculate increments corresponding to tup """
		# generate input
		exp.core_idx, exp.cas_idx = tools.core_cas(mol, calc.ref_space, tup)
		# perform calc
		inc_tup = kernel.main(mol, calc, exp, calc.model['method'])
		if calc.base['method'] is not None:
			inc_tup -= kernel.main(mol, calc, exp, calc.base['method'])
		inc_tup -= calc.prop['ref'][calc.target]
		if exp.order > 1:
			if np.any(inc_tup != 0.0):
				inc_tup -= _sum(calc, exp, tup, calc.target)
		# debug print
		if mol.debug >= 1:
			tup_lst = [i for i in tup]
			tup_sym = [symm.addons.irrep_id2name(mol.symmetry, i) for i in calc.orbsym[tup]]
			string = ' INC: order = {:} , tup = {:}\n'
			string += '      symmetry = {:}\n'
			form = (exp.order, tup_lst, tup_sym)
			if calc.target in ['energy', 'excitation']:
				string += '      increment for root {:} = {:.4e}\n'
				form += (calc.state['root'], inc_tup,)
			else:
				string += '      increment for root {:} = ({:.4e}, {:.4e}, {:.4e})\n'
				form += (calc.state['root'], *inc_tup,)
			print(string.format(*form))
		return inc_tup


def _sum(calc, exp, tup, target):
		""" recursive summation """
		# init res
		if target in ['energy', 'excitation']:
			res = 0.0
		else:
			res = np.zeros(3, dtype=np.float64)
		# compute contributions from lower-order increments
		for k in range(exp.order-1, 0, -1):
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, k)], dtype=np.int32)
			# sigma pruning
			if calc.extra['sigma']:
				combs = combs[[tools.sigma_prune(calc.mo_energy, calc.orbsym, combs[comb, :]) for comb in range(combs.shape[0])]]
			# convert to sorted hashes
			combs_hash = tools.hash_2d(combs)
			combs_hash.sort()
			# get indices
			indx = tools.hash_compare(exp.hashes[k-1], combs_hash)
			tools.assertion(indx is not None, 'error in recursive increment calculation (tuple not found)')
			# add up lower-order increments
			if target in ['energy', 'excitation']:
				res += tools.fsum(exp.prop[calc.target]['inc'][k-1][indx])
			else:
				res += tools.fsum(exp.prop[calc.target]['inc'][k-1][indx, :])
		return res


def _init_inc(n_tuples, target):
		""" init array of increments """
		if target in ['energy', 'excitation']:
			return np.zeros(n_tuples, dtype=np.float64)
		else:
			return np.zeros([n_tuples, 3], dtype=np.float64)


