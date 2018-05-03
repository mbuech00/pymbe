#!/usr/bin/env python
# -*- coding: utf-8 -*

""" mbe.py: mbe module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import sys
import itertools
import scipy.misc
import math

import kernel
import output
import expansion
import driver
import parallel


# mbe parameters
TAGS = parallel.enum('start', 'ready', 'exit')


def main(mpi, mol, calc, exp):
		""" mbe phase """
		# print header
		if mpi.global_master: output.mbe_header(exp)
		# init increments
		if len(exp.property['energy']['inc']) < exp.order - (exp.start_order - 1):
			exp.property['energy']['inc'].append(np.zeros(len(exp.tuples[-1]), dtype=np.float64))
			exp.property['dipmom']['inc'].append(np.zeros([len(exp.tuples[-1]), 3], dtype=np.float64))
		# sanity check
		assert exp.tuples[-1].flags['F_CONTIGUOUS']
		# mpi parallel or serial version
		if mpi.parallel:
			if mpi.global_master:
				# master
				_master(mpi, mol, calc, exp)
			else:
				# slaves
				_slave(mpi, mol, calc, exp)
				return
		else:
			_serial(mpi, mol, calc, exp)
		# sum up total quantities
		exp.property['energy']['tot'].append(math.fsum(exp.property['energy']['inc'][-1]))
		exp.property['dipmom']['tot'].append(np.zeros(3, dtype=np.float64))
		for i in range(3):
			exp.property['dipmom']['tot'][-1][i] += math.fsum(exp.property['dipmom']['inc'][-1][:, i])
		if exp.order > exp.start_order:
			exp.property['energy']['tot'][-1] += exp.property['energy']['tot'][-2]
			for i in range(3):
				exp.property['dipmom']['tot'][-1][i] += exp.property['dipmom']['tot'][-2][i]


def _serial(mpi, mol, calc, exp):
		""" serial version """
		# start time
		time = MPI.Wtime()
		# loop over tuples
		for i in range(len(exp.tuples[-1])):
			# calculate increments
			exp.property['energy']['inc'][-1][i], \
				exp.property['dipmom']['inc'][-1][i] = _inc(mpi, mol, calc, exp, exp.tuples[-1][i])
			# print status
			output.mbe_status(exp, float(i+1) / float(len(exp.tuples[-1])))
		# collect time
		exp.time['mbe'].append(MPI.Wtime() - time)


def _master(mpi, mol, calc, exp):
		""" master function """
		# set communicator
		comm = mpi.local_comm
		# wake up slaves
		msg = {'task': 'mbe', 'order': exp.order}
		comm.bcast(msg, root=0)
		# start time
		time = MPI.Wtime()
		num_slaves = slaves_avail = mpi.local_size - 1
		# init tasks
		tasks = parallel.tasks(len(exp.tuples[-1]), mpi.local_size)
		if mol.verbose: print(' tasks = {0:}'.format(tasks))
		i = 0
		# init job_info array
		job_info = np.zeros(2, dtype=np.int32)
		# distribute initial set of tasks to slaves
		for j in range(num_slaves):
			if tasks:
				# batch
				batch = tasks.pop(0)
				# store job indices
				job_info[0] = i; job_info[1] = i+batch
				# send job info
				comm.Isend([job_info, MPI.INT], dest=j+1, tag=TAGS.start)
				# increment job index
				i += batch
			else:
				# send exit signal
				comm.Isend([None, MPI.INT], dest=j+1, tag=TAGS.exit)
				# remove slave
				slaves_avail -= 1
		# init request
		req = MPI.Request()
		# loop until no tasks left
		while True:
			# probe for available slaves
			if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive slave status
				req = comm.Irecv([None, MPI.INT], source=mpi.stat.source, tag=TAGS.ready)
				# any tasks left?
				if tasks:
					# batch
					batch = tasks.pop(0)
					# store job indices
					job_info[0] = i; job_info[1] = i+batch
					# send job info
					comm.Isend([job_info, MPI.INT], dest=mpi.stat.source, tag=TAGS.start)
					# increment job index
					i += batch
					# wait for completion
					req.Wait()
				else:
					# send exit signal
					comm.Isend([None, MPI.INT], dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# any slaves left?
					if slaves_avail == 0:
						# wait for completion
						req.Wait()
						# exit loop
						break
			else:
				if tasks:
					# batch
					batch = tasks.pop()
					# store job indices
					job_info[0] = i; job_info[1] = i+batch
					# loop over tuples
					for count, idx in enumerate(range(job_info[0], job_info[1])):
						# calculate increments
						exp.property['energy']['inc'][-1][idx] = _inc(mpi, mol, calc, exp, exp.tuples[-1][idx])
					# increment job index
					i += batch
		# allreduce energies
		parallel.energy(exp, comm)
		# collect time
		exp.time['mbe'].append(MPI.Wtime() - time)


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# set communicator
		comm = mpi.local_comm
		# init job_info array
		job_info = np.zeros(2, dtype=np.int32)
		# receive work from master
		while True:
			# receive job info
			comm.Recv([job_info, MPI.INT], source=0, status=mpi.stat)
			# do job
			if mpi.stat.tag == TAGS.start:
				# loop over tuples
				for count, idx in enumerate(range(job_info[0], job_info[1])):
					# send availability to master
					if idx == max(job_info[1] - 2, job_info[0]):
						comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
					# calculate increments
					exp.property['energy']['inc'][-1][idx] = _inc(mpi, mol, calc, exp, exp.tuples[-1][idx])
			elif mpi.stat.tag == TAGS.exit:
				break
		# receive energies
		parallel.energy(exp, comm)


def _inc(mpi, mol, calc, exp, tup):
		""" calculate increments corresponding to tup """
		# generate input
		exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, tup)
		# perform calc
		e, dipmom = kernel.main(mol, calc, exp, calc.model['METHOD'])
		e_model = e + (calc.property['energy']['hf'] - calc.property['energy']['ref'])
		dipmom_model = dipmom - calc.property['dipmom']['hf']
		if calc.base['METHOD'] is None:
			e_base = 0.0
			dipmom_base = np.zeros(3, dtype=np.float64)
		else:
			e, dipmom = kernel.main(mol, calc, exp, calc.base['METHOD'])
			e_base = e + (calc.property['energy']['hf'] - calc.property['energy']['ref_base'])
			dipmom_base = dipmom - calc.property['dipmom']['hf']
		# calc increments
		e_inc = e_model - e_base
		dipmom_inc = dipmom_model - dipmom_base
		if exp.order > exp.start_order:
			e, dipmom = _sum(calc, exp, tup)
			e_inc -= e
			dipmom_inc -= dipmom
		# verbose print
		if mol.verbose:
			print(' proc = {0:} , core = {1:} , cas = {2:} , e_model = {3:.4e} , e_base = {4:.4e} , e_inc = {5:.4e}'.\
					format(mpi.local_rank, exp.core_idx, exp.cas_idx, e_model, e_base, e_inc))
		return e_inc, dipmom_inc


def _sum(calc, exp, tup):
		""" recursive summation """
		# init res
		e_res = 0.0
		dipmom_res = np.zeros(3, dtype=np.float64)
		# compute contributions from lower-order increments
		for count, i in enumerate(range(exp.order-exp.start_order, 0, -1)):
			# generate array with all subsets of particular tuple (manually adding active orbitals)
			if calc.no_exp > 0:
				combs = np.array([tuple(exp.tuples[0][0])+comb for comb in itertools.\
									combinations(tup[calc.no_exp:], i-1)], dtype=np.int32)
			else:
				combs = np.array([comb for comb in itertools.combinations(tup, i)], dtype=np.int32)
			# init masks
			mask = np.zeros(exp.tuples[i-1].shape[0], dtype=np.bool)
			# loop over subset combinations
			for j in range(combs.shape[0]):
				# update mask
				mask |= (combs[j, calc.no_exp:] == exp.tuples[i-1][:, calc.no_exp:]).all(axis=1)
			# recover indices
			mask = np.where(mask)[0]
			assert mask.size == combs.shape[0]
			# add up lower-order increments
			e_res += math.fsum(exp.property['energy']['inc'][i-1][mask])
			for j in range(3):
				dipmom_res[j] += math.fsum(exp.property['dipmom']['inc'][i-1][mask, j])
		return e_res, dipmom_res


