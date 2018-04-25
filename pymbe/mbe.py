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


def _enum(*sequential, **named):
		""" hardcoded enums
		see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
		"""
		enums = dict(zip(sequential, range(len(sequential))), **named)
		return type('Enum', (), enums)


# mbe parameters
TAGS = _enum('ready', 'exit', 'start')


def main(mpi, mol, calc, exp):
		""" energy mbe phase """
		# print header
		if mpi.global_master: output.mbe_header(exp)
		# init energies
		if len(exp.energy['inc']) < exp.order - (exp.start_order - 1):
			exp.energy['inc'].append(np.zeros(len(exp.tuples[-1]), dtype=np.float64))
		# sanity check
		assert exp.tuples[-1].flags['F_CONTIGUOUS']
		# mpi parallel or serial version
		if mpi.parallel:
			if mpi.global_master:
				# master
				_master(mpi, mol, calc, exp)
				# sum up total energy
				e_tmp = math.fsum(exp.energy['inc'][-1])
				if exp.order > exp.start_order: e_tmp += exp.energy['tot'][-1]
				# add to total energy list
				exp.energy['tot'].append(e_tmp)
			else:
				# slaves
				_slave(mpi, mol, calc, exp)
		else:
			_serial(mol, calc, exp)
			# sum up total energy
			e_tmp = math.fsum(exp.energy['inc'][-1])
			if exp.order > exp.start_order: e_tmp += exp.energy['tot'][-1]
			# add to total energy list
			exp.energy['tot'].append(e_tmp)


def _serial(mol, calc, exp):
		""" serial version """
		# start time
		time = MPI.Wtime()
		# loop over tuples
		for i in range(len(exp.tuples[-1])):
			# calculate energy increment
			exp.energy['inc'][-1][i] = _e_inc(mol, calc, exp, exp.tuples[-1][i])
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
		tasks = _tasks(len(exp.tuples[-1]), mpi.local_size)
		if mol.verbose: print(' tasks = {0:}'.format(tasks))
		i = 0
		# init job_info and book-keeping arrays
		job_info = np.zeros(2, dtype=np.int32)
		book = np.zeros([num_slaves, 2], dtype=np.int32)
		# distribute tasks to slaves
		for j in range(num_slaves):
			if tasks:
				# batch
				batch = tasks.pop(0)
				# store job indices
				job_info[0] = i; job_info[1] = i+batch
				book[j, :] = job_info
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
		req = None
		# loop until no tasks left
		while True:
			#
			# wait for Irecv
			if req is not None:
				req.Wait()
			# probe for available slaves
			if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# get source
				source = mpi.stat.Get_source()
				# receive data
				req = comm.Irecv([None, MPI.INT], source=source, tag=TAGS.ready)
				# any tasks left?
				if tasks:
					# batch
					batch = tasks.pop()
					# store job indices
					job_info[0] = i; job_info[1] = i+batch
					book[source-1, :] = job_info
					# send job info
					comm.Isend([job_info, MPI.INT], dest=source, tag=TAGS.start)
					# increment job index
					i += batch
				else:
					# send exit signal
					comm.Isend([None, MPI.INT], dest=source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# any slaves left?
					if slaves_avail == 0:
						break
			if tasks:
				# batch
				batch = tasks.pop()
				# store job indices
				job_info[0] = i; job_info[1] = i+batch
				# loop over tuples
				for count, idx in enumerate(range(job_info[0], job_info[1])):
					# calculate energy increments
					exp.energy['inc'][-1][idx] = _e_inc(mpi, mol, calc, exp, exp.tuples[-1][idx])
				# increment job index
				i += batch
		# wait for Irecv
		if req is not None:
			req.Wait()
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
			# get tag
			tag = mpi.stat.Get_tag()
			# do job
			if tag == TAGS.start:
				# loop over tuples
				for count, idx in enumerate(range(job_info[0], job_info[1])):
					# send availability to master
					if idx == max(job_info[1] - 2, job_info[0])::
						comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
					# calculate energy increments
					exp.energy['inc'][-1][idx] = _e_inc(mpi, mol, calc, exp, exp.tuples[-1][idx])
			elif tag == TAGS.exit:
				break
		# receive energies
		parallel.energy(exp, comm)


def _e_inc(mpi, mol, calc, exp, tup):
		""" calculate energy increment corresponding to tup """
		# generate input
		exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, tup)
		# perform calc
		e_model = kernel.corr(mol, calc, exp, calc.model['METHOD']) \
					+ (calc.energy['hf'] - calc.energy['ref'])
		if calc.base['METHOD'] is None:
			e_base = 0.0
		else:
			e_base = kernel.corr(mol, calc, exp, calc.base['METHOD']) \
						+ (calc.energy['hf'] - calc.energy['ref_base'])
		e_inc = e_model - e_base
		# calc increment
		if exp.order > exp.start_order:
			e_inc -= _sum(calc, exp, tup)
		# verbose print
		if mol.verbose:
			print(' proc = {0:} , core = {1:} , cas = {2:} , e_model = {3:.4e} , e_base = {4:.4e} , e_inc = {5:.4e}'.\
					format(mpi.local_rank, exp.core_idx, exp.cas_idx, e_model, e_base, e_inc))
		return e_inc


def _sum(calc, exp, tup):
		""" energy summation """
		# init res
		res = np.zeros(len(exp.energy['inc'])-1, dtype=np.float64)
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
			res[count] = math.fsum(exp.energy['inc'][i-1][mask])
		return math.fsum(res)


def _tasks(n_tasks, procs):
		""" determine batch sizes """
		base = n_tasks // procs
		leftover = n_tasks % procs
		tasks = [base for p in range(procs-1) if base > 0]
		tasks += [1 for i in range(leftover+base)]
#		tasks = [base for p in range(procs) if base > 0]
#		if leftover > 0:
#			if len(tasks) > 0:
#				for i in range(leftover):
#					tasks[i] += 1
#			else:
#				tasks = [1 for i in range(leftover)]
		return tasks


