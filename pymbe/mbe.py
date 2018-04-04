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

import restart
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
TAGS = _enum('ready', 'done', 'exit', 'start')


def main(mpi, mol, calc, exp):
		""" energy mbe phase """
		# print header
		if mpi.global_master: output.mbe_header(exp)
		# mpi parallel or serial version
		if mpi.parallel:
			if mpi.global_master:
				_master(mpi, mol, calc, exp)
				# sum up total energy
				e_tmp = math.fsum(exp.energy['inc'][-1])
				if exp.order > exp.start_order: e_tmp += exp.energy['tot'][-1]
				# add to total energy list
				exp.energy['tot'].append(e_tmp)
			else:
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
		# init time
		if len(exp.time['mbe']) < (exp.order-exp.start_order)+1: exp.time['mbe'].append(0.0)
		# determine start index
		start = np.argmax(np.isnan(exp.energy['inc'][-1]))
		# loop over tuples
		for i in range(start, len(exp.tuples[-1])):
			# start time
			time = MPI.Wtime()
			# generate input
			exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, exp.tuples[-1][i])
			# model calc
			e_model = kernel.corr(mol, calc, exp, calc.model['METHOD']) \
						+ (calc.energy['hf'] - calc.energy['ref'])
			# base calc
			if calc.base['METHOD'] is None:
				e_base = 0.0
			else:
				e_base = kernel.corr(mol, calc, exp, calc.base['METHOD']) \
							+ (calc.energy['hf'] - calc.energy['ref_base'])
			exp.energy['inc'][-1][i] = e_model - e_base
			# calc increment
			exp.energy['inc'][-1][i] -= _sum(calc, exp, i)
			# verbose print
			if mol.verbose:
				print(' core = {0:} , cas = {1:} , e_model = {2:.4e} , e_base = {3:.4e} , e_inc = {4:.4e}'.\
						format(exp.core_idx, exp.cas_idx, e_model, e_base, exp.energy['inc'][-1][i]))
			# print status
			output.mbe_status(exp, float(i+1) / float(len(exp.tuples[-1])))
			# collect time
			exp.time['mbe'][-1] += MPI.Wtime() - time
			# write restart files
			restart.mbe_write(calc, exp, False)


def _master(mpi, mol, calc, exp):
		""" master function """
		# wake up slaves
		msg = {'task': 'mbe', 'order': exp.order}
		# set communicator
		comm = mpi.local_comm
		# number of workers
		slaves_avail = num_slaves = mpi.local_size - 1
		# bcast msg
		comm.bcast(msg, root=0)
		# init job index
		i = np.argmax(np.isnan(exp.energy['inc'][-1]))
		# init stat and restart counters
		counter_stat = i
		counter_rst = i
		# print status for START
		if mpi.global_master: output.mbe_status(exp, float(i) / float(len(exp.tuples[-1])))
		# init time
		if mpi.global_master:
			if len(exp.time['mbe']) < (exp.order - exp.start_order) + 1: exp.time['mbe'].append(0.0)
			time = MPI.Wtime()
		# init tasks
		tasks = _tasks(i, len(exp.tuples[-1]), num_slaves)
		# init job_info, data, and book-keeping array
		job_info = np.zeros(2, dtype=np.int32)
		data = np.zeros(tasks[0], dtype=np.float64) # largest possible batch
		book = np.zeros([num_slaves, 2], dtype=np.int32)
		# loop until no slaves left
		while (slaves_avail >= 1):
			# receive data
			comm.Recv(data, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi.stat)
			# collect time
			if mpi.global_master:
				exp.time['mbe'][-1] += MPI.Wtime() - time
				time = MPI.Wtime()
			# probe for source and tag
			source = mpi.stat.Get_source(); tag = mpi.stat.Get_tag()
			# slave is ready
			if tag == TAGS.ready:
				# any jobs left?
				if i <= len(exp.tuples[-1])-1:
					# batch
					batch = tasks.pop(0)
					# store job indices
					job_info[0] = i; job_info[1] = i+batch
					book[source-1, :] = job_info
					# send job info
					comm.Send([job_info, MPI.INT], dest=source, tag=TAGS.start)
					# increment job index
					i += batch
				else:
					# send exit signal
					comm.Send(np.array([], dtype=np.int32), dest=source, tag=TAGS.exit)
			# receive result from slave
			elif tag == TAGS.done:
				# collect energies
				for idx, val in enumerate(data[:(book[source-1, 1]-book[source-1, 0])]):
					exp.energy['inc'][-1][book[source-1, 0] + idx] = val
				if mpi.global_master:
					# write restart files
					if book[source-1, 1] // exp.rst_freq > counter_rst:
						counter_rst += 1
						restart.mbe_write(calc, exp, False)
					# print status
					if book[source-1, 1] // 1000 > counter_stat:
						counter_stat += 1
						output.mbe_status(exp, float(counter_stat * 1000) / float(len(exp.tuples[-1])))
			# put slave to sleep
			elif tag == TAGS.exit:
				slaves_avail -= 1
		# print 100.0 %
		if mpi.global_master:
			if len(exp.tuples[-1]) % 1000 != 0:
				output.mbe_status(exp, 1.0)
		# bcast energies
		parallel.energy(exp, comm)


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# set communicator
		comm = mpi.local_comm
		# init energies
		if len(exp.energy['inc']) < (exp.order - exp.start_order) + 1:
			inc = np.empty(len(exp.tuples[-1]), dtype=np.float64)
			inc.fill(np.nan)
			exp.energy['inc'].append(inc)
		# init job_info and data arrays
		job_info = np.zeros(2, dtype=np.int32)
		data = None
		# receive work from master
		while (True):
			# ready for task
			comm.Send(np.array([], dtype=np.float64), dest=0, tag=TAGS.ready)
			# receive job info
			comm.Recv([job_info, MPI.INT], source=0, tag=MPI.ANY_TAG, status=mpi.stat)
			# recover tag
			tag = mpi.stat.Get_tag()
			# do job
			if tag == TAGS.start:
				# init data (first time)
				if data is None: data = np.zeros(job_info[1]-job_info[0], dtype=np.float64)
				# calculate energy increments
				for count, idx in enumerate(range(job_info[0], job_info[1])):
					# generate input
					exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, exp.tuples[-1][idx])
					# perform calc
					e_model = kernel.corr(mol, calc, exp, calc.model['METHOD']) \
								+ (calc.energy['hf'] - calc.energy['ref'])
					if calc.base['METHOD'] is None:
						e_base = 0.0
					else:
						e_base = kernel.corr(mol, calc, exp, calc.base['METHOD']) \
									+ (calc.energy['hf'] - calc.energy['ref_base'])
					exp.energy['inc'][-1][idx] = e_model - e_base
					# calc increment
					exp.energy['inc'][-1][idx] -= _sum(calc, exp, idx)
					# verbose print
					if mol.verbose:
						print(' core = {0:} , cas = {1:} , e_model = {2:.4e} , e_base = {3:.4e} , e_inc = {4:.4e}'.\
								format(exp.core_idx, exp.cas_idx, e_model, e_base, exp.energy['inc'][-1][idx]))
					# write data
					data[count] = exp.energy['inc'][-1][idx]
				# send data back to local master
				comm.Send([data, job_info[1]-job_info[0], MPI.DOUBLE], dest=0, tag=TAGS.done)
			# exit
			elif tag == TAGS.exit:
				break
		# send exit signal to master
		comm.Send(np.array([], dtype=np.float64), dest=0, tag=TAGS.exit)
		# receive energies
		parallel.energy(exp, comm)


def _sum(calc, exp, idx):
		""" energy summation """
		# init res
		res = np.zeros(len(exp.energy['inc'])-1, dtype=np.float64)
		# compute contributions from lower-order increments
		for count, i in enumerate(range(exp.order-1, exp.start_order-1, -1)):
			# test if tuple is a subset
			combs = exp.tuples[-1][idx, _comb_index(exp.order, i)]
			dt = np.dtype((np.void, exp.tuples[i-exp.start_order].dtype.itemsize * \
							exp.tuples[i-exp.start_order].shape[1]))
			match = np.nonzero(np.in1d(exp.tuples[i-exp.start_order].view(dt).reshape(-1),
								combs.view(dt).reshape(-1)))[0]
			# add up lower-order increments
			res[count] = math.fsum(exp.energy['inc'][i-exp.start_order][match])
		return math.fsum(res)


def _comb_index(n, k):
		""" calculate combined index """
		count = scipy.misc.comb(n, k, exact=True)
		index = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n), k)), int,count=count * k)
		return index.reshape(-1, k)


def _tasks(start, n_tasks, procs):
		""" determine batch sizes """
		lst = []
		for i in range(n_tasks-start):
			lst += [i+1 for p in range(procs)]
			if np.sum(lst) > float(n_tasks-start):
				lst = lst[:-procs]
				lst = lst[::-1]
				lst += [1 for j in range((n_tasks-start) - int(np.sum(lst)))]
				return lst


