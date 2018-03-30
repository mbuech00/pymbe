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
		# init job_info dictionary
		job_info = {}
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
		# loop until no slaves left
		while (slaves_avail >= 1):
			# receive data dict
			data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi.stat)
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
					if tasks[source-1]:
						batch = tasks[source-1].pop(0)
					else:
						batch = 1
					# store job indices
					job_info['i_s'] = i; job_info['i_e'] = min(i+batch, len(exp.tuples[-1])-1)
					# send string dict
					comm.send(job_info, dest=source, tag=TAGS.start)
					# increment job index
					i += batch
				else:
					# send exit signal
					comm.send(None, dest=source, tag=TAGS.exit)
			# receive result from slave
			elif tag == TAGS.done:
				# collect energies
				for idx, val in enumerate(data['e_inc']):
					exp.energy['inc'][-1][data['i_s']+idx] = val
				if mpi.global_master:
					# write restart files
					if data['i_e'] // exp.rst_freq > counter_rst:
						counter_rst += 1
						restart.mbe_write(calc, exp, False)
					# print status
					if data['i_e'] // 1000 > counter_stat or mol.verbose:
						counter_stat += 1
						output.mbe_status(exp, float(counter_stat * 1000) / float(len(exp.tuples[-1])))
			# put slave to sleep
			elif tag == TAGS.exit:
				slaves_avail -= 1
		# print 100.0 %
		if mpi.global_master and not mol.verbose:
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
		# init data dict
		data = {}
		# receive work from master
		while (True):
			# ready for task
			comm.send(None, dest=0, tag=TAGS.ready)
			# receive drop string
			job_info = comm.recv(source=0, tag=MPI.ANY_TAG, status=mpi.stat)
			# recover tag
			tag = mpi.stat.Get_tag()
			# do job
			if tag == TAGS.start:
				# init data['e_inc']
				data['e_inc'] = np.zeros(job_info['i_e']-job_info['i_s'], dtype=np.float64)
				# calculate energy increments
				for count, idx in enumerate(range(job_info['i_s'], job_info['i_e'])):
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
					# write info into data dict
					data['e_inc'][count] = exp.energy['inc'][-1][idx]
				# store indices
				data['i_s'] = job_info['i_s']; data['i_e'] = job_info['i_e']
				# send data back to local master
				comm.send(data, dest=0, tag=TAGS.done)
			# exit
			elif tag == TAGS.exit:
				break
		# send exit signal to master
		comm.send(None, dest=0, tag=TAGS.exit)
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


def _tasks(start, size, slaves):
		""" determine batch sizes """
		# b1
		if start < size // 10 * 8:
			b1 = max(1, ((size // 10 * 8) - start) // slaves) #  0 % - 80 %
			lst = [b1]
			new_start = size // 10 * 8
		else:
			lst = []
			new_start = start
		# b2
		if new_start < size // 20 * 19:
			b2 = max(1, ((size // 20 * 19) - new_start) // slaves // 2) #  80 % - 95 %
			lst += [b2] * 2
			new_start = size // 20 * 19
		# b4
		if new_start < size // 50 * 49:
			b4 = max(1, ((size // 50 * 49) - new_start) // slaves // 4) #  95 % - 98 %
			lst += [b4] * 4
			new_start = size // 50 * 49
		# b6
		if new_start < size // 1000 * 990:
			b6 = max(1, ((size // 1000 * 990) - new_start) // slaves // 6) #  98 % - 99.9 %
			lst += [b6] * 6
		return [lst + [1] for idx in range(slaves)]


