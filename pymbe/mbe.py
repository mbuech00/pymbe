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
			if calc.prop['DIPOLE']:
				exp.property['dipole']['inc'].append(np.zeros([len(exp.tuples[-1]), 4], dtype=np.float64))
			if calc.prop['EXCITATION']:
				exp.property['excitation']['inc'].append(np.zeros(len(exp.tuples[-1]), dtype=np.float64))
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
		if calc.prop['DIPOLE']:
			exp.property['dipole']['tot'].append(np.zeros(4, dtype=np.float64))
		if calc.prop['EXCITATION']:
			exp.property['excitation']['tot'].append(math.fsum(exp.property['excitation']['inc'][-1]))
		if calc.prop['DIPOLE']:
			for i in range(3):
				exp.property['dipole']['tot'][-1][i] += math.fsum(exp.property['dipole']['inc'][-1][:, i])
			exp.property['dipole']['tot'][-1][-1] = np.sqrt(np.sum(exp.property['dipole']['tot'][-1][:3]**2))
		if exp.order > exp.start_order:
			exp.property['energy']['tot'][-1] += exp.property['energy']['tot'][-2]
			if calc.prop['DIPOLE']:
				for i in range(3):
					exp.property['dipole']['tot'][-1][i] += exp.property['dipole']['tot'][-2][i]
				exp.property['dipole']['tot'][-1][-1] = np.sqrt(np.sum(exp.property['dipole']['tot'][-1][:3]**2))
			if calc.prop['EXCITATION']:
				exp.property['excitation']['tot'][-1] += exp.property['excitation']['tot'][-2]


def _serial(mpi, mol, calc, exp):
		""" serial version """
		# start time
		time = MPI.Wtime()
		# loop over tuples
		for i in range(len(exp.tuples[-1])):
			# calculate increments
			_calc(mpi, mol, calc, exp, i)
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
					for idx in range(job_info[0], job_info[1]):
						# calculate increments
						_calc(mpi, mol, calc, exp, idx)
					# increment job index
					i += batch
		# allreduce properties
		parallel.property(calc, exp, comm)
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
				for idx in range(job_info[0], job_info[1]):
					# send availability to master
					if idx == max(job_info[1] - 2, job_info[0]):
						comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
					# calculate increments
					_calc(mpi, mol, calc, exp, idx)
			elif mpi.stat.tag == TAGS.exit:
				break
		# receive properties
		parallel.property(calc, exp, comm)


def _calc(mpi, mol, calc, exp, idx):
		""" calculate increments """
		res = _inc(mpi, mol, calc, exp, exp.tuples[-1][idx])
		exp.property['energy']['inc'][-1][idx] = res['e_corr']
		if calc.prop['DIPOLE']:
			exp.property['dipole']['inc'][-1][idx][:3] = res['dipole']
			exp.property['dipole']['inc'][-1][idx][-1] = np.sqrt(np.sum(exp.property['dipole']['inc'][-1][idx][:3]**2))
		if calc.prop['EXCITATION']:
			exp.property['excitation']['inc'][-1][idx] = res['e_exc']


def _inc(mpi, mol, calc, exp, tup):
		""" calculate increments corresponding to tup """
		# generate input
		exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, tup)
		# perform calc
		res = kernel.main(mol, calc, exp, calc.model['METHOD'])
		inc = {'e_corr': res['e_corr'] - calc.property['energy']['ref']}
		if calc.prop['DIPOLE']:
			inc['dipole'] = res['dipole'] - calc.property['dipole']['ref']
		if calc.prop['EXCITATION']:
			inc['e_exc'] = res['e_exc'] - calc.property['excitation']['ref']
		if calc.base['METHOD'] is None:
			e_base = 0.0
		else:
			res = kernel.main(mol, calc, exp, calc.base['METHOD'])
			e_base = res['e_corr']
		# calc increments
		inc['e_corr'] -= e_base
		if exp.order > exp.start_order:
			res = _sum(calc, exp, tup)
			inc['e_corr'] -= res['e_corr']
			if calc.prop['DIPOLE']:
				inc['dipole'] -= res['dipole']
			if calc.prop['EXCITATION']:
				inc['e_exc'] -= res['e_exc']
		# debug print
		if mol.debug:
			string = ' INC: proc = {:} , core = {:} , cas = {:} , e_corr = {:.4e}'
			form = (mpi.local_rank, exp.core_idx.tolist(), exp.cas_idx.tolist(), inc['e_corr'])
			if calc.prop['EXCITATION']:
				string += ' , e_exc = {:.4e}'
				form += (inc['e_exc'],)
			if calc.prop['DIPOLE']:
				string += ' , dipole = {:.4e}'
				form += (np.sqrt(np.sum(inc['dipole']**2)),)
			print(string.format(*form))
		return inc


def _sum(calc, exp, tup):
		""" recursive summation """
		# init res
		res = {'e_corr': 0.0}
		if calc.prop['DIPOLE']:
			res['dipole'] = np.zeros(3, dtype=np.float64)
		if calc.prop['EXCITATION']:
			res['e_exc'] = 0.0
		# compute contributions from lower-order increments
		for count, i in enumerate(range(exp.order-exp.start_order, 0, -1)):
			# generate array with all subsets of particular tuple (manually adding active orbs)
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
			res['e_corr'] += math.fsum(exp.property['energy']['inc'][i-1][mask])
			if calc.prop['DIPOLE']:
				for j in range(3):
					res['dipole'][j] += math.fsum(exp.property['dipole']['inc'][i-1][mask, j])
			if calc.prop['EXCITATION']:
				res['e_exc'] += math.fsum(exp.property['excitation']['inc'][i-1][mask])
		return res


