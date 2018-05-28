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

import kernel
import output
import expansion
import driver
import parallel
import tools


# mbe parameters
TAGS = tools.enum('start', 'ready', 'exit')


def main(mpi, mol, calc, exp):
		""" mbe phase """
		# print header
		if mpi.global_master: output.mbe_header(exp)
		# init increments
		if len(exp.prop['energy'][0]['inc']) < exp.order - (exp.start_order - 1):
			for i in range(calc.nroots):
				exp.prop['energy'][i]['inc'].append(np.zeros(len(exp.tuples[-1]), dtype=np.float64))
				if calc.target['dipole']:
					exp.prop['dipole'][i]['inc'].append(np.zeros([len(exp.tuples[-1]), 3], dtype=np.float64))
				if calc.target['trans']:
					if i < calc.nroots - 1:
						exp.prop['trans'][i]['inc'].append(np.zeros([len(exp.tuples[-1]), 3], dtype=np.float64))
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
		for i in range(calc.nroots):
			exp.prop['energy'][i]['tot'].append(tools.fsum(exp.prop['energy'][i]['inc'][-1]))
			if calc.target['dipole']:
				exp.prop['dipole'][i]['tot'].append(tools.fsum(exp.prop['dipole'][i]['inc'][-1]))
			if calc.target['trans']:
				if i < calc.nroots - 1:
					exp.prop['trans'][i]['tot'].append(tools.fsum(exp.prop['trans'][i]['inc'][-1]))
		if exp.order > exp.start_order:
			for i in range(calc.nroots):
				exp.prop['energy'][i]['tot'][-1] += exp.prop['energy'][i]['tot'][-2]
				if calc.target['dipole']:
					exp.prop['dipole'][i]['tot'][-1] += exp.prop['dipole'][i]['tot'][-2]
				if calc.target['trans']:
					if i < calc.nroots - 1:
						exp.prop['trans'][i]['tot'][-1] += exp.prop['trans'][i]['tot'][-2]


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
		tasks = tools.tasks(len(exp.tuples[-1]), mpi.local_size)
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
		parallel.prop(calc, exp, comm)
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
		parallel.prop(calc, exp, comm)


def _calc(mpi, mol, calc, exp, idx):
		""" calculate increments """
		res = _inc(mpi, mol, calc, exp, exp.tuples[-1][idx])
		for i in range(calc.nroots):
			exp.prop['energy'][i]['inc'][-1][idx] = res['energy'][i]
			if calc.target['dipole']:
				exp.prop['dipole'][i]['inc'][-1][idx] = res['dipole'][i]
			if calc.target['trans']:
				if i < calc.nroots - 1:
					exp.prop['trans'][i]['inc'][-1][idx] = res['trans'][i]


def _inc(mpi, mol, calc, exp, tup):
		""" calculate increments corresponding to tup """
		# generate input
		exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, tup)
		# perform calc
		res = kernel.main(mol, calc, exp, calc.model['method'])
		inc = {'energy': [res['energy'][i] - calc.prop['ref']['energy'][i] for i in range(calc.nroots)]}
		if calc.target['dipole']:
			inc['dipole'] = [res['dipole'][i] - calc.prop['ref']['dipole'][i] for i in range(calc.nroots)]
		if calc.target['trans']:
			inc['trans'] = [res['trans'][i] - calc.prop['ref']['trans'][i] for i in range(calc.nroots-1)]
		if calc.base['method'] is None:
			e_base = 0.0
		else:
			res = kernel.main(mol, calc, exp, calc.base['method'])
			e_base = res['energy'][0]
		# calc increments
		inc['energy'][0] -= e_base
		if exp.order > exp.start_order:
			res = _sum(calc, exp, tup)
			inc['energy'] = [inc['energy'][i] - res['energy'][i] for i in range(calc.nroots)]
			if calc.target['dipole']:
				inc['dipole'] = [inc['dipole'][i] - res['dipole'][i] for i in range(calc.nroots)]
			if calc.target['trans']:
				inc['trans'] = [inc['trans'][i] - res['trans'][i] for i in range(calc.nroots-1)]
		# debug print
		if mol.debug:
			string = ' INC: proc = {:} , core = {:} , cas = {:}\n'
			form = (mpi.local_rank, exp.core_idx.tolist(), exp.cas_idx.tolist())
			string += '      ground state correlation energy = {:.4e}\n'
			form += (inc['energy'][0],)
			if calc.nroots > 1:
				for i in range(1, calc.nroots):
					string += '      excitation energy for root {:} = {:.4e}\n'
					form += (i, inc['energy'][i],)
			if calc.target['dipole']:
				for i in range(calc.nroots):
					string += '      dipole moment for root {:} = ({:.4e}, {:.4e}, {:.4e})\n'
					if calc.prot['specific']:
						form += (calc.state['root'], *inc['dipole'][i],)
					else:
						form += (i, *inc['dipole'][i],)
			if calc.target['trans']:
				for i in range(1, calc.nroots):
					string += '      transition dipole moment for excitation {:} > {:} = ({:.4e}, {:.4e}, {:.4e})\n'
					if calc.prot['specific']:
						form += (0, calc.state['root'], *inc['trans'][i-1],)
					else:
						form += (0, i, *inc['trans'][i-1],)
			print(string.format(*form))
		return inc


def _sum(calc, exp, tup):
		""" recursive summation """
		# init res
		res = {'energy': [0.0 for i in range(calc.nroots)]}
		if calc.target['dipole']:
			res['dipole'] = [np.zeros(3, dtype=np.float64) for i in range(calc.nroots)]
		if calc.target['trans']:
			res['trans'] = [np.zeros(3, dtype=np.float64) for i in range(calc.nroots-1)]
		# compute contributions from lower-order increments
		for i in range(exp.order-exp.start_order, 0, -1):
			# generate array with all subsets of particular tuple (manually adding active orbs)
			if calc.no_exp > 0:
				combs = np.array([tuple(exp.tuples[0][0])+comb for comb in itertools.\
									combinations(tup[calc.no_exp:], i-1)], dtype=np.int32)
			else:
				combs = np.array([comb for comb in itertools.combinations(tup, i)], dtype=np.int32)
			# convert to sorted hashes
			combs = tools.hash_2d(combs)
			combs.sort()
			# get index
			diff, left, right = tools.hash_compare(exp.hashes[i-1], combs)
			if diff.size == combs.size:
				indx = left
				# add up lower-order increments
				for j in range(calc.nroots):
					res['energy'][j] += tools.fsum(exp.prop['energy'][j]['inc'][i-1][indx])
					if calc.target['dipole']:
						res['dipole'][j] += tools.fsum(exp.prop['dipole'][j]['inc'][i-1][indx, :])
					if calc.target['trans']:
						if j < calc.nroots - 1:
							res['trans'][j] += tools.fsum(exp.prop['trans'][j]['inc'][i-1][indx, :])
			else:
				raise RuntimeError('\nin mbe.py:_sum()\ndiff  = {:}\nleft = {:}\nright = {:}\n'.format(diff, left, right))
		return res


