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
TAGS = _enum('ready', 'done', 'exit', 'start')


def main(mpi, mol, calc, exp):
		""" energy mbe phase """
		# print header
		if mpi.global_master: output.mbe_header(exp)
		# init energies
		if len(exp.energy['inc']) < exp.order - (exp.start_order - 1):
			exp.energy['inc'].append(np.empty(len(exp.tuples[-1]), dtype=np.float64))
		# mpi parallel or serial version
		if mpi.parallel:
			_parallel(mpi, mol, calc, exp)
			# sum up total energy
			if mpi.global_master:
				e_tmp = math.fsum(exp.energy['inc'][-1])
				if exp.order > exp.start_order: e_tmp += exp.energy['tot'][-1]
				# add to total energy list
				exp.energy['tot'].append(e_tmp)
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
			exp.energy['inc'][-1][i] -= _sum(calc, exp, exp.tuples[-1][i])
			# verbose print
			if mol.verbose:
				print(' core = {0:} , cas = {1:} , e_model = {2:.4e} , e_base = {3:.4e} , e_inc = {4:.4e}'.\
						format(exp.core_idx, exp.cas_idx, e_model, e_base, exp.energy['inc'][-1][i]))
			# print status
			output.mbe_status(exp, float(i+1) / float(len(exp.tuples[-1])))
		# collect time
		exp.time['mbe'].append(MPI.Wtime() - time)


def _parallel(mpi, mol, calc, exp):
		""" parallel function """
		# set communicator
		comm = mpi.local_comm
		# master only
		if mpi.global_master:
			# wake up slaves
			msg = {'task': 'mbe', 'order': exp.order}
			comm.bcast(msg, root=0)
			# start time
			time = MPI.Wtime()
		# determine tasks
		tasks, offsets = _task_dist(mpi, exp)
		# get start index, number of tasks, and local energy array
		start = int(offsets[mpi.local_rank])
		n_tasks = int(tasks[mpi.local_rank])
		e_inc = np.empty(n_tasks, dtype=np.float64)
		# perform tasks
		for count, idx in enumerate(range(start, start+n_tasks)):
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
			e_inc[count] = e_model - e_base
			# calc increment
			e_inc[count] -= _sum(calc, exp, exp.tuples[-1][idx])
			# verbose print
			if mol.verbose:
				print(' core = {0:} , cas = {1:} , e_model = {2:.4e} , e_base = {3:.4e} , e_inc = {4:.4e}'.\
						format(exp.core_idx, exp.cas_idx, e_model, e_base, e_inc[count]))
		# collect time
		if mpi.global_master:
			exp.time['mbe'].append(MPI.Wtime() - time)
		# bcast energies
		parallel.energy(e_inc, tasks, offsets, exp, comm)


def _sum(calc, exp, tup):
		""" energy summation """
		# init res
		res = np.zeros(len(exp.energy['inc'])-1, dtype=np.float64)
		# compute contributions from lower-order increments
		for count, i in enumerate(range(exp.order-1, exp.start_order-1, -1)):
			# generate array with all subsets of particular tuple (excluding the active orbitals)
			combs = np.array([comb for comb in itertools.combinations(tup, i)], dtype=np.int32)
			# recover datatype
			dt = np.dtype((np.void, exp.tuples[i-exp.start_order].dtype.itemsize * \
							exp.tuples[i-exp.start_order].shape[1]))
			# compute mask (from flattened views of tuples and combs)
			mask = np.in1d(exp.tuples[i-exp.start_order].view(dt).reshape(-1), combs.view(dt).reshape(-1))
			# add up lower-order increments
			res[count] = math.fsum(exp.energy['inc'][i-exp.start_order][mask])
		return math.fsum(res)


def _task_dist(mpi, exp):
		""" distribution of tasks """
		base = len(exp.energy['inc'][-1]) // mpi.local_size
		leftover = len(exp.energy['inc'][-1]) % mpi.local_size
		tasks = np.ones(mpi.local_size) * base
		tasks[:leftover] += 1
		offsets = np.zeros(mpi.local_size)
		offsets[1:] = np.cumsum(tasks)[:-1]
		return tasks, offsets


