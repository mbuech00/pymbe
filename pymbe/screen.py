#!/usr/bin/env python
# -*- coding: utf-8 -*

""" screen.py: screening module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import itertools

import parallel
import tools
import output


# tags
TAGS = tools.enum('start', 'ready', 'exit')


def main(mpi, mol, calc, exp):
		""" input generation for subsequent order """
		# update expansion threshold
		exp.thres = update(exp.order, calc.thres['init'], calc.thres['relax'])
		# master and slave functions
		if mpi.master:
			# master
			_master(mpi, mol, calc, exp)
		else:
			# slaves
			_slave(mpi, mol, calc, exp)
			return


def _master(mpi, mol, calc, exp):
		""" master function """
		# print header
		output.screen_header(exp, exp.thres)
		if exp.count[-1] == 0:
			# converged
			exp.tuples.append(np.array([], dtype=np.int32).reshape(-1, exp.order+1))
			exp.time['screen'].append(0.0)
			return
		# wake up slaves
		msg = {'task': 'screen', 'order': exp.order}
		mpi.comm.bcast(msg, root=0)
		# start time
		time = MPI.Wtime()
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, len(exp.tuples[-1]))
		# task list and number of tasks
		tasks = tools.tasks(len(exp.tuples[-1]), num_slaves, calc.mpi['task_size'])
		n_tasks = len(tasks)
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
		# init child_tup/child_hash lists
		child_tup = []; child_hash = []
		# allgatherv tuples/hashes
		tuples, hashes = parallel.screen(mpi, child_tup, child_hash, exp.order-calc.no_exp)
		# append tuples and hashes
		exp.tuples.append(tuples)
		exp.hashes.append(hashes)
		# collect time
		exp.time['screen'].append(MPI.Wtime() - time)


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# init idx
		idx = np.empty(1, dtype=np.int32)
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, len(exp.tuples[-1]))
		# task list
		tasks = tools.tasks(len(exp.tuples[-1]), num_slaves, calc.mpi['task_size'])
		# send availability to master
		if mpi.rank <= num_slaves:
			mpi.comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
		# init child_tup/child_hash lists
		child_tup = []; child_hash = []
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
					# compute child tuples/hashes
					lst = _test(mol, calc, exp, exp.tuples[-1][task_idx])
					parent_tup = exp.tuples[-1][task_idx].tolist()
					for m in lst:
						tup = parent_tup+[m]
						if not calc.extra['sigma'] or (calc.extra['sigma'] and tools.sigma_prune(calc.mo_energy, calc.orbsym, np.asarray(tup, dtype=np.int32))):
							child_tup += tup
							child_hash.append(tools.hash_1d(np.asarray(tup, dtype=np.int32)))
						else:
							if mol.debug >= 2:
								print('screen [sigma]: parent_tup = {:} , m = {:}'.format(parent_tup, m))
			elif mpi.stat.tag == TAGS.exit:
				# exit
				break
		# allgatherv tuples/hashes
		tuples, hashes = parallel.screen(mpi, child_tup, child_hash, exp.order-calc.no_exp)
		# append tuples and hashes
		exp.tuples.append(tuples)
		exp.hashes.append(hashes)


def _test(mol, calc, exp, tup):
		""" screening test """
		if exp.order == exp.start_order:
			return [m for m in range(tup[-1]+1, calc.exp_space[-1]+1)]
		else:
			# init return list
			lst = []
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, (exp.order-calc.no_exp)-1)], dtype=np.int32)
			# loop over new orbs 'm'
			for m in range(tup[-1]+1, calc.exp_space[-1]+1):
				# add orbital m to combinations
				combs_m = np.concatenate((combs, m * np.ones(combs.shape[0], dtype=np.int32)[:, None]), axis=1)
				# sigma pruning
				if calc.extra['sigma']:
					combs_m = combs_m[[tools.sigma_prune(calc.mo_energy, calc.orbsym, combs_m[comb, :]) for comb in range(combs_m.shape[0])]]
				# convert to sorted hashes
				combs_m_hash = tools.hash_2d(combs_m)
				combs_m_hash.sort()
				# get indices
				indx = tools.hash_compare(exp.hashes[-1], combs_m_hash)
				if calc.extra['sigma']:
					# deep pruning (to check validity of tup + [m])
					for k in range(exp.order-exp.start_order, 0, -1):
						combs_sigma = np.array([comb for comb in itertools.combinations(tup, k)], dtype=np.int32)
						# add orbital m to combinations
						combs_sigma = np.concatenate((combs_sigma, m * np.ones(combs_sigma.shape[0], dtype=np.int32)[:, None]), axis=1)
						combs_sigma = combs_sigma[[tools.sigma_prune(calc.mo_energy, calc.orbsym, combs_sigma[comb, :]) for comb in range(combs_sigma.shape[0])]]
						# convert to sorted hashes
						combs_sigma_hash = tools.hash_2d(combs_sigma)
						combs_sigma_hash.sort()
						# get indices
						indx_sigma = tools.hash_compare(exp.hashes[k], combs_sigma_hash)
						# break if disallowed
						if indx_sigma is None:
							indx = None
							break
				if indx is not None:
					if not _prot_screen(exp.thres, calc.prot['scheme'], calc.target, exp.prop, indx):
						lst += [m]
					else:
						if mol.debug >= 2:
							print('screen [prot_screen]: parent_tup = {:} , m = {:}, combs_m = {:}'.format(tup, m, combs_m))
				else:
					if mol.debug >= 2:
						print('screen [indx is None]: parent_tup = {:} , m = {:}, combs_m = {:}'.format(tup, m, combs_m))
			return lst


def _prot_screen(thres, scheme, target, prop, indx):
		""" protocol check """
		screen = True
		for t in ['energy', 'excitation', 'dipole', 'trans']:
			if target[t]:
				if t in ['energy', 'excitation']:
					screen = _prot_scheme(thres, scheme, prop[t]['inc'][-1][indx])
				elif t in ['dipole', 'trans']:
					for dim in range(3):
						# (x,y,z) = (0,1,2)
						if np.sum(prop[t]['inc'][-1][indx, dim]) != 0.0:
							screen = _prot_scheme(thres, scheme, prop[t]['inc'][-1][indx, dim])
						if not screen:
							break
				if not screen:
					break
		return screen


def _prot_scheme(thres, scheme, prop):
		""" screen according to chosen scheme """
		if np.sum(prop) == 0.0:
			# sigma pruning
			return False
		else:
			# are *all* increments below the threshold?
			if scheme == 'new':
				return np.max(np.abs(prop)) < thres
			# are *any* increments below the threshold?
			elif scheme == 'old':
				return np.min(np.abs(prop)) < thres


def update(order, thres_init, thres_relax):
		""" update expansion threshold """
		if order < 3:
			return 0.0
		else:
			return thres_init * thres_relax ** (order - 3)


