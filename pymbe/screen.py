#!/usr/bin/env python
# -*- coding: utf-8 -*

""" screen.py: screening module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import functools
import itertools

import parallel
import tools
import output


# tags
TAGS = tools.enum('start', 'ready', 'exit')


def main(mpi, mol, calc, exp):
		""" input generation for subsequent order """
		# master and slave functions
		if mpi.master:
			# start time
			time = MPI.Wtime()
			# master function
			tuples, hashes = _master(mpi, mol, calc, exp)
			# collect time
			exp.time['screen'].append(MPI.Wtime() - time)
		else:
			# slave function
			tuples, hashes = _slave(mpi, mol, calc, exp)
		# append tuples and hashes
		exp.tuples.append(tuples)
		exp.hashes.append(hashes)


def _master(mpi, mol, calc, exp):
		""" master function """
		# print header
		print(output.screen_header(exp.order))
		# converged due to pi screening
		if exp.order > 1 and exp.count[-1] == 0:
			tuples = np.array([], dtype=np.int32).reshape(-1, exp.order+1)
			hashes = np.array([], dtype=np.int64)
			return tuples, hashes
		# wake up slaves
		msg = {'task': 'screen', 'order': exp.order}
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
			if mpi.comm.iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive slave status
				req = mpi.comm.irecv(None, source=mpi.stat.source, tag=TAGS.ready)
				# any tasks left?
				if i < n_tasks:
					# send index
					mpi.comm.isend(i, dest=mpi.stat.source, tag=TAGS.start)
					# increment index
					i += 1
					# wait for completion
					req.wait()
				else:
					# send exit signal
					mpi.comm.isend(None, dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# wait for completion
					req.wait()
					# any slaves left?
					if slaves_avail == 0:
						# exit loop
						break
		# init child_tup/child_hash lists
		child_tup = []; child_hash = []
		# allgatherv tuples/hashes
		return parallel.screen(mpi, child_tup, child_hash, exp.order)


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, exp.tuples[-1].shape[0])
		# send availability to master
		if mpi.rank <= num_slaves:
			mpi.comm.isend(None, dest=0, tag=TAGS.ready)
		# init child_tup/child_hash lists
		child_tup = []; child_hash = []
		# receive work from master
		while True:
			# early exit in case of large proc count
			if mpi.rank > num_slaves:
				break
			# receive index
			task_idx = mpi.comm.recv(source=0, status=mpi.stat)
			# do jobs
			if mpi.stat.tag == TAGS.start:
				# compute child tuples/hashes
				lst = _test(mol, calc, exp, exp.tuples[-1][task_idx])
				parent_tup = exp.tuples[-1][task_idx].tolist()
				for m in lst:
					tup = parent_tup+[m]
					if not calc.extra['pruning'] or \
					tools.pi_orb_pruning(calc.mo_energy, calc.orbsym, np.asarray(tup, dtype=np.int32)):
						child_tup += tup
						child_hash.append(tools.hash_1d(np.asarray(tup, dtype=np.int32)))
					else:
						if mol.debug >= 2:
							print('screen [pi-pruned]: parent_tup = {:} , m = {:}'.format(parent_tup, m))
				mpi.comm.isend(None, dest=0, tag=TAGS.ready)
			elif mpi.stat.tag == TAGS.exit:
				# exit
				break
		# allgatherv tuples/hashes
		return parallel.screen(mpi, child_tup, child_hash, exp.order)


def _test(mol, calc, exp, tup):
		""" screening test """
		if exp.order == 1:
			return [m for m in calc.exp_space[np.where(calc.exp_space > tup[-1])]]
		else:
			# set threshold
			n_virt = np.count_nonzero(calc.occup[calc.ref_space] == 0.)
			n_virt += np.count_nonzero(calc.occup[tup] == 0.)
			if n_virt < 3:
				thres = 0.0
			else:
				thres = calc.thres['init'] * calc.thres['relax'] ** (n_virt - 3)
			# init return list
			lst = []
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, exp.order-1)], dtype=np.int32)
			# 1st pi-orbital pruning
			if calc.extra['pruning']:
				combs = combs[np.fromiter(map(functools.partial(tools.pi_orb_pruning, \
									calc.mo_energy, calc.orbsym), combs), \
									dtype=bool, count=combs.shape[0])]
			# loop over new orbs 'm'
			for m in calc.exp_space[np.where(calc.exp_space > tup[-1])]:
				# add orbital m to combinations
				combs_m = np.concatenate((combs, m * np.ones(combs.shape[0], dtype=np.int32)[:, None]), axis=1)
				# 2nd pi-orbital pruning
				if calc.extra['pruning']:
					combs_m = combs_m[np.fromiter(map(functools.partial(tools.pi_orb_pruning, \
										calc.mo_energy, calc.orbsym), combs_m), \
										dtype=bool, count=combs_m.shape[0])]
				# convert to sorted hashes
				combs_m_hash = tools.hash_2d(combs_m)
				combs_m_hash.sort()
				# get indices
				indx = tools.hash_compare(exp.hashes[-1], combs_m_hash)
				# add m to lst
				if indx is not None:
					if not _prot_screen(thres, calc.prot['scheme'], calc.target, exp.prop, indx):
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
		if target in ['energy', 'excitation']:
			return _prot_scheme(thres, scheme, prop[target]['inc'][-1][indx])
		else:
			screen = True
			for dim in range(3):
				# (x,y,z) = (0,1,2)
				if np.sum(prop[target]['inc'][-1][indx, dim]) != 0.0:
					screen = _prot_scheme(thres, scheme, prop[target]['inc'][-1][indx, dim])
				if not screen:
					break
			return screen


def _prot_scheme(thres, scheme, prop):
		""" screen according to chosen scheme """
		if np.sum(prop) == 0.0:
			return False
		else:
			# are *all* increments below the threshold?
			if scheme == 'new':
				return np.max(np.abs(prop)) < thres
			# are *any* increments below the threshold?
			elif scheme == 'old':
				return np.min(np.abs(prop)) < thres


