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
		# wake up slaves
		msg = {'task': 'screen', 'order': exp.order}
		mpi.comm.bcast(msg, root=0)
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, exp.tuples[-1].shape[0])
		# number of tasks
		n_tasks = exp.tuples[-1].shape[0]
		# start index
		i = 0
		# loop until no tasks left
		while True:
			# probe for available slaves
			if mpi.comm.probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive slave status
				mpi.comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)
				# any tasks left?
				if i < n_tasks:
					# send index
					mpi.comm.send(i, dest=mpi.stat.source, tag=TAGS.start)
					# increment index
					i += 1
				else:
					# send exit signal
					mpi.comm.send(None, dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# any slaves left?
					if slaves_avail == 0:
						# exit loop
						break
		# init child_tup/child_hash lists
		if not calc.extra['pruning']:
			child_tup = np.array([], dtype=np.int32)
		else:
			child_tup = []
			if exp.order == 1:
				for j in range(exp.pi_orbs.shape[0]):
					child_tup += exp.pi_orbs[j].tolist()
			else:
				for i in range(exp.tuples[-2].shape[0]):
					for j in range(exp.pi_orbs.shape[0]):
						if exp.tuples[-2][i, -1] < exp.pi_orbs[j, 0]:
							child_tup += exp.tuples[-2][i].tolist() + exp.pi_orbs[j].tolist()
			child_tup = np.array(child_tup, dtype=np.int32)
		# allgatherv tuples
		return parallel.screen(mpi, child_tup, exp.order)


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# number of slaves
		num_slaves = slaves_avail = min(mpi.size - 1, exp.tuples[-1].shape[0])
		# init child_tup list
		child_tup = []
		# send availability to master
		if mpi.rank <= num_slaves:
			mpi.comm.send(None, dest=0, tag=TAGS.ready)
		# receive work from master
		while True:
			# early exit in case of large proc count
			if mpi.rank > num_slaves:
				break
			# receive index
			task_idx = mpi.comm.recv(source=0, status=mpi.stat)
			# do jobs
			if mpi.stat.tag == TAGS.start:
				# compute child tuples
				for m in _orbs(mol, calc, exp, exp.tuples[-1][task_idx]):
					child_tup += exp.tuples[-1][task_idx].tolist() + [m]
				mpi.comm.send(None, dest=0, tag=TAGS.ready)
			elif mpi.stat.tag == TAGS.exit:
				# exit
				break
		# allgatherv tuples
		child_tup = np.array(child_tup, dtype=np.int32)
		return parallel.screen(mpi, child_tup, exp.order)


def _orbs(mol, calc, exp, tup):
		""" determine list of child tuple orbitals """
		if exp.order == 1:
			return [m for m in calc.exp_space[np.where(calc.exp_space > tup[-1])]]
		else:
			# check for missing occupied orbitals
			if not tools.cas_occ(calc.occup, calc.ref_space, tup):
				return []
			# set threshold
			thres = _thres(calc.occup, calc.ref_space, calc.thres, tup)
			# init return list
			lst = []
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, exp.order-1)], dtype=np.int32)
			# prune combinations with no occupied orbitals
			combs = combs[np.fromiter(map(functools.partial(tools.cas_occ, \
								calc.occup, calc.ref_space), combs), \
								dtype=bool, count=combs.shape[0])]
			# pi-orbital pruning
			if calc.extra['pruning']:
				combs = combs[np.fromiter(map(functools.partial(tools.pruning, \
									calc.mo_energy, calc.orbsym), combs), \
									dtype=bool, count=combs.shape[0])]
				if combs.size == 0:
					# tup consists entirely of pi-orbitals, automatically allow for all child tuples
					return [m for m in calc.exp_space[np.where(calc.exp_space > tup[-1])]]
			# loop over new orbs 'm'
			for m in calc.exp_space[np.where(calc.exp_space > tup[-1])]:
				# add orbital m to combinations
				orb = np.empty(combs.shape[0], dtype=np.int32)
				orb[:] = m
				combs_orb = np.concatenate((combs, orb[:, None]), axis=1)
				# convert to sorted hashes
				combs_orb_hash = tools.hash_2d(combs_orb)
				combs_orb_hash.sort()
				# get indices
				indx = tools.hash_compare(exp.hashes[-1], combs_orb_hash)
				# add m to lst
				if indx is not None:
					if thres == 0.0:
						lst += [m]
					else:
						if not _prot_screen(thres, calc.prot['scheme'], calc.target, exp.prop, indx):
							lst += [m]
						else:
							if mol.debug >= 2:
								print('screen [prot_screen]\nparent_tup:\n{:}\nm:\n{:}\ncombs_m:\n{:}'.format(tup, m, combs_m))
				else:
					if mol.debug >= 2:
						print('screen [indx is None]\nparent_tup:\n{:}\nm:\n{:}\ncombs_m:\n{:}'.format(tup, m, combs_m))
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
		# are *all* increments below the threshold?
		if scheme == 'new':
			return np.max(np.abs(prop)) < thres
		# are *any* increments below the threshold?
		elif scheme == 'old':
			return np.min(np.abs(prop)) < thres


def _thres(occup, ref_space, thres, tup):
		""" set screening threshold for tup """
		n_virt = np.count_nonzero(occup[ref_space] == 0.)
		n_virt += np.count_nonzero(occup[tup] == 0.)
		if n_virt < 3:
			return 0.0
		else:
			return thres['init'] * thres['relax'] ** (n_virt - 3)


