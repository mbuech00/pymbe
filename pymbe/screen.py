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
		# init child_tup list
		child_tup = []
		# add pi-orbitals if pi-pruning is requested
		if calc.extra['pruning']:
			# start index
			j = 0
			if exp.order == 1:
				# add degenerate pairs of occupied pi-orbitals
				for k in range(calc.pi_orbs.shape[0]):
					if tools.cas_occ(calc.occup, calc.ref_space, calc.pi_orbs[k]):
						child_tup += calc.pi_orbs[k].tolist()
				# number of tasks
				n_tasks_pi = 0
			else:
				# number of tasks
				n_tasks_pi = exp.tuples[-2].shape[0]
		# loop until no tasks left
		while True:
			# probe for available slaves
			mpi.comm.Probe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat)
			# receive slave status
			mpi.comm.recv(None, source=mpi.stat.source, tag=TAGS.ready)
			# any tasks left?
			if i < n_tasks:
				# send task
				mpi.comm.send({'idx': i, 'pi': False}, dest=mpi.stat.source, tag=TAGS.start)
				# increment index
				i += 1
			else:
				if calc.extra['pruning']:
					if j < n_tasks_pi:
						# send task
						mpi.comm.send({'idx': j, 'pi': True}, dest=mpi.stat.source, tag=TAGS.start)
						# increment index
						j += 1
					else:
						# send exit signal
						mpi.comm.send(None, dest=mpi.stat.source, tag=TAGS.exit)
						# remove slave
						slaves_avail -= 1
						# any slaves left?
						if slaves_avail == 0:
							# exit loop
							break
				else:
					# send exit signal
					mpi.comm.send(None, dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# any slaves left?
					if slaves_avail == 0:
						# exit loop
						break
		# allgatherv tuples
		child_tup = np.array(child_tup, dtype=np.int32)
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
			# receive task
			task = mpi.comm.recv(source=0, status=mpi.stat)
			# do jobs
			if mpi.stat.tag == TAGS.start:
				if not task['pi']:
					# child tuples wrt order k-1
					orb_lst = _orbs(mol, calc, exp, exp.tuples[-1][task['idx']], exp.order)
					if calc.extra['pruning']:
						# deep pruning wrt orders k-2, k-4, etc.
						orb_lst = _deep_pruning(mol, calc, exp, exp.tuples[-1][task['idx']], orb_lst)
					for m in orb_lst:
						child_tup += exp.tuples[-1][task['idx']].tolist() + [m]
				else:
					# child tuples wrt order k-2
					orb_lst = _orbs_pi(mol, calc, exp, exp.tuples[-2][task['idx']], exp.order-1)
					# deep pruning wrt orders k-3, k-5, etc.
					orb_lst = _deep_pruning(mol, calc, exp, exp.tuples[-2][task['idx']], orb_lst, master=True)
					# add degenerate pairs of pi-orbitals
					for m_1, m_2 in orb_lst.reshape(-1, 2):
						child_tup += exp.tuples[-2][task['idx']].tolist() + [m_1, m_2]
				mpi.comm.send(None, dest=0, tag=TAGS.ready)
			elif mpi.stat.tag == TAGS.exit:
				# exit
				break
		# allgatherv tuples
		child_tup = np.array(child_tup, dtype=np.int32)
		return parallel.screen(mpi, child_tup, exp.order)


def _orbs(mol, calc, exp, tup, order):
		""" determine list of child tuple orbitals """
		if order == 1:
			lst = [m for m in calc.exp_space[np.where(calc.exp_space > tup[order-1])]]
		else:
			# set threshold
			thres = _thres(calc.occup, calc.ref_space, calc.thres, calc.prot['scheme'], tup)
			# init return list
			lst = []
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, order-1)], dtype=np.int32)
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
				lst = [m for m in calc.exp_space[np.where(tup[-1] < calc.exp_space)]]
			else:
				# loop over new orbs 'm'
				for m in calc.exp_space[np.where(tup[-1] < calc.exp_space)]:
					# add orbital to combinations
					orb = np.empty(combs.shape[0], dtype=np.int32)
					orb[:] = m
					combs_orb = np.concatenate((combs, orb[:, None]), axis=1)
					# convert to sorted hashes
					combs_orb_hash = tools.hash_2d(combs_orb)
					combs_orb_hash.sort()
					# get indices
					idx = tools.hash_compare(exp.hashes[order-1], combs_orb_hash)
					# add orbital to lst
					if idx is not None:
						if thres == 0.0 or not _prot_screen(thres, calc.prot['scheme'], calc.target, exp.prop, order, idx):
							lst += [m]
		return np.array(lst, dtype=np.int32)


def _orbs_pi(mol, calc, exp, tup, order):
		""" determine list of child tuple pi-orbitals """
		if order == 1:
			# init return list
			lst = []
			# loop over pairs of degenerate pi-orbitals
			for j in range(calc.pi_orbs.shape[0]):
				if tup[-1] < calc.pi_orbs[j, 0]:
					lst += calc.pi_orbs[j].tolist()
		else:
			# set threshold
			thres = _thres(calc.occup, calc.ref_space, calc.thres, calc.prot['scheme'], tup)
			# init return list
			lst = []
			# generate array with all subsets of particular tuple
			combs = np.array([comb for comb in itertools.combinations(tup, order-1)], dtype=np.int32)
			# prune combinations with no occupied orbitals
			combs = combs[np.fromiter(map(functools.partial(tools.cas_occ, \
								calc.occup, calc.ref_space), combs), \
								dtype=bool, count=combs.shape[0])]
			# pi-orbital pruning
			combs = combs[np.fromiter(map(functools.partial(tools.pruning, \
								calc.mo_energy, calc.orbsym), combs), \
								dtype=bool, count=combs.shape[0])]
			if combs.size == 0:
				# loop over pairs of degenerate pi-orbitals
				for j in range(calc.pi_orbs.shape[0]):
					if tup[-1] < calc.pi_orbs[j, 0]:
						lst += calc.pi_orbs[j].tolist()
			else:
				# loop over pairs of degenerate pi-orbitals
				for j in range(calc.pi_orbs.shape[0]):
					if tup[-1] < calc.pi_orbs[j, 0]:
						# add pi-orbitals to combinations
						orb = np.empty([combs.shape[0], 2], dtype=np.int32)
						orb[:, 0] = calc.pi_orbs[j, 0]
						orb[:, 1] = calc.pi_orbs[j, 1]
						combs_orb = np.concatenate((combs, orb), axis=1)
						# convert to sorted hashes
						combs_orb_hash = tools.hash_2d(combs_orb)
						combs_orb_hash.sort()
						# get indices
						idx = tools.hash_compare(exp.hashes[(order+1)-1], combs_orb_hash)
						# add orbitals to lst
						if idx is not None:
							if thres == 0.0 or not _prot_screen(thres, calc.prot['scheme'], calc.target, exp.prop, order+1, idx):
								lst += calc.pi_orbs[j].tolist()
		return np.array(lst, dtype=np.int32)


def _prot_screen(thres, scheme, target, prop, order, idx):
		""" protocol check """
		if target in ['energy', 'excitation']:
			return _prot_scheme(thres, scheme, prop[target]['inc'][order-1][idx])
		else:
			screen = True
			for dim in range(3):
				# (x,y,z) = (0,1,2)
				if np.sum(prop[target]['inc'][order-1][idx, dim]) != 0.0:
					screen = _prot_scheme(thres, scheme, prop[target]['inc'][order-1][idx, dim])
				if not screen:
					break
			return screen


def _prot_scheme(thres, scheme, prop):
		""" screen according to chosen scheme """
		# are *any* increments below the threshold?
		if scheme == 1:
			return np.min(np.abs(prop)) < thres
		# are *all* increments below the threshold?
		elif scheme > 1:
			return np.max(np.abs(prop)) < thres


def _deep_pruning(mol, calc, exp, tup, orb_lst, master=False):
		""" deep pruning """
		# deep pruning wrt to lower orders
		for k in range(tools.n_pi_orbs(calc.orbsym, tup) // 2):
			if master:
				orb_lst = np.intersect1d(orb_lst, _orbs_pi(mol, calc, exp, tup, exp.order - (2*k+2)))
			else:
				orb_lst = np.intersect1d(orb_lst, _orbs(mol, calc, exp, tup, exp.order - (2*k+1)))
		return orb_lst


def _thres(occup, ref_space, thres, scheme, tup):
		""" set screening threshold for tup """
		nocc_ref = np.count_nonzero(occup[ref_space] > 0.0)
		nocc_tup = np.count_nonzero(occup[tup] > 0.0)
		nvirt_ref = np.count_nonzero(occup[ref_space] == 0.0)
		nvirt_tup = np.count_nonzero(occup[tup] == 0.0)
		if scheme < 3:
			if nvirt_ref + nvirt_tup < 3:
				return 0.0
			else:
				return thres['init'] * thres['relax'] ** ((nvirt_ref + nvirt_tup) - 3)
		elif scheme == 3:
			if max(nocc_tup, nvirt_tup) < 3:
				return 0.0
			else:
				if nvirt_ref + nvirt_tup == 0:
					return 0.0
				else:
					return thres['init'] * thres['relax'] ** (max(nocc_tup, nvirt_tup) - 3)


