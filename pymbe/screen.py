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


# mbe parameters
TAGS = tools.enum('start', 'ready', 'exit', 'collect')


def main(mpi, mol, calc, exp):
		""" input generation for subsequent order """
		# update expansion threshold
		exp.thres = update(calc, exp)
		# print header
		if mpi.global_master: output.screen_header(exp, exp.thres)
		# mpi parallel or serial version
		if mpi.parallel:
			if mpi.global_master:
				_master(mpi, mol, calc, exp)
				# update expansion threshold
				exp.thres = update(calc, exp)
			else:
				_slave(mpi, mol, calc, exp)
		else:
			_serial(mol, calc, exp)


def _serial(mol, calc, exp):
		""" serial version """
		# start time
		time = MPI.Wtime()
		# init child tuples list
		child_tup = []
		# screen
		if exp.count[-1] > 0:
	        # loop over parent tuples
			for i in range(len(exp.tuples[-1])):
				lst = _test(calc, exp, exp.tuples[-1][i])
				parent_tup = exp.tuples[-1][i].tolist()
				for m in lst:
					if calc.model['type'] == 'occ':
						child_tup += [m]+parent_tup
					elif calc.model['type'] == 'virt':
						child_tup += parent_tup+[m]
		# convert child tuple list to array
		tuples = np.asarray(child_tup, dtype=np.int32).reshape(-1, exp.order+1)
		# collect time
		exp.time['screen'].append(MPI.Wtime() - time)
		# when done, write to tup list if expansion has not converged
		if tuples.shape[0] > 0:
			# get hashes
			hashes = tools.hash_2d(tuples)
			# sort wrt hashes
			exp.tuples.append(tuples[hashes.argsort()])
			exp.hashes.append(np.sort(hashes))
		else:
			exp.tuples.append(np.array([], dtype=np.int32))


def _master(mpi, mol, calc, exp):
		""" master routine """
		# wake up slaves
		msg = {'task': 'screen', 'order': exp.order, 'thres': exp.thres}
		# set communicator
		comm = mpi.local_comm
		# set number of workers
		num_slaves = slaves_avail = min(mpi.local_size - 1, len(exp.tuples[-1]))
		# bcast
		comm.bcast(msg, root=0)
		# start index
		i = 0
		# init request
		req = MPI.Request()
		# start time
		time = MPI.Wtime()
		# loop until no tasks left
		while True:
			# probe for available slaves
			if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive slave status
				req = comm.Irecv([None, MPI.INT], source=mpi.stat.source, tag=TAGS.ready)
				# any tasks left?
				if i < len(exp.tuples[-1]) and exp.count[-1] > 0:
					# send index
					comm.Isend([np.array([i], dtype=np.int32), MPI.INT], dest=mpi.stat.source, tag=TAGS.start)
					# increment index
					i += 1
					# wait for completion
					req.Wait()
				else:
					# send exit signal
					comm.Isend([None, MPI.INT], dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# wait for completion
					req.Wait()
					# any slaves left?
					if slaves_avail == 0:
						# exit loop
						break
		# gather tuples
		recv_counts = np.array(comm.allgather(0), dtype=np.int32)
		tuples = np.empty(np.sum(recv_counts, dtype=np.int64), dtype=np.int32)
		comm.Gatherv(np.array([], dtype=np.int32), [tuples, recv_counts], root=0)
		tuples = tuples.reshape(-1, exp.order+1)
		# collect time
		exp.time['screen'].append(MPI.Wtime() - time)
		# finally, bcast tuples and compute hashes if expansion has not converged 
		if tuples.shape[0] > 0:
			# compute hashes
			hashes = tools.hash_2d(tuples)
			# sort wrt hashes
			tuples = tuples[hashes.argsort()]
			# bcast tuples
			parallel.tuples(tuples, comm)
			# append tuples and (sorted) hashes
			exp.tuples.append(tuples)
			exp.hashes.append(np.sort(hashes))
		else:
			exp.tuples.append(np.array([], dtype=np.int32))


def _slave(mpi, mol, calc, exp):
		""" slave routine """
		# set communicator
		comm = mpi.local_comm
		# init idx array and child_tup list
		idx = np.empty(1, dtype=np.int32)
		child_tup = []
		# send availability to master
		if mpi.local_rank <= len(exp.tuples[-1]):
			comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
		# receive work from master
		while True:
			# early exit in case of large proc count
			if mpi.local_rank > len(exp.tuples[-1]):
				break
			# receive index
			comm.Recv([idx, MPI.INT], source=0, status=mpi.stat)
			# do job
			if mpi.stat.tag == TAGS.start:
				# send availability to master
				comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
				lst = _test(calc, exp, exp.tuples[-1][idx[0]])
				parent_tup = exp.tuples[-1][idx[0]].tolist()
				for m in lst:
					if calc.model['type'] == 'occ':
						child_tup += [m]+parent_tup
					elif calc.model['type'] == 'virt':
						child_tup += parent_tup+[m]
			elif mpi.stat.tag == TAGS.exit:
				break
		# gather tuples
		child_tup = np.asarray(child_tup, dtype=np.int32)
		recv_counts = np.array(comm.allgather(child_tup.size), dtype=np.int32)
		tup_size = np.sum(recv_counts, dtype=np.int64)
		comm.Gatherv(child_tup, [None, None], root=0)
		# receive tuples
		if tup_size > 0:
			# init tuples
			tuples = np.empty(tup_size, dtype=np.int32).reshape(-1, exp.order+1)
			# receive tuples
			parallel.tuples(tuples, comm)
			# append tuples
			exp.tuples.append(tuples)
			# compute and append hashes
			hashes = tools.hash_2d(tuples)
			exp.hashes.append(hashes)


def _test(calc, exp, tup):
		""" screening test """
		if exp.thres == 0.0 or exp.order == exp.start_order:
			if calc.model['type'] == 'occ':
				return [m for m in range(calc.exp_space[0], tup[0])]
			elif calc.model['type'] == 'virt':
				return [m for m in range(tup[-1]+1, calc.exp_space[-1]+1)]
		else:
			# init return list
			lst = []
			# generate array with all subsets of particular tuple (manually adding active orbs)
			if calc.no_exp > 0:
				if calc.model['type'] == 'occ':
					combs = np.array([comb+tuple(exp.tuples[0][0]) for comb in itertools.\
										combinations(tup[:calc.no_exp], (exp.order-calc.no_exp)-1)], dtype=np.int32)
				elif calc.model['type'] == 'virt':
					combs = np.array([tuple(exp.tuples[0][0])+comb for comb in itertools.\
										combinations(tup[calc.no_exp:], (exp.order-calc.no_exp)-1)], dtype=np.int32)
			else:
				combs = np.array([comb for comb in itertools.combinations(tup, exp.order-1)], dtype=np.int32)
			# loop over new orbs 'm'
			if calc.model['type'] == 'occ':
				for m in range(calc.exp_space[0], tup[0]):
					# add orbital m to combinations
					combs_m = np.concatenate((m * np.ones(combs.shape[0], dtype=np.int32)[:, None], combs), axis=1)
					# convert to sorted hashes
					combs_m = tools.hash_2d(combs_m)
					combs_m.sort()
					# get index
					diff, left, right = tools.hash_compare(exp.hashes[-1], combs_m)
					if diff.size == combs_m.size:
						indx = left
						lst += _prot_check(exp, calc, indx, m)
			elif calc.model['type'] == 'virt':
				for m in range(tup[-1]+1, calc.exp_space[-1]+1):
					# add orbital m to combinations
					combs_m = np.concatenate((combs, m * np.ones(combs.shape[0], dtype=np.int32)[:, None]), axis=1)
					# convert to sorted hashes
					combs_m = tools.hash_2d(combs_m)
					combs_m.sort()
					# get index
					diff, left, right = tools.hash_compare(exp.hashes[-1], combs_m)
					if diff.size == combs_m.size:
						indx = left
						lst += _prot_check(exp, calc, indx, m)
			return lst


def _prot_check(exp, calc, indx, m):
		""" protocol check """
		screen = True
		for i in ['energy', 'excitation', 'dipole', 'trans']:
			if calc.target[i]:
				if i == 'energy':
					prop = exp.prop['energy']['inc'][-1][indx]
					screen = _prot_scheme(prop, exp.thres, calc.prot['scheme'])
					if not screen: break
				elif i == 'excitation':
					prop = exp.prop['excitation']['inc'][-1][indx]
					screen = _prot_scheme(prop, exp.thres, calc.prot['scheme'])
					if not screen: break
				elif i == 'dipole':
					for k in range(3):
						# (x,y,z) = (0,1,2)
						prop = exp.prop['dipole']['inc'][-1][indx, k]
						screen = _prot_scheme(prop, exp.thres, calc.prot['scheme'])
						if not screen: break
					if not screen: break
				elif i == 'trans':
					for k in range(3):
						# (x,y,z) = (0,1,2)
						prop = exp.prop['trans']['inc'][-1][indx, k]
						screen = _prot_scheme(prop, exp.thres, calc.prot['scheme'])
						if not screen: break
					if not screen: break
			if not screen: break
		if not screen:
			return [m]
		else:
			return []


def _prot_scheme(prop, thres, scheme):
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


def update(calc, exp):
		""" update expansion threshold """
		if exp.order < 3:
			return 0.0
		else:
			return calc.thres['init'] * calc.thres['relax'] ** (exp.order - 3)


