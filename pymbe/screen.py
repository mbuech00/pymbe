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
		# init child tuples list
		child_tup = []
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
		exp.tuples.append(np.asarray(child_tup, dtype=np.int32).reshape(-1, exp.order+1))
		# when done, write to tup list if expansion has not converged
		if exp.tuples[-1].shape[0] > 0:
			# get hashes
			exp.hashes.append(tools.hash_2d(exp.tuples[-1]))
			# sort wrt hashes
			exp.tuples[-1] = exp.tuples[-1][exp.hashes[-1].argsort()]
			exp.hashes[-1].sort()


def _master(mpi, mol, calc, exp):
		""" master routine """
		# wake up slaves
		msg = {'task': 'screen', 'order': exp.order, 'thres': exp.thres}
		# set communicator
		comm = mpi.local_comm
		# set number of workers
		slaves_avail = num_slaves = mpi.local_size - 1
		# bcast
		comm.bcast(msg, root=0)
		# start index
		i = 0
		# init tasks
		tasks = tools.tasks(len(exp.tuples[-1]), mpi.local_size)
		# init job_info array and child_tup list
		job_info = np.zeros(2, dtype=np.int32)
		child_tup = []
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
					# calculate child tuples
					for idx in range(job_info[0], job_info[1]):
						lst = _test(calc, exp, exp.tuples[-1][idx])
						parent_tup = exp.tuples[-1][idx].tolist()
						for m in lst:
							if calc.model['type'] == 'occ':
								child_tup += [m]+parent_tup
							elif calc.model['type'] == 'virt':
								child_tup += parent_tup+[m]
					# increment job index
					i += batch
		# convert child tuple list to array
		exp.tuples.append(np.asarray(child_tup, dtype=np.int32).reshape(-1, exp.order+1))
		# collect child tuples from participating slaves
		slaves_avail = num_slaves
		while slaves_avail > 0:
			# probe for source
			comm.probe(source=MPI.ANY_SOURCE, tag=TAGS.collect, status=mpi.stat)
			# init tmp array
			tmp = np.empty(mpi.stat.Get_elements(MPI.INT), dtype=np.int32)
			comm.Recv(tmp, source=mpi.stat.source, tag=TAGS.collect)
			# add child tuples
			exp.tuples[-1] = np.vstack((exp.tuples[-1], tmp.reshape(-1, exp.order+1)))
			slaves_avail -= 1
		# finally, bcast tuples and hashes if expansion has not converged 
		comm.Bcast([np.asarray([exp.tuples[-1].shape[0]], dtype=np.int32), MPI.INT], root=0)
		if exp.tuples[-1].shape[0] > 0:
			parallel.tuples(exp, comm)


def _slave(mpi, mol, calc, exp):
		""" slave routine """
		# set communicator
		comm = mpi.local_comm
		# init job_info array and child_tup list
		job_info = np.zeros(2, dtype=np.int32)
		child_tup = []
		# receive work from master
		while True:
			# receive job info
			comm.Recv([job_info, MPI.INT], source=0, status=mpi.stat)
			# do job
			if mpi.stat.tag == TAGS.start:
				# calculate child tuples
				for idx in range(job_info[0], job_info[1]):
					# send availability to master
					if idx == max(job_info[1] - 2, job_info[0]):
						comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
					lst = _test(calc, exp, exp.tuples[-1][idx])
					parent_tup = exp.tuples[-1][idx].tolist()
					for m in lst:
						if calc.model['type'] == 'occ':
							child_tup += [m]+parent_tup
						elif calc.model['type'] == 'virt':
							child_tup += parent_tup+[m]
			elif mpi.stat.tag == TAGS.exit:
				# send tuples to master
				comm.Send([np.asarray(child_tup, dtype=np.int32), MPI.INT], dest=0, tag=TAGS.collect)
				break
		# receive tuples and hashes
		tup_size = np.empty(1, dtype=np.int32)
		comm.Bcast([tup_size, MPI.INT], root=0)
		if tup_size[0] >= 1:
			exp.tuples.append(np.empty([tup_size[0], exp.order+1], dtype=np.int32))
			parallel.tuples(exp, comm)


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
				combs = np.array([tuple(exp.tuples[0][0])+comb for comb in itertools.\
									combinations(tup[calc.no_exp:], (exp.order-calc.no_exp)-1)], dtype=np.int32)
			else:
				combs = np.array([comb for comb in itertools.combinations(tup, exp.order-1)], dtype=np.int32)
			# loop over new orbs 'm'
			if calc.model['type'] == 'occ':
				for m in range(calc.exp_space[0], tup[0]):
					raise NotImplementedError('pymbe/screen.py: _test()')
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
		for i in ['energy', 'dipole', 'trans']:
			if calc.target[i]:
				if i == 'energy':
					for j in range(calc.nroots):
						prop = exp.prop['energy'][j]['inc'][-1][indx]
						screen = _prot_scheme(prop, exp.thres, calc.prot['scheme'])
						if not screen: break
				elif i == 'dipole':
					for j in range(calc.nroots):
						for k in range(3):
							# (x,y,z) = (0,1,2)
							prop = exp.prop['dipole'][j]['inc'][-1][indx, k]
							screen = _prot_scheme(prop, exp.thres, calc.prot['scheme'])
							if not screen: break
						if not screen: break
				elif i == 'trans':
					for j in range(calc.nroots-1):
						for k in range(3):
							# (x,y,z) = (0,1,2)
							prop = exp.prop['trans'][j]['inc'][-1][indx, k]
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


