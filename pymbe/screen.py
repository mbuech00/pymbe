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
import output


# mbe parameters
TAGS = parallel.enum('start', 'ready', 'exit', 'collect')


def main(mpi, mol, calc, exp):
		""" input generation for subsequent order """
		# update expansion threshold
		exp.thres = update(calc, exp)
		# print header
		if mpi.global_master: output.screen_header(exp, exp.thres)
		# sanity check
		assert exp.tuples[-1].flags['F_CONTIGUOUS']
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
				if calc.typ == 'occupied':
					child_tup += [m]+parent_tup
				elif calc.typ == 'virtual':
					child_tup += parent_tup+[m]
		# convert child tuple list to array
		exp.tuples.append(np.asarray(child_tup, dtype=np.int32).reshape(-1, exp.order+1))
		# when done, write to tup list or mark expansion as converged
		exp.conv_orb.append(exp.tuples[-1].shape[0] == 0)
		if not exp.conv_orb[-1]:
			# recast tuples as Fortran order array
			exp.tuples[-1] = np.asfortranarray(exp.tuples[-1])


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
		tasks = parallel.tasks(len(exp.tuples[-1]), mpi.local_size)
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
		# register number of participating slaves
		slaves_part = slaves_avail
		# init request
		req = MPI.Request()
		# loop until no tasks left
		while True:
			# probe for available slaves
			if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive data
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
				else:
					# send exit signal
					comm.Isend([None, MPI.INT], dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# any slaves left?
					if slaves_avail == 0:
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
							if calc.typ == 'occupied':
								child_tup += [m]+parent_tup
							elif calc.typ == 'virtual':
								child_tup += parent_tup+[m]
					# increment job index
					i += batch
		# convert child tuple list to array
		exp.tuples.append(np.asarray(child_tup, dtype=np.int32).reshape(-1, exp.order+1))
		# collect child tuples from participating slaves
		while slaves_part > 0:
			# probe for available calls
			if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.collect, status=mpi.stat):
				# init tmp array
				tmp = np.empty(mpi.stat.Get_elements(MPI.INT), dtype=np.int32)
				comm.Recv(tmp, source=mpi.stat.source, tag=TAGS.collect)
				# add child tuples
				exp.tuples[-1] = np.vstack((exp.tuples[-1], tmp.reshape(-1, exp.order+1)))
				slaves_part -= 1
		# finally, bcast tuples or mark expansion as converged 
		exp.conv_orb.append(exp.tuples[-1].shape[0] == 0)
		comm.Bcast([np.asarray([exp.tuples[-1].shape[0]], dtype=np.int32), MPI.INT], root=0)
		if not exp.conv_orb[-1]:
			parallel.tup(exp, comm)


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
						if calc.typ == 'occupied':
							child_tup += [m]+parent_tup
						elif calc.typ == 'virtual':
							child_tup += parent_tup+[m]
			elif mpi.stat.tag == TAGS.exit:
				# send tuples to master
				comm.Isend([np.asarray(child_tup, dtype=np.int32), MPI.INT], dest=0, tag=TAGS.collect)
				break
		# receive tuples
		tup_size = np.empty(1, dtype=np.int32)
		comm.Bcast([tup_size, MPI.INT], root=0)
		if tup_size[0] >= 1:
			exp.tuples.append(np.empty([tup_size[0], exp.order+1], dtype=np.int32))
			parallel.tup(exp, comm)


def _test(calc, exp, tup):
		""" screening test """
		if exp.order < 3:
			if calc.typ == 'occupied':
				return [m for m in range(calc.exp_space[0], tup[0])]
			elif calc.typ == 'virtual':
				return [m for m in range(tup[-1]+1, calc.exp_space[-1]+1)]
		else:
			# init return list
			lst = []
			# generate array with all subsets of particular tuple (manually adding active orbitals)
			if calc.no_exp > 0:
				combs = np.array([tuple(exp.tuples[0][0])+comb for comb in itertools.\
									combinations(tup[calc.no_exp:], (exp.order-exp.start_order)-1)], dtype=np.int32)
			else:
				combs = np.array([comb for comb in itertools.combinations(tup, exp.order-exp.start_order)], dtype=np.int32)
			# init masks
			mask_i = np.ones(exp.tuples[-1].shape[0], dtype=np.bool)
			mask_j = np.zeros(exp.tuples[-1].shape[0], dtype=np.bool)
			# loop over subset combinations
			for j in range(combs.shape[0]):
				# re-init mask_i
				mask_i.fill(True)
				# compute mask_c
				for i in range(calc.no_exp, exp.order-1):
					mask_i &= combs[j, i] == exp.tuples[-1][:, i]
					if not mask_i.any():
						return []
				# update mask
				mask_j |= mask_i
			# loop over new orbitals 'm'
			if calc.typ == 'occupied':
				for m in range(calc.exp_space[0], tup[0]):
					raise NotImplementedError('pymbe/screen.py: _test()')
			elif calc.typ == 'virtual':
				for m in range(tup[-1]+1, calc.exp_space[-1]+1):
					# get index
					indx = np.where(mask_j & (np.int32(m) == exp.tuples[-1][:, -1]))[0]
					if (indx.size+calc.no_exp) == exp.order:
						lst += _prot(exp, calc.protocol, indx, m)
			return lst


def _prot(exp, prot, indx, m):
		""" protocol check """
		if indx.size == 0:
			return []
		else:
			# conservative protocol
			if prot == 1:
				# are *all* increments below the threshold?
				if np.all(np.abs(exp.energy['inc'][-1][indx]) < exp.thres):
					return []
				else:
					return [m]
			# aggressive protocol
			elif prot == 2:
				# are *any* increments below the threshold?
				if np.any(np.abs(exp.energy['inc'][-1][indx]) < exp.thres):
					return []
				else:
					return [m]


def update(calc, exp):
		""" update expansion threshold """
		if exp.order < 3:
			return 0.0
		else:
			return calc.thres * calc.relax ** (exp.order - 3)


