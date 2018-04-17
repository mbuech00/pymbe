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


def _enum(*sequential, **named):
		""" hardcoded enums
		see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
		"""
		enums = dict(zip(sequential, range(len(sequential))), **named)
		return type('Enum', (), enums)


# mbe parameters
TAGS = _enum('ready', 'done', 'exit', 'start')


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
		# init bookkeeping variables
		tmp = []; combs = []
        # loop over parent tuples
		for i in range(len(exp.tuples[-1])):
			# loop through possible orbitals to augment the combinations with
			if calc.typ == 'occupied':
				for m in range(calc.exp_space[0], exp.tuples[-1][i][0]):
					# if tuple is allowed, add to child tuple list, otherwise screen away
					if not _test(calc, exp, exp.tuples[-1][i], m):
						tmp.append(sorted(exp.tuples[-1][i].tolist()+[m]))
			elif calc.typ == 'virtual':
				for m in range(exp.tuples[-1][i][-1]+1, calc.exp_space[-1]+1):
					# if tuple is allowed, add to child tuple list, otherwise screen away
					if not _test(calc, exp, exp.tuples[-1][i], m):
						tmp.append(sorted(exp.tuples[-1][i].tolist()+[m]))
		# when done, write to tup list or mark expansion as converged
		if len(tmp) == 0:
			exp.conv_orb.append(True)
		else:
			tmp.sort()
			exp.tuples.append(np.array(tmp, dtype=np.int32))


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
		n_tasks = len(exp.tuples[-1])
		tasks = _tasks(n_tasks, num_slaves)
		# init job_info and book-keeping arrays
		job_info = np.zeros(2, dtype=np.int32)
		book = np.zeros([num_slaves, 2], dtype=np.int32)
		# init tuples
		exp.tuples.append(np.empty([0, exp.order+1], dtype=np.int32))
		# loop until no slaves left
		while (slaves_avail >= 1):
			# probe for source and tag
			comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi.stat)
			source = mpi.stat.Get_source(); tag = mpi.stat.Get_tag()
			# init data
			data = np.empty([mpi.stat.Get_elements(MPI.INT) // (exp.order+1), exp.order+1], dtype=np.int32)
			# receive data
			comm.Recv(data, source=source, tag=tag)
			# slave is ready
			if tag == TAGS.ready:
				# any jobs left?
				if i <= n_tasks-1:
					# batch
					batch = tasks.pop(0)
					# store job indices
					job_info[0] = i; job_info[1] = i+batch
					book[source-1, :] = job_info
					# send parent tuple index
					comm.Send([job_info, MPI.INT], dest=source, tag=TAGS.start)
					# increment job index
					i += batch
				else:
					# send exit signal
					comm.Send([None, MPI.INT], dest=source, tag=TAGS.exit)
			# receive result from slave
			elif tag == TAGS.done:
				# append child tuples
				exp.tuples[-1] = np.append(exp.tuples[-1], data, axis=0)
			# put slave to sleep
			elif tag == TAGS.exit:
				# remove slave
				slaves_avail -= 1
		# finally we sort the tuples or mark expansion as converged 
		if exp.tuples[-1].shape[0] == 0:
			exp.conv_orb.append(True)
			# bcast tuples
			info = {'len': 0}
			comm.bcast(info, root=0)
		else:
			# bcast tuples
			info = {'len': len(exp.tuples[-1])}
			comm.bcast(info, root=0)
			parallel.tup(exp, comm)


def _slave(mpi, mol, calc, exp):
		""" slave routine """
		# set communicator
		comm = mpi.local_comm
		# init job_info array and data list
		job_info = np.zeros(2, dtype=np.int32)
		data = []
		# receive work from master
		while (True):
			# send status to master
			comm.Send([None, MPI.INT], dest=0, tag=TAGS.ready)
			# probe for tag
			comm.Probe(source=0, tag=MPI.ANY_TAG, status=mpi.stat)
			tag = mpi.stat.Get_tag()
			# receive job info
			comm.Recv([job_info, MPI.INT], source=0, tag=tag)
			# do job
			if tag == TAGS.start:
				# re-init data
				data[:] = []
				# calculate energy increments
				for idx in range(job_info[0], job_info[1]):
					if calc.typ == 'occupied':
						for m in range(calc.exp_space[0], exp.tuples[-1][idx][0]):
							# if tuple is allowed, add to child tuple list, otherwise screen away
							if not _test(calc, exp, exp.tuples[-1][idx], m):
								data.append(sorted(exp.tuples[-1][idx].tolist()+[m]))
					elif calc.typ == 'virtual':
						for m in range(exp.tuples[-1][idx][-1]+1, calc.exp_space[-1]+1):
							# if tuple is allowed, add to child tuple list, otherwise screen away
							if not _test(calc, exp, exp.tuples[-1][idx], m):
								data.append(sorted(exp.tuples[-1][idx].tolist()+[m]))
				# send data back to master
				comm.Send([np.asarray(data, dtype=np.int32), MPI.INT], dest=0, tag=TAGS.done)
			# exit
			elif tag == TAGS.exit:
				break
		# send exit signal to master
		comm.Send([None, MPI.INT], dest=0, tag=TAGS.exit)
		# receive tuples
		info = comm.bcast(None, root=0)
		if info['len'] >= 1:
			exp.tuples.append(np.empty([info['len'], exp.order+1], dtype=np.int32))
			parallel.tup(exp, comm)


def _test(calc, exp, tup, m):
		""" screening test """
		if exp.order < 3:
			return False
		else:
			# generate list with all subsets of particular tuple that include the active orbitals
			combs = [comb for comb in itertools.combinations(tup[calc.no_exp:], (exp.order-calc.no_exp)-1)]
			# init mask_m
			mask_m = m == exp.tuples[-1][:, -1]
			# conservative protocol
			if calc.protocol == 1:
				# init screening logical
				screen = True
				# loop over subset combinations
				for j in range(len(combs)):
					# init mask_comb
					mask_comb = np.copy(mask_m)
					# compute mask_comb
					for idx, i in enumerate(range(calc.no_exp, exp.order-1)):
						mask_comb = mask_comb & (combs[j][idx]==exp.tuples[-1][:, i])
						# does it exist?
						if np.count_nonzero(mask_comb) == 0:
							# screen away
							screen = True
							break
					# is the increment above threshold?
					if np.abs(exp.energy['inc'][-1][mask_comb]) >= exp.thres:
						# mark as 'allowed'
						screen = False
			# aggressive protocol
			elif calc.protocol == 2:
				# init screening logical
				screen = False
				# loop over subset combinations
				for j in range(len(combs)):
					# init mask_comb
					mask_comb = np.copy(mask_m)
					# compute mask_comb
					for idx, i in enumerate(range(calc.no_exp, exp.order-1)):
						mask_comb = mask_comb & (combs[j][idx]==exp.tuples[-1][:, i])
						# does it exist?
						if np.count_nonzero(mask_comb) == 0:
							# screen away
							screen = True
							break
					# is the increment below threshold?
					if np.abs(exp.energy['inc'][-1][mask_comb]) < exp.thres:
						# screen away
						screen = True
						break
			return screen


def update(calc, exp):
		""" update expansion threshold """
		if exp.order < 3:
			return 0.0
		else:
			return calc.thres * calc.relax ** (exp.order - 3)


def _tasks(n_tasks, procs):
		""" determine batch sizes """
		lst = []
		for i in range(n_tasks):
			lst += [i+1 for p in range(procs)]
			if np.sum(lst) > float(n_tasks):
				lst = lst[:-procs]
				lst = lst[::-1]
				lst += [1 for j in range(n_tasks - int(np.sum(lst)))]
				return lst


