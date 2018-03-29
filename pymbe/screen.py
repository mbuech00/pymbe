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
		# init job_info dictionary
		job_info = {}
		# init bookkeeping variables
		i = 0; tmp = []
		# init tasks
		tasks = _tasks(len(exp.tuples[-1]), num_slaves)
		# loop until no slaves left
		while (slaves_avail >= 1):
			# receive data dict
			data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi.stat)
			# probe for source and tag
			source = mpi.stat.Get_source(); tag = mpi.stat.Get_tag()
			# slave is ready
			if tag == TAGS.ready:
				# any jobs left?
				if i <= len(exp.tuples[-1]) - 1:
					# batch
					if tasks[source-1]:
						batch = tasks[source-1].pop(0)
					else:
						batch = 1
					# store job indices
					job_info['i_s'] = i; job_info['i_e'] = i + batch
					# send parent tuple index
					comm.send(job_info, dest=source, tag=TAGS.start)
					# increment job index
					i += batch
				else:
					# send exit signal
					comm.send(None, dest=source, tag=TAGS.exit)
			# receive result from slave
			elif tag == TAGS.done:
				# write tmp child tuple list
				tmp += data['child']
			# put slave to sleep
			elif tag == TAGS.exit:
				# remove slave
				slaves_avail -= 1
		# finally we sort the tuples or mark expansion as converged 
		if len(tmp) == 0:
			exp.conv_orb.append(True)
			# bcast tuples
			info = {'len': 0}
			comm.bcast(info, root=0)
		else:
			tmp.sort()
			exp.tuples.append(np.array(tmp, dtype=np.int32))
			# bcast tuples
			info = {'len': len(exp.tuples[-1])}
			comm.bcast(info, root=0)
			parallel.tup(exp, comm)


def _slave(mpi, mol, calc, exp):
		""" slave routine """
		# init data dict and combs list
		data = {'child': []}; combs = []
		# set communicator
		comm = mpi.local_comm
		# receive work from master
		while (True):
			# send status to master
			comm.send(None, dest=0, tag=TAGS.ready)
			# receive parent tuple
			job_info = comm.recv(source=0, tag=MPI.ANY_TAG, status=mpi.stat)
			# recover tag
			tag = mpi.stat.Get_tag()
			# do job
			if tag == TAGS.start:
				# init child tuple list
				data['child'][:] = []
				# calculate energy increments
				for idx in range(job_info['i_s'], job_info['i_e']):
					if calc.typ == 'occupied':
						for m in range(calc.exp_space[0], exp.tuples[-1][idx][0]):
							# if tuple is allowed, add to child tuple list, otherwise screen away
							if not _test(calc, exp, exp.tuples[-1][idx], m):
								data['child'].append(sorted(exp.tuples[-1][idx].tolist()+[m]))
					elif calc.typ == 'virtual':
						for m in range(exp.tuples[-1][idx][-1]+1, calc.exp_space[-1]+1):
							# if tuple is allowed, add to child tuple list, otherwise screen away
							if not _test(calc, exp, exp.tuples[-1][idx], m):
								data['child'].append(sorted(exp.tuples[-1][idx].tolist()+[m]))
				# send data back to master
				comm.send(data, dest=0, tag=TAGS.done)
			# exit
			elif tag == TAGS.exit:
				break
		# send exit signal to master
		comm.send(None, dest=0, tag=TAGS.exit)
		# receive tuples
		info = comm.bcast(None, root=0)
		if info['len'] >= 1:
			exp.tuples.append(np.empty([info['len'], exp.order+1], dtype=np.int32))
			parallel.tup(exp, comm)


def _test(calc, exp, tup, m):
		""" screening test """
		if exp.order == exp.start_order:
			return False
		else:
			# generate list with all subsets of particular tuple
			combs = np.array(list(list(comb) for comb in itertools.combinations(tup, exp.order-1)))
			# select only those combinations that include the active orbitals
			if calc.no_exp > 0:
				cond = np.zeros(len(combs), dtype=bool)
				for j in range(len(combs)): cond[j] = set(exp.tuples[0][0]) <= set(combs[j])
				combs = combs[cond]
			# conservative protocol
			if calc.protocol == 1:
				# init screening logical
				screen = True
				# loop over subset combinations
				for j in range(len(combs)):
					# recover index of particular tuple
					comb_idx = np.where(np.all(sorted(np.append(combs[j], [m])) == exp.tuples[-1], axis=1))[0]
					# does it exist?
					if len(comb_idx) == 0:
						# screen away
						screen = True
						break
					else:
						# is the increment above threshold?
						if np.abs(exp.energy['inc'][-1][comb_idx]) >= exp.thres:
							# mark as 'allowed'
							screen = False
			# aggressive protocol
			elif calc.protocol == 2:
				# init screening logical
				screen = False
				# loop over subset combinations
				for j in range(len(combs)):
					# recover index of particular tuple
					comb_idx = np.where(np.all(sorted(np.append(combs[j], [m])) == exp.tuples[-1], axis=1))[0]
					# does it exist?
					if len(comb_idx) == 0:
						# screen away
						screen = True
						break
					else:
						# is the increment above threshold?
						if np.abs(exp.energy['inc'][-1][comb_idx]) < exp.thres:
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


def _tasks(size, slaves):
		""" determine batch sizes """
		b1 = max(1, (size//10*8) // slaves) #  0 % - 80 %
		b2 = max(1, ((size//20*19) - (size//10*8)) // slaves // 2) #  80 % - 95 %
		b4 = max(1, ((size//50*49) - (size//20*19)) // slaves // 4) #  95 % - 98 %
		b6 = max(1, ((size//1000*990) - (size//50*49)) // slaves // 6) #  98 % - 99.9 %
		return [[b1] + [b2]*2 + [b4]*4 + [b6]*6 + [1] for idx in range(slaves)]


