#!/usr/bin/env python
# -*- coding: utf-8 -*

""" screen.py: screening module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import itertools

import parallel


def _enum(*sequential, **named):
		""" hardcoded enums
		see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
		"""
		enums = dict(zip(sequential, range(len(sequential))), **named)
		return type('Enum', (), enums)


# mbe parameters
_tags = _enum('ready', 'done', 'exit', 'start')


def main(mpi, mol, calc, exp):
		""" input generation for subsequent order """
		# mpi parallel or serial version
		if mpi.parallel:
			if mpi.global_master:
				_master(mpi, mol, calc, exp)
			else:
				_slave(mpi, mol, calc, exp)
		else:
			_serial(mol, calc, exp)
		# update expansion threshold
		exp.thres = update(calc, exp)


def _serial(mol, calc, exp):
		""" serial version """
		# init bookkeeping variables
		tmp = []; combs = []
        # loop over parent tuples
		for i in range(len(exp.tuples[-1])):
			# loop through possible orbitals to augment the combinations with
			for m in range(exp.tuples[-1][i][-1]+1, calc.exp_space[-1]+1):
				# if tuple is allowed, add to child tuple list, otherwise screen away
				if not _test(calc, exp, exp.tuples[-1][i]):
					tmp.append(exp.tuples[-1][i].tolist()+[m])
		# when done, write to tup list or mark expansion as converged
		if len(tmp) >= 1:
			tmp.sort()
			exp.tuples.append(np.array(tmp, dtype=np.int32))
		else:
			exp.conv_orb.append(True)


def _master(mpi, mol, calc, exp):
		""" master routine """
		# wake up slaves
		if exp.level == 'macro':
			msg = {'task': 'screen_local_master', 'exp_order': exp.order, 'thres': exp.thres}
			# set communicator
			comm = mpi.master_comm
			# set number of workers
			slaves_avail = num_slaves = mpi.num_local_masters
		else:
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
		# loop until no slaves left
		while (slaves_avail >= 1):
			# receive data dict
			data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi.stat)
			# probe for source and tag
			source = mpi.stat.Get_source(); tag = mpi.stat.Get_tag()
			# slave is ready
			if tag == _tags.ready:
				# any jobs left?
				if i <= len(exp.tuples[-1]) - 1:
					# save parent tuple index
					job_info['index'] = i
					# send parent tuple index
					comm.send(job_info, dest=source, tag=_tags.start)
					# increment job index
					i += 1
				else:
					# send exit signal
					comm.send(None, dest=source, tag=_tags.exit)
			# receive result from slave
			elif tag == _tags.done:
				# write tmp child tuple list
				tmp += data['child'] 
			# put slave to sleep
			elif tag == _tags.exit:
				# remove slave
				slaves_avail -= 1
		# finally we sort the tuples or mark expansion as converged 
		if len(tmp) == 1:
			exp.conv_orb.append(True)
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
		# set communicator and number of workers
		if exp.level == 'macro':
			comm = mpi.master_comm
		else:
			comm = mpi.local_comm
		# receive work from master
		while (True):
			# send status to master
			comm.send(None, dest=0, tag=_tags.ready)
			# receive parent tuple
			job_info = comm.recv(source=0, tag=MPI.ANY_TAG, status=mpi.stat)
			# recover tag
			tag = mpi.stat.Get_tag()
			# do job
			if tag == _tags.start:
				# init child tuple list
				data['child'][:] = []
				for m in range(exp.tuples[-1][job_info['index']][-1]+1, calc.exp_space[-1]+1):
					# if tuple is allowed, add to child tuple list, otherwise screen away
					if not _test(calc, exp, exp.tuples[-1][job_info['index']]):
						data['child'].append(exp.tuples[-1][job_info['index']].tolist()+[m])
				# send data back to master
				comm.send(data, dest=0, tag=_tags.done)
			# exit
			elif tag == _tags.exit:
				break
		# send exit signal to master
		comm.send(None, dest=0, tag=_tags.exit)
		# receive tuples
		info = comm.bcast(None, root=0)
		exp.tuples.append(np.empty([info['len'], exp.order+1], dtype=np.int32))
		parallel.tup(exp, comm)


def _test(calc, exp, tup):
		""" screening test """
		if exp.order == exp.start_order:
			screen = False
		else:
			# generate list with all subsets of particular tuple
			combs = np.array(list(list(comb) for comb in itertools.combinations(tup, exp.order-1)))
			# select only those combinations that include the active orbitals
			if calc.no_act > len(calc.ref_space):
				cond = np.zeros(len(combs), dtype=bool)
				for j in range(len(combs)): cond[j] = set(exp.tuples[0][0]) <= set(combs[j])
				combs = combs[cond]
			# loop through possible orbitals to augment the combinations with
			for m in range(tup[-1]+1, calc.exp_space[-1]+1):
				# init screening logical
				screen = True
				# loop over subset combinations
				for j in range(len(combs)):
					# recover index of particular tuple
					comb_idx = np.where(np.all(np.append(combs[j], [m]) == exp.tuples[-1], axis=1))[0]
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
		return screen


def update(calc, exp):
		""" update expansion threshold """
		if exp.order == 1:
			return 0.0
		else:
			return calc.thres * calc.relax ** (exp.order - 2)


