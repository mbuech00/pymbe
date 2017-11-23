#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_screen.py: screening class for Bethe-Goldstone correlation calculations. """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from itertools import combinations


class ScrCls():
		""" screening class """
		def __init__(self, _mol, _type):
				""" init parameters """
				# store type
				self.exp_type = _type
				# set tags
				self.tags = self.enum('ready', 'done', 'exit', 'start') 
				# set u_limit
				if (_type in ['occupied','combined']):
					self.l_limit = 0
					self.u_limit = _mol.nocc
				else:
					self.l_limit = _mol.nocc
					self.u_limit = _mol.nvirt 
				#
				return


		def enum(self, *sequential, **named):
				""" hardcoded enums
				see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
				"""
				enums = dict(zip(sequential, range(len(sequential),len(sequential)*2)), **named)
				#
				return type('Enum', (), enums)

	
		def update(self, _calc, _exp):
				""" update expansion threshold according to start order """
				if (_exp.order < 2):
					return _calc.exp_thres
				else:
					return _calc.exp_thres * _calc.exp_relax ** (_exp.order - 2)

		
		def main(self, _mpi, _mol, _calc, _exp, _rst):
				""" input generation for subsequent order """
				# start screening
				if (_mpi.parallel):
					# mpi parallel version
					self.master(_mpi, _mol, _calc, _exp)
				else:
					# init bookkeeping variables
					tmp = []; combs = []
					# generate list of allowed tuples
					if ((_exp.order <= 2) or (_exp.thres == 0.0)):
						# compute all (k+1)-order increments
						_exp.allow_tuples = _exp.tuples[-1]
					else:
						# screen away tuples with increments below threshold
						_exp.allow_tuples = _exp.tuples[-1][np.where(np.abs(_exp.energy_inc[-1]) >= _exp.thres)]
			        # loop over parent tuples
					for i in range(len(_exp.allow_tuples)):
						if (_exp.order == _exp.start_order):
							# loop through possible orbitals to augment the combinations with
							for m in range(_exp.allow_tuples[i][-1]+1, self.l_limit+self.u_limit):
								if (not (m in _exp.allow_tuples[i])):
									tmp.append(_exp.allow_tuples[i].tolist()+[m])
						else:
							# generate list with all subsets of particular tuple
							combs = np.array(list(list(comb) for comb in combinations(_exp.allow_tuples[i], _exp.order-1)))
							# select only those combinations that include the active orbitals
							cond = np.zeros(len(combs), dtype=bool)
							for j in range(len(combs)): cond[j] = set(_calc.act_orbs) <= set(combs[j])
							combs = combs[cond]
							# loop through possible orbitals to augment the combinations with
							for m in range(_exp.allow_tuples[i][-1]+1, self.l_limit+self.u_limit):
								if (not (m in _exp.allow_tuples[i])):
									# init screening logical
									screen = False
									# loop over subset combinations
									for j in range(len(combs)):
										# check whether or not the particular tuple is allowed
										if (not np.equal(np.append(combs[j], [m]), _exp.allow_tuples).all(axis=1).any()):
											# screen away
											screen = True
											break
									# if tuple is allowed, add to child tuple list, otherwise screen away
									if (not screen): tmp.append(_exp.allow_tuples[i].tolist()+[m])
					# when done, write to tup list or mark expansion as converged
					if (len(tmp) >= 1):
						tmp.sort()
						_exp.tuples.append(np.array(tmp, dtype=np.int32))
					else:
						_exp.conv_orb.append(True)
				# update expansion threshold
				_exp.thres = self.update(_calc, _exp)
				#
				return
	
	
		def master(self, _mpi, _mol, _calc, _exp):
				""" master routine """
				# wake up slaves
				if (_exp.level == 'macro'):
					msg = {'task': 'screen_local_master', 'exp_order': _exp.order, 'thres': _exp.thres}
					# set communicator
					comm = _mpi.master_comm
					# set number of workers
					slaves_avail = num_slaves = _mpi.num_local_masters
				else:
					msg = {'task': 'screen_slave', 'exp_order': _exp.order, 'thres': _exp.thres}
					# set communicator
					comm = _mpi.local_comm
					# set number of workers
					slaves_avail = num_slaves = _mpi.local_size - 1
				# bcast
				comm.bcast(msg, root=0)
				# init job_info dictionary
				job_info = {}
				# init bookkeeping variables
				i = 0; tmp = []
				# generate list of allowed tuples
				if ((_exp.order <= 2) or (_exp.thres == 0.0)):
					# compute all (k+1)-order increments
					_exp.allow_tuples = _exp.tuples[-1]
				else:
					# screen away tuples with increments below threshold
					_exp.allow_tuples = _exp.tuples[-1][np.where(np.abs(_exp.energy_inc[-1]) >= _exp.thres)]
				# loop until no slaves left
				while (slaves_avail >= 1):
					# receive data dict
					data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=_mpi.stat)
					# probe for source and tag
					source = _mpi.stat.Get_source(); tag = _mpi.stat.Get_tag()
					# slave is ready
					if (tag == self.tags.ready):
						# any jobs left?
						if (i <= len(_exp.allow_tuples)-1):
							# save parent tuple index
							job_info['index'] = i
							# send parent tuple index
							comm.send(job_info, dest=source, tag=self.tags.start)
							# increment job index
							i += 1
						else:
							# send exit signal
							comm.send(None, dest=source, tag=self.tags.exit)
					# receive result from slave
					elif (tag == self.tags.done):
						# write tmp child tuple list
						tmp += data['child_tuple'] 
					# put slave to sleep
					elif (tag == self.tags.exit):
						# remove slave
						slaves_avail -= 1
				# finally we sort the tuples or mark expansion as converged 
				if (len(tmp) >= 1):
					tmp.sort()
				else:
					_exp.conv_orb.append(True)
				# make numpy array out of tmp
				buff = np.array(tmp, dtype=np.int32)
				# bcast buff
				_mpi.bcast_tup(_exp, buff, comm)
				#
				return
		
		
		def slave(self, _mpi, _mol, _calc, _exp):
				""" slave routine """
				# init data dict and combs list
				data = {'child_tuple': []}; combs = []
				# set communicator and number of workers
				if (_exp.level == 'macro'):
					comm = _mpi.master_comm
				else:
					comm = _mpi.local_comm
				# generate list of allowed tuples
				if ((_exp.order <= 2) or (_exp.thres == 0.0)):
					# compute all (k+1)-order increments
					_exp.allow_tuples = _exp.tuples[-1]
				else:
					# screen away tuples with increments below threshold
					_exp.allow_tuples = _exp.tuples[-1][np.where(np.abs(_exp.energy_inc[-1]) >= _exp.thres)]
				# receive work from master
				while (True):
					# send status to master
					comm.send(None, dest=0, tag=self.tags.ready)
					# receive parent tuple
					job_info = comm.recv(source=0, tag=MPI.ANY_TAG, status=_mpi.stat)
					# recover tag
					tag = _mpi.stat.Get_tag()
					# do job
					if (tag == self.tags.start):
						# init child tuple list
						data['child_tuple'][:] = []
						if (_exp.order == _exp.start_order):
							# loop through possible orbitals to augment the combinations with
							for m in range(_exp.allow_tuples[job_info['index']][-1]+1, self.l_limit+self.u_limit):
								if (not (m in _exp.allow_tuples[job_info['index']])):
									data['child_tuple'].append(_exp.allow_tuples[job_info['index']].tolist()+[m])
						else:
							# generate list with all subsets of particular tuple
							combs = np.array(list(list(comb) for comb in combinations(_exp.allow_tuples[job_info['index']], _exp.order-1)))
							# select only those combinations that include the active orbitals
							cond = np.zeros(len(combs), dtype=bool)
							for j in range(len(combs)): cond[j] = set(_calc.act_orbs) <= set(combs[j])
							combs = combs[cond]
							# loop through possible orbitals to augment the combinations with
							for m in range(_exp.allow_tuples[job_info['index']][-1]+1, self.l_limit+self.u_limit):
								if (not (m in _exp.allow_tuples[job_info['index']])):
									# init screening logical
									screen = False
									# loop over subset combinations
									for j in range(len(combs)):
										# check whether or not the particular tuple is allowed
										if (not np.equal(np.append(combs[j], [m]), _exp.allow_tuples).all(axis=1).any()):
											# screen away
											screen = True
											break
									# if tuple is allowed, add to child tuple list, otherwise screen away
									if (not screen): data['child_tuple'].append(_exp.allow_tuples[job_info['index']].tolist()+[m])
						# send data back to master
						comm.send(data, dest=0, tag=self.tags.done)
					# exit
					elif (tag == self.tags.exit):
						break
				# send exit signal to master
				comm.send(None, dest=0, tag=self.tags.exit)
				# init buffer
				tup_info = comm.bcast(None, root=0)
				buff = np.empty([tup_info['tup_len'],_exp.order+1], dtype=np.int32)
				# receive buffer
				_mpi.bcast_tup(_exp, buff, comm)
				#
				return


