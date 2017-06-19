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
		def __init__(self, _exp):
				""" init tags """
				self.tags = _exp.enum('ready', 'done', 'exit', 'start') 
				#
				return
		
		
		def main(self, _mpi, _calc, _exp, _time, _rst):
				""" input generation for subsequent order """
				if (_mpi.parallel):
					# mpi parallel version
					self.master(_mpi, _calc, _exp, _time)
					_time.coll_screen_time(_mpi, _rst, _exp.order, True)
				else:
					# start time
					_time.timer('work_screen', _exp.order)
					# determine which tuples have contributions below the threshold
					if (_exp.order == 1):
						allow_tuple = _exp.tuples[-1]
					else:
						allow_tuple = _exp.tuples[-1][np.where(np.abs(_exp.energy_inc[-1]) >= _calc.exp_thres)]
					# init bookkeeping variables
					_exp.screen_count.append(0); tmp = []; combs = []
			        # loop over parent tuples
					for i in range(len(_exp.tuples[-1])):
						# generate list with all subsets of particular tuple
						combs = list(list(comb) for comb in combinations(_exp.tuples[-1][i], _exp.order-1))
						# loop through possible orbitals to augment the combinations with
						for m in range(_exp.tuples[-1][i][-1]+1, _exp.l_limit+_exp.u_limit):
							# init screening logical
							screen = False
							# loop over subset combinations
							for j in range(len(combs)):
								# check whether or not the particular tuple is actually allowed
								if (not np.equal(combs[j]+[m],_exp.tuples[-1]).all(axis=1).any()):
									# screen away
									screen = True
									break
							if (not screen):
				                # loop over subset combinations
								for j in range(len(combs)):
									# check whether the particular tuple among negligible tuples
									if (not np.equal(combs[j]+[m],allow_tuple).all(axis=1).any()):
										# screen away
										screen = True
										break
							# if tuple is allowed, add to child tuple list, otherwise screen away
							if (not screen):
								tmp.append(_exp.tuples[-1][i].tolist()+[m])
							else:
								_exp.screen_count[-1] += 1
					# when done, write to tup list or mark expansion as converged
					if (len(tmp) >= 1):
						_exp.tuples.append(np.array(tmp, dtype=np.int32))
					else:
						_exp.conv_orb.append(True)
					# end time
					_time.timer('work_screen', _exp.order, True)
				#
				return
	
	
		def master(self, _mpi, _calc, _exp, _time):
				""" master routine """
				# start idle time
				_time.timer('idle_screen', _exp.order)
				# wake up slaves
				msg = {'task': 'screen_slave', 'order': _exp.order}
				# bcast
				_mpi.comm.bcast(msg, root=0)
				# start work time
				_time.timer('work_screen', _exp.order)
				# init job_info dictionary
				job_info = {}
				# number of slaves
				num_slaves = _mpi.size - 1
				# number of available slaves
				slaves_avail = num_slaves
				# init job index, tmp list, and screen_count
				i = 0; tmp = []; _exp.screen_count.append(0)
				# loop until no slaves left
				while (slaves_avail >= 1):
					# start idle time
					_time.timer('idle_screen', _exp.order)
					# receive data dict
					data = _mpi.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
											status=_mpi.stat)
					# start work time
					_time.timer('work_screen', _exp.order)
					# probe for source and tag
					source = _mpi.stat.Get_source(); tag = _mpi.stat.Get_tag()
					# slave is ready
					if (tag == self.tags.ready):
						# any jobs left?
						if (i <= len(_exp.tuples[-1])-1):
							# start comm time
							_time.timer('comm_screen', _exp.order)
							# save parent tuple index
							job_info['index'] = i
							# send parent tuple index
							_mpi.comm.send(job_info, dest=source, tag=self.tags.start)
							# start work time
							_time.timer('work_screen', _exp.order)
							# increment job index
							i += 1
						else:
							# start comm time#
							_time.timer('comm_screen', _exp.order)
							# send None info
							_mpi.comm.send(None, dest=source, tag=self.tags.exit)
							# start work time
							_time.timer('work_screen', _exp.order)
					# receive result from slave
					elif (tag == self.tags.done):
						# write tmp child tuple list
						tmp += data['child_tuple'] 
						# increment number of screened tuples
						_exp.screen_count[-1] += data['screen_count']
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
				_mpi.bcast_tup(_exp, _time, buff)
				#
				return
		
		
		def slave(self, _mpi, _calc, _exp, _time):
				""" slave routine """
				# start work time
				_time.timer('work_screen', _exp.order)
				# init data dict and combs list
				data = {'child_tuple': [], 'screen_count': 0}; combs = []
				# determine which tuples have contributions larger than the threshold
				if (_exp.order == 1):
					allow_tuple = _exp.tuples[-1]
				else:
					allow_tuple = _exp.tuples[-1][np.where(np.abs(_exp.energy_inc[-1]) >= _calc.exp_thres)]
				# receive work from master
				while (True):
					# start comm time
					_time.timer('comm_screen', _exp.order)
					# send status to master
					_mpi.comm.send(None, dest=0 ,tag=self.tags.ready)
					# receive parent tuple
					job_info = _mpi.comm.recv(source=0, tag=MPI.ANY_SOURCE, status=_mpi.stat)
					# start work time
					_time.timer('work_screen', _exp.order)
					# recover tag
					tag = _mpi.stat.Get_tag()
					# do job
					if (tag == self.tags.start):
						# init child tuple list and screen counter
						data['child_tuple'][:] = []; data['screen_count'] = 0
						# generate list with all subsets of particular tuple
						combs = list(list(comb) for comb in combinations(_exp.tuples[-1][job_info['index']], _exp.order-1))
						# loop through possible orbitals to augment the combinations with
						for m in range(_exp.tuples[-1][job_info['index']][-1]+1, _exp.l_limit+_exp.u_limit):
							# init screening logical
							screen = False
							# loop over subset combinations
							for j in range(len(combs)):
								# check whether or not the particular tuple is actually allowed
								if (not np.equal(combs[j]+[m],_exp.tuples[-1]).all(axis=1).any()):
									# screen away
									screen = True
									break
							if (not screen):
			                    # loop over subset combinations
								for j in range(len(combs)):
									# check whether the particular tuple among negligible tuples
									if (not np.equal(combs[j]+[m],allow_tuple).all(axis=1).any()):
										# screen away
										screen = True
										break
							# if tuple is allowed, add to child tuple list, otherwise screen away
							if (not screen):
								data['child_tuple'].append(_exp.tuples[-1][job_info['index']].tolist()+[m])
							else:
								data['screen_count'] += 1
						# start comm time
						_time.timer('comm_screen', _exp.order)
						# send data back to master
						_mpi.comm.send(data, dest=0, tag=self.tags.done)
						# start work time
						_time.timer('work_screen', _exp.order)
					# exit
					elif (tag == self.tags.exit):
						break
				# start comm time
				_time.timer('comm_screen', _exp.order)
				# send exit signal to master
				_mpi.comm.send(None, dest=0, tag=self.tags.exit)
				# start work time
				_time.timer('work_screen', _exp.order)
				# init buffer
				tup_info = _mpi.comm.bcast(None, root=0)
				buff = np.empty([tup_info['tup_len'],_exp.order+1], dtype=np.int32)
				# receive buffer
				_mpi.bcast_tup(_exp, _time, buff)
				#
				return


