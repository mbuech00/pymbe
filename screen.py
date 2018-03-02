#!/usr/bin/env python
# -*- coding: utf-8 -*

""" screen.py: screening class """

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
from itertools import combinations


class ScrCls():
		""" screening class """
		def __init__(self, mol, variant):
				""" init parameters """
				# store type
				self.exp_type = variant
				# set tags
				self.tags = self.enum('ready', 'done', 'exit', 'start') 
				#
				return


		def enum(self, *sequential, **named):
				""" hardcoded enums
				see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
				"""
				enums = dict(zip(sequential, range(len(sequential),len(sequential)*2)), **named)
				#
				return type('Enum', (), enums)

	
		def update(self, calc, exp):
				""" update expansion threshold """
				if (exp.order == 1):
					return 0.0
				else:
					return calc.exp_thres * calc.exp_relax ** (exp.order - 2)

		
		def main(self, mpi, mol, calc, exp, rst):
				""" input generation for subsequent order """
				# start screening
				if (mpi.parallel):
					# mpi parallel version
					self.master(mpi, mol, calc, exp)
				else:
					# init bookkeeping variables
					tmp = []; combs = []
			        # loop over parent tuples
					for i in range(len(exp.tuples[-1])):
						if (exp.order == exp.start_order):
							# loop through possible orbitals to augment the combinations with
							for m in range(exp.tuples[-1][i][-1]+1, calc.exp_space[-1]+1):
								tmp.append(exp.tuples[-1][i].tolist()+[m])
						else:
							# generate list with all subsets of particular tuple
							combs = np.array(list(list(comb) for comb in combinations(exp.tuples[-1][i], exp.order-1)))
							# select only those combinations that include the active orbitals
							if (calc.no_act > len(calc.ref_space)):
								cond = np.zeros(len(combs), dtype=bool)
								for j in range(len(combs)): cond[j] = set(exp.tuples[0][0]) <= set(combs[j])
								combs = combs[cond]
							# loop through possible orbitals to augment the combinations with
							for m in range(exp.tuples[-1][i][-1]+1, calc.exp_space[-1]+1):
								# init screening logical
								screen = True
								# loop over subset combinations
								for j in range(len(combs)):
									# recover index of particular tuple
									comb_idx = np.where(np.all(np.append(combs[j], [m]) == exp.tuples[-1], axis=1))[0]
									# does it exist?
									if (len(comb_idx) == 0):
										# screen away
										screen = True
										break
									else:
										# is the increment above threshold?
										if (np.abs(exp.energy['inc'][-1][comb_idx]) >= exp.thres):
											# mark as 'allowed'
											screen = False
								# if tuple is allowed, add to child tuple list, otherwise screen away
								if (not screen): tmp.append(exp.tuples[-1][i].tolist()+[m])
					# when done, write to tup list or mark expansion as converged
					if (len(tmp) >= 1):
						tmp.sort()
						exp.tuples.append(np.array(tmp, dtype=np.int32))
					else:
						exp.conv_orb.append(True)
				# update expansion threshold
				exp.thres = self.update(calc, exp)
				#
				return
	
	
		def master(self, mpi, mol, calc, exp):
				""" master routine """
				# wake up slaves
				if (exp.level == 'macro'):
					msg = {'task': 'screen_local_master', 'exp_order': exp.order, 'thres': exp.thres}
					# set communicator
					comm = mpi.master_comm
					# set number of workers
					slaves_avail = num_slaves = mpi.num_local_masters
				else:
					msg = {'task': 'screen_slave', 'exp_order': exp.order, 'thres': exp.thres}
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
					if (tag == self.tags.ready):
						# any jobs left?
						if (i <= len(exp.tuples[-1])-1):
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
					exp.conv_orb.append(True)
				# make numpy array out of tmp
				buff = np.array(tmp, dtype=np.int32)
				# bcast buff
				mpi.bcast_tup(exp, buff, comm)
				#
				return
		
		
		def slave(self, mpi, mol, calc, exp):
				""" slave routine """
				# init data dict and combs list
				data = {'child_tuple': []}; combs = []
				# set communicator and number of workers
				if (exp.level == 'macro'):
					comm = mpi.master_comm
				else:
					comm = mpi.local_comm
				# receive work from master
				while (True):
					# send status to master
					comm.send(None, dest=0, tag=self.tags.ready)
					# receive parent tuple
					job_info = comm.recv(source=0, tag=MPI.ANY_TAG, status=mpi.stat)
					# recover tag
					tag = mpi.stat.Get_tag()
					# do job
					if (tag == self.tags.start):
						# init child tuple list
						data['child_tuple'][:] = []
						if (exp.order == exp.start_order):
							# loop through possible orbitals to augment the combinations with
							for m in range(exp.tuples[-1][job_info['index']][-1]+1, calc.exp_space[-1]+1):
								data['child_tuple'].append(exp.tuples[-1][job_info['index']].tolist()+[m])
						else:
							# generate list with all subsets of particular tuple
							combs = np.array(list(list(comb) for comb in combinations(exp.tuples[-1][job_info['index']], exp.order-1)))
							# select only those combinations that include the active orbitals
							if (calc.no_act > len(calc.ref_space)):
								cond = np.zeros(len(combs), dtype=bool)
								for j in range(len(combs)): cond[j] = set(exp.tuples[0][0]) <= set(combs[j])
								combs = combs[cond]
							# loop through possible orbitals to augment the combinations with
							for m in range(exp.tuples[-1][job_info['index']][-1]+1, calc.exp_space[-1]+1):
								# init screening logical
								screen = True
								# loop over subset combinations
								for j in range(len(combs)):
									# recover index of particular tuple
									comb_idx = np.where(np.all(np.append(combs[j], [m]) == exp.tuples[-1], axis=1))[0]
									# does it exist?
									if (len(comb_idx) == 0):
										# screen away
										screen = True
										break
									else:
										# is the increment above threshold?
										if (np.abs(exp.energy['inc'][-1][comb_idx]) >= exp.thres):
											# mark as 'allowed'
											screen = False
								# if tuple is allowed, add to child tuple list, otherwise screen away
								if (not screen): data['child_tuple'].append(exp.tuples[-1][job_info['index']].tolist()+[m])
						# send data back to master
						comm.send(data, dest=0, tag=self.tags.done)
					# exit
					elif (tag == self.tags.exit):
						break
				# send exit signal to master
				comm.send(None, dest=0, tag=self.tags.exit)
				# init buffer
				tup_info = comm.bcast(None, root=0)
				buff = np.empty([tup_info['tup_len'],exp.order+1], dtype=np.int32)
				# receive buffer
				mpi.bcast_tup(exp, buff, comm)
				#
				return


