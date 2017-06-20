#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_kernel.py: kernel class for Bethe-Goldstone correlation calculations."""

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


class KernCls():
		""" kernel class """
		def __init__(self, _exp):
				""" init tags """
				self.tags = _exp.enum('ready', 'done', 'data', 'exit', 'start')
				#
				return


		def main(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst):
				""" energy kernel phase """
				# mpi parallel version
				if (_mpi.parallel):
					self.master(_mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst)
					_time.coll_kernel_time(_mpi, _rst, _exp.order)
				else:
					# determine start index
					start = np.argmax(_exp.energy_inc[-1] == 0.0)
					# loop over tuples
					for i in range(start,len(_exp.tuples[-1])):
						# start work time
						_time.timer('work_kernel', _exp.order)
						# generate input
						_exp.cas_idx, _exp.core_idx, _exp.h1e_cas, _exp.h2e_cas, _exp.e_core = \
								_pyscf.corr_input(_mol, _calc, _exp, _exp.tuples[-1][i])
						# run correlated calc
						_exp.energy_inc[-1][i] = _pyscf.corr_calc(_mol, _calc, _exp)
						# print status
						_prt.kernel_status(float(i+1) / float(len(_exp.tuples[-1])))
#						# error handling
#						if (molecule['error'][-1]):
#							molecule['error_rank'] = 0
#							molecule['error_drop'] = string['drop']
#							term_calc(molecule)
						# collect work time
						_time.timer('work_kernel', _exp.order, True)
						# write restart files
						if (((i+1) % _rst.rst_freq) == 0): _rst.write_kernel(_mpi, _exp, _time)
				#
				return
		
		
		def master(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst):
				""" master function """
				# start idle time
				_time.timer('idle_kernel', _exp.order)
				# wake up slaves
				msg = {'task': 'kernel_slave', 'order': _exp.order}
				# bcast
				_mpi.comm.bcast(msg, root=0)
				# start work time
				_time.timer('work_kernel', _exp.order)
				# init job_info dictionary
				job_info = {}
				# number of slaves
				num_slaves = _mpi.size - 1
				# number of available slaves
				slaves_avail = num_slaves
				# init job index
				i = np.argmax(_exp.energy_inc[-1] == 0.0)
				# init stat counter
				counter = i
				# init timings
				if ((not _rst.restart) or (_exp.order != _calc.exp_min_order)):
					for j in range(_mpi.size):
						_time.time_work[0][j].append(0.0)
						_time.time_comm[0][j].append(0.0)
						_time.time_idle[0][j].append(0.0)
				# print status for START
				_prt.kernel_status(float(counter) / float(len(_exp.tuples[-1])))
				# loop until no slaves left
				while (slaves_avail >= 1):
					# start idle time
					_time.timer('idle_kernel', _exp.order)
					# receive data dict
					stat = _mpi.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=_mpi.stat)
					# start work time
					_time.timer('work_kernel', _exp.order)
					# probe for source and tag
					source = _mpi.stat.Get_source(); tag = _mpi.stat.Get_tag()
					# slave is ready
					if (tag == self.tags.ready):
						# any jobs left?
						if (i <= (len(_exp.tuples[-1]) - 1)):
							# generate input
							_exp.cas_idx, _exp.core_idx, _exp.h1e_cas, _exp.h2e_cas, _exp.e_core = \
									_pyscf.corr_input(_mol, _calc, _exp, _exp.tuples[-1][i])
							# store job info
							job_info['index'] = i
							job_info['cas_idx'] = _exp.cas_idx
							job_info['core_idx'] = _exp.core_idx
							job_info['h1e_cas'] = _exp.h1e_cas
							job_info['h2e_cas'] = _exp.h2e_cas
							job_info['e_core'] = _exp.e_core
							# start comm time
							_time.timer('comm_kernel', _exp.order)
							# send string dict
							_mpi.comm.send(job_info, dest=source, tag=self.tags.start)
							# start work time
							_time.timer('work_kernel', _exp.order)
							# increment job index
							i += 1
						else:
							# start comm time
							_time.timer('comm_kernel', _exp.order)
							# send exit signal
							_mpi.comm.send(None, dest=source, tag=self.tags.exit)
							# start work time
							_time.timer('work_kernel', _exp.order)
					# receive result from slave
					elif (tag == self.tags.done):
						# start comm time
						_time.timer('comm_kernel', _exp.order)
						# receive data
						data = _mpi.comm.recv(source=source, tag=self.tags.data, status=_mpi.stat)
						# start work time
						_time.timer('work_kernel', _exp.order)
						# write to e_inc
						_exp.energy_inc[-1][data['index']] = data['e_corr']
						# store timings
						_time.time_work[0][source][-1] = data['t_work']
						_time.time_comm[0][source][-1] = data['t_comm']
						_time.time_idle[0][source][-1] = data['t_idle']
						# write restart files
						if (((data['index']+1) % int(_rst.rst_freq)) == 0):
							_time.time_work[0][0][-1] = _time.timings['work_kernel'][-1]
							_time.time_comm[0][0][-1] = _time.timings['comm_kernel'][-1]
							_time.time_idle[0][0][-1] = _time.timings['idle_kernel'][-1]
							_rst.write_kernel(_mpi, _exp, _time)
						# increment stat counter
						counter += 1
						# print status
						if (((data['index']+1) % 1000) == 0):
							_prt.kernel_status(float(counter) / float(len(_exp.tuples[-1])))
#						# error handling
#						if (data['error']):
#							molecule['error'].append(True)
#							molecule['error_code'] = data['error_code']
#							molecule['error_msg'] = data['error_msg']
#							molecule['error_rank'] = source
#							molecule['error_drop'] = data['error_drop']
#							term_calc(molecule)
					# put slave to sleep
					elif (tag == self.tags.exit):
						slaves_avail -= 1
				# print 100.0 %
				_prt.kernel_status(1.0)
				# collect work time
				_time.timer('work_kernel', _exp.order, True)
				#
				return
		
		
		def slave(self, _mpi, _mol, _calc, _pyscf, _exp, _time):
				""" slave function """
				# start work time
				_time.timer('work_kernel', _exp.order)
				# init e_inc list
				if (len(_exp.energy_inc) != _exp.order):
					_exp.energy_inc.append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
				# init data dict
				data = {}
				# receive work from master
				while (True):
					# start comm time
					_time.timer('comm_kernel', _exp.order)
					# ready for task
					_mpi.comm.send(None, dest=0, tag=self.tags.ready)
					# receive drop string
					job_info = _mpi.comm.recv(source=0, tag=MPI.ANY_SOURCE, status=_mpi.stat)
					# start work time
					_time.timer('work_kernel', _exp.order)
					# recover tag
					tag = _mpi.stat.Get_tag()
					# do job
					if (tag == self.tags.start):
						# load job info
						_exp.cas_idx = job_info['cas_idx']
						_exp.core_idx = job_info['core_idx']
						_exp.h1e_cas = job_info['h1e_cas']
						_exp.h2e_cas = job_info['h2e_cas']
						_exp.e_core = job_info['e_core']
						# run correlated calc
						_exp.energy_inc[-1][job_info['index']] = _pyscf.corr_calc(_mol, _calc, _exp)
						# start comm time
						_time.timer('comm_kernel', _exp.order)
						# report status back to master
						_mpi.comm.send(None, dest=0, tag=self.tags.done)
						# start work time
						_time.timer('work_kernel', _exp.order)
						# write info into data dict
						data['index'] = job_info['index']
						data['e_corr'] = _exp.energy_inc[-1][job_info['index']]
						data['t_work'] = _time.timings['work_kernel'][-1]
						data['t_comm'] = _time.timings['comm_kernel'][-1]
						data['t_idle'] = _time.timings['idle_kernel'][-1]
#						data['error'] = molecule['error'][-1]
#						data['error_code'] = molecule['error_code']
#						data['error_msg'] = molecule['error_msg']
#						data['error_drop'] = string['drop']
						# start comm time
						_time.timer('comm_kernel', _exp.order)
						# send data back to master
						_mpi.comm.send(data, dest=0, tag=self.tags.data)
						# start work time
						_time.timer('work_kernel', _exp.order)
					# exit
					elif (tag == self.tags.exit):
						break
				# start comm time
				_time.timer('comm_kernel', _exp.order)
				# send exit signal to master
				_mpi.comm.send(None, dest=0, tag=self.tags.exit)
				# collect comm time
				_time.timer('comm_kernel', _exp.order, True)
				#
				return


