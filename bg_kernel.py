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
import sys
from itertools import combinations, chain
from scipy.misc import comb

from bg_exp import ExpCls
from bg_time import TimeCls
import bg_driver


class KernCls():
		""" kernel class """
		def __init__(self):
				""" init parameters """
				# set tags
				self.tags = self.enum('ready', 'done', 'data', 'exit', 'start')
				#
				return


		def enum(self, *sequential, **named):
				""" hardcoded enums
				see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
				"""
				enums = dict(zip(sequential, range(len(sequential))), **named)
				#
				return type('Enum', (), enums)


		def comb_index(self, _n, _k):
				""" calculate combined index """
				count = comb(_n, _k, exact=True)
				index = np.fromiter(chain.from_iterable(combinations(range(_n), _k)),
									int,count=count * _k)
				#
				return index.reshape(-1, _k)


		def summation(self, _exp, _idx):
				""" energy summation """
				for i in range(_exp.order-1, 0, -1):
					# test if tuple is a subset
					combs = _exp.tuples[-1][_idx, self.comb_index(_exp.order, i)]
					dt = np.dtype((np.void, _exp.tuples[i-1].dtype.itemsize * \
									_exp.tuples[i-1].shape[1]))
					match = np.nonzero(np.in1d(_exp.tuples[i-1].view(dt).reshape(-1),
										combs.view(dt).reshape(-1)))[0]
					for j in match: _exp.energy_inc[-1][_idx] -= _exp.energy_inc[i-1][j]
				#
				return


		def macro_core(self, _mpi, _mol, _calc, _pyscf, _time, _prt, _rst, _driver, _order, _tup):
				""" core procedure for level == macro """
				# micro exp instantiation
				exp_micro = ExpCls(_mpi, _mol, _calc, 'virtual')
				# mark expansion as micro 
				exp_micro.level = 'micro'; exp_micro.order_macro = _order; exp_micro.incl_idx = _tup
				# make recursive call to driver with micro exp
				_driver.master(_mpi, _mol, _calc, _pyscf, exp_micro, _time, _prt, _rst)
				#
				return exp_micro.energy_tot[-1], exp_micro.order


		def micro_core(self, _mpi, _mol, _calc, _pyscf, _exp, _tup):
				""" core procedure for level == micro """
				# generate input
				_exp.core_idx, _exp.cas_idx, _exp.h1e_cas, _exp.h2e_cas, _exp.e_core = \
						_pyscf.corr_input(_mol, _calc, _exp, _tup)
				try:
					return _pyscf.corr_calc(_mol, _calc, _exp)
				except Exception as err:
					try:
						raise RuntimeError
					except RuntimeError:
						sys.stderr.write('\nCASCI Error : MPI proc. = {0:} (host = {1:})\n'
											'input: core_idx = {2:} , cas_idx = {3:}\n'
											'PySCF error : {4:}\n\n'.\
											format(_mpi.rank, _mpi.host, _exp.core_idx, _exp.cas_idx, err))
						raise


		def main(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst):
				""" energy kernel phase """
				# mpi parallel version
				if (_mpi.parallel):
					self.master(_mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst)
					_time.coll_phase_time(_mpi, _rst, _time.order, 'kernel')
				else:
					# micro driver instantiation
					if (_calc.exp_type == 'combined'):
						if (_exp.level == 'macro'): driver_micro = bg_driver.DrvCls(_mol, 'virtual') 
						_exp.micro_conv_res = np.zeros(len(_exp.tuples[-1]), dtype=np.int32)
					# determine start index
					start = np.argmax(_exp.energy_inc[-1] == 0.0)
					# loop over tuples
					for i in range(start, len(_exp.tuples[-1])):
						# start work time
						_time.timer('work_kernel', _time.order)
						# run correlated calc
						if (_exp.level == 'macro'):
							# store e_inc and micro convergence results
							_exp.energy_inc[-1][i], _exp.micro_conv_res[i] = \
									self.macro_core(_mpi, _mol, _calc, _pyscf, _time, _prt, _rst, \
														driver_micro, _exp.order_macro, _exp.tuples[-1][i].tolist()) 
						else:
							# store e_inc result
							_exp.energy_inc[-1][i] = self.micro_core(_mpi, _mol, _calc, _pyscf, \
																		_exp, _exp.tuples[-1][i])
						# sum up energy increment
						self.summation(_exp, i)
						# print status
						_prt.kernel_status(_calc, _exp, float(i+1) / float(len(_exp.tuples[-1])))
						# collect work time
						_time.timer('work_kernel', _time.order, True)
						# write restart files
						_rst.write_kernel(_mpi, _calc, _exp, _time, False)
				# sum of energy increments
				e_tmp = np.sum(_exp.energy_inc[-1][np.where(np.abs(_exp.energy_inc[-1]) >= _calc.tolerance)])
				# sum of total energy
				if (_exp.order >= 2): e_tmp += _exp.energy_tot[-1]
				# add to total energy list
				_exp.energy_tot.append(e_tmp)
				# check for convergence wrt total energy
				if ((_exp.order >= 2) and (abs(_exp.energy_tot[-1] - _exp.energy_tot[-2]) < _calc.energy_thres)):
					_exp.conv_energy.append(True)
				#
				return
		
		
		def master(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst):
				""" master function """
				# start idle time
				_time.timer('idle_kernel', _time.order)
				# wake up slaves
				msg = {'task': 'kernel_slave', 'exp_order': _exp.order, 'time_order': _time.order}
				# bcast
				_mpi.comm.bcast(msg, root=0)
				# start work time
				_time.timer('work_kernel', _time.order)
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
				if (i == 0):
					for j in range(_mpi.size):
						_time.time_work[0][j].append(0.0)
						_time.time_comm[0][j].append(0.0)
						_time.time_idle[0][j].append(0.0)
				# print status for START
				_prt.kernel_status(_calc, _exp, float(counter) / float(len(_exp.tuples[-1])))
				# loop until no slaves left
				while (slaves_avail >= 1):
					# start idle time
					_time.timer('idle_kernel', _time.order)
					# receive data dict
					stat = _mpi.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=_mpi.stat)
					# start work time
					_time.timer('work_kernel', _time.order)
					# probe for source and tag
					source = _mpi.stat.Get_source(); tag = _mpi.stat.Get_tag()
					# slave is ready
					if (tag == self.tags.ready):
						# any jobs left?
						if (i <= (len(_exp.tuples[-1]) - 1)):
							# generate input
							_exp.core_idx, _exp.cas_idx, _exp.h1e_cas, _exp.h2e_cas, _exp.e_core = \
									_pyscf.corr_input(_mol, _calc, _exp, _exp.tuples[-1][i])
							# store job info
							job_info['index'] = i
							job_info['core_idx'] = _exp.core_idx
							job_info['cas_idx'] = _exp.cas_idx
							job_info['h1e_cas'] = _exp.h1e_cas
							job_info['h2e_cas'] = _exp.h2e_cas
							job_info['e_core'] = _exp.e_core
							# start comm time
							_time.timer('comm_kernel', _time.order)
							# send string dict
							_mpi.comm.send(job_info, dest=source, tag=self.tags.start)
							# start work time
							_time.timer('work_kernel', _time.order)
							# increment job index
							i += 1
						else:
							# start comm time
							_time.timer('comm_kernel', _time.order)
							# send exit signal
							_mpi.comm.send(None, dest=source, tag=self.tags.exit)
							# start work time
							_time.timer('work_kernel', _time.order)
					# receive result from slave
					elif (tag == self.tags.done):
						# start comm time
						_time.timer('comm_kernel', _time.order)
						# receive data
						data = _mpi.comm.recv(source=source, tag=self.tags.data, status=_mpi.stat)
						# error handling
						if (data['error']):
							try:
								raise RuntimeError('\nCASCI Error : MPI proc. = {0:} (host = {1:})\n'
													'input: core_idx = {2:} , cas_idx = {3:}\n'
													'PySCF error : {4:}\n\n'.\
													format(source, data['host'], data['core_idx'],\
															 data['cas_idx'], data['pyscf_err']))
							except Exception as err:
								sys.stderr.write(str(err))
								raise
						# start work time
						_time.timer('work_kernel', _time.order)
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
							_rst.write_kernel(_mpi, _calc, _exp, _time, False)
						# increment stat counter
						counter += 1
						# print status
						if (((data['index']+1) % 1000) == 0):
							_prt.kernel_status(_calc, _exp, float(counter) / float(len(_exp.tuples[-1])))
					# put slave to sleep
					elif (tag == self.tags.exit):
						slaves_avail -= 1
				# print 100.0 %
				_prt.kernel_status(_calc, _exp, 1.0)
				# bcast e_inc[-1]
				_mpi.bcast_e_inc(_exp, _time)
				# collect work time
				_time.timer('work_kernel', _time.order, True)
				#
				return
		
		
		def slave(self, _mpi, _mol, _calc, _pyscf, _exp, _time):
				""" slave function """
				# start work time
				_time.timer('idle_kernel', _time.order)
				# init e_inc list
				if (len(_exp.energy_inc) != _exp.order):
					_exp.energy_inc.append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
				# init data dict
				data = {'error': False}
				# receive work from master
				while (True):
					# start comm time
					_time.timer('comm_kernel', _time.order)
					# ready for task
					_mpi.comm.send(None, dest=0, tag=self.tags.ready)
					# receive drop string
					job_info = _mpi.comm.recv(source=0, tag=MPI.ANY_SOURCE, status=_mpi.stat)
					# start work time
					_time.timer('work_kernel', _time.order)
					# recover tag
					tag = _mpi.stat.Get_tag()
					# do job
					if (tag == self.tags.start):
						# load job info
						_exp.core_idx = job_info['core_idx']
						_exp.cas_idx = job_info['cas_idx']
						_exp.h1e_cas = job_info['h1e_cas']
						_exp.h2e_cas = job_info['h2e_cas']
						_exp.e_core = job_info['e_core']
						# run correlated calc
						try:
							_exp.energy_inc[-1][job_info['index']] = _pyscf.corr_calc(_mol, _calc, _exp)
							# sum up energy increment
							self.summation(_exp, job_info['index'])
						except Exception as err:
							data['error'] = True
							data['host'] = _mpi.host
							data['core_idx'] = _exp.core_idx
							data['cas_idx'] = _exp.cas_idx
							data['pyscf_err'] = err
							pass
						finally:
							# start comm time
							_time.timer('comm_kernel', _time.order)
							# report status back to master
							_mpi.comm.send(None, dest=0, tag=self.tags.done)
							# start work time
							_time.timer('work_kernel', _time.order)
							# write info into data dict
							data['index'] = job_info['index']
							data['e_corr'] = _exp.energy_inc[-1][job_info['index']]
							data['t_work'] = _time.timings['work_kernel'][-1]
							data['t_comm'] = _time.timings['comm_kernel'][-1]
							data['t_idle'] = _time.timings['idle_kernel'][-1]
							# start comm time
							_time.timer('comm_kernel', _time.order)
							# send data back to master
							_mpi.comm.send(data, dest=0, tag=self.tags.data)
							# start work time
							_time.timer('work_kernel', _time.order)
					# exit
					elif (tag == self.tags.exit):
						break
				# start comm time
				_time.timer('comm_kernel', _time.order)
				# send exit signal to master
				_mpi.comm.send(None, dest=0, tag=self.tags.exit)
				# bcast e_inc[-1]
				_mpi.bcast_e_inc(_exp, _time)
				# collect comm time
				_time.timer('work_kernel', _time.order, True)
				#
				return


