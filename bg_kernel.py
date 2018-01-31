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
import bg_drv


class KernCls():
		""" kernel class """
		def __init__(self):
				""" init parameters """
				# set tags
				self.tags = self.enum('ready', 'done', 'exit', 'start')
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
				index = np.fromiter(chain.from_iterable(combinations(range(_n), _k)), int,count=count * _k)
				#
				return index.reshape(-1, _k)


		def summation(self, _calc, _exp, _idx):
				""" energy summation """
				# init res
				res = np.zeros(len(_exp.energy_inc)-1, dtype=np.float64)
				# compute contributions from lower-order increments
				for count, i in enumerate(range(_exp.order-1, _exp.start_order-1, -1)):
					# test if tuple is a subset
					combs = _exp.tuples[-1][_idx, self.comb_index(_exp.order, i)]
					dt = np.dtype((np.void, _exp.tuples[i-_exp.start_order].dtype.itemsize * \
									_exp.tuples[i-_exp.start_order].shape[1]))
					match = np.nonzero(np.in1d(_exp.tuples[i-_exp.start_order].view(dt).reshape(-1),
										combs.view(dt).reshape(-1)))[0]
					# add up lower-order increments
					for j in match: res[count] += _exp.energy_inc[i-_exp.start_order][j]
				# now compute increment
				_exp.energy_inc[-1][_idx] -= np.sum(res)
				#
				return


		def main(self, _mpi, _mol, _calc, _pyscf, _exp, _prt, _rst):
				""" energy kernel phase """
				# init micro_conv list
				if (_mpi.global_master and (_exp.level == 'macro')):
					if (len(_exp.micro_conv) < _exp.order):
						_exp.micro_conv.append(np.zeros(len(_exp.tuples[-1]), dtype=np.int32))
				# mpi parallel version
				if (_mpi.parallel):
					self.master(_mpi, _mol, _calc, _pyscf, _exp, _prt, _rst)
				else:
					self.serial(_mpi, _mol, _calc, _pyscf, _exp, _prt, _rst)
				# sum of total energy
				e_tmp = np.sum(_exp.energy_inc[-1][np.where(np.abs(_exp.energy_inc[-1]) >= _calc.tolerance)])
				if (_exp.order > _exp.start_order): e_tmp += _exp.energy_tot[-1]
				# add to total energy list
				_exp.energy_tot.append(e_tmp)
				# check for convergence wrt total energy
#				if ((_exp.order > _exp.start_order+1) and \
#					(abs(_exp.energy_tot[-1] - _exp.energy_tot[-2]) < 1.0e-06)):
#						_exp.conv_energy.append(True)
				#
				return
		
	
		def serial(self, _mpi, _mol, _calc, _pyscf, _exp, _prt, _rst):
				""" energy kernel phase """
				# perform calculations
				if ((_exp.order == _exp.start_order) and (_calc.exp_ref['METHOD'] in ['CASCI','CASSCF'])):
					# start time
					time = MPI.Wtime()
					# print status
					_prt.kernel_status(_calc, _exp, 0.0)
					# generate input
					_exp.core_idx, _exp.cas_idx = _pyscf.core_cas(_mol, _exp, _exp.tuples[0][0])
					# perform calc
					_exp.energy_inc[0][0] = _pyscf.main_calc(_mol, _calc, _exp)
					# print status
					_prt.kernel_status(_calc, _exp, 1.0)
					# collect time
					_exp.time_kernel.append(MPI.Wtime() - time)
				else:
					# print and time logical
					do_print = _mpi.global_master and (not ((_calc.exp_type == 'combined') and (_exp.level == 'micro')))
					# init time
					if (do_print):
						if (len(_exp.time_kernel) < _exp.order): _exp.time_kernel.append(0.0)
					# micro driver instantiation
					if (_exp.level == 'macro'):
						drv_micro = bg_drv.DrvCls(_mol, 'virtual') 
					# determine start index
					start = np.argmax(_exp.energy_inc[-1] == 0.0)
					# loop over tuples
					for i in range(start, len(_exp.tuples[-1])):
						# start time
						if (do_print): time = MPI.Wtime()
						# run correlated calc
						if (_exp.level == 'macro'):
							# micro exp instantiation
							exp_micro = ExpCls(_mpi, _mol, _calc, 'virtual')
							# mark expansion as micro 
							exp_micro.level = 'micro'
							# transfer incl_idx
							exp_micro.incl_idx = _exp.tuples[-1][i].tolist()
							# make recursive call to driver with micro exp
							drv_micro.main(_mpi, _mol, _calc, _pyscf, exp_micro, _prt, _rst)
							# store results
							_exp.energy_inc[-1][i] = exp_micro.energy_tot[-1]
							_exp.micro_conv[-1][i] = exp_micro.order
						else:
							# generate input
							_exp.core_idx, _exp.cas_idx = _pyscf.core_cas(_mol, _exp, _exp.tuples[-1][i])
							# perform calc
							_exp.energy_inc[-1][i] = _pyscf.main_calc(_mol, _calc, _exp)
						# sum up energy increment
						self.summation(_calc, _exp, i)
						if (do_print):
							# print status
							_prt.kernel_status(_calc, _exp, float(i+1) / float(len(_exp.tuples[-1])))
							# collect time
							_exp.time_kernel[-1] += MPI.Wtime() - time
							# write restart files
							_rst.write_kernel(_calc, _exp, False)
				#
				return

	
		def master(self, _mpi, _mol, _calc, _pyscf, _exp, _prt, _rst):
				""" master function """
				# wake up slaves
				if (_exp.level == 'macro'):
					msg = {'task': 'kernel_local_master', 'exp_order': _exp.order}
					# set communicator
					comm = _mpi.master_comm
					# number of workers
					slaves_avail = num_slaves = _mpi.num_local_masters
				else:
					msg = {'task': 'kernel_slave', 'exp_order': _exp.order}
					# set communicator
					comm = _mpi.local_comm
					# number of workers
					slaves_avail = num_slaves = _mpi.local_size - 1
				# bcast msg
				comm.bcast(msg, root=0)
				# perform calculations
				if (_exp.order == _exp.start_order):
					# start time
					time = MPI.Wtime()
					# print status
					_prt.kernel_status(_calc, _exp, 0.0)
					# master calculates increments
					for i in range(len(_exp.tuples[0])):
						# generate input
						_exp.core_idx, _exp.cas_idx = _pyscf.core_cas(_mol, _exp, _exp.tuples[0][i])
						# perform calc
						_exp.energy_inc[0][i] = _pyscf.main_calc(_mol, _calc, _exp)
						# print status
						if (_mol.verbose_prt): _prt.kernel_status(_calc, _exp, float(i+1) / float(len(_exp.tuples[0])))
					# collect time
					_exp.time_kernel.append(MPI.Wtime() - time)
					# print status
					if (not _mol.verbose_prt): _prt.kernel_status(_calc, _exp, 1.0)
					# bcast energy
					_mpi.bcast_e_inc(_mol, _calc, _exp, comm)
				else:
					# init job_info dictionary
					job_info = {}
					# init job index
					i = np.argmax(_exp.energy_inc[-1] == 0.0)
					# init stat counter
					counter = i
					# print status for START
					if (_mpi.global_master and (not (_exp.level == 'macro'))):
						_prt.kernel_status(_calc, _exp, float(counter) / float(len(_exp.tuples[-1])))
					# init time
					if (_mpi.global_master and (len(_exp.time_kernel) < _exp.order)):
						_exp.time_kernel.append(0.0)
					time = MPI.Wtime()
					# loop until no slaves left
					while (slaves_avail >= 1):
						# receive data dict
						data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=_mpi.stat)
						# collect time
						if (_mpi.global_master):
							_exp.time_kernel[-1] += MPI.Wtime() - time
							time = MPI.Wtime()
						# probe for source and tag
						source = _mpi.stat.Get_source(); tag = _mpi.stat.Get_tag()
						# slave is ready
						if (tag == self.tags.ready):
							# any jobs left?
							if (i <= (len(_exp.tuples[-1]) - 1)):
								# start time
								if (_mpi.global_master): time = MPI.Wtime()
								# store job index
								job_info['index'] = i
								# send string dict
								comm.send(job_info, dest=source, tag=self.tags.start)
								# increment job index
								i += 1
							else:
								# send exit signal
								comm.send(None, dest=source, tag=self.tags.exit)
						# receive result from slave
						elif (tag == self.tags.done):
							# write to e_inc
							_exp.energy_inc[-1][data['index']] = data['e_inc']
							# write to micro_conv
							if (_mpi.global_master and (_exp.level == 'macro')):
								self.summation(_calc, _exp, data['index'])
								_exp.micro_conv[-1][data['index']] = data['micro_order']
							# write restart files
							if (_mpi.global_master and ((((data['index']+1) % int(_rst.rst_freq)) == 0) or (_exp.level == 'macro'))):
								_rst.write_kernel(_calc, _exp, False)
							# increment stat counter
							counter += 1
							# print status
							if (_mpi.global_master and (((((data['index']+1) % 1000) == 0) or (_exp.level == 'macro')) or _mol.verbose_prt)):
								_prt.kernel_status(_calc, _exp, float(counter) / float(len(_exp.tuples[-1])))
						# put slave to sleep
						elif (tag == self.tags.exit):
							slaves_avail -= 1
					# print 100.0 %
					if (_mpi.global_master and (not (_exp.level == 'macro'))): _prt.kernel_status(_calc, _exp, 1.0)
					# bcast e_inc[-1]
					_mpi.bcast_e_inc(_mol, _calc, _exp, comm)
				#
				return
		
		
		def slave(self, _mpi, _mol, _calc, _pyscf, _exp, _rst=None):
				""" slave function """
				# set communicator and possible micro driver instantiation
				if (_exp.level == 'macro'):
					comm = _mpi.master_comm
					# micro driver instantiation
					drv_micro = bg_drv.DrvCls(_mol, 'virtual') 
				else:
					comm = _mpi.local_comm
				# init e_inc list
				if (len(_exp.energy_inc) < _exp.order):
					_exp.energy_inc.append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
				# ref_calc
				if (_exp.order == _exp.start_order):
					# receive energy
					_mpi.bcast_e_inc(_mol, _calc, _exp, comm)
				else:
					# init data dict
					data = {}
					# receive work from master
					while (True):
						# ready for task
						comm.send(None, dest=0, tag=self.tags.ready)
						# receive drop string
						job_info = comm.recv(source=0, tag=MPI.ANY_TAG, status=_mpi.stat)
						# recover tag
						tag = _mpi.stat.Get_tag()
						# do job
						if (tag == self.tags.start):
							# load job info
							if (_exp.level == 'macro'):
								# micro exp instantiation
								exp_micro = ExpCls(_mpi, _mol, _calc, 'virtual')
								# mark expansion as micro 
								exp_micro.level = 'micro'
								# transfer incl_idx
								exp_micro.incl_idx = sorted(_exp.tuples[-1][job_info['index']].tolist())
								# make recursive call to driver with micro exp
								drv_micro.main(_mpi, _mol, _calc, _pyscf, exp_micro, None, _rst)
								# store micro convergence
								data['micro_order'] = exp_micro.order
								# write info into data dict
								data['index'] = job_info['index']
								data['e_inc'] = exp_micro.energy_tot[-1]
								# send data back to local master
								comm.send(data, dest=0, tag=self.tags.done)
							else:
								# generate input
								_exp.core_idx, _exp.cas_idx = _pyscf.core_cas(_mol, _exp, _exp.tuples[-1][job_info['index']])
								# run correlated calc
								_exp.energy_inc[-1][job_info['index']] = _pyscf.main_calc(_mol, _calc, _exp)
								# sum up energy increment
								self.summation(_calc, _exp, job_info['index'])
								# write info into data dict
								data['index'] = job_info['index']
								data['e_inc'] = _exp.energy_inc[-1][job_info['index']]
								# send data back to local master
								comm.send(data, dest=0, tag=self.tags.done)
						# exit
						elif (tag == self.tags.exit):
							break
					# send exit signal to master
					comm.send(None, dest=0, tag=self.tags.exit)
					# bcast e_inc[-1]
					_mpi.bcast_e_inc(_mol, _calc, _exp, comm)
				#
				return


