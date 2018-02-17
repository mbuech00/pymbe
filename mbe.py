#!/usr/bin/env python
# -*- coding: utf-8 -*

""" mbe.py: mbe class """

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
from math import fsum

from exp import ExpCls
import drv


class MBECls():
		""" mbe class """
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
				res = np.zeros(len(_exp.energy['inc'])-1, dtype=np.float64)
				# compute contributions from lower-order increments
				for count, i in enumerate(range(_exp.order-1, _exp.start_order-1, -1)):
					# test if tuple is a subset
					combs = _exp.tuples[-1][_idx, self.comb_index(_exp.order, i)]
					dt = np.dtype((np.void, _exp.tuples[i-_exp.start_order].dtype.itemsize * \
									_exp.tuples[i-_exp.start_order].shape[1]))
					match = np.nonzero(np.in1d(_exp.tuples[i-_exp.start_order].view(dt).reshape(-1),
										combs.view(dt).reshape(-1)))[0]
					# add up lower-order increments
					res[count] = fsum(_exp.energy['inc'][i-_exp.start_order][match])
				# now compute increment
				_exp.energy['inc'][-1][_idx] -= fsum(res)
				#
				return


		def main(self, _mpi, _mol, _calc, _kernel, _exp, _prt, _rst):
				""" energy mbe phase """
				# init micro_conv list
				if (_mpi.global_master and (_exp.level == 'macro')):
					if (len(_exp.micro_conv) < _exp.order):
						_exp.micro_conv.append(np.zeros(len(_exp.tuples[-1]), dtype=np.int32))
				# mpi parallel version
				if (_mpi.parallel):
					self.master(_mpi, _mol, _calc, _kernel, _exp, _prt, _rst)
				else:
					self.serial(_mpi, _mol, _calc, _kernel, _exp, _prt, _rst)
				# sum up total energy
				e_tmp = fsum(_exp.energy['inc'][-1])
				if (_exp.order > _exp.start_order): e_tmp += _exp.energy['tot'][-1]
				# add to total energy list
				_exp.energy['tot'].append(e_tmp)
				#
				return
		
	
		def serial(self, _mpi, _mol, _calc, _kernel, _exp, _prt, _rst):
				""" energy mbe phase """
				# print and time logical
				do_print = _mpi.global_master and (not ((_calc.exp_type == 'combined') and (_exp.level == 'micro')))
				# init time
				if (do_print):
					if (len(_exp.time_mbe) < _exp.order): _exp.time_mbe.append(0.0)
				# micro driver instantiation
				if (_exp.level == 'macro'):
					drv_micro = bg_drv.DrvCls(_mol, 'virtual') 
				# determine start index
				start = np.argmax(_exp.energy['inc'][-1] == 0.0)
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
						drv_micro.main(_mpi, _mol, _calc, _kernel, exp_micro, _prt, _rst)
						# store results
						_exp.energy['inc'][-1][i] = exp_micro.energy['tot'][-1]
						_exp.micro_conv[-1][i] = exp_micro.order
						# sum up energy increment
						self.summation(_calc, _exp, i)
					else:
						# generate input
						_exp.core_idx, _exp.cas_idx = _kernel.core_cas(_mol, _exp, _exp.tuples[-1][i])
						# perform calc
						if ((_exp.order == _exp.start_order) and (_calc.exp_ref['METHOD'] == 'CASSCF')):
							e_model = _calc.energy['cas_model']
						else:
							e_model = _kernel.main_calc(_mol, _calc, _exp, _calc.exp_model['METHOD'])
						if (_calc.exp_base['METHOD'] is None):
							_exp.energy['inc'][-1][i] = e_model
						else:
							if ((_exp.order == _exp.start_order) and (_calc.exp_ref['METHOD'] == 'CASSCF')):
								e_base = _calc.energy['cas_base']
							else:
								e_base = _kernel.main_calc(_mol, _calc, _exp, _calc.exp_base['METHOD'])
							_exp.energy['inc'][-1][i] = e_model - e_base
						# calc increment
						self.summation(_calc, _exp, i)
					if (do_print):
						# print status
						_prt.mbe_status(_calc, _exp, float(i+1) / float(len(_exp.tuples[-1])))
						# collect time
						_exp.time_mbe[-1] += MPI.Wtime() - time
						# write restart files
						_rst.write_mbe(_calc, _exp, False)
				#
				return

	
		def master(self, _mpi, _mol, _calc, _kernel, _exp, _prt, _rst):
				""" master function """
				# wake up slaves
				if (_exp.level == 'macro'):
					msg = {'task': 'mbe_local_master', 'exp_order': _exp.order}
					# set communicator
					comm = _mpi.master_comm
					# number of workers
					slaves_avail = num_slaves = _mpi.num_local_masters
				else:
					msg = {'task': 'mbe_slave', 'exp_order': _exp.order}
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
					_prt.mbe_status(_calc, _exp, 0.0)
					# master calculates increments
					for i in range(len(_exp.tuples[0])):
						# generate input
						_exp.core_idx, _exp.cas_idx = _kernel.core_cas(_mol, _exp, _exp.tuples[0][i])
						# perform calc
						if (_calc.exp_ref['METHOD'] == 'CASSCF'):
							e_model = _calc.energy['cas_model']
						else:
							e_model = _kernel.main_calc(_mol, _calc, _exp, _calc.exp_model['METHOD'])
						if (_calc.exp_base['METHOD'] is None):
							e_base = 0.0
						else:
							if (_calc.exp_ref['METHOD'] == 'CASSCF'):
								e_base = _calc.energy['cas_base']
							else:
								e_base = _kernel.main_calc(_mol, _calc, _exp, _calc.exp_base['METHOD'])
						_exp.energy['inc'][0][i] = e_model - e_base
						# calc increment
						self.summation(_calc, _exp, i)
						# verbose print
						if (_mol.verbose_prt):
							print(' cas = {0:} , e_model = {1:.6f} , e_base = {2:.6f} , e_inc = {3:.6f}'.\
									format(_exp.cas_idx, e_model, e_base, _exp.energy['inc'][0][i]))
						# print status
						if (_mol.verbose_prt): _prt.mbe_status(_calc, _exp, float(i+1) / float(len(_exp.tuples[0])))
					# collect time
					_exp.time_mbe.append(MPI.Wtime() - time)
					# print status
					if (not _mol.verbose_prt): _prt.mbe_status(_calc, _exp, 1.0)
					# bcast energy
					_mpi.bcast_energy(_mol, _calc, _exp, comm)
					# bcast mo coefficients
					if (_calc.exp_ref['METHOD'] == 'CASSCF'): _mpi.bcast_mo_info(_mol, _calc, _mpi.global_comm)
				else:
					# init job_info dictionary
					job_info = {}
					# init job index
					i = np.argmax(_exp.energy['inc'][-1] == 0.0)
					# init stat counter
					counter = i
					# print status for START
					if (_mpi.global_master and (not (_exp.level == 'macro'))):
						_prt.mbe_status(_calc, _exp, float(counter) / float(len(_exp.tuples[-1])))
					# init time
					if (_mpi.global_master and (len(_exp.time_mbe) < _exp.order)):
						_exp.time_mbe.append(0.0)
					time = MPI.Wtime()
					# loop until no slaves left
					while (slaves_avail >= 1):
						# receive data dict
						data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=_mpi.stat)
						# collect time
						if (_mpi.global_master):
							_exp.time_mbe[-1] += MPI.Wtime() - time
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
							# collect energies
							_exp.energy['inc'][-1][data['index']] = data['e_inc']
							# write to micro_conv
							if (_mpi.global_master and (_exp.level == 'macro')):
								self.summation(_calc, _exp, 'inc', data['index'])
								_exp.micro_conv[-1][data['index']] = data['micro_order']
							# write restart files
							if (_mpi.global_master and ((((data['index']+1) % int(_rst.rst_freq)) == 0) or (_exp.level == 'macro'))):
								_rst.write_mbe(_calc, _exp, False)
							# increment stat counter
							counter += 1
							# print status
							if (_mpi.global_master and (((((data['index']+1) % 1000) == 0) or (_exp.level == 'macro')) or _mol.verbose_prt)):
								_prt.mbe_status(_calc, _exp, float(counter) / float(len(_exp.tuples[-1])))
						# put slave to sleep
						elif (tag == self.tags.exit):
							slaves_avail -= 1
					# print 100.0 %
					if (_mpi.global_master and (not (_exp.level == 'macro'))):
						if (not _mol.verbose_prt):
							_prt.mbe_status(_calc, _exp, 1.0)
					# bcast energies
					_mpi.bcast_energy(_mol, _calc, _exp, comm)
				#
				return
		
		
		def slave(self, _mpi, _mol, _calc, _kernel, _exp, _rst=None):
				""" slave function """
				# set communicator and possible micro driver instantiation
				if (_exp.level == 'macro'):
					comm = _mpi.master_comm
					# micro driver instantiation
					drv_micro = bg_drv.DrvCls(_mol, 'virtual') 
				else:
					comm = _mpi.local_comm
				# init energies
				if (len(_exp.energy['inc']) < _exp.order):
					_exp.energy['inc'].append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
				# ref_calc
				if (_exp.order == _exp.start_order):
					# receive energy
					_mpi.bcast_energy(_mol, _calc, _exp, comm)
					# receive mo coefficients
					if (_calc.exp_ref['METHOD'] == 'CASSCF'): _mpi.bcast_mo_info(_mol, _calc, _mpi.global_comm)
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
								drv_micro.main(_mpi, _mol, _calc, _kernel, exp_micro, None, _rst)
								# store micro convergence
								data['micro_order'] = exp_micro.order
								# write info into data dict
								data['index'] = job_info['index']
								data['e_inc'] = exp_micro.energy['tot'][-1]
								# send data back to local master
								comm.send(data, dest=0, tag=self.tags.done)
							else:
								# generate input
								_exp.core_idx, _exp.cas_idx = _kernel.core_cas(_mol, _exp, _exp.tuples[-1][job_info['index']])
								# perform calc
								e_model = _kernel.main_calc(_mol, _calc, _exp, _calc.exp_model['METHOD'])
								if (_calc.exp_base['METHOD'] is None):
									e_base = 0.0
								else:
									e_base = _kernel.main_calc(_mol, _calc, _exp, _calc.exp_base['METHOD'])
								_exp.energy['inc'][-1][job_info['index']] = e_model - e_base
								# calc increment
								self.summation(_calc, _exp, job_info['index'])
								# verbose print
								if (_mol.verbose_prt):
									print(' cas = {0:} , e_model = {1:.6f} , e_base = {2:.6f} , e_inc = {3:.6f}'.\
											format(_exp.cas_idx, e_model, e_base, _exp.energy['inc'][-1][job_info['index']]))
								# write info into data dict
								data['index'] = job_info['index']
								data['e_inc'] = _exp.energy['inc'][-1][job_info['index']]
								# send data back to local master
								comm.send(data, dest=0, tag=self.tags.done)
						# exit
						elif (tag == self.tags.exit):
							break
					# send exit signal to master
					comm.send(None, dest=0, tag=self.tags.exit)
					# bcast energies
					_mpi.bcast_energy(_mol, _calc, _exp, comm)
				#
				return


