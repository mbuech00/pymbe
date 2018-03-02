#!/usr/bin/env python
# -*- coding: utf-8 -*

""" mbe.py: mbe class """

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


		def summation(self, calc, exp, _idx):
				""" energy summation """
				# init res
				res = np.zeros(len(exp.energy['inc'])-1, dtype=np.float64)
				# compute contributions from lower-order increments
				for count, i in enumerate(range(exp.order-1, exp.start_order-1, -1)):
					# test if tuple is a subset
					combs = exp.tuples[-1][_idx, self.comb_index(exp.order, i)]
					dt = np.dtype((np.void, exp.tuples[i-exp.start_order].dtype.itemsize * \
									exp.tuples[i-exp.start_order].shape[1]))
					match = np.nonzero(np.in1d(exp.tuples[i-exp.start_order].view(dt).reshape(-1),
										combs.view(dt).reshape(-1)))[0]
					# add up lower-order increments
					res[count] = fsum(exp.energy['inc'][i-exp.start_order][match])
				# now compute increment
				exp.energy['inc'][-1][_idx] -= fsum(res)
				#
				return


		def main(self, mpi, mol, calc, kernel, exp, prt, rst):
				""" energy mbe phase """
				# init micro_conv list
				if (mpi.global_master and (exp.level == 'macro')):
					if (len(exp.micro_conv) < exp.order):
						exp.micro_conv.append(np.zeros(len(exp.tuples[-1]), dtype=np.int32))
				# mpi parallel version
				if (mpi.parallel):
					self.master(mpi, mol, calc, kernel, exp, prt, rst)
				else:
					self.serial(mpi, mol, calc, kernel, exp, prt, rst)
				# sum up total energy
				e_tmp = fsum(exp.energy['inc'][-1])
				if (exp.order > exp.start_order): e_tmp += exp.energy['tot'][-1]
				# add to total energy list
				exp.energy['tot'].append(e_tmp)
				#
				return
		
	
		def serial(self, mpi, mol, calc, kernel, exp, prt, rst):
				""" energy mbe phase """
				# print and time logical
				do_print = mpi.global_master and (not ((calc.exp_type == 'combined') and (exp.level == 'micro')))
				# init time
				if (do_print):
					if (len(exp.timembe) < exp.order): exp.timembe.append(0.0)
				# micro driver instantiation
				if (exp.level == 'macro'):
					drv_micro = drv.DrvCls(mol, 'virtual') 
				# determine start index
				start = np.argmax(np.isnan(exp.energy['inc'][-1]))
				# loop over tuples
				for i in range(start, len(exp.tuples[-1])):
					# start time
					if (do_print): time = MPI.Wtime()
					# run correlated calc
					if (exp.level == 'macro'):
						# micro exp instantiation
						exp_micro = ExpCls(mpi, mol, calc, 'virtual')
						# mark expansion as micro 
						exp_micro.level = 'micro'
						# transfer incl_idx
						exp_micro.incl_idx = exp.tuples[-1][i].tolist()
						# make recursive call to driver with micro exp
						drv_micro.main(mpi, mol, calc, kernel, exp_micro, prt, rst)
						# store results
						exp.energy['inc'][-1][i] = exp_micro.energy['tot'][-1]
						exp.micro_conv[-1][i] = exp_micro.order
						# sum up energy increment
						self.summation(calc, exp, i)
					else:
						# generate input
						exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, exp.tuples[-1][i])
						# model calc
						e_model = kernel.calc(mol, calc, exp, calc.exp_model['METHOD']) \
									+ (calc.energy['hf'] - calc.energy['ref'])
						# base calc
						if (calc.exp_base['METHOD'] is None):
							e_base = 0.0
						else:
							if ((exp.order == 1) and (mol.spin == 0)):
								e_base = e_model
							else:
								e_base = kernel.calc(mol, calc, exp, calc.exp_base['METHOD']) \
											+ (calc.energy['hf'] - calc.energy['ref_base'])
						exp.energy['inc'][-1][i] = e_model - e_base
						# calc increment
						self.summation(calc, exp, i)
						# verbose print
						if (mol.verboseprt):
							print(' cas = {0:} , e_model = {1:.6f} , e_base = {2:.6f} , e_inc = {3:.6f}'.\
									format(exp.cas_idx, e_model, e_base, exp.energy['inc'][-1][i]))
					if (do_print):
						# print status
						prt.mbe_status(calc, exp, float(i+1) / float(len(exp.tuples[-1])))
						# collect time
						exp.timembe[-1] += MPI.Wtime() - time
						# write restart files
						rst.writembe(calc, exp, False)
				#
				return

	
		def master(self, mpi, mol, calc, kernel, exp, prt, rst):
				""" master function """
				# wake up slaves
				if (exp.level == 'macro'):
					msg = {'task': 'mbe_local_master', 'exp_order': exp.order}
					# set communicator
					comm = mpi.master_comm
					# number of workers
					slaves_avail = num_slaves = mpi.num_local_masters
				else:
					msg = {'task': 'mbe_slave', 'exp_order': exp.order}
					# set communicator
					comm = mpi.local_comm
					# number of workers
					slaves_avail = num_slaves = mpi.local_size - 1
				# bcast msg
				comm.bcast(msg, root=0)
				# perform calculations
				if (exp.order == exp.start_order):
					# start time
					time = MPI.Wtime()
					# print status
					prt.mbe_status(calc, exp, 0.0)
					# master calculates increments
					for i in range(len(exp.tuples[0])):
						# generate input
						exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, exp.tuples[0][i])
						# model calc
						e_model = kernel.calc(mol, calc, exp, calc.exp_model['METHOD']) \
									+ (calc.energy['hf'] - calc.energy['ref'])
						# base calc
						if (calc.exp_base['METHOD'] is None):
							e_base = 0.0
						else:
							if ((exp.order == 1) and (mol.spin == 0)):
								e_base = e_model
							else:
								e_base = kernel.calc(mol, calc, exp, calc.exp_base['METHOD']) \
											+ (calc.energy['hf'] - calc.energy['ref_base'])
						exp.energy['inc'][0][i] = e_model - e_base
						# calc increment
						self.summation(calc, exp, i)
						# verbose print
						if (mol.verboseprt):
							print(' cas = {0:} , e_model = {1:.6f} , e_base = {2:.6f} , e_inc = {3:.6f}'.\
									format(exp.cas_idx, e_model, e_base, exp.energy['inc'][0][i]))
						# print status
						if (mol.verboseprt): prt.mbe_status(calc, exp, float(i+1) / float(len(exp.tuples[0])))
					# collect time
					exp.timembe.append(MPI.Wtime() - time)
					# print status
					if (not mol.verboseprt): prt.mbe_status(calc, exp, 1.0)
					# bcast energy
					mpi.bcast_energy(mol, calc, exp, comm)
				else:
					# init job_info dictionary
					job_info = {}
					# init job index
					i = np.argmax(np.isnan(exp.energy['inc'][-1]))
					# init stat counter
					counter = i
					# print status for START
					if (mpi.global_master and (not (exp.level == 'macro'))):
						prt.mbe_status(calc, exp, float(counter) / float(len(exp.tuples[-1])))
					# init time
					if (mpi.global_master and (len(exp.timembe) < exp.order)):
						exp.timembe.append(0.0)
					time = MPI.Wtime()
					# loop until no slaves left
					while (slaves_avail >= 1):
						# receive data dict
						data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi.stat)
						# collect time
						if (mpi.global_master):
							exp.timembe[-1] += MPI.Wtime() - time
							time = MPI.Wtime()
						# probe for source and tag
						source = mpi.stat.Get_source(); tag = mpi.stat.Get_tag()
						# slave is ready
						if (tag == self.tags.ready):
							# any jobs left?
							if (i <= (len(exp.tuples[-1]) - 1)):
								# start time
								if (mpi.global_master): time = MPI.Wtime()
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
							exp.energy['inc'][-1][data['index']] = data['e_inc']
							# write to micro_conv
							if (mpi.global_master and (exp.level == 'macro')):
								self.summation(calc, exp, 'inc', data['index'])
								exp.micro_conv[-1][data['index']] = data['micro_order']
							# write restart files
							if (mpi.global_master and ((((data['index']+1) % rst.rst_freq) == 0) or (exp.level == 'macro'))):
								rst.writembe(calc, exp, False)
							# increment stat counter
							counter += 1
							# print status
							if (mpi.global_master and (((((data['index']+1) % 1000) == 0) or (exp.level == 'macro')) or mol.verboseprt)):
								prt.mbe_status(calc, exp, float(counter) / float(len(exp.tuples[-1])))
						# put slave to sleep
						elif (tag == self.tags.exit):
							slaves_avail -= 1
					# print 100.0 %
					if (mpi.global_master and (not (exp.level == 'macro'))):
						if (not mol.verboseprt):
							prt.mbe_status(calc, exp, 1.0)
					# bcast energies
					mpi.bcast_energy(mol, calc, exp, comm)
				#
				return
		
		
		def slave(self, mpi, mol, calc, kernel, exp, rst=None):
				""" slave function """
				# set communicator and possible micro driver instantiation
				if (exp.level == 'macro'):
					comm = mpi.master_comm
					# micro driver instantiation
					drv_micro = drv.DrvCls(mol, 'virtual') 
				else:
					comm = mpi.local_comm
				# init energies
				if (len(exp.energy['inc']) < (exp.order - (exp.start_order - 1))):
					inc = np.empty(len(exp.tuples[-1]), dtype=np.float64)
					inc.fill(np.nan)
					exp.energy['inc'].append(inc)
				# refcalc
				if (exp.order == exp.start_order):
					# receive energy
					mpi.bcast_energy(mol, calc, exp, comm)
				else:
					# init data dict
					data = {}
					# receive work from master
					while (True):
						# ready for task
						comm.send(None, dest=0, tag=self.tags.ready)
						# receive drop string
						job_info = comm.recv(source=0, tag=MPI.ANY_TAG, status=mpi.stat)
						# recover tag
						tag = mpi.stat.Get_tag()
						# do job
						if (tag == self.tags.start):
							# load job info
							if (exp.level == 'macro'):
								# micro exp instantiation
								exp_micro = ExpCls(mpi, mol, calc, 'virtual')
								# mark expansion as micro 
								exp_micro.level = 'micro'
								# transfer incl_idx
								exp_micro.incl_idx = sorted(exp.tuples[-1][job_info['index']].tolist())
								# make recursive call to driver with micro exp
								drv_micro.main(mpi, mol, calc, kernel, exp_micro, None, rst)
								# store micro convergence
								data['micro_order'] = exp_micro.order
								# write info into data dict
								data['index'] = job_info['index']
								data['e_inc'] = exp_micro.energy['tot'][-1]
								# send data back to local master
								comm.send(data, dest=0, tag=self.tags.done)
							else:
								# generate input
								exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, exp.tuples[-1][job_info['index']])
								# perform calc
								e_model = kernel.calc(mol, calc, exp, calc.exp_model['METHOD']) \
											+ (calc.energy['hf'] - calc.energy['ref'])
								if (calc.exp_base['METHOD'] is None):
									e_base = 0.0
								else:
									e_base = kernel.calc(mol, calc, exp, calc.exp_base['METHOD']) \
												+ (calc.energy['hf'] - calc.energy['ref_base'])
								exp.energy['inc'][-1][job_info['index']] = e_model - e_base
								# calc increment
								self.summation(calc, exp, job_info['index'])
								# verbose print
								if (mol.verboseprt):
									print(' cas = {0:} , e_model = {1:.6f} , e_base = {2:.6f} , e_inc = {3:.6f}'.\
											format(exp.cas_idx, e_model, e_base, exp.energy['inc'][-1][job_info['index']]))
								# write info into data dict
								data['index'] = job_info['index']
								data['e_inc'] = exp.energy['inc'][-1][job_info['index']]
								# send data back to local master
								comm.send(data, dest=0, tag=self.tags.done)
						# exit
						elif (tag == self.tags.exit):
							break
					# send exit signal to master
					comm.send(None, dest=0, tag=self.tags.exit)
					# bcast energies
					mpi.bcast_energy(mol, calc, exp, comm)
				#
				return


