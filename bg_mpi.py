#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi.py: MPI class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from os import getcwd, mkdir, chdir
from os.path import isfile
from shutil import copy, rmtree


class MPICls():
		""" mpi parameters """
		def __init__(self):
				""" init parameters """
				self.comm = MPI.COMM_WORLD
				self.parallel = self.comm.Get_size() > 1
				self.size = self.comm.Get_size()
				self.rank = self.comm.Get_rank()
				self.master = (self.rank == 0)
				self.name = MPI.Get_processor_name()
				self.stat = MPI.Status()


		def bcast_hf_int(self, _mol):
				""" bcast hf and int info """
				if (self.master):
					# bcast to slaves
					self.comm.bcast(_mol.e_hf, root=0)
					self.comm.bcast(_mol.norb, root=0)
					self.comm.bcast(_mol.nocc, root=0)
					self.comm.bcast(_mol.nvirt, root=0)
					self.comm.bcast(_mol.h1e, root=0)
					self.comm.bcast(_mol.h2e, root=0)
				else:
					# receive from master
					_mol.e_hf = self.comm.bcast(None, root=0)
					_mol.norb = self.com.bcast(None, root=0)
					_mol.nocc = self.comm.bcast(None, root=0)
					_mol.nvirt = self.comm.bcast(None, root=0)
					_mol.h1e = self.comm.bcast(None, root=0)
					_mol.h2e = self.comm.bcast(None, root=0)
				#
				return


		def bcast_rst(self, _calc, _exp, _time):
				""" bcast restart files """
				if (self.master):
					# wake up slaves 
					msg = {'task': 'bcast_rst'}
					# bcast
					self.comm.bcast(msg, root=0)
					# determine start index for energy kernel phase
					e_inc_end = np.argmax(_exp.energy_inc[-1] == 0.0)
					if (e_inc_end == 0): e_inc_end = len(_exp.energy_inc[-1])
					# collect exp_info
					exp_info = {'len_tup': [len(_exp.tuples[i]) for i in range(1,len(_exp.tuples))],\
								'len_e_inc': [len(_exp.energy_inc[i]) for i in range(len(_exp.energy_inc))],\
								'min_order': _calc.exp_min_order, 'e_inc_end': e_inc_end}
					# bcast info
					self.comm.bcast(exp_info, root=0)
					# bcast tuples
					for i in range(1,len(_exp.tuples)):
						self.comm.Bcast([_exp.tuples[i],MPI.INT], root=0)
					# bcast energy increments
					for i in range(len(_exp.energy_inc)):
						if (i < (len(_exp.energy_inc)-1)):
							self.comm.Bcast([_exp.energy_inc[i],MPI.DOUBLE], root=0)
						else:
							self.comm.Bcast([_exp.energy_inc[i][:e_inc_end],MPI.DOUBLE], root=0)
					# collect time_info
					for i in range(1,self.size):
						time_info = {'kernel': [_time.mpi_time_work[1][i],
									_time.mpi_time_comm[1][i],_time.mpi_time_idle[1][i]],\
									'summation': [_time.mpi_time_work[2][i],
									_time.mpi_time_comm[2][i],_time.mpi_time_idle[2][i]],\
									'screen': [_time.mpi_time_work[0][i],
									_time.mpi_time_comm[0][i],_time.mpi_time_idle[0][i]]}
						self.comm.send(time_info, dest=i)
				else:
					# receive exp_info
					info = self.comm.bcast(None, root=0)
					# set min_order
					_calc.exp_min_order = info['min_order']
					# receive tuples
					for i in range(len(info['len_tup'])):
						buff = np.empty([info['len_tup'][i],i+2], dtype=np.int32)
						self.comm.Bcast([buff,MPI.INT], root=0)
						_exp.tuples.append(buff)
					# receive e_inc
					for i in range(len(info['len_e_inc'])):
						buff = np.zeros(info['len_e_inc'][i], dtype=np.float64)
						if (i < (len(info['len_e_inc'])-1)):
							self.comm.Bcast([buff,MPI.DOUBLE], root=0)
						else:
							self.comm.Bcast([buff[:info['e_inc_end']],MPI.DOUBLE], root=0)
						_exp.energy_inc.append(buff)
					# for e_inc[-1], make sure that this is distributed among the slaves
					for i in range(info['e_inc_end']):
						if ((i % (self.size-1)) != (self.rank-1)): _exp.energy_inc[-1][i] = 0.0 
					# receive time_info
					_time_info = self.comm.recv(source=0, status=self.stat)
					_time.time_work_kernel = time_info['kernel'][0]
					_time.time_comm_kernel = time_info['kernel'][1]
					_time.time_idle_kernel = time_info['kernel'][2]
					_time.time_work_summation = time_info['summation'][0]
					_time.time_comm_summation = time_info['summation'][1]
					_time.time_idle_summation = time_info['summation'][2]
					_time.time_work_screen = time_info['screen'][0]
					_time.time_comm_screen = time_info['screen'][1]
					_time.time_idle_screen = time_info['screen'][2]
				#
				return

	
		def allred_e_inc(self, _exp, _time):
				""" allreduce e_inc[-1] """
				# start idle time
				_time.timer('idle_summation', _exp.order)
				# barrier
				self.comm.Barrier()
				# start comm time
				_time.timer('comm_summation', _exp.order)
				# init receive buffer
				recv_buff = np.zeros(len(_exp.energy_inc[-1]), dtype=np.float64)
				# now do Allreduce
				self.comm.Allreduce([_exp.energy_inc[-1],MPI.DOUBLE], [recv_buff,MPI.DOUBLE], op=MPI.SUM)
				# start work time
				_time.timer('work_summation', _exp.order)
				# finally, overwrite e_inc[-1]
				_exp.energy_inc[-1] = recv_buff
				#
				return

	
		def red_orb_ent(self, _exp, _time, _send_buff, _recv_buff):
				""" reduce orb_ent onto master proc. """
				# start idle time
				_time.timer('idle_screen', _exp.order)
				# collect idle time
				self.comm.Barrier()
				# start comm time
				_time.timer('comm_screen', _exp.order)
				# reduce tmp into recv_buff
				self.comm.Reduce([_send_buff,MPI.DOUBLE], [_recv_buff,MPI.DOUBLE], op=MPI.SUM, root=0)
				# collect comm time
				_time.timer('comm_screen', _exp.order, True)
				#
				return


		def bcast_tup(self, _exp, _time, _buff):
				""" master/slave routine for bcasting total number of tuples """
				if (self.master):
					# start comm time
					_time.timer('comm_screen', _exp.order)
					# init bcast dict
					tup_info = {'tup_len': len(_buff)}
					# bcast
					self.comm.bcast(tup_info, root=0)
				# start idle time
				_time.timer('idle_screen', _exp.order)
				# all meet at barrier
				self.comm.Barrier()
				# start comm time
				_time.timer('comm_screen', _exp.order)
				# bcast buffer
				self.comm.Bcast([_buff,MPI.INT], root=0)
				# start work time
				_time.timer('work_screen', _exp.order)
				# append tup[-1] with buff
				if (len(_buff) >= 1): _exp.tuples.append(_buff)
				# end work time
				_time.timer('work_screen', _exp.order, True)
				#
				return

	
