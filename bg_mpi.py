#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi.py: MPI class for Bethe-Goldstone correlation calculations."""

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
from os import getcwd, mkdir, chdir
from os.path import isfile
from shutil import copy, rmtree


class MPICls():
		""" mpi parameters """
		def __init__(self):
				""" init parameters """
				self.comm = MPI.COMM_WORLD
				self.parallel = (self.comm.Get_size() > 1)
				self.size = self.comm.Get_size()
				self.rank = self.comm.Get_rank()
				self.master = (self.rank == 0)
				self.host = MPI.Get_processor_name()
				self.stat = MPI.Status()
				# set custom exception hook
				if (self.master): self.set_exc_hook()
				#
				return


		def set_exc_hook(self):
				""" set an exception hook for aborting mpi """
				# save sys.excepthook
				sys_excepthook = sys.excepthook
				# define mpi exception hook
				def mpi_excepthook(_t, _v, _tb):
					sys_excepthook(_t, _v, _tb)
					self.comm.Abort(1)
				# overwrite sys.excepthook
				sys.excepthook = mpi_excepthook
				#
				return


		def bcast_rst_info(self, _rst):
				""" bcast rst info """
				if (self.master):
					self.comm.bcast(_rst.restart, root=0)
					self.comm.bcast(_rst.rst_freq, root=0)
				else:
					_rst.restart = self.comm.bcast(None, root=0)
					_rst.rst_freq = self.comm.bcast(None, root=0)
				#
				return


		def bcast_mol_info(self, _mol):
				""" bcast mol info """
				if (self.master):
					self.comm.bcast(_mol.atom, root=0)
					self.comm.bcast(_mol.charge, root=0)
					self.comm.bcast(_mol.spin, root=0)
					self.comm.bcast(_mol.symmetry, root=0)
					self.comm.bcast(_mol.basis, root=0)
					self.comm.bcast(_mol.unit, root=0)
					self.comm.bcast(_mol.frozen, root=0)
					self.comm.bcast(_mol.verbose, root=0)
				else:
					_mol.atom = self.comm.bcast(None, root=0)
					_mol.charge = self.comm.bcast(None, root=0)
					_mol.spin = self.comm.bcast(None, root=0)
					_mol.symmetry = self.comm.bcast(None, root=0)
					_mol.basis = self.comm.bcast(None, root=0)
					_mol.unit = self.comm.bcast(None, root=0)
					_mol.frozen = self.comm.bcast(None, root=0)
					_mol.verbose = self.comm.bcast(None, root=0)
				#
				return


		def bcast_calc_info(self, _calc):
				""" bcast calc info """
				if (self.master):
					# bcast to slaves
					self.comm.bcast(_calc.exp_model, root=0)
					self.comm.bcast(_calc.exp_type, root=0)
					self.comm.bcast(_calc.exp_base, root=0)
					self.comm.bcast(_calc.exp_thres, root=0)
					self.comm.bcast(_calc.exp_damp, root=0)
					self.comm.bcast(_calc.exp_max_order, root=0)
					self.comm.bcast(_calc.exp_occ, root=0)
					self.comm.bcast(_calc.exp_virt, root=0)
					self.comm.bcast(_calc.energy_thres, root=0)
				else:
					# receive from master
					_calc.exp_model = self.comm.bcast(None, root=0)
					_calc.exp_type = self.comm.bcast(None, root=0)
					_calc.exp_base = self.comm.bcast(None, root=0)
					_calc.exp_thres = self.comm.bcast(None, root=0)
					_calc.exp_damp = self.comm.bcast(None, root=0)
					_calc.exp_max_order = self.comm.bcast(None, root=0)
					_calc.exp_occ = self.comm.bcast(None, root=0)
					_calc.exp_virt = self.comm.bcast(None, root=0)
					_calc.energy_thres = self.comm.bcast(None, root=0)
				#
				return


		def bcast_hf_base(self, _mol):
				""" bcast hf and base info """
				if (self.master):
					# bcast to slaves
					self.comm.bcast(_mol.e_hf, root=0)
					self.comm.bcast(_mol.e_ref, root=0)
					self.comm.bcast(_mol.norb, root=0)
					self.comm.bcast(_mol.nocc, root=0)
					self.comm.bcast(_mol.nvirt, root=0)
				else:
					# receive from master
					_mol.e_hf = self.comm.bcast(None, root=0)
					_mol.e_ref = self.comm.bcast(None, root=0)
					_mol.norb = self.comm.bcast(None, root=0)
					_mol.nocc = self.comm.bcast(None, root=0)
					_mol.nvirt = self.comm.bcast(None, root=0)
				#
				return


		def bcast_rst(self, _calc, _exp, _time):
				""" bcast restart files """
				if (self.master):
					# determine start index for energy kernel phase
					e_inc_end = np.argmax(_exp.energy_inc[-1] == 0.0)
					if (e_inc_end == 0): e_inc_end = len(_exp.energy_inc[-1])
					# collect exp_info
					exp_info = {'len_tup': [len(_exp.tuples[i]) for i in range(len(_exp.tuples))],\
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
						time_info = {'kernel': [_time.time_work[0][i],
									_time.time_comm[0][i],_time.time_idle[0][i]],\
									'summation': [_time.time_work[1][i],
									_time.time_comm[1][i],_time.time_idle[1][i]],\
									'screen': [_time.time_work[2][i],
									_time.time_comm[2][i],_time.time_idle[2][i]]}
						self.comm.send(time_info, dest=i)
				else:
					# receive exp_info
					exp_info = self.comm.bcast(None, root=0)
					# set min_order
					_calc.exp_min_order = exp_info['min_order']
					# receive tuples
					for i in range(1,len(exp_info['len_tup'])):
						buff = np.empty([exp_info['len_tup'][i],i+1], dtype=np.int32)
						self.comm.Bcast([buff,MPI.INT], root=0)
						_exp.tuples.append(buff)
					# receive e_inc
					for i in range(len(exp_info['len_e_inc'])):
						buff = np.zeros(exp_info['len_e_inc'][i], dtype=np.float64)
						if (i < (len(exp_info['len_e_inc'])-1)):
							self.comm.Bcast([buff,MPI.DOUBLE], root=0)
						else:
							self.comm.Bcast([buff[:exp_info['e_inc_end']],MPI.DOUBLE], root=0)
						_exp.energy_inc.append(buff)
					# for e_inc[-1], make sure that this is distributed among the slaves
					for i in range(exp_info['e_inc_end']):
						if ((i % (self.size-1)) != (self.rank-1)): _exp.energy_inc[-1][i] = 0.0 
					# receive time_info
					time_info = self.comm.recv(source=0, status=self.stat)
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


		def final(self, _rst):
				""" terminate calculation """
				if (self.master):
					_rst.rm_rst()
					self.comm.bcast({'task': 'exit_slave'}, root=0)
				self.comm.Barrier()
				MPI.Finalize()
	

