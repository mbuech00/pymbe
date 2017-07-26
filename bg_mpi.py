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
				self.global_comm = MPI.COMM_WORLD
				self.global_group = self.global_comm.Get_group()
				self.parallel = (self.global_comm.Get_size() > 1)
				self.global_size = self.global_comm.Get_size()
				self.global_rank = self.global_comm.Get_rank()
				self.global_master = (self.global_rank == 0)
				self.host = MPI.Get_processor_name()
				self.stat = MPI.Status()
				# default value
				self.num_groups = 1
				# set custom exception hook
				if (self.global_master): self.set_exc_hook()
				#
				return


		def set_local_groups(self):
				""" define local groups """
				if (self.num_groups == 1):
					self.master_comm = self.local_comm = self.global_comm
					self.local_master = self.global_master
				else:
					# array of ranks (global master excluded)
					ranks = np.arange(1, self.global_size)
					# split into local groups and add global master
					groups = np.array_split(ranks, self.num_groups)
					groups = [np.array([0])] + groups
					# extract local master indices and append to global master index
					masters = [groups[i][0] for i in range(self.num_groups + 1)]
					# define local masters (global master excluded)
					self.local_master = (self.global_rank in masters[1:])
					# define master group and intracomm
					self.master_group = self.global_group.Incl(masters)
					self.master_comm = self.global_comm.Create(self.master_group)
					# define local intracomm based on color
					for i in range(len(groups)):
						if (self.global_rank in groups[i]): color = i
					self.local_comm = self.global_comm.Split(color)
				#
				return


		def set_exc_hook(self):
				""" set an exception hook for aborting mpi """
				# save sys.excepthook
				sys_excepthook = sys.excepthook
				# define mpi exception hook
				def mpi_excepthook(_t, _v, _tb):
					sys_excepthook(_t, _v, _tb)
					self.global_comm.Abort(1)
				# overwrite sys.excepthook
				sys.excepthook = mpi_excepthook
				#
				return


		def bcast_mol_info(self, _mol):
				""" bcast mol info """
				if (self.global_master):
					self.global_comm.bcast(_mol.atom, root=0)
					self.global_comm.bcast(_mol.charge, root=0)
					self.global_comm.bcast(_mol.spin, root=0)
					self.global_comm.bcast(_mol.symmetry, root=0)
					self.global_comm.bcast(_mol.basis, root=0)
					self.global_comm.bcast(_mol.unit, root=0)
					self.global_comm.bcast(_mol.frozen, root=0)
					self.global_comm.bcast(_mol.verbose, root=0)
				else:
					_mol.atom = self.global_comm.bcast(None, root=0)
					_mol.charge = self.global_comm.bcast(None, root=0)
					_mol.spin = self.global_comm.bcast(None, root=0)
					_mol.symmetry = self.global_comm.bcast(None, root=0)
					_mol.basis = self.global_comm.bcast(None, root=0)
					_mol.unit = self.global_comm.bcast(None, root=0)
					_mol.frozen = self.global_comm.bcast(None, root=0)
					_mol.verbose = self.global_comm.bcast(None, root=0)
				#
				return


		def bcast_calc_info(self, _calc):
				""" bcast calc info """
				if (self.global_master):
					# bcast to slaves
					self.global_comm.bcast(_calc.exp_model, root=0)
					self.global_comm.bcast(_calc.exp_type, root=0)
					self.global_comm.bcast(_calc.exp_base, root=0)
					self.global_comm.bcast(_calc.exp_thres, root=0)
					self.global_comm.bcast(_calc.exp_max_order, root=0)
					self.global_comm.bcast(_calc.exp_occ, root=0)
					self.global_comm.bcast(_calc.exp_virt, root=0)
					self.global_comm.bcast(_calc.energy_thres, root=0)
				else:
					# receive from master
					_calc.exp_model = self.global_comm.bcast(None, root=0)
					_calc.exp_type = self.global_comm.bcast(None, root=0)
					_calc.exp_base = self.global_comm.bcast(None, root=0)
					_calc.exp_thres = self.global_comm.bcast(None, root=0)
					_calc.exp_max_order = self.global_comm.bcast(None, root=0)
					_calc.exp_occ = self.global_comm.bcast(None, root=0)
					_calc.exp_virt = self.global_comm.bcast(None, root=0)
					_calc.energy_thres = self.global_comm.bcast(None, root=0)
				#
				return


		def bcast_mpi_info(self):
				""" bcast mpi info """
				if (self.global_master):
					# bcast to slaves
					self.global_comm.bcast(self.num_groups, root=0)
				else:
					# receive from master
					self.num_groups = self.global_comm.bcast(None, root=0)
				#
				return


		def bcast_hf_base(self, _mol):
				""" bcast hf and base info """
				if (self.global_master):
					# bcast to slaves
					self.global_comm.bcast(_mol.e_hf, root=0)
					self.global_comm.bcast(_mol.e_ref, root=0)
					self.global_comm.bcast(_mol.norb, root=0)
					self.global_comm.bcast(_mol.nocc, root=0)
					self.global_comm.bcast(_mol.nvirt, root=0)
				else:
					# receive from master
					_mol.e_hf = self.global_comm.bcast(None, root=0)
					_mol.e_ref = self.global_comm.bcast(None, root=0)
					_mol.norb = self.global_comm.bcast(None, root=0)
					_mol.nocc = self.global_comm.bcast(None, root=0)
					_mol.nvirt = self.global_comm.bcast(None, root=0)
				#
				return


		def bcast_rst(self, _calc, _exp, _time):
				""" bcast restart files """
				if (self.global_master):
					# determine start index for energy kernel phase
					e_inc_end = np.argmax(_exp.energy_inc[-1] == 0.0)
					if (e_inc_end == 0): e_inc_end = len(_exp.energy_inc[-1])
					# collect exp_info
					exp_info = {'len_tup': [len(_exp.tuples[i]) for i in range(len(_exp.tuples))],\
								'len_e_inc': [len(_exp.energy_inc[i]) for i in range(len(_exp.energy_inc))],\
								'min_order': _exp.min_order, 'e_inc_end': e_inc_end}
					# bcast info
					self.master_comm.bcast(exp_info, root=0)
					# bcast tuples
					for i in range(1,len(_exp.tuples)):
						self.master_comm.Bcast([_exp.tuples[i],MPI.INT], root=0)
					# bcast energy increments
					for i in range(len(_exp.energy_inc)):
						self.master_comm.Bcast([_exp.energy_inc[i],MPI.DOUBLE], root=0)
					# collect time_info
					for i in range(1,self.global_size):
						time_info = {'kernel': [_time.time_work[0][i],
									_time.time_comm[0][i],_time.time_idle[0][i]],\
									'screen': [_time.time_work[1][i],
									_time.time_comm[1][i],_time.time_idle[1][i]]}
						self.master_comm.send(time_info, dest=i)
				else:
					# receive exp_info
					exp_info = self.master_comm.bcast(None, root=0)
					# set min_order
					_exp.min_order = exp_info['min_order']
					# receive tuples
					for i in range(1,len(exp_info['len_tup'])):
						buff = np.empty([exp_info['len_tup'][i],i+1], dtype=np.int32)
						self.master_comm.Bcast([buff,MPI.INT], root=0)
						_exp.tuples.append(buff)
					# receive e_inc
					for i in range(len(exp_info['len_e_inc'])):
						buff = np.zeros(exp_info['len_e_inc'][i], dtype=np.float64)
						self.master_comm.Bcast([buff,MPI.DOUBLE], root=0)
						_exp.energy_inc.append(buff)
					# for e_inc[-1], make sure that this is distributed among the slaves *if* incomplete
					if (exp_info['e_inc_end'] != exp_info['len_e_inc'][-1]):
						for i in range(exp_info['e_inc_end']):
							if ((i % (self.global_size-1)) != (self.global_rank-1)): _exp.energy_inc[-1][i] = 0.0 
						_exp.energy_inc[-1][exp_info['e_inc_end']:] = 0.0
					# receive time_info
					time_info = self.master_comm.recv(source=0, status=self.stat)
					_time.time_work_kernel = time_info['kernel'][0]
					_time.time_comm_kernel = time_info['kernel'][1]
					_time.time_idle_kernel = time_info['kernel'][2]
					_time.time_work_screen = time_info['screen'][0]
					_time.time_comm_screen = time_info['screen'][1]
					_time.time_idle_screen = time_info['screen'][2]
				#
				return


		def bcast_e_inc(self, _exp, _time):
				""" bcast e_inc[-1] """
				# start idle time
				_time.timer('idle_kernel', _time.order)
				# barrier
				self.global_comm.Barrier()
				# start comm time
				_time.timer('comm_kernel', _time.order)
				# now do Bcast
				self.global_comm.Bcast([_exp.energy_inc[-1],MPI.DOUBLE], root=0)
				# start work time
				_time.timer('work_kernel', _time.order)
				#
				return


		def bcast_tup(self, _exp, _time, _buff):
				""" master/slave routine for bcasting total number of tuples """
				if (self.local_master):
					# start comm time
					_time.timer('comm_screen', _time.order)
					# init bcast dict
					tup_info = {'tup_len': len(_buff)}
					# bcast
					self.local_comm.bcast(tup_info, root=0)
				# start idle time
				_time.timer('idle_screen', _time.order)
				# all meet at barrier
				self.local_comm.Barrier()
				# start comm time
				_time.timer('comm_screen', _time.order)
				# bcast buffer
				self.local_comm.Bcast([_buff,MPI.INT], root=0)
				# start work time
				_time.timer('work_screen', _time.order)
				# append tup[-1] with buff
				if (len(_buff) >= 1): _exp.tuples.append(_buff)
				# end work time
				_time.timer('work_screen', _time.order, True)
				#
				return


		def final(self, _rst):
				""" terminate calculation """
				if (self.global_master):
					_rst.rm_rst()
					if (self.num_groups == 1):
						self.local_comm.bcast({'task': 'exit_slave'}, root=0)
					else:
						self.master_comm.bcast({'task': 'exit_master'}, root=0)
				elif (self.local_master):
					self.local_comm.bcast({'task': 'exit_slave'}, root=0)
				self.global_comm.Barrier()
				MPI.Finalize()
	

