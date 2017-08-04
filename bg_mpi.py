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
import sys, traceback
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
				self.num_local_masters = 0
				# set custom exception hook
				self.set_exc_hook()
				#
				return


		def set_local_groups(self):
				""" define local groups """
				if (not self.parallel):
					self.prim_master = True
				else:
					if (self.num_local_masters == 0):
						self.master_comm = self.local_comm = self.global_comm
						self.local_size = self.global_size
						self.local_master = False
						self.prim_master = self.global_master
					else:
						# array of ranks (global master excluded)
						ranks = np.arange(1, self.global_size)
						# split into local groups and add global master
						groups = np.array_split(ranks, self.num_local_masters)
						groups = [np.array([0])] + groups
						# extract local master indices and append to global master index
						masters = [groups[i][0] for i in range(len(groups))]
						# define local masters (global master excluded)
						self.local_master = (self.global_rank in masters[1:])
						# define primary local master
						self.prim_master = (self.global_rank == 1)
						# set master group and intracomm
						self.master_group = self.global_group.Incl(masters)
						self.master_comm = self.global_comm.Create(self.master_group)
						# set local master group and intracomm
						self.local_master_group = self.master_group.Excl([0])
						self.local_master_comm = self.global_comm.Create(self.local_master_group)
						# set local intracomm based on color
						for i in range(len(groups)):
							if (self.global_rank in groups[i]): color = i
						self.local_comm = self.global_comm.Split(color)
						# determine size of local group
						self.local_size = self.local_comm.Get_size()
					# define slave
					if ((not self.global_master) and (not self.local_master)):
						self.slave = True
					else:
						self.slave = False
				#
				return


		def set_exc_hook(self):
				""" set an exception hook for aborting mpi """
				# save sys.excepthook
				sys_excepthook = sys.excepthook
				# define mpi exception hook
				def mpi_excepthook(_type, _value, _traceback):
					""" custom mpi exception hook """
					if (not issubclass(_type, OSError)):
						print('\n-- Error information --')
						print('\ntype:\n\n  {0:}'.format(_type))
						print('\nvalue:\n\n  {0:}'.format(_value))
						print('\ntraceback:\n\n{0:}'.format(''.join(traceback.format_tb(_traceback))))
					sys_excepthook(_type, _value, _traceback)
					self.global_comm.Abort(1)
				# overwrite sys.excepthook
				sys.excepthook = mpi_excepthook
				#
				return


		def bcast_mol_info(self, _mol):
				""" bcast mol info """
				if (self.global_master):
					mol = {'atom': _mol.atom, 'charge': _mol.charge, 'spin': _mol.spin, \
							'symmetry': _mol.symmetry, 'basis': _mol.basis, 'unit': _mol.unit, \
							'frozen': _mol.frozen, 'verbose': _mol.verbose}
					self.global_comm.bcast(mol, root=0)
				else:
					mol = self.global_comm.bcast(None, root=0)
					_mol.atom = mol['atom']; _mol.charge = mol['charge']; _mol.spin = mol['spin']
					_mol.symmetry = mol['symmetry']; _mol.basis = mol['basis']; _mol.unit = mol['unit']
					_mol.frozen = mol['frozen']; _mol.verbose = mol['verbose']
				#
				return


		def bcast_calc_info(self, _calc):
				""" bcast calc info """
				if (self.global_master):
					# bcast to slaves
					calc = {'exp_model': _calc.exp_model, 'exp_type': _calc.exp_type, 'exp_base': _calc.exp_base, \
							'exp_thres': _calc.exp_thres, 'exp_max_order': _calc.exp_max_order, 'exp_occ': _calc.exp_occ, \
							'exp_virt': _calc.exp_virt, 'energy_thres': _calc.energy_thres}
					self.global_comm.bcast(calc, root=0)
				else:
					# receive from master
					calc = self.global_comm.bcast(None, root=0)
					_calc.exp_model = calc['exp_model']; _calc.exp_type = calc['exp_type']; _calc.exp_base = calc['exp_base']
					_calc.exp_thres = calc['exp_thres']; _calc.exp_max_order = calc['exp_max_order']; _calc.exp_occ = calc['exp_occ']
					_calc.exp_virt = calc['exp_virt']; _calc.energy_thres = calc['energy_thres']
				#
				return


		def bcast_mpi_info(self):
				""" bcast mpi info """
				if (self.global_master):
					# bcast to slaves
					self.global_comm.bcast(self.num_local_masters, root=0)
				else:
					# receive from master
					self.num_local_masters = self.global_comm.bcast(None, root=0)
				#
				return


		def bcast_hf_info(self, _mol):
				""" bcast hf and base info """
				if (self.num_local_masters >= 2):
					# prim master has everything - bcast to rest of local masters
					if (self.prim_master):
						# bcast to local masters
						hf = {'e_hf': _mol.e_hf, 'norb': _mol.norb, 'nocc': _mol.nocc, 'nvirt': _mol.nvirt}
						self.local_master_comm.bcast(hf, root=0)
						self.local_master_comm.Bcast([_mol.trans_mat, MPI.DOUBLE], root=0)
						self.local_master_comm.Bcast([_mol.hcore, MPI.DOUBLE], root=0)
					elif (self.local_master and (not self.prim_master)):
						# receive from primary master
						hf = self.local_master_comm.bcast(None, root=0)
						_mol.e_hf = hf['e_hf']; _mol.norb = hf['norb']
						_mol.nocc = hf['nocc']; _mol.nvirt = hf['nvirt']
						buff_trans = np.empty([_mol.norb,_mol.norb], dtype=np.float64)
						self.local_master_comm.Bcast([buff_trans, MPI.DOUBLE], root=0)
						_mol.trans_mat = buff_trans
						buff_hcore = np.empty([_mol.norb,_mol.norb], dtype=np.float64)
						self.local_master_comm.Bcast([buff_hcore, MPI.DOUBLE], root=0)
						_mol.hcore = buff_hcore
				# now bcast to global master and slaves
				if (((self.num_local_masters == 0) and self.global_master) or \
						((self.num_local_masters >= 1) and self.local_master)):
					hf = {'e_hf': _mol.e_hf, 'norb': _mol.norb, 'nocc': _mol.nocc, 'nvirt': _mol.nvirt}
					self.local_comm.bcast(hf, root=0)
					if (self.num_local_masters >= 1): self.master_comm.send(hf, dest=0, tag=0)
				elif (self.global_master):
					if (self.num_local_masters >= 1):
						hf = self.master_comm.recv(source=1, tag=0, status=self.stat)	
						_mol.e_hf = hf['e_hf']; _mol.norb = hf['norb']
						_mol.nocc = hf['nocc']; _mol.nvirt = hf['nvirt']
				elif (self.slave):
					hf = self.local_comm.bcast(None, root=0)
					_mol.e_hf = hf['e_hf']; _mol.norb = hf['norb']
					_mol.nocc = hf['nocc']; _mol.nvirt = hf['nvirt']
				#
				return


		def send_e_zero(self, _mol):
				""" send zeroth-order energy """
				if (self.num_local_masters >= 1):
					if (self.prim_master):
						self.master_comm.send(_mol.e_zero, dest=0, tag=0)	
					elif (self.global_master):
						_mol.e_zero = self.master_comm.recv(source=1, tag=0, status=self.stat)	
				#
				return


		def bcast_rst(self, _calc, _exp):
				""" bcast restart files """
				if (_exp.level == 'macro'):
					comm = self.master_comm
				elif (_exp.level == 'micro'):
					comm = self.global_comm
				if (self.global_master):
					# determine start index for energy kernel phase
					e_inc_end = np.argmax(_exp.energy_inc[-1] == 0.0)
					if (e_inc_end == 0): e_inc_end = len(_exp.energy_inc[-1])
					# collect exp_info
					exp_info = {'len_tup': [len(_exp.tuples[i]) for i in range(len(_exp.tuples))],\
								'len_e_inc': [len(_exp.energy_inc[i]) for i in range(len(_exp.energy_inc))],\
								'min_order': _exp.min_order, 'e_inc_end': e_inc_end}
					# bcast info
					comm.bcast(exp_info, root=0)
					# bcast tuples
					for i in range(1,len(_exp.tuples)):
						comm.Bcast([_exp.tuples[i],MPI.INT], root=0)
					# bcast energy increments
					for i in range(len(_exp.energy_inc)):
						comm.Bcast([_exp.energy_inc[i],MPI.DOUBLE], root=0)
				else:
					# receive exp_info
					exp_info = comm.bcast(None, root=0)
					# set min_order
					_exp.min_order = exp_info['min_order']
					# receive tuples
					for i in range(1,len(exp_info['len_tup'])):
						buff = np.empty([exp_info['len_tup'][i],i+1], dtype=np.int32)
						comm.Bcast([buff,MPI.INT], root=0)
						_exp.tuples.append(buff)
					# receive e_inc
					for i in range(len(exp_info['len_e_inc'])):
						buff = np.zeros(exp_info['len_e_inc'][i], dtype=np.float64)
						comm.Bcast([buff,MPI.DOUBLE], root=0)
						_exp.energy_inc.append(buff)
				#
				return


		def bcast_e_inc(self, _exp, _comm):
				""" bcast e_inc[-1] """
				# now do Bcast
				_comm.Bcast([_exp.energy_inc[-1], MPI.DOUBLE], root=0)
				#
				return


		def bcast_tup(self, _exp, _buff, _comm):
				""" master/slave routine for bcasting total number of tuples """
				if ((self.global_master and (self.num_local_masters == 0)) or \
						(self.global_master and (_exp.level == 'macro')) or \
						(self.local_master and (_exp.level == 'micro'))):
					# init bcast dict
					tup_info = {'tup_len': len(_buff)}
					# bcast
					_comm.bcast(tup_info, root=0)
				# bcast buffer
				_comm.Bcast([_buff, MPI.INT], root=0)
				# append tup[-1] with buff
				if (len(_buff) >= 1): _exp.tuples.append(_buff)
				#
				return


		def final(self, _rst):
				""" terminate calculation """
				if (self.global_master):
					_rst.rm_rst()
					if (self.parallel):
						if (self.num_local_masters == 0):
							self.local_comm.bcast({'task': 'exit_slave'}, root=0)
						else:
							self.master_comm.bcast({'task': 'exit_local_master'}, root=0)
				elif (self.local_master):
					self.local_comm.bcast({'task': 'exit_slave'}, root=0)
				self.global_comm.Barrier()
				MPI.Finalize()


