#!/usr/bin/env python
# -*- coding: utf-8 -*

""" mpi.py: mpi class """

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


		def set_mpi(self):
				""" set mpi info """
				# communicate mpi info
				self.bcast_mpi_info()
				# set local groups
				self.set_local_groups()
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
							'symmetry': _mol.symmetry, 'irrep_nelec': _mol.irrep_nelec, 'basis': _mol.basis, \
							'unit': _mol.unit, 'frozen': _mol.frozen, 'verbose_prt': _mol.verbose_prt}
					self.global_comm.bcast(mol, root=0)
				else:
					mol = self.global_comm.bcast(None, root=0)
					_mol.atom = mol['atom']; _mol.charge = mol['charge']; _mol.spin = mol['spin']
					_mol.symmetry = mol['symmetry']; _mol.irrep_nelec = mol['irrep_nelec']
					_mol.basis = mol['basis']; _mol.unit = mol['unit']; _mol.frozen = mol['frozen']
					_mol.verbose_prt = mol['verbose_prt']
				#
				return


		def bcast_calc_info(self, _calc):
				""" bcast calc info """
				if (self.global_master):
					# bcast to slaves
					calc = {'exp_model': _calc.exp_model['METHOD'], 'exp_type': _calc.exp_type, \
							'exp_ref': _calc.exp_ref['METHOD'], 'exp_base': _calc.exp_base['METHOD'], \
							'exp_thres': _calc.exp_thres, 'exp_relax': _calc.exp_relax, \
							'wfnsym': _calc.wfnsym, 'exp_max_order': _calc.exp_max_order, \
							'exp_occ': _calc.exp_occ, 'exp_virt': _calc.exp_virt}
					self.global_comm.bcast(calc, root=0)
				else:
					# receive from master
					calc = self.global_comm.bcast(None, root=0)
					_calc.exp_model = {'METHOD': calc['exp_model']}; _calc.exp_type = calc['exp_type']
					_calc.exp_ref = {'METHOD': calc['exp_ref']}; _calc.exp_base = {'METHOD': calc['exp_base']}
					_calc.exp_thres = calc['exp_thres']; _calc.exp_relax = calc['exp_relax']
					_calc.wfnsym = calc['wfnsym']; _calc.exp_max_order = calc['exp_max_order']
					_calc.exp_occ = calc['exp_occ']; _calc.exp_virt = calc['exp_virt']
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


		def bcast_hf_ref_info(self, _mol, _calc):
				""" bcast hf and ref info """
				if (self.global_master):
					# collect dimensions, reference energies, and mo_occ
					info = {'e_hf': _calc.energy['hf'], 'e_base': _calc.energy['base'], \
								'norb': _mol.norb, 'nocc': _mol.nocc, 'nvirt': _mol.nvirt, \
								'ref_space': _calc.ref_space, 'exp_space': _calc.exp_space, \
								'occup': _calc.occup, 'no_act': _calc.no_act}
					# bcast info
					self.global_comm.bcast(info, root=0)
					# bcast mo
					if (self.num_local_masters >= 1):
						self.master_comm.Bcast([_calc.mo, MPI.DOUBLE], root=0)
				else:
					# receive info
					info = self.global_comm.bcast(None, root=0)
					_calc.energy['hf'] = info['e_hf']; _calc.energy['base'] = info['e_base']
					_mol.norb = info['norb']; _mol.nocc = info['nocc']; _mol.nvirt = info['nvirt']
					_calc.ref_space = info['ref_space']; _calc.exp_space = info['exp_space']
					_calc.occup = info['occup']; _calc.no_act = info['no_act']
					# receive mo
					if (self.local_master):
						buff = np.zeros([_mol.norb, _mol.norb], dtype=np.float64)
						self.master_comm.Bcast([buff, MPI.DOUBLE], root=0)
						_calc.mo = buff
				#
				return


		def bcast_mo_info(self, _mol, _calc, _comm):
				""" bcast mo coefficients """
				if (_comm.Get_rank() == 0):
					# bcast mo
					_comm.Bcast([_calc.mo, MPI.DOUBLE], root=0)
				else:
					# receive mo
					buff = np.zeros([_mol.norb, _mol.norb], dtype=np.float64)
					_comm.Bcast([buff, MPI.DOUBLE], root=0)
					_calc.mo = buff
				#
				return


		def bcast_rst(self, _calc, _exp):
				""" bcast restart files """
				if (_exp.level == 'macro'):
					comm = self.master_comm
				elif (_exp.level == 'micro'):
					comm = self.global_comm
				if (self.global_master):
					# collect exp_info
					exp_info = {'len_tup': [len(_exp.tuples[i]) for i in range(len(_exp.tuples))], \
								'len_e_inc': [len(_exp.energy['inc'][i]) for i in range(len(_exp.energy['inc']))], \
								'min_order': _exp.min_order, 'start_order': _exp.start_order}
					# bcast info
					comm.bcast(exp_info, root=0)
					# bcast tuples
					for i in range(1,len(_exp.tuples)):
						comm.Bcast([_exp.tuples[i], MPI.INT], root=0)
					# bcast energy increments
					for i in range(len(_exp.energy['inc'])):
						comm.Bcast([_exp.energy['inc'][i], MPI.DOUBLE], root=0)
				else:
					# receive exp_info
					exp_info = comm.bcast(None, root=0)
					# set min_order and start_order
					_exp.min_order = exp_info['min_order']
					_exp.start_order = exp_info['start_order']
					# receive tuples
					for i in range(1, len(exp_info['len_tup'])):
						buff = np.empty([exp_info['len_tup'][i], _exp.start_order+i], dtype=np.int32)
						comm.Bcast([buff, MPI.INT], root=0)
						_exp.tuples.append(buff)
					# receive e_inc
					for i in range(len(exp_info['len_e_inc'])):
						buff = np.zeros(exp_info['len_e_inc'][i], dtype=np.float64)
						comm.Bcast([buff, MPI.DOUBLE], root=0)
						_exp.energy['inc'].append(buff)
				#
				return


		def bcast_energy(self, _mol, _calc, _exp, _comm):
				""" bcast energies """
				# Bcast
				_comm.Bcast([_exp.energy['inc'][-1], MPI.DOUBLE], root=0)
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


