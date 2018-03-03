#!/usr/bin/env python
# -*- coding: utf-8 -*

""" parallel.py: mpi class """

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
import sys, traceback
from os import getcwd, mkdir, chdir
from os.path import isfile
from shutil import copy, rmtree

import restart


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
				self.bcast_mpi()
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
				def mpi_excepthook(type, _value, _traceback):
					""" custom mpi exception hook """
					if (not issubclass(type, OSError)):
						print('\n-- Error information --')
						print('\ntype:\n\n  {0:}'.format(type))
						print('\nvalue:\n\n  {0:}'.format(_value))
						print('\ntraceback:\n\n{0:}'.format(''.join(traceback.format_tb(_traceback))))
					sys_excepthook(type, _value, _traceback)
					self.global_comm.Abort(1)
				# overwrite sys.excepthook
				sys.excepthook = mpi_excepthook
				#
				return


		def bcast_mol(self, mol):
				""" bcast mol info """
				if (self.global_master):
					info = {'atom': mol.atom, 'charge': mol.charge, 'spin': mol.spin, \
							'symmetry': mol.symmetry, 'irrep_nelec': mol.irrep_nelec, 'basis': mol.basis, \
							'unit': mol.unit, 'frozen': mol.frozen, 'verbose': mol.verbose}
					self.global_comm.bcast(info, root=0)
				else:
					info = self.global_comm.bcast(None, root=0)
					mol.atom = info['atom']; mol.charge = info['charge']; mol.spin = info['spin']
					mol.symmetry = info['symmetry']; mol.irrep_nelec = info['irrep_nelec']
					mol.basis = info['basis']; mol.unit = info['unit']; mol.frozen = info['frozen']
					mol.verbose = info['verbose']
				#
				return


		def bcast_calc(self, calc):
				""" bcast calc info """
				if (self.global_master):
					# bcast to slaves
					info = {'exp_model': calc.exp_model['METHOD'], 'exp_type': calc.exp_type, \
							'exp_ref': calc.exp_ref['METHOD'], 'exp_base': calc.exp_base['METHOD'], \
							'exp_thres': calc.exp_thres, 'exp_relax': calc.exp_relax, \
							'wfnsym': calc.wfnsym, 'exp_max_order': calc.exp_max_order, \
							'exp_occ': calc.exp_occ, 'exp_virt': calc.exp_virt}
					self.global_comm.bcast(info, root=0)
				else:
					# receive from master
					info = self.global_comm.bcast(None, root=0)
					calc.exp_model = {'METHOD': info['exp_model']}; calc.exp_type = info['exp_type']
					calc.exp_ref = {'METHOD': info['exp_ref']}; calc.exp_base = {'METHOD': info['exp_base']}
					calc.exp_thres = info['exp_thres']; calc.exp_relax = info['exp_relax']
					calc.wfnsym = info['wfnsym']; calc.exp_max_order = info['exp_max_order']
					calc.exp_occ = info['exp_occ']; calc.exp_virt = info['exp_virt']
				#
				return


		def bcast_mpi(self):
				""" bcast mpi info """
				if (self.global_master):
					# bcast to slaves
					self.global_comm.bcast(self.num_local_masters, root=0)
				else:
					# receive from master
					self.num_local_masters = self.global_comm.bcast(None, root=0)
				#
				return


		def bcast_fund(self, mol, calc):
				""" bcast fundamental info """
				if (self.global_master):
					# collect dimensions, reference energies, and mo_occ
					info = {'e_hf': calc.energy['hf'], 'e_base': calc.energy['base'], \
								'e_ref': calc.energy['ref'], 'e_ref_base': calc.energy['ref_base'], \
								'norb': mol.norb, 'nocc': mol.nocc, 'nvirt': mol.nvirt, \
								'ref_space': calc.ref_space, 'exp_space': calc.exp_space, \
								'occup': calc.occup, 'no_act': calc.no_act}
					# bcast info
					self.global_comm.bcast(info, root=0)
					# bcast mo
					self.global_comm.Bcast([calc.mo, MPI.DOUBLE], root=0)
				else:
					# receive info
					info = self.global_comm.bcast(None, root=0)
					calc.energy['hf'] = info['e_hf']; calc.energy['base'] = info['e_base']
					calc.energy['ref'] = info['e_ref']; calc.energy['ref_base'] = info['e_ref_base']
					mol.norb = info['norb']; mol.nocc = info['nocc']; mol.nvirt = info['nvirt']
					calc.ref_space = info['ref_space']; calc.exp_space = info['exp_space']
					calc.occup = info['occup']; calc.no_act = info['no_act']
					# receive mo
					buff = np.zeros([mol.norb, mol.norb], dtype=np.float64)
					self.global_comm.Bcast([buff, MPI.DOUBLE], root=0)
					calc.mo = buff
				#
				return


		def bcast_exp(self, calc, exp):
				""" bcast exp info """
				if (exp.level == 'macro'):
					comm = self.master_comm
				elif (exp.level == 'micro'):
					comm = self.global_comm
				if (self.global_master):
					# collect info
					info = {'len_tup': [len(exp.tuples[i]) for i in range(len(exp.tuples))], \
								'len_e_inc': [len(exp.energy['inc'][i]) for i in range(len(exp.energy['inc']))], \
								'min_order': exp.min_order, 'start_order': exp.start_order}
					# bcast info
					comm.bcast(info, root=0)
					# bcast tuples
					for i in range(1,len(exp.tuples)):
						comm.Bcast([exp.tuples[i], MPI.INT], root=0)
					# bcast energy increments
					for i in range(len(exp.energy['inc'])):
						comm.Bcast([exp.energy['inc'][i], MPI.DOUBLE], root=0)
				else:
					# receive info
					info = comm.bcast(None, root=0)
					# set min_order and start_order
					exp.min_order = info['min_order']
					exp.start_order = info['start_order']
					# receive tuples
					for i in range(1, len(info['len_tup'])):
						buff = np.empty([info['len_tup'][i], exp.start_order+i], dtype=np.int32)
						comm.Bcast([buff, MPI.INT], root=0)
						exp.tuples.append(buff)
					# receive e_inc
					for i in range(len(info['len_e_inc'])):
						buff = np.zeros(info['len_e_inc'][i], dtype=np.float64)
						comm.Bcast([buff, MPI.DOUBLE], root=0)
						exp.energy['inc'].append(buff)
				#
				return


		def bcast_energy(self, mol, calc, exp, comm):
				""" bcast energies """
				# Bcast
				comm.Bcast([exp.energy['inc'][-1], MPI.DOUBLE], root=0)
				#
				return


		def bcast_tup(self, exp, buff, comm):
				""" master/slave routine for bcasting total number of tuples """
				if ((self.global_master and (self.num_local_masters == 0)) or \
						(self.global_master and (exp.level == 'macro')) or \
						(self.local_master and (exp.level == 'micro'))):
					# init bcast dict
					tup_info = {'tup_len': len(buff)}
					# bcast
					comm.bcast(tup_info, root=0)
				# bcast buffer
				comm.Bcast([buff, MPI.INT], root=0)
				# append tup[-1] with buff
				if (len(buff) >= 1): exp.tuples.append(buff)
				#
				return


		def final(self):
				""" terminate calculation """
				if (self.global_master):
					restart.rm()
					if (self.parallel):
						if (self.num_local_masters == 0):
							self.local_comm.bcast({'task': 'exit_slave'}, root=0)
						else:
							self.master_comm.bcast({'task': 'exit_local_master'}, root=0)
				elif (self.local_master):
					self.local_comm.bcast({'task': 'exit_slave'}, root=0)
				self.global_comm.Barrier()
				MPI.Finalize()


