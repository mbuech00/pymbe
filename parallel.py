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
import sys
import traceback

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
				# save sys.excepthook
				sys_excepthook = sys.excepthook
				# define mpi exception hook
				def mpi_excepthook(variant, value, trace):
					""" custom mpi exception hook """
					if not issubclass(variant, OSError):
						print('\n-- Error information --')
						print('\ntype:\n\n  {0:}'.format(variant))
						print('\nvalue:\n\n  {0:}'.format(value))
						print('\ntraceback:\n\n{0:}'.format(''.join(traceback.format_tb(trace))))
					sys_excepthook(variant, value, trace)
					self.global_comm.Abort(1)
				# overwrite sys.excepthook
				sys.excepthook = mpi_excepthook


def set_mpi(mpi):
		""" set mpi info """
		# bcast num_local_masters
		mpi.global_comm.bcast(mpi.num_local_masters, root=0)
		# now set info
		if not mpi.parallel:
			mpi.prim_master = True
		else:
			if mpi.num_local_masters == 0:
				mpi.master_comm = mpi.local_comm = mpi.global_comm
				mpi.local_size = mpi.global_size
				mpi.local_master = False
				mpi.prim_master = mpi.global_master
			else:
				# array of ranks (global master excluded)
				ranks = np.arange(1, mpi.global_size)
				# split into local groups and add global master
				groups = np.array_split(ranks, mpi.num_local_masters)
				groups = [np.array([0])] + groups
				# extract local master indices and append to global master index
				masters = [groups[i][0] for i in range(len(groups))]
				# define local masters (global master excluded)
				mpi.local_master = (mpi.global_rank in masters[1:])
				# define primary local master
				mpi.prim_master = (mpi.global_rank == 1)
				# set master group and intracomm
				mpi.master_group = mpi.global_group.Incl(masters)
				mpi.master_comm = mpi.global_comm.Create(mpi.master_group)
				# set local master group and intracomm
				mpi.local_master_group = mpi.master_group.Excl([0])
				mpi.local_master_comm = mpi.global_comm.Create(mpi.local_master_group)
				# set local intracomm based on color
				for i in range(len(groups)):
					if mpi.global_rank in groups[i]: color = i
				mpi.local_comm = mpi.global_comm.Split(color)
				# determine size of local group
				mpi.local_size = mpi.local_comm.Get_size()
			# define slave
			if not mpi.global_master and not mpi.local_master:
				mpi.slave = True
			else:
				mpi.slave = False


def mol(mpi, mol):
		""" bcast mol info """
		if mpi.global_master:
			info = {'atom': mol.atom, 'charge': mol.charge, 'spin': mol.spin, \
					'symmetry': mol.symmetry, 'irrep_nelec': mol.irrep_nelec, 'basis': mol.basis, \
					'unit': mol.unit, 'frozen': mol.frozen, 'verbose': mol.verbose}
			mpi.global_comm.bcast(info, root=0)
		else:
			info = mpi.global_comm.bcast(None, root=0)
			mol.atom = info['atom']; mol.charge = info['charge']; mol.spin = info['spin']
			mol.symmetry = info['symmetry']; mol.irrep_nelec = info['irrep_nelec']
			mol.basis = info['basis']; mol.unit = info['unit']; mol.frozen = info['frozen']
			mol.verbose = info['verbose']


def calc(mpi, calc):
		""" bcast calc info """
		if mpi.global_master:
			info = {'exp_model': calc.exp_model['METHOD'], 'exp_type': calc.exp_type, \
					'exp_ref': calc.exp_ref['METHOD'], 'exp_base': calc.exp_base['METHOD'], \
					'exp_thres': calc.exp_thres, 'exp_relax': calc.exp_relax, \
					'wfnsym': calc.wfnsym, 'exp_max_order': calc.exp_max_order, \
					'exp_occ': calc.exp_occ, 'exp_virt': calc.exp_virt}
			mpi.global_comm.bcast(info, root=0)
		else:
			info = mpi.global_comm.bcast(None, root=0)
			calc.exp_model = {'METHOD': info['exp_model']}; calc.exp_type = info['exp_type']
			calc.exp_ref = {'METHOD': info['exp_ref']}; calc.exp_base = {'METHOD': info['exp_base']}
			calc.exp_thres = info['exp_thres']; calc.exp_relax = info['exp_relax']
			calc.wfnsym = info['wfnsym']; calc.exp_max_order = info['exp_max_order']
			calc.exp_occ = info['exp_occ']; calc.exp_virt = info['exp_virt']


def fund(mpi, mol, calc):
		""" bcast fundamental info """
		if mpi.global_master:
			info = {'e_hf': calc.energy['hf'], 'e_base': calc.energy['base'], \
						'e_ref': calc.energy['ref'], 'e_ref_base': calc.energy['ref_base'], \
						'norb': mol.norb, 'nocc': mol.nocc, 'nvirt': mol.nvirt, \
						'ref_space': calc.ref_space, 'exp_space': calc.exp_space, \
						'occup': calc.occup, 'no_act': calc.no_act}
			mpi.global_comm.bcast(info, root=0)
			# bcast mo
			mpi.global_comm.Bcast([calc.mo, MPI.DOUBLE], root=0)
		else:
			info = mpi.global_comm.bcast(None, root=0)
			calc.energy['hf'] = info['e_hf']; calc.energy['base'] = info['e_base']
			calc.energy['ref'] = info['e_ref']; calc.energy['ref_base'] = info['e_ref_base']
			mol.norb = info['norb']; mol.nocc = info['nocc']; mol.nvirt = info['nvirt']
			calc.ref_space = info['ref_space']; calc.exp_space = info['exp_space']
			calc.occup = info['occup']; calc.no_act = info['no_act']
			# receive mo
			buff = np.zeros([mol.norb, mol.norb], dtype=np.float64)
			mpi.global_comm.Bcast([buff, MPI.DOUBLE], root=0)
			calc.mo = buff


def exp(mpi, calc, exp, comm):
		""" bcast exp info """
		if mpi.global_master:
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


def energy(exp, comm):
		""" bcast energies """
		# Bcast
		comm.Bcast([exp.energy['inc'][-1], MPI.DOUBLE], root=0)


def tup(exp, comm):
		""" bcast tuples """
		# Bcast
		comm.Bcast([exp.tuples[-1], MPI.INT], root=0)


def final(mpi):
		""" terminate calculation """
		if mpi.global_master:
			restart.rm()
			if mpi.parallel:
				if mpi.num_local_masters == 0:
					mpi.local_comm.bcast({'task': 'exit_slave'}, root=0)
				else:
					mpi.master_comm.bcast({'task': 'exit_local_master'}, root=0)
		elif mpi.local_master:
			mpi.local_comm.bcast({'task': 'exit_slave'}, root=0)
		mpi.global_comm.Barrier()
		MPI.Finalize()


