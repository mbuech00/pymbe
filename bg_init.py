#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_init.py: setup utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from mpi4py import MPI
from os import getcwd, mkdir, chdir
from os.path import isdir
from shutil import rmtree 

from bg_pyscf import PySCFCls
from bg_time import TimeCls
from bg_print import PrintCls

from bg_mpi_wrapper import set_exception_hook
from bg_mpi_utils import bcast_mol_dict, init_slave_env
from bg_mpi_time import init_mpi_timings
from bg_info import init_mol, init_param, init_backend_prog, sanity_chk
from bg_utils import run_calc_hf, term_calc
from bg_print import redirect_stdout
from bg_rst_main import rst_init_env


class InitCls():
		""" initialization class """
		def __init__(self):
				""" init calculation """
				# init output and rst dirs
				self.wrk_dir, self.out_dir, self.rst_dir, self.rst = \
						self.init_out_rst()
				# init error handling
				self.err = ErrCls(self.out_dir)
				# init molecule, calculation, and mpi parameters
				self.mol = MolCls(self.err)
				self.calc = CalcCls(self.err)
				self.mpi = MPICls()
				# pyscf instance
				self.pyscf = PySCFCls()
				# hf calculation and integral transformation
				if (self.mpi.master):
					self.mol.hf, self.mol.norb, self.mol.nocc, self.mol.nvirt = \
							self.pyscf.hf_calc(self.mol)
					self.calc.h1e, self.calc.h2e = \
							self.pyscf.int_trans(self.mol, self.calc.orbs == 'natural')
				# bcast to slaves
				self.mpi.bcast_hf_int(self.mol, self.calc)
				# init timings
				self.time = TimeCls()
				# print instance
				if (self.mpi.master): self.prt = PrintCls()
				#
				return self


		def init_out_rst(self):
				""" init output and restart environments"""
				# get work dir
				_wrk_dir = getcwd()
				# set output dir
				_out_dir = _wrk_dir+'/output'
				# rm out_dir if present
				if (isdir(_out_dir)): rmtree(_out_dir,ignore_errors=True)
				# mk out_dir
				mkdir(_out_dir)
				# set rst dir
		        _rst_dir = _wrk_dir+'/rst'
		        # mk rst_dir and set rst logical
		        if (not isdir(_rst_dir)):
					_rst = False
					mkdir(_rst_dir)
				else:
					_rst = True
				#
				return _wrk_dir, _out_dir, _rst_dir, _rst


class MolCls(gto.Mole):
		""" make instance of pyscf mole class """
		def __init__(err):
				""" init parameters """
				# set geometric parameters
				self.atom = self.set_geo()
				# set molecular parameters
				self.charge, self.spin, self.symmetry, self.basis, \
					self.unit, self.frozen, self.verbose  = \
							self.set_mol()
				# build mol
				self.build()
				# set number of core orbs
				self.ncore = self.set_ncore()
				#
				return self


		def set_geo(self, err):
				""" set geometry from bg-geo.inp file """
				# error handling
				if (not isfile('bg-geo.inp')):
					err.error_msg = 'bg-geo.inp not found'
					err.abort()
				# read input file
				with open('bg-geo.inp') as f:
					content = f.readlines()
					_atom = ''
					for i in range(len(content)):
						_atom += content[i]
				#
				return _atom


		def set_mol(self, err):
				""" set molecular parameters from bg-mol.inp file """
				# error handling
				if (not isfile('bg-mol.inp')):
					err.error_msg = 'bg-mol.inp not found'
					err.abort()
				# read input file
				with open('bg-mol.inp') as f:
					content = f.readlines()
					for i in range(len(content)):
						if (content[i].split()[0] == 'charge'):
							_charge = int(content[i].split()[2])
						elif (content[i].split()[0] == 'spin'):
							_spin = int(content[i].split()[2])
						elif (content[i].split()[0] == 'symmetry'):
							_symmetry = content[i].split()[2].upper() == 'TRUE'
						elif (content[i].split()[0] == 'basis'):
							_basis = content[i].split()[2]
						elif (content[i].split()[0] == 'unit'):
							_unit = content[i].split()[2]
						elif (content[i].split()[0] == 'frozen'):
							_frozen = content[i].split()[2].upper() == 'TRUE'
						# error handling
						else:
							err.error_msg = content[i].split()[2]+' keyword in bg-mol.inp not recognized'
							err.abort()
				# silence pyscf output
				_verbose = 0
				#
				return _charge, _spin, _symmetry, _basis, \
							_unit, _frozen, _verbose


		def set_ncore(self):
				""" set ncore """
				_ncore = 0
				if (self.frozen):
					for i in range(self.natm):
						if (self.atom_charge(i) > 2): _ncore += 1
						if (self.atom_charge(i) > 12): _ncore += 4
						if (self.atom_charge(i) > 20): _ncore += 4
						if (self.atom_charge(i) > 30): _ncore += 6
				#
				return _ncore


class CalcCls():
		""" calculation parameters """
		def __init__(err):
				""" init parameters """
				self.exp_model = 'fci'
				self.exp_type = 'virtual'
				self.exp_thres = 1.0e-06
				self.exp_damp = 1.0
				self.exp_max_order = 6
				self.exp_orbs = 'canonical'
				self.energy_thres = 3.8e-04
                self.ref = False
				# hardcoded parameters
				self.rst_freq = 50000.0
				# set calculation parameters
				self.exp_model, self.exp_type, self.exp_thres, self.exp_damp, \
					self.exp_order, self.exp_occ, self.exp_virt, \
					self.energy_thres, self.ref = \
						self.set_calc(err)
				# sanity check
				self.sanity_chk(err)
				#
				return self


		def set_calc(self, err):
				""" set calculation parameters from bg-calc.inp file """
                # error handling
   				if (not isfile('bg-calc.inp')):
   					err.error_msg = 'bg-calc.inp not found'
   					err.abort()
   				# read input file
   				with open('bg-calc.inp') as f:
					content = f.readlines()
					for i in range(len(content)):
						if (content[i].split()[0] == 'exp_model'):
							_exp_model = content[i].split()[2].upper()
						elif (content[i].split()[0] == 'exp_type'):
							_exp_type = content[i].split()[2]
						elif (content[i].split()[0] == 'exp_thres'):
							_exp_thres = float(content[i].split()[2])
						elif (content[i].split()[0] == 'exp_damp'):
							_exp_damp = float(content[i].split()[2])
						elif (content[i].split()[0] == 'exp_max_order'):
							_exp_max_order = int(content[i].split()[2])
						elif (content[i].split()[0] == 'orbitals'):
							_exp_orbs = content[i].split()[2]
						elif (content[i].split()[0] == 'energy_thres'):
							_energy_thres = float(content[i].split()[2])
						elif (content[i].split()[0] == 'ref'):
							_ref = content[i].split()[2].upper() == 'TRUE'
						# error handling
						else:
							err.error_msg = content[i].split()[2]+' keyword in bg-calc.inp not recognized'
							err.abort()
				#
				return _exp_model, _exp_type, _exp_thres, _exp_damp, \
							_exp_max_order, _exp_orbs, _energy_thres, _ref


		def sanity_chk(self, err):
				""" sanity check for calculation parameters """
				# type of expansion
				if (not (self.exp_type in ['occupied','virtual'])):
					err.error_msg = 'wrong input -- valid choices for expansion scheme are occupied and virtual'
				# expansion model
				if (not (self.exp_model in ['CCSD','FCI'])):
					err.error_msg = 'wrong input -- valid expansion models are currently: CCSD and FCI'
				# max order
				if (self.exp_max_order < 0):
					err.error_msg = 'wrong input -- wrong maximum expansion order (must be integer >= 1)'
				# expansion thresholds
				if (self.exp_thres < 0.0):
					err.error_msg = 'wrong input -- expansion threshold (exp_thres) must be float >= 0.0'
				if (self.exp_damp < 1.0):
					err.error_msg = 'wrong input -- expansion dampening (exp_damp) must be float > 1.0'
				if (self.energy_thres < 0.0):
					err.error_msg = 'wrong input -- energy threshold (energy_thres) must be float >= 0.0'
				# orbital representation
				if (not (self.exp_orbs in ['canonical','local','natural'])):
					err.error_msg = 'wrong input -- orbital representations must be chosen for occupied and virtual orbitals'
				#
				if (err.error_msg != ''):
					err.abort()
				return


class MPICls():
		""" mpi parameters """
		def __init__():
				""" init parameters """
				self.parallel = self.comm.Get_size() > 1
				if (self.parallel):
					self.comm = MPI.COMM_WORLD
					self.size = self.comm.Get_size()
					self.rank = self.comm.Get_rank()
					self.master = self.rank == 0
					self.name = MPI.Get_processor_name()
					self.stat = MPI.Status()
				#
				return self


		def bcast_hf_int(mol, calc):
				""" bcast hf and int info """
				if (self.master):
					# bcast to slaves
					self.comm.bcast(mol.hf,root=0)
					self.comm.bcast(mol.norb,root=0)
					self.comm.bcast(mol.nocc,root=0)
					self.comm.bcast(mol.nvirt,root=0)
					self.comm.bcast(calc.h1e,root=0)
					self.comm.bcast(calc.h2e,root=0)
				else:
					# receive from master
					mol.hf = self.comm.bcast(None,root=0)
					mol.norb = self.com.bcast(None,root=0)
					mol.nocc = self.comm.bcast(None,root=0)
					mol.nvirt = self.comm.bcast(None,root=0)
					calc.h1e = self.comm.bcast(None,root=0)
					calc.h2e = self.comm.bcast(None,root=0)
				#
				return


class ErrCls():
		""" error handling """
		def __init__(out_dir):
				""" init parameters """
				self.error_msg = ''
				self.error_tup = ''
				self.error_rank = -1
				self.error_out = out_dir+'/bg_output.out'
				# set custom exception hook
				self.set_exc_hook()
				#
				return self


		def set_exc_hook(self):
				""" set an exception hook for aborting mpi """
				# save sys.excepthook
				sys_excepthook = sys.excepthook
				# define mpi exception hook
				def mpi_excepthook(t, v, tb):
					sys_excepthook(t, v, tb)
					traceback.print_last(file=self.error_out)
					self.abort()
				# overwrite sys.excepthook
				sys.excepthook = mpi_excepthook
				#
				return


		def abort():
				""" abort bg calculation in case of error """
				# write error log to bg_output.out
				with open(self.error_out,'a') as f:
					with redirect_stdout(f):
						print('')
						print('!!!!!!!!!!!!!')
						print('ERROR\n')
						if (self.error_tup == ''):
							print(' - master quits with error:\n')
						else:
							print(' - mpi proc. # {0:} quits with correlated calc. error:\n'.format(self.error_rank))
							print(self.error_msg)
							print('\nprint of the string of dropped MOs:\n')
							print(self.error_tup)
						print('\nERROR')
						print('!!!!!!!!!!!!!')
						print('')
				# abort
				MPI.COMM_WORLD.Abort()
				#
				return


