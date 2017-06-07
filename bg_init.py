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

from bg_rst import RstCls
from bg_error import ErrCls
from bg_mpi import MPICls
from bg_pyscf import PySCFCls
from bg_time import TimeCls
from bg_print import PrintCls


class InitCls():
		""" initialization class """
		def __init__(self):
				""" init calculation """
				# output and rst instances
				self.out = OutCls()
				self.rst = RstCls(self.out.wrk_dir)
				# init error handling
				self.err = ErrCls(self.out.out_dir)
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
				self.time = TimeCls(self.mpi, self.rst)
				# print instance
				if (self.mpi.master): self.prt = PrintCls(self.out.out_dir)
				#
				return self


class OutCls():
		""" output class """
		def __init__(self):
				""" init output environment """
				# get work dir
				self.wrk_dir = getcwd()
				# set output dir
				self.out_dir = self.wrk_dir+'/output'
				# rm out_dir if present
				if (isdir(self.out_dir)): rmtree(self.out_dir, ignore_errors=True)
				# mk out_dir
				mkdir(self.out_dir)
				#
				return self


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
							err.error_msg = content[i].split()[2] + \
											' keyword in bg-mol.inp not recognized'
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
				# hardcoded parameters
				self.exp_thres_init = self.exp_thres
				self.rst_freq = 50000.0
				# set calculation parameters
				self.exp_model, self.exp_type, self.exp_thres, self.exp_damp, \
					self.exp_order, self.exp_occ, self.exp_virt, \
					self.energy_thres, = self.set_calc(err)
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
						# error handling
						else:
							err.error_msg = content[i].split()[2] + \
											' keyword in bg-calc.inp not recognized'
							err.abort()
				#
				return _exp_model, _exp_type, _exp_thres, _exp_damp, \
							_exp_max_order, _exp_orbs, _energy_thres


		def sanity_chk(self, err):
				""" sanity check for calculation parameters """
				# type of expansion
				if (not (self.exp_type in ['occupied','virtual'])):
					err.error_msg = 'wrong input -- valid choices for ' + \
									'expansion scheme are occupied and virtual'
				# expansion model
				if (not (self.exp_model in ['CCSD','FCI'])):
					err.error_msg = 'wrong input -- valid expansion models ' + \
									'are currently: CCSD and FCI'
				# max order
				if (self.exp_max_order < 0):
					err.error_msg = 'wrong input -- wrong maximum ' + \
									'expansion order (must be integer >= 1)'
				# expansion thresholds
				if (self.exp_thres < 0.0):
					err.error_msg = 'wrong input -- expansion threshold ' + \
									'(exp_thres) must be float >= 0.0'
				if (self.exp_damp < 1.0):
					err.error_msg = 'wrong input -- expansion dampening ' + \
									'(exp_damp) must be float >= 1.0'
				if (self.energy_thres < 0.0):
					err.error_msg = 'wrong input -- energy threshold ' + \
									'(energy_thres) must be float >= 0.0'
				# orbital representation
				if (not (self.exp_orbs in ['canonical','local','natural'])):
					err.error_msg = 'wrong input -- valid orbital ' + \
									'representations are currently: canonical, ' + \
									'local, and natural (CCSD)'
				#
				if (err.error_msg != ''):
					err.abort()
				return


