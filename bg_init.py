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
from bg_driver import DrvCls
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
				# init expansion parameters and next search for restart files 
				if (self.calc.exp_type in ['occupied','virtual']):
					self.exp = ExpCls(self.mol, self.calc, self.rst)
					self.rst.rst_main(self.mpi, self.calc, self.exp, self.time)
				# init timings
				self.time = TimeCls(self.mpi, self.rst)
				# driver and print instance
				if (self.mpi.master):
					self.drv = DrvCls()
					self.prt = PrintCls(self.out.out_dir)
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
		""" molecule class (inherited from pyscf gto.Mole class) """
		def __init__(self, _err):
				""" init parameters """
				# set geometric parameters
				self.atom = self.set_geo(_err)
				# set molecular parameters
				self.charge, self.spin, self.symmetry, self.basis, \
					self.unit, self.frozen, self.verbose  = \
							self.set_mol(_err)
				# build mol
				self.build()
				# set number of core orbs
				self.ncore = self.set_ncore()
				#
				return self


		def set_geo(self, _err):
				""" set geometry from bg-geo.inp file """
				# error handling
				if (not isfile('bg-geo.inp')):
					_err.error_msg = 'bg-geo.inp not found'
					_err.abort()
				# read input file
				with open('bg-geo.inp') as f:
					content = f.readlines()
					atom = ''
					for i in range(len(content)):
						atom += content[i]
				#
				return atom


		def set_mol(self, _err):
				""" set molecular parameters from bg-mol.inp file """
				# error handling
				if (not isfile('bg-mol.inp')):
					_err.error_msg = 'bg-mol.inp not found'
					_err.abort()
				# read input file
				with open('bg-mol.inp') as f:
					content = f.readlines()
					for i in range(len(content)):
						if (content[i].split()[0] == 'charge'):
							charge = int(content[i].split()[2])
						elif (content[i].split()[0] == 'spin'):
							spin = int(content[i].split()[2])
						elif (content[i].split()[0] == 'symmetry'):
							symmetry = content[i].split()[2].upper() == 'TRUE'
						elif (content[i].split()[0] == 'basis'):
							basis = content[i].split()[2]
						elif (content[i].split()[0] == 'unit'):
							unit = content[i].split()[2]
						elif (content[i].split()[0] == 'frozen'):
							frozen = content[i].split()[2].upper() == 'TRUE'
						# error handling
						else:
							_err.error_msg = content[i].split()[2] + \
											' keyword in bg-mol.inp not recognized'
							_err.abort()
				# silence pyscf output
				verbose = 0
				#
				return charge, spin, symmetry, basis, unit, frozen, verbose


		def set_ncore(self):
				""" set ncore """
				ncore = 0
				if (self.frozen):
					for i in range(self.natm):
						if (self.atom_charge(i) > 2): ncore += 1
						if (self.atom_charge(i) > 12): ncore += 4
						if (self.atom_charge(i) > 20): ncore += 4
						if (self.atom_charge(i) > 30): ncore += 6
				#
				return ncore


class CalcCls():
		""" calculation class """
		def __init__(self, _err):
				""" init parameters """
				self.exp_model = 'fci'
				self.exp_type = 'virtual'
				self.exp_thres = 1.0e-06
				self.exp_damp = 1.0
				self.exp_max_order = 0
				self.exp_orbs = 'canonical'
				self.energy_thres = 3.8e-04
				# hardcoded parameters
				self.exp_thres_init = self.exp_thres
				self.rst_freq = 50000.0
				# set calculation parameters
				self.exp_model, self.exp_type, self.exp_thres, self.exp_damp, \
					self.exp_order, self.exp_occ, self.exp_virt, \
					self.energy_thres, = self.set_calc(_err)
				# sanity check
				self.sanity_chk(_err)
				#
				return self


		def set_calc(self, _err):
				""" set calculation parameters from bg-calc.inp file """
                # error handling
   				if (not isfile('bg-calc.inp')):
   					_err.error_msg = 'bg-calc.inp not found'
   					_err.abort()
   				# read input file
   				with open('bg-calc.inp') as f:
					content = f.readlines()
					for i in range(len(content)):
						if (content[i].split()[0] == 'exp_model'):
							exp_model = content[i].split()[2].upper()
						elif (content[i].split()[0] == 'exp_type'):
							exp_type = content[i].split()[2]
						elif (content[i].split()[0] == 'exp_thres'):
							exp_thres = float(content[i].split()[2])
						elif (content[i].split()[0] == 'exp_damp'):
							exp_damp = float(content[i].split()[2])
						elif (content[i].split()[0] == 'exp_max_order'):
							exp_max_order = int(content[i].split()[2])
						elif (content[i].split()[0] == 'orbitals'):
							exp_orbs = content[i].split()[2]
						elif (content[i].split()[0] == 'energy_thres'):
							energy_thres = float(content[i].split()[2])
						# error handling
						else:
							_err.error_msg = content[i].split()[2] + \
											' keyword in bg-calc.inp not recognized'
							_err.abort()
				#
				return exp_model, exp_type, exp_thres, exp_damp, \
							exp_max_order, exp_orbs, energy_thres


		def sanity_chk(self, _err):
				""" sanity check for calculation parameters """
				# type of expansion
				if (not (self.exp_type in ['occupied','virtual'])):
					_err.error_msg = 'wrong input -- valid choices for ' + \
									'expansion scheme are occupied and virtual'
				# expansion model
				if (not (self.exp_model in ['CCSD','FCI'])):
					_err.error_msg = 'wrong input -- valid expansion models ' + \
									'are currently: CCSD and FCI'
				# max order
				if (self.exp_max_order < 0):
					_err.error_msg = 'wrong input -- wrong maximum ' + \
									'expansion order (must be integer >= 1)'
				# expansion thresholds
				if (self.exp_thres < 0.0):
					_err.error_msg = 'wrong input -- expansion threshold ' + \
									'(exp_thres) must be float >= 0.0'
				if (self.exp_damp < 1.0):
					_err.error_msg = 'wrong input -- expansion dampening ' + \
									'(exp_damp) must be float >= 1.0'
				if (self.energy_thres < 0.0):
					_err.error_msg = 'wrong input -- energy threshold ' + \
									'(energy_thres) must be float >= 0.0'
				# orbital representation
				if (not (self.exp_orbs in ['canonical','local','natural'])):
					_err.error_msg = 'wrong input -- valid orbital ' + \
									'representations are currently: canonical, ' + \
									'local, and natural (CCSD)'
				#
				if (_err.error_msg != ''):
					_err.abort()
				return


class ExpCls():
		""" expansion class """
		def __init__(self, _mol, _calc, _rst):
				""" init parameters """
				# set params and lists for occ expansion
				if (_calc.exp_type == 'occupied'):
					# set lower and upper limits
					self.l_limit = 0
					self.u_limit = _mol.nocc
					# init tuples and e_inc
					self.tuples = [np.array(list([i] for i in range(_mol.ncore,
										self.u_limit)), dtype=np.int32)]
				# set params and lists for virt expansion
				elif (_calc.exp_type == 'virtual'):
					# set lower and upper limits
					self.l_limit = _mol.nocc
					self.u_limit = _mol.nvirt
					# init prim tuple and e_inc
					self.tuples = [np.array(list([i] for i in range(self.l_limit,
										self.l_limit + self.u_limit)), dtype=np.int32)]
				# init energy_inc
				if (_rst.restart):
					self.energy_inc = []
				else:
					self.energy_inc = [np.zeros(len(self.tuples[0]),
								dtype=np.float64)]
				# set max_order
				if ((_calc.exp_max_order == 0) or (_calc.exp_max_order > self.u_limit)):
					_calc.exp_max_order = self.u_limit
					if ((_calc.exp_type == 'occupied') and _mol.frozen): _calc.exp_max_order -= _mol.ncore
				# determine max theoretical work
				self.theo_work = []
				for k in range(calc.exp_max_order):
					self.theo_work.append(int(factorial(_calc.exp_max_order) / \
											(factorial(k + 1) * factorial(_calc.exp_max_order - (k + 1)))))
				# init convergence lists
				self.conv_orb = [False]
				self.conv_energy = [False]
				# init orb_ent and orb_con lists
				self.orb_ent_abs = []; self.orb_ent_rel = []
				self.orb_con_abs = []; self.orb_con_rel = []
				# init total energy lists for prim exp
				self.energy_tot = []
				#
				return
		
		
		def enum(self, *sequential, **named):
				""" hardcoded enums """
				enums = dict(zip(sequential, range(len(sequential))), **named)
				#
				return type('Enum', (), enums)


