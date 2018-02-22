#!/usr/bin/env python
# -*- coding: utf-8 -*

""" init.py: initialization class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from os import getcwd, mkdir
from os.path import isdir
from shutil import rmtree 
import sys

from rst import RstCls
from mol import MolCls
from calc import CalcCls
from mpi import MPICls
from kernel import KernCls
from exp import ExpCls
from drv import DrvCls
from prt import PrintCls
from res import ResCls


class InitCls():
		""" initialization class """
		def __init__(self):
				""" init calculation """
				# mpi instantiation
				self.mpi = MPICls()
				# output instantiation
				self.out = OutCls(self.mpi)
				# restart instantiation
				self.rst = RstCls(self.out, self.mpi)
				# molecule instantiation
				self.mol = MolCls(self.mpi, self.rst)
				# build and communicate molecule
				if (self.mpi.global_master):
					self.mol.make(self.mpi, self.rst)
					self.mpi.bcast_mol_info(self.mol)
				else:
					self.mpi.bcast_mol_info(self.mol)
					self.mol.make(self.mpi, self.rst)
				# calculation instantiation
				self.calc = CalcCls(self.mpi, self.rst, self.mol)
				# kernel instantiation
				self.kernel = KernCls()
				# set core region
				self.mol.ncore = self.mol.set_ncore()
				# communicate calc info 
				self.mpi.bcast_calc_info(self.calc)
				# init mpi
				self.mpi.set_mpi()
				# hf and ref calculations
				if (self.mpi.global_master):
					# hf calculation
					self.calc.hf, self.calc.mo = self.kernel.hf(self.mol, self.calc)
					# get hcore and eri
					self.mol.hcore, self.mol.eri = self.kernel.hcore_eri(self.mol)
					# reference and expansion spaces
					self.calc.ref_space, self.calc.exp_space, \
						self.calc.no_act, self.calc.ne_act = self.kernel.active(self.mol, self.calc)
					# expansion instantiation
					if (self.calc.exp_type in ['occupied','virtual']):
						self.exp = ExpCls(self.mol, self.calc, self.calc.exp_type)
						# mark expansion as micro
						self.exp.level = 'micro'
					elif (self.calc.exp_type == 'combined'):
						self.exp = ExpCls(self.mol, self.calc, 'occupied')
						# mark expansion as macro
						self.exp.level = 'macro'
					# base energy and transformation matrix
					self.calc.energy['base'], self.calc.mo = self.kernel.main_mo(self.mol, self.calc, self.exp)
				else:
					# get hcore and eri
					self.mol.hcore, self.mol.eri = self.kernel.hcore_eri(self.mol)
				# bcast hf and transformation info
				if (self.mpi.parallel):
					self.mpi.bcast_hf_ref_info(self.mol, self.calc)
					self.mpi.bcast_mo_info(self.mol, self.calc, self.mpi.global_comm)
					# in case of combined expansion, have local masters perform hf calc
					if (self.mpi.local_master):
						self.calc.hf = self.kernel.hf(self.mol, self.calc)
				# driver instantiations
				if (self.mpi.global_master):
					if (self.calc.exp_type in ['occupied','virtual']):
						self.drv = DrvCls(self.mol, self.calc.exp_type)
					elif (self.calc.exp_type == 'combined'):
						self.drv = DrvCls(self.mol, 'occupied')
					# print and result instantiations
					self.prt = PrintCls(self.out)
					self.res = ResCls(self.mpi, self.mol, self.calc, self.out)
				else:
					if (self.calc.exp_type in ['occupied','virtual']):
						self.drv = DrvCls(self.mol, self.calc.exp_type)
					elif (self.calc.exp_type == 'combined'):
						if (self.mpi.local_master):
							self.drv = DrvCls(self.mol, 'occupied')
						else:
							self.drv = DrvCls(self.mol, 'virtual')
					# prt as None type
					self.prt = None
				#
				return


class OutCls():
		""" output class """
		def __init__(self, _mpi):
				""" init output environment """
				# get work dir
				self.wrk_dir = getcwd()
				# set output dir
				self.out_dir = self.wrk_dir+'/output'
				if (_mpi.global_master):
					# rm out_dir if present
					if (isdir(self.out_dir)): rmtree(self.out_dir, ignore_errors=True)
					# mk out_dir
					mkdir(self.out_dir)


