#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_init.py: initialization class for Bethe-Goldstone correlation calculations."""

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

from bg_rst import RstCls
from bg_mol import MolCls
from bg_calc import CalcCls
from bg_mpi import MPICls
from bg_pyscf import PySCFCls
from bg_exp import ExpCls
from bg_drv import DrvCls
from bg_print import PrintCls
from bg_results import ResCls


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
				# molecule and calculation instantiations
				self.mol = MolCls(self.mpi, self.rst)
				self.calc = CalcCls(self.mpi, self.rst)
				# pyscf instantiation
				self.pyscf = PySCFCls()
				# hf calculation
				try:
					self.mol.hf, self.mol.norb, self.mol.nocc, self.mol.nvirt = self.pyscf.hf(self.mol, self.calc)
				except Exception as err:
					sys.stderr.write('\nHF Error : problem with HF calculation\n'
										'PySCF error : {0:}\n\n'.\
										format(err))
					raise
				# expansion and driver instantiations
				if (self.mpi.global_master):
					if (self.calc.exp_type in ['occupied','virtual']):
						self.exp = ExpCls(self.mpi, self.mol, self.calc, self.calc.exp_type)
						self.drv = DrvCls(self.mol, self.calc.exp_type)
						# mark expansion as micro
						self.exp.level = 'micro'
					elif (self.calc.exp_type == 'combined'):
						self.exp = ExpCls(self.mpi, self.mol, self.calc, 'occupied')
						self.drv = DrvCls(self.mol, 'occupied')
						# mark expansion as macro
						self.exp.level = 'macro'
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


