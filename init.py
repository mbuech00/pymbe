#!/usr/bin/env python
# -*- coding: utf-8 -*

""" init.py: initialization class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from os import getcwd, mkdir
from os.path import isdir
from shutil import rmtree 
import sys

from mol import MolCls
from calc import CalcCls
from mpi import MPICls
import rst
import kernel
from exp import ExpCls
from drv import DrvCls


class InitCls():
		""" initialization class """
		def __init__(self):
				""" init calculation """
				# mpi instantiation
				self.mpi = MPICls()
				# output instantiation
				self.out = OutCls(self.mpi)
				# molecule instantiation
				self.mol = MolCls(self.mpi)
				# build and communicate molecule
				if (self.mpi.global_master):
					self.mol.make(self.mpi)
					self.mpi.bcast_mol(self.mol)
				else:
					self.mpi.bcast_mol(self.mol)
					self.mol.make(self.mpi)
				# calculation instantiation
				self.calc = CalcCls(self.mpi, self.mol)
				# set core region
				self.mol.ncore = self.mol.set_ncore()
				# communicate calc info 
				self.mpi.bcast_calc(self.calc)
				# init mpi
				self.mpi.set_mpi()
				# restart logical
				self.calc.restart = rst.restart()
				# hf and ref calculations
				if (self.mpi.global_master):
					# restart
					if (self.calc.restart):
						# read fundamental info
						rst.read_fund(self.mol, self.calc)
						# expansion instantiation
						if (self.calc.exp_type in ['occupied','virtual']):
							self.exp = ExpCls(self.mol, self.calc)
							# mark expansion as micro
							self.exp.level = 'micro'
#						elif (self.calc.exp_type == 'combined'):
#							self.exp = ExpCls(self.mol, self.calc)
#							# mark expansion as macro
#							self.exp.level = 'macro'
					# no restart
					else:
						# hf calculation
						self.calc.hf, self.calc.energy['hf'], self.calc.occup, \
							self.calc.orbsym, self.calc.mo = kernel.hf(self.mol, self.calc)
						# get hcore and eri
						self.mol.hcore, self.mol.eri = kernel.hcore_eri(self.mol)
						# reference and expansion spaces
						self.calc.ref_space, self.calc.exp_space, self.calc.no_act = kernel.active(self.mol, self.calc)
						# expansion instantiation
						if (self.calc.exp_type in ['occupied','virtual']):
							self.exp = ExpCls(self.mol, self.calc)
							# mark expansion as micro
							self.exp.level = 'micro'
#						elif (self.calc.exp_type == 'combined'):
#							self.exp = ExpCls(self.mol, self.calc, 'occupied')
#							# mark expansion as macro
#							self.exp.level = 'macro'
						# reference calculation
						self.calc.energy['ref'], self.calc.energy['ref_base'], \
							self.calc.mo = kernel.ref(self.mol, self.calc, self.exp)
						# base energy and transformation matrix
						self.calc.energy['base'], self.calc.mo = kernel.base(self.mol, self.calc, self.exp)
						# write fundamental info
						rst.write_fund(self.mol, self.calc)
				else:
					# get hcore and eri
					self.mol.hcore, self.mol.eri = kernel.hcore_eri(self.mol)
				# bcast fundamental info
				if (self.mpi.parallel): self.mpi.bcast_fund(self.mol, self.calc)
				# driver instantiations
				if (self.mpi.global_master):
					if (self.calc.exp_type in ['occupied','virtual']):
						self.drv = DrvCls(self.mol, self.calc)
#					elif (self.calc.exp_type == 'combined'):
#						self.drv = DrvCls(self.mol, 'occupied')
				else:
					if (self.calc.exp_type in ['occupied','virtual']):
						self.drv = DrvCls(self.mol, self.calc)
#					elif (self.calc.exp_type == 'combined'):
#						if (self.mpi.local_master):
#							self.drv = DrvCls(self.mol, 'occupied')
#						else:
#							self.drv = DrvCls(self.mol, 'virtual')
				#
				return


class OutCls():
		""" output class """
		def __init__(self, mpi):
				""" init output environment """
				# get work dir
				self.wrk_dir = getcwd()
				# set output dir
				self.out_dir = self.wrk_dir+'/output'
				if (mpi.global_master):
					# rm out_dir if present
					if (isdir(self.out_dir)): rmtree(self.out_dir, ignore_errors=True)
					# mk out_dir
					mkdir(self.out_dir)


