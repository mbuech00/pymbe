#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_init.py: initialization class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from os import getcwd, mkdir
from os.path import isdir
from shutil import rmtree 

from bg_rst import RstCls
from bg_error import ErrCls
from bg_mol import MolCls
from bg_calc import CalcCls
from bg_mpi import MPICls
from bg_pyscf import PySCFCls
from bg_exp import ExpCls
from bg_time import TimeCls
from bg_driver import DrvCls
from bg_print import PrintCls
from bg_results import ResCls


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
				# driver, print, and result instances
				if (self.mpi.master):
					self.drv = DrvCls()
					self.prt = PrintCls(self.out.out_dir)
					self.res = ResCls(self.out.out_dir)
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


