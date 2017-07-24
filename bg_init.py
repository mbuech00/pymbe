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
from bg_time import TimeCls
from bg_driver import DrvCls
from bg_print import PrintCls
from bg_results import ResCls


class InitCls():
		""" initialization class """
		def __init__(self):
				""" init calculation """
				# mpi instance
				self.mpi = MPICls()
				# output instance
				self.out = OutCls(self.mpi)
				# restart instance
				self.rst = RstCls(self.out, self.mpi)
				# molecule and calculation instances
				self.mol = MolCls(self.mpi, self.rst)
				self.calc = CalcCls(self.mpi, self.rst)
				# pyscf instance
				self.pyscf = PySCFCls()
				# hf calculation and integral transformation
				if (self.mpi.master):
					try:
						self.mol.hf, self.mol.e_hf, self.mol.norb, self.mol.nocc, self.mol.nvirt = \
								self.pyscf.hf_calc(self.mol)
					except Exception as err:
						sys.stderr.write('\nHF Error : problem with HF calculation\n'
											'PySCF error : {0:}\n\n'.\
											format(err))
					try:
						self.mol.e_ref, self.mol.h1e, self.mol.h2e = \
								self.pyscf.int_trans(self.mol, self.calc)
					except Exception as err:
						sys.stderr.write('\nINT-TRANS Error : problem with integral transformation\n'
											'PySCF error : {0:}\n\n'.\
											format(err))
				# bcast to slaves
				if (self.mpi.parallel): self.mpi.bcast_hf_base(self.mol)
				# time instance
				self.time = TimeCls(self.mpi, self.rst)
				# expansion instance
				self.exp_prim = ExpCls(self.mpi, self.mol, self.calc, self.rst, self.calc.exp_type)
				self.rst.rst_main(self.mpi, self.calc, self.exp_prim, self.time)
				# driver instance
				self.driver = DrvCls()
				# print and result instances
				if (self.mpi.master):
					self.prt = PrintCls(self.out)
					self.res = ResCls(self.out)
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
				if (_mpi.master):
					# rm out_dir if present
					if (isdir(self.out_dir)): rmtree(self.out_dir, ignore_errors=True)
					# mk out_dir
					mkdir(self.out_dir)


