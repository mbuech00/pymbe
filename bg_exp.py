#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_init.py: expansion class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from scipy.misc import factorial


class ExpCls():
		""" expansion class """
		def __init__(self, _mpi, _mol, _calc, _type):
				""" init parameters """
				# init_tuples and incl_idx
				if (_type == 'occupied'):
					init_tuples = _mol.occ
					self.incl_idx = _mol.virt
				# set params and lists for virt expansion
				elif (_type == 'virtual'):
					init_tuples = _mol.virt
					self.incl_idx = _mol.occ
				# set frozen_idx
				self.frozen_idx = list(range(_mol.ncore))
				# update init_tuples and incl_idx
				init_tuples = sorted(list(set(init_tuples) - set(self.frozen_idx)))
				init_tuples = list([i] for i in init_tuples)
				self.incl_idx = sorted(list(set(self.incl_idx) - set(self.frozen_idx))) 
				# append to self.tuples
				self.tuples = [np.array(init_tuples, dtype=np.int32)]
				# verbose print
				if (_mol.verbose > 1):
					print('mo_occ = {0:} , incl_idx = {1:} , init_tuples = {2:}'.format(_calc.hf_mo_occ,self.incl_idx,init_tuples))
				# init energy_inc
				self.energy_inc = []
				# set max_order (derived from calc class)
				self.max_order = _calc.exp_max_order
				if (_type == 'occupied'):
					if ((self.max_order == 0) or (self.max_order > (_mol.nocc-_mol.ncore))):
						self.max_order = _mol.nocc - _mol.ncore
				else:
					if ((self.max_order == 0) or (self.max_order > _mol.nvirt)):
						self.max_order = _mol.nvirt
				# determine max theoretical work
				self.theo_work = []
				if (_type == 'occupied'):
					for k in range(_mol.nocc-_mol.ncore):
						self.theo_work.append(int(factorial(_mol.nocc-_mol.ncore) / \
												(factorial(k+1) * factorial((_mol.nocc-_mol.ncore) - (k+1)))))
				else:
					for k in range(_mol.nvirt):
						self.theo_work.append(int(factorial(_mol.nvirt) / \
												(factorial(k+1) * factorial(_mol.nvirt - (k+1)))))
				# init micro_conv list
				if (_mpi.global_master): self.micro_conv = []
				# init convergence lists
				self.conv_orb = [False]
				self.conv_energy = [False]
				# init total energy list
				self.energy_tot = []
				# init timings
				if (_mpi.global_master):
					self.time_kernel = []
					self.time_screen = []
				# init e_core
				self.e_core = None
				#
				return


