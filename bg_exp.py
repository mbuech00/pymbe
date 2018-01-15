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
				# init tuples and incl_idx
				self.incl_idx, self.tuples = self.init_tuples(_mol, _calc, _type)
				# set start order
				self.start_order = self.tuples[0].shape[1]
				# init energy_inc
				self.energy_inc = []
				# set max_order (derived from calc class) and determine max theoretical work
				self.theo_work = []
				if (_type == 'occupied'):
					self.max_order = min(_mol.nocc - _mol.ncore, _calc.exp_max_order)
					for k in range(self.start_order, (_mol.nocc - _mol.ncore)+1):
						self.theo_work.append(int(factorial(_mol.nocc - _mol.ncore) / \
												(factorial(k) * factorial((_mol.nocc - _mol.ncore) - k))))
				else:
					self.max_order = min(_mol.nvirt, _calc.exp_max_order)
					for k in range(self.start_order, _mol.nvirt+1):
						self.theo_work.append(int(factorial(_mol.nvirt) / \
												(factorial(k) * factorial(_mol.nvirt - k))))
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
				# init thres
				self.thres = _calc.exp_thres
				#
				return


		def init_tuples(self, _mol, _calc, _type):
				""" init tuples and incl_idx """
				# incl_idx
				if (_type == 'occupied'):
					incl_idx = _mol.virt.tolist()
				elif (_type == 'virtual'):
					incl_idx = _mol.occ.tolist()
				# tuples
				if (_calc.exp_ref['METHOD'] == 'HF'):
					# set params and lists for occ expansion
					if (_type == 'occupied'):
						init = _mol.occ
					# set params and lists for virt expansion
					elif (_type == 'virtual'):
						init = _mol.virt
					tuples = [np.array(list([i] for i in init), dtype=np.int32)]
				else:
					tuples = [np.array([_calc.act_orbs.tolist()], dtype=np.int32)]
				#
				return incl_idx, tuples


