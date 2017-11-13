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
				for k in range(len(self.tuples[0][0]), self.max_order+1):
					self.theo_work.append(int(factorial(self.max_order) / \
											(factorial(k) * factorial(self.max_order - k))))
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
				if (_calc.exp_ref['METHOD'] == 'HF'):
					if (_type == 'occupied'):
						init = _mol.occ
						incl_idx = _mol.virt.tolist()
					# set params and lists for virt expansion
					elif (_type == 'virtual'):
						init = _mol.virt
						incl_idx = _mol.occ.tolist()
					# append to tuples
					if ((_calc.exp_base['METHOD'] is None) or (_mol.spin > 0)):
						tuples = [np.array(list([i] for i in init), dtype=np.int32)]
					else:
						tmp = []
						for i in range(len(init)):
							for m in range(init[i]+1, init[-1]+1):
								tmp.append([init[i]]+[m])
						tmp.sort()
						tuples = [np.array(tmp, dtype=np.int32)]
				elif (_calc.exp_ref['METHOD'] in ['CASCI','CASSCF']):
					init = []
					if (_type == 'occupied'):
						for i in range(len(_mol.occ)):
							if (_mol.occ[i] not in _calc.act_orbs):
								init.append(_calc.act_orbs.tolist() + [_mol.occ[i]])
						incl_idx = _mol.virt.tolist()
					# set params and lists for virt expansion
					elif (_type == 'virtual'):
						for i in range(len(_mol.virt)):
							if (_mol.virt[i] not in _calc.act_orbs):
								init.append(_calc.act_orbs.tolist() + [_mol.virt[i]])
						incl_idx = _mol.occ.tolist()
					tuples = [np.array(init, dtype=np.int32)]
				#
				return incl_idx, tuples


