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
from itertools import combinations, chain
from scipy.misc import comb, factorial


class ExpCls():
		""" expansion class """
		def __init__(self, _mpi, _mol, _calc, _rst):
				""" init parameters """
				# set params and lists for occ expansion
				if (_calc.exp_type == 'occupied'):
					# set lower and upper limits
					self.l_limit = 0
					self.u_limit = _mol.nocc
					# init tuples and incl_idx
					self.tuples = []; self.tuples.append(np.array(list([i] for i in range(_mol.ncore,
										_mol.nocc)), dtype=np.int32))
					self.incl_idx = list(range(_mol.nocc, _mol.norb))
				# set params and lists for virt expansion
				elif (_calc.exp_type == 'virtual'):
					# set lower and upper limits
					self.l_limit = _mol.nocc
					self.u_limit = _mol.nvirt
					# init tuples and incl_idx
					self.tuples = []; self.tuples.append(np.array(list([i] for i in range(_mol.nocc,
										_mol.norb)), dtype=np.int32))
					self.incl_idx = list(range(_mol.nocc))
				# set frozen_idx
				self.frozen_idx = list(range(_mol.ncore))
				# init energy_inc
				if (_rst.restart):
					self.energy_inc = []
				else:
					self.energy_inc = []; self.energy_inc.append(np.zeros(len(self.tuples[0]),
								dtype=np.float64))
				# set max_order (in calc class)
				if ((_calc.exp_max_order == 0) or (_calc.exp_max_order > self.u_limit)):
					_calc.exp_max_order = self.u_limit
					if ((_calc.exp_type == 'occupied') and _mol.frozen):
						_calc.exp_max_order -= _mol.ncore
				# determine max theoretical work
				self.theo_work = []
				for k in range(_calc.exp_max_order):
					self.theo_work.append(int(factorial(self.u_limit) / \
											(factorial(k+1) * factorial(self.u_limit - (k+1)))))
				# init screen_count list
				self.screen_count = []
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


		def comb_index(self, _n, _k):
				""" calculate combined index """
				count = comb(_n, _k, exact=True)
				index = np.fromiter(chain.from_iterable(combinations(range(_n), _k)),
									int,count=count * _k)
				#
				return index.reshape(-1, _k)
		
		
		def enum(self, *sequential, **named):
				""" hardcoded enums
				see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
				"""
				enums = dict(zip(sequential, range(len(sequential))), **named)
				#
				return type('Enum', (), enums)


