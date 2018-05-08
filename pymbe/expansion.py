#!/usr/bin/env python
# -*- coding: utf-8 -*

""" expansion.py: expansion class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np


class ExpCls():
		""" expansion class """
		def __init__(self, mol, calc, typ):
				""" init parameters """
				# set type of expansion
				self.typ = typ
				# init tuples and incl_idx
				self.incl_idx, self.tuples = _init_tup(mol, calc)
				# init property dict
				self.property = {}
				self.property['energy'] = {'inc': [], 'tot': []}
				if calc.prop['DIPOLE']:
					self.property['dipole'] = {'inc': [], 'tot': []}
				if calc.prop['EXCITATION']:
					self.property['excitation'] = {'inc': [], 'tot': []}
				# set start_order/max_order
				self.start_order = self.tuples[0].shape[1]
				self.max_order = min(len(calc.exp_space), calc.max_order)
				# init convergence list
				self.conv_orb = [False]
				# init timings
				self.time = {'mbe': [], 'screen': []}
				# init thres
				if self.start_order < 3:
					self.thres = 0.0
				else:
					self.thres = calc.thres * calc.relax ** (self.start_order - 3)
				# init order (pre-calc)
				self.order = 0
				# restart frequency
				self.rst_freq = 50000


def _init_tup(mol, calc):
		""" init tuples and incl_idx """
		# incl_idx
		incl_idx = calc.ref_space.tolist()
		# tuples
		if calc.no_exp == 0:
			tuples = [np.array(list([i] for i in calc.exp_space), dtype=np.int32, order='F')]
		else:
			if calc.typ == 'occupied':
				tuples = [np.array([calc.exp_space[-calc.no_exp:]], dtype=np.int32, order='F')]
			elif calc.typ == 'virtual':
				tuples = [np.array([calc.exp_space[:calc.no_exp]], dtype=np.int32, order='F')]
		return incl_idx, tuples


