#!/usr/bin/env python
# -*- coding: utf-8 -*

""" expansion.py: expansion class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import copy
import numpy as np

import tools


class ExpCls():
		""" expansion class """
		def __init__(self, mol, calc, typ):
				""" init parameters """
				# set expansion model dict
				self.model = copy.deepcopy(calc.model)
				self.model['type'] = typ
				# init prop dict
				self.prop = {}
				if calc.target['energy']:
					self.prop['energy'] = {'inc': [], 'tot': []}
				if calc.target['excitation']:
					self.prop['excitation'] = {'inc': [], 'tot': []}
				if calc.target['dipole']:
					self.prop['dipole'] = {'inc': [], 'tot': []}
				if calc.target['trans']:
					self.prop['trans'] = {'inc': [], 'tot': []}
				# set start_order/max_order
				self.start_order = calc.no_exp + 1
				if calc.misc['order'] is not None:
					self.max_order = min(calc.exp_space.size + calc.no_exp, calc.misc['order'])
				else:
					self.max_order = calc.exp_space.size + calc.no_exp
				# init timings and calculation counter
				self.count = []
				self.time = {'mbe': [], 'screen': []}
				# init thres
				if self.start_order < 3:
					self.thres = 0.0
				else:
					self.thres = calc.thres['init'] * calc.thres['relax'] ** (self.start_order - 3)
				# init order (pre-calc)
				self.order = 0
				# restart frequency
				self.rst_freq = 50000


def init_tup(mol, calc):
		""" init tuples and hashes """
		# tuples
		if calc.extra['sigma']:
			tuples = [np.array([[i] for i in calc.exp_space if tools.sigma_prune(calc.mo_energy, calc.orbsym, np.asarray([i], dtype=np.int32))], dtype=np.int32)]
		else:
			tuples = [np.array([[i] for i in calc.exp_space], dtype=np.int32)]
		# hashes
		hashes = [tools.hash_2d(tuples[0])]
		# sort wrt hashes
		tuples[0] = tuples[0][hashes[0].argsort()]
		hashes[0].sort()
		return tuples, hashes


