#!/usr/bin/env python
# -*- coding: utf-8 -*

""" expansion.py: expansion class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
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
				# init incl_idx, tuples, and hashes
				self.incl_idx, self.tuples, self.hashes = _init_tup(mol, calc)
				# init prop dict
				self.prop = {'energy': [{'inc': [], 'tot': []} for i in range(calc.state['root']+1)]}
				if calc.target['dipole']:
					self.prop['dipole'] = [{'inc': [], 'tot': []} for i in range(calc.state['root']+1)]
				if calc.target['trans']:
					self.prop['trans'] = [{'inc': [], 'tot': []} for i in range(calc.state['root'])]
				# set start_order/max_order
				self.start_order = self.tuples[0].shape[1]
				if calc.misc['order'] is not None:
					self.max_order = min(calc.exp_space.size, calc.misc['order'])
				else:
					self.max_order = calc.exp_space.size
				# init timings
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


def _init_tup(mol, calc):
		""" init tuples and incl_idx """
		# incl_idx
		incl_idx = calc.ref_space.tolist()
		# tuples
		if calc.no_exp == 0:
			tuples = [np.array(list([i] for i in calc.exp_space), dtype=np.int32, order='F')]
		else:
			if calc.model['type'] == 'occ':
				tuples = [np.array([calc.exp_space[-calc.no_exp:]], dtype=np.int32, order='F')]
			elif calc.model['type'] == 'virt':
				tuples = [np.array([calc.exp_space[:calc.no_exp]], dtype=np.int32, order='F')]
		# hashes
		hashes = [tools.hash_2d(tuples[0])]
		return incl_idx, tuples, hashes


