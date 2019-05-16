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


class ExpCls(object):
		""" expansion class """
		def __init__(self, mol, calc):
				""" init parameters """
				# set expansion model dict
				self.model = copy.deepcopy(calc.model)
				# init prop dict
				self.prop = {}
				self.prop[calc.target] = {'inc': [], 'tot': []}
				# set max_order
				if calc.misc['order'] is not None:
					self.max_order = min(calc.exp_space['tot'].size, calc.misc['order'])
				else:
					self.max_order = calc.exp_space['tot'].size
				# init timings, calculation counter, and ndets lists
				self.count = []
				self.ndets = []
				self.time = {'mbe': [], 'screen': []}
				# init order (pre-calc)
				self.order = 0


def init_tup(mol, calc):
		""" init tuples and hashes """
		# tuples
		if calc.ref_space.size == 0:
			tuples = np.array([[i, a] for i in calc.exp_space['occ'] for a in calc.exp_space['virt']], \
								dtype=np.int32)
		else:
			tuples = np.array([[p] for p in calc.exp_space['tot'] if tools.cas_allow(calc.occup, calc.ref_space, p)], \
								dtype=np.int32)
		# hashes
		hashes = tools.hash_2d(tuples)
		# sort wrt hashes
		tuples = tuples[hashes.argsort()]
		hashes.sort()
		return [hashes], [tuples]


