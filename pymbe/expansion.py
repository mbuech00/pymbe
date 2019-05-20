#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
expansion module containing all expansion attributes
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.6'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import copy
import numpy as np

import tools


class ExpCls(object):
		"""
		this class contains the pymbe expansion attributes
		"""
		def __init__(self, mol, calc):
				"""
				init expansion attributes

				:param mol: pymbe mol object
				:param calc: pymbe calc object
				"""
				# set expansion model dict
				self.model = copy.deepcopy(calc.model)

				# init prop dict
				self.prop = {calc.target: {'inc': [], 'tot': []}}

				# set max_order
				if calc.misc['order'] is not None:
					self.max_order = min(calc.exp_space['tot'].size, calc.misc['order'])
				else:
					self.max_order = calc.exp_space['tot'].size

				# init timings and and ndets lists
				self.time = {'mbe': [], 'screen': []}
				self.ndets = []

				# init order
				self.order = 0


def init_tup(mol, calc):
		"""
		this function initializes tuples and hashes

		:param mol: pymbe mol object
		:param calc: pymbe calc object
		:return: two lists with numpy arrays of shapes (n_tuples,) [hashes] and (n_tuples, min_order) [tuples]
		"""
		# init tuples
		if calc.ref_space.size == 0:
			tuples = np.array([[i, a] for i in calc.exp_space['occ'] for a in calc.exp_space['virt']], \
								dtype=np.int32)
		else:
			tuples = np.array([[p] for p in calc.exp_space['tot'] if tools.cas_corr(calc.occup, calc.ref_space, p)], \
								dtype=np.int32)

		# init hashes
		hashes = tools.hash_2d(tuples)

		# sort wrt hashes
		tuples = tuples[hashes.argsort()]
		hashes.sort()

		return [hashes], [tuples]


