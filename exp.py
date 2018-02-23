#!/usr/bin/env python
# -*- coding: utf-8 -*

""" exp.py: expansion class """

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
		def __init__(self, _mol, _calc, _type):
				""" init parameters """
				# init tuples and incl_idx
				self.incl_idx, self.tuples = self.init_tuples(_mol, _calc, _type)
				# init energy dict
				self.energy = {}
				self.energy['inc'] = []
				self.energy['tot'] = []
				# set start_order/max_order and determine max theoretical work
				self.theo_work = []
				self.start_order = self.tuples[0].shape[1]
				self.max_order = min(len(_calc.exp_space), _calc.exp_max_order)
				for k in range(self.start_order, len(_calc.exp_space)+1):
					self.theo_work.append(int(factorial(len(_calc.exp_space)) / \
											(factorial(k) * factorial(len(_calc.exp_space) - k))))
				# init micro_conv list
				self.micro_conv = []
				# init convergence list
				self.conv_orb = [False]
				# init timings
				self.time_mbe = []
				self.time_screen = []
				# init thres
				self.thres = _calc.exp_thres
				#
				return


		def init_tuples(self, _mol, _calc, _type):
				""" init tuples and incl_idx """
				# incl_idx
				incl_idx = _calc.ref_space.tolist()
				# tuples
				if (_calc.no_act == 0):
					tuples = [np.array(list([i] for i in _calc.exp_space), dtype=np.int32)]
				else:
					if (_type == 'occupied'):
						tuples = [np.array([_calc.exp_space[-(_calc.no_act-len(_calc.ref_space)):]], dtype=np.int32)]
					elif (_type == 'virtual'):
						tuples = [np.array([_calc.exp_space[:(_calc.no_act-len(_calc.ref_space))]], dtype=np.int32)]
				#
				return incl_idx, tuples


