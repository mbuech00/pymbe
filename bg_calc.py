#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_calc.py: calculation class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from os.path import isfile


class CalcCls():
		""" calculation class """
		def __init__(self, _err):
				""" init parameters """
				self.exp_model = 'fci'
				self.exp_type = 'virtual'
				self.exp_thres = 1.0e-06
				self.exp_damp = 1.0
				self.exp_max_order = 0
				self.exp_orbs = 'canonical'
				self.energy_thres = 3.8e-04
				self.exp_thres_init = self.exp_thres
				# set calculation parameters
				self.exp_model, self.exp_type, self.exp_thres, self.exp_damp, \
					self.exp_max_order, self.exp_orbs, self.energy_thres = self.set_calc(_err)
				# sanity check
				self.sanity_chk(_err)


		def set_calc(self, _err):
				""" set calculation parameters from bg-calc.inp file """
                # error handling
				if (not isfile('bg-calc.inp')):
					_err.error_msg = 'bg-calc.inp not found'
					_err.abort()
				# read input file
				with open('bg-calc.inp') as f:
					content = f.readlines()
					for i in range(len(content)):
						if (content[i].split()[0] == 'exp_model'):
							exp_model = content[i].split()[2].upper()
						elif (content[i].split()[0] == 'exp_type'):
							exp_type = content[i].split()[2]
						elif (content[i].split()[0] == 'exp_thres'):
							exp_thres = float(content[i].split()[2])
						elif (content[i].split()[0] == 'exp_damp'):
							exp_damp = float(content[i].split()[2])
						elif (content[i].split()[0] == 'exp_max_order'):
							exp_max_order = int(content[i].split()[2])
						elif (content[i].split()[0] == 'orbitals'):
							exp_orbs = content[i].split()[2]
						elif (content[i].split()[0] == 'energy_thres'):
							energy_thres = float(content[i].split()[2])
						# error handling
						else:
							_err.error_msg = content[i].split()[2] + \
											' keyword in bg-calc.inp not recognized'
							_err.abort()
				#
				return exp_model, exp_type, exp_thres, exp_damp, \
							exp_max_order, exp_orbs, energy_thres


		def sanity_chk(self, _err):
				""" sanity check for calculation parameters """
				# type of expansion
				if (not (self.exp_type in ['occupied','virtual'])):
					_err.error_msg = 'wrong input -- valid choices for ' + \
									'expansion scheme are occupied and virtual'
				# expansion model
				if (not (self.exp_model in ['CCSD','FCI'])):
					_err.error_msg = 'wrong input -- valid expansion models ' + \
									'are currently: CCSD and FCI'
				# max order
				if (self.exp_max_order < 0):
					_err.error_msg = 'wrong input -- wrong maximum ' + \
									'expansion order (must be integer >= 1)'
				# expansion thresholds
				if (self.exp_thres < 0.0):
					_err.error_msg = 'wrong input -- expansion threshold ' + \
									'(exp_thres) must be float >= 0.0'
				if (self.exp_damp < 1.0):
					_err.error_msg = 'wrong input -- expansion dampening ' + \
									'(exp_damp) must be float >= 1.0'
				if (self.energy_thres < 0.0):
					_err.error_msg = 'wrong input -- energy threshold ' + \
									'(energy_thres) must be float >= 0.0'
				# orbital representation
				if (not (self.exp_orbs in ['canonical','local','natural'])):
					_err.error_msg = 'wrong input -- valid orbital ' + \
									'representations are currently: canonical, ' + \
									'local, and natural (CCSD)'
				#
				if (_err.error_msg != ''):
					_err.abort()
				return


