#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np


class DrvCls():
		""" driver class """
		def driver(self, _mpi, _mol, _calc, _exp, _prt, _time, _rst, _err):
				""" main driver routine """
				# print expansion header
				_prt.exp_header()
				for _exp.order in range(_calc.min_exp_order,_calc.max_exp_order + 1):
					#
					#** energy kernel phase **#
					#
					# print kernel header
					_prt.kernel_header(_exp)
					# init e_inc
					if (len(_exp.energy_inc) != _exp.order):
						_exp.energy_inc.append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
					# kernel calculations
					_exp.kernel.main(_mpi, _mol, _calc, _exp, _time, _err)
					# print kernel end
					_prt.kernel_end(_exp)
					#
					#** energy summation phase **#
					#
					# print summation header
					_prt.summation_header(_exp)
					# energy summation
					_exp.summation.main(_mpi, _exp, _time)
					# write restart files
					_rst.write_summation(_mpi, _exp, _time)
					# print summation end
					_prt.summation_end(_calc, _exp)
					# print results
					_prt.summation_results(_exp)
					#
					#** screening phase **#
					#
					# print screen header
					_prt.screen_header(_exp)
					# orbital entanglement
					_exp.ent_main(_mpi, _exp, _time)
					# orbital screening
					if (not _exp.conv_energy[-1]):
						# perform screening
						_exp.screening.main(_mpi, _calc, _exp, _time)
						# write restart files
						if (not _exp.conv_orb[-1]):
							_rst.write_screen(_mpi, _exp, _time)
					# print screen end
					_prt.screen_end(_exp)
					#
					#** convergence check **#
					#
					if (_exp.conv_energy[-1] or _exp.conv_orb[-1]): break
				#
				return
		
		
