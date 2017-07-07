#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI

from bg_kernel import KernCls
from bg_sum import SumCls
from bg_ent import EntCls
from bg_screen import ScrCls


class DrvCls():
		""" driver class """
		def master(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst):
				""" main driver routine """
				# make kernel, summation, entanglement, and screening instances
				self.kernel = KernCls(_exp)
				self.summation = SumCls()
				self.entanglement = EntCls()
				self.screening = ScrCls(_exp)
				# print expansion header
				_prt.exp_header()
				# now do expansion
				for _exp.order in range(_calc.exp_min_order,_calc.exp_max_order + 1):
					#
					#** energy kernel phase **#
					#
					# print kernel header
					_prt.kernel_header(_exp)
					# init e_inc
					if (len(_exp.energy_inc) != _exp.order):
						_exp.energy_inc.append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
					# kernel calculations
					self.kernel.main(_mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst)
					# print kernel end
					_prt.kernel_end(_exp)
					#
					#** energy summation phase **#
					#
					# print summation header
					_prt.summation_header(_exp)
					# energy summation
					self.summation.main(_mpi, _calc, _exp, _time, _rst)
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
					self.entanglement.main(_mpi, _mol, _calc, _exp, _time, _rst)
					# orbital screening
					if ((not _exp.conv_energy[-1]) and (_exp.order < _calc.exp_max_order)):
						# perform screening
						self.screening.main(_mpi, _calc, _exp, _time, _rst)
						# write restart files
						if (not _exp.conv_orb[-1]):
							_rst.write_screen(_mpi, _exp, _time)
						# print screen results
						_prt.screen_results(_calc, _exp)
						# print screen end
						_prt.screen_end(_exp)
					else:
						# print screen end
						_prt.screen_end(_exp)
						break
					# update threshold and restart frequency
					_rst.update(_calc, _exp.order)
					#
					#** convergence check **#
					#
					if (_exp.conv_energy[-1] or _exp.conv_orb[-1]): break
				#
				return
	
	
		def slave(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _rst):
				""" main slave routine """
				# make kernel, summation, entanglement, and screening instances
				self.kernel = KernCls(_exp)
				self.summation = SumCls()
				self.entanglement = EntCls()
				self.screening = ScrCls(_exp)
				# set loop/waiting logical
				slave = True
				# enter slave state
				while (slave):
					# task id
					msg = _mpi.comm.bcast(None, root=0)
					#
					#** energy kernel phase **#
					#
					if (msg['task'] == 'kernel_slave'):
						_exp.order = msg['order']
						self.kernel.slave(_mpi, _mol, _calc, _pyscf, _exp, _time)
						_time.coll_kernel_time(_mpi, None, _exp.order)
					#
					#** energy summation phase **#
					#
					elif (msg['task'] == 'sum_par'):
						_exp.order = msg['order']
						self.summation.sum_par(_mpi, _calc, _exp, _time)
						_time.coll_summation_time(_mpi, None, _exp.order)
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'ent_abs_par'):
						_exp.order = msg['order']
						self.entanglement.ent_abs_par(_mpi, _exp, _time)
						_time.coll_screen_time(_mpi, None, _exp.order, msg['conv_energy'])
					elif (msg['task'] == 'screen_slave'):
						_exp.order = msg['order']
						self.screening.slave(_mpi, _calc, _exp, _time)
						_time.coll_screen_time(_mpi, None, _exp.order, True)
						_rst.update(_calc, _exp.order)
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_slave'):
						slave = False
				# finalize
				_mpi.final(None)
	
	
