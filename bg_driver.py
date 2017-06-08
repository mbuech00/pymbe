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
from mpi4py import MPI

from bg_kernel import KernCls
from bg_sum import SumCls
from bg_ent import EntCls
from bg_screen import ScrCls


class DrvCls():
		""" driver class """
		def master(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _err, _prt, _rst):
				""" main driver routine """
				# make kernel, summation, entanglement, and screening instances
				self.kernel = KernCls(_exp)
				self.summation = SumCls()
 				self.entanglement = EntCls()
				self.screening = ScrCls()
				# print expansion header
				_exp.exp_header()
				for _exp.order in range(_calc.min_exp_order,_calc.max_exp_order + 1):
					#
					#** energy kernel phase **#
					#
					# print kernel header
					_exp.kernel_header(_exp)
					# init e_inc
					if (len(_exp.energy_inc) != _exp.order):
						_exp.energy_inc.append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
					# kernel calculations
					_exp.kernel.main(_mpi, _mol, _calc, _pyscf, _exp, _time, _err, _prt, _rst)
					# print kernel end
					_exp.kernel_end(_exp)
					#
					#** energy summation phase **#
					#
					# print summation header
					_exp.summation_header(_exp)
					# energy summation
					_exp.summation.main(_mpi, _calc, _exp, _time, _rst)
					# write restart files
					_rst.write_summation(_mpi, _exp, _time)
					# print summation end
					_exp.summation_end(_calc, _exp)
					# print results
					_exp.summation_results(_exp)
					#
					#** screening phase **#
					#
					# print screen header
					_exp.screen_header(_exp)
					# orbital entanglement
					self.entanglement.main(_mpi, _mol, _calc, _exp, _time, _rst)
					# orbital screening
					if (not _exp.conv_energy[-1]):
						# perform screening
						_exp.screening.main(_mpi, _calc, _exp, _time)
						# write restart files
						if (not _exp.conv_orb[-1]):
							_rst.write_screen(_mpi, _exp, _time)
					# print screen end
					_exp.screen_end(_exp)
					#
					#** convergence check **#
					#
					if (_exp.conv_energy[-1] or _exp.conv_orb[-1]): break
				#
				return
	
	
		def slave(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _err):
				""" main slave routine """
				# make kernel, summation, entanglement, and screening instances
				self.kernel = KernCls(_exp)
				self.summation = SumCls()
 				self.entanglement = EntCls()
				self.screening = ScrCls()
				# set loop/waiting logical
				slave = True
				# start waiting
				while (slave):
					# receive task
					msg = _mpi.comm.bcast(None, root=0)
					# branch depending on task id
					if (msg['task'] == 'bcast_rst'):
						# distribute (receive) restart files
						_mpi.bcast_rst(_calc, _exp, _time) 
					#
					#** energy kernel phase **#
					#
					elif (msg['task'] == 'kernel_slave'):
						_exp.order = msg['order']
						self.kernel.slave(_mpi, _mol, _calc, _pyscf, _exp, _time, _err)
						_time.coll_kernel_time(_mpi, None, _exp.order)
					#
					#** energy summation phase **#
					#
					elif (msg['task'] == 'energy_summation_par'):
						_exp.order = msg['order']
						self.summation.sum_par(_mpi, _calc, _exp, _time)
						_time.coll_summation_time(_mpi, None, _exp.order)
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'ent_abs_par'):
						_exp.order = msg['order']
						self.entanglement.ent_abs_par(_mpi, _exp, _time)
						_time.coll_screen_time(self, None, _exp.order, msg['conv_energy'])
					elif (msg['task'] == 'tuple_generation_par'):
						tuple_generation_slave(molecule,molecule['prim_tuple'],molecule['prim_energy_inc'],msg['thres'],msg['l_limit'],msg['u_limit'],msg['order'],'MACRO')
						collect_screen_mpi_time(molecule,msg['order'],True)
					elif (msg['task'] == 'exit_slave'):
						slave = False
				#
				return
	
	
