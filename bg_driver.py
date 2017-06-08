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


class DrvCls():
		""" driver class """
		def master(self, _mpi, _mol, _calc, _exp, _time, _rst, _err):
				""" main driver routine """
				# make kernel, summation, entanglement, and screening instances
				self.kernel = KernCls()
				self.summation = SummCls()
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
					_exp.kernel.main(_mpi, _mol, _calc, _exp, _time, _err)
					# print kernel end
					_exp.kernel_end(_exp)
					#
					#** energy summation phase **#
					#
					# print summation header
					_exp.summation_header(_exp)
					# energy summation
					_exp.summation.main(_mpi, _exp, _time)
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
					_exp.ent_main(_mpi, _exp, _time)
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
	
	
		def slave(self, _mpi, _mol, _calc, _exp, _time, _err):
				""" main slave routine """
				# make kernel, summation, entanglement, and screening instances
				self.kernel = KernCls()
				self.summation = SummCls()
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
					elif (msg['task'] == 'energy_kernel_par'):
						# energy kernel
						energy_kernel_slave(molecule,molecule['prim_tuple'],molecule['prim_energy_inc'],msg['l_limit'],msg['u_limit'],msg['order'],'MACRO')
						collect_kernel_mpi_time(molecule,msg['order'])
					#
					#** energy summation phase **#
					#
					elif (msg['task'] == 'energy_summation_par'):
						# energy summation
						energy_summation_par(molecule,molecule['prim_tuple'],molecule['prim_energy_inc'],None,None,msg['order'],'MACRO')
						collect_summation_mpi_time(molecule,msg['order'])
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'ent_abs_par'):
						# orbital entanglement
						_exp.order = msg['order']
						self.entanglement.ent_abs_par(self, _exp, _time)
						_time.coll_screen_time(self, None, _exp.order, msg['conv_energy'])
					elif (msg['task'] == 'tuple_generation_par'):
						# generate tuples
						tuple_generation_slave(molecule,molecule['prim_tuple'],molecule['prim_energy_inc'],msg['thres'],msg['l_limit'],msg['u_limit'],msg['order'],'MACRO')
						collect_screen_mpi_time(molecule,msg['order'],True)
					elif (msg['task'] == 'exit_slave'):
						slave = False
				#
				return
	
	
