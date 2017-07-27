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

#from bg_kernel import KernCls
import bg_kernel
from bg_screen import ScrCls
from bg_exp import ExpCls


class DrvCls():
		""" driver class """
		def __init__(self, _mol, _type):
				""" init parameters and classes """
				# init required classes
				self.kernel = bg_kernel.KernCls()
				self.screening = ScrCls(_mol, _type)
				#
				return


		def main(self, _mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst):
				""" main driver routine """
				# exp class instantiation on slaves
				if (_mpi.parallel):
					if (_calc.exp_type in ['occupied','virtual']):
						msg = {'task': 'exp_cls', 'type': _calc.exp_type, 'rst': _rst.restart}
						# bcast msg
						_mpi.local_comm.bcast(msg, root=0)
					else:
						if (_exp.level == 'macro'):
							msg = {'task': 'exp_cls', 'rst': _rst.restart}
							# bcast msg
							_mpi.master_comm.bcast(msg, root=0)
						else:
							msg = {'task': 'exp_cls', 'order_macro': _exp.order_macro, 'incl_idx': _exp.incl_idx}
							# bcast msg
							_mpi.local_comm.bcast(msg, root=0)
				# restart
				_rst.rst_main(_mpi, _calc, _exp, _time)
				# print expansion header
				_prt.exp_header(_calc, _exp)
				# now do expansion
				for _exp.order in range(_exp.min_order, _exp.max_order+1):
					#
					# set order for timings
					#
					if (_calc.exp_type in ['occupied','virtual']):
						_time.order = _exp.order
					else:
						if (_exp.level == 'macro'): _exp.order_macro = _exp.order
						_time.order = _exp.order_macro
					#
					#** energy kernel phase **#
					#
					# print kernel header
					_prt.kernel_header(_calc, _exp)
					# init e_inc
					if (len(_exp.energy_inc) != _exp.order):
						_exp.energy_inc.append(np.zeros(len(_exp.tuples[-1]), dtype=np.float64))
					# kernel calculations
					self.kernel.main(_mpi, _mol, _calc, _pyscf, _exp, _time, _prt, _rst)
					# print micro results
					_prt.kernel_micro_results(_calc, _exp)
					# print kernel end
					_prt.kernel_end(_calc, _exp)
					# write restart files
					_rst.write_kernel(_mpi, _calc, _exp, _time, True)
					# print kernel results
					_prt.kernel_results(_calc, _exp)
					#
					#** screening phase **#
					#
					# print screen header
					_prt.screen_header(_calc, _exp)
					# orbital screening
					if ((not _exp.conv_energy[-1]) and (_exp.order < _exp.max_order)):
						# perform screening
						self.screening.main(_mpi, _calc, _exp, _time, _rst)
						# write restart files
						if (not _exp.conv_orb[-1]):
							_rst.write_screen(_mpi, _calc, _exp, _time)
						# print screen results
						_prt.screen_results(_calc, _exp)
						# print screen end
						_prt.screen_end(_calc, _exp)
					else:
						# print screen end
						_prt.screen_end(_calc, _exp)
					# update restart frequency
					_rst.rst_freq = _rst.update()
					#
					#** convergence check **#
					#
					if (_exp.conv_energy[-1] or _exp.conv_orb[-1]): break
				#
				return


		def local_master(self, _mpi, _mol, _calc, _pyscf, _time):
				""" local master routine """
				# set loop/waiting logical
				local_master = True
				# enter local master state
				while (local_master):
					# task id
					msg = _mpi.master_comm.bcast(None, root=0)
					#
					#** exp class instantiation **#
					#
					if (msg['task'] == 'exp_cls'):
						exp = ExpCls(_mpi, _mol, _calc, 'occupied')
						exp.level = 'macro'
						# receive rst data
						if (msg['rst']): _mpi.bcast_rst(_calc, exp, _time)
					#
					#** energy kernel phase **#
					#
					if (msg['task'] == 'kernel_slave'):
						exp.order = msg['exp_order']
						_time.order = msg['time_order']
						self.kernel.slave(_mpi, _mol, _calc, _pyscf, exp, _time)
						_time.coll_phase_time(_mpi, None, _time.order, 'kernel')
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_local_master'):
						local_master = False
				# finalize
				_mpi.final(None)
	
	
		def slave(self, _mpi, _mol, _calc, _pyscf, _time):
				""" slave routine """
				# set loop/waiting logical
				slave = True
				# enter slave state
				while (slave):
					# task id
					msg = _mpi.local_comm.bcast(None, root=0)
					#
					#** exp class instantiation **#
					#
					if (msg['task'] == 'exp_cls'):
						exp = ExpCls(_mpi, _mol, _calc, msg['type'])
						exp.level = 'micro'
						if (_calc.exp_type == 'combined'):
							exp.order_macro = msg['order_macro']
							exp.incl_idx = msg['incl_idx']
						else:
							# receive rst data
							if (msg['rst']): _mpi.bcast_rst(_calc, exp, _time)
					#
					#** energy kernel phase **#
					#
					if (msg['task'] == 'kernel_slave'):
						exp.order = msg['exp_order']
						_time.order = msg['time_order']
						self.kernel.slave(_mpi, _mol, _calc, _pyscf, exp, _time)
						_time.coll_phase_time(_mpi, None, _time.order, 'kernel')
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'screen_slave'):
						exp.order = msg['exp_order']
						_time.order = msg['time_order']
						exp.thres = msg['thres']
						self.screening.slave(_mpi, _calc, exp, _time)
						_time.coll_phase_time(_mpi, None, _time.order, 'screen')
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_slave'):
						slave = False
				# finalize
				_mpi.final(None)
	
	
