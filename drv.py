#!/usr/bin/env python
# -*- coding: utf-8 -*

""" drv.py: driver class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import numpy as np
from mpi4py import MPI

import mbe
from screen import ScrCls
from exp import ExpCls


class DrvCls():
		""" driver class """
		def __init__(self, _mol, _type):
				""" init parameters and classes """
				# init required classes
				self.mbe = mbe.MBECls()
				self.screening = ScrCls(_mol, _type)
				#
				return


		def main(self, _mpi, _mol, _calc, _kernel, _exp, _prt, _rst):
				""" main driver routine """
				# print and time logical
				do_print = _mpi.global_master and (not ((_calc.exp_type == 'combined') and (_exp.level == 'micro')))
				# restart
				_rst.rst_main(_mpi, _calc, _exp)
				# exp class instantiation on slaves
				if (_mpi.parallel):
					if (_calc.exp_type in ['occupied','virtual']):
						msg = {'task': 'exp_cls', 'type': _calc.exp_type, 'rst': _rst.restart}
						# bcast msg
						_mpi.local_comm.bcast(msg, root=0)
					else:
						if ((_exp.level == 'macro') and (_mpi.num_local_masters >= 1)):
							msg = {'task': 'exp_cls', 'rst': _rst.restart, 'min_order': _exp.min_order}
							# bcast msg
							_mpi.master_comm.bcast(msg, root=0)
						else:
							msg = {'task': 'exp_cls', 'type': 'virtual', 'incl_idx': _exp.incl_idx, 'min_order': _exp.min_order}
							# bcast msg
							_mpi.local_comm.bcast(msg, root=0)
							# compute and communicate distinct natural virtual orbitals
							if (_calc.exp_virt == 'DNO'):
								_kernel.trans_dno(_mol, _calc, _exp) 
								_mpi.bcast_mo_info(_mol, _calc, _mpi.local_comm)
				# print expansion header
				if (do_print): _prt.exp_header(_calc, _exp)
				# restart
				if (_rst.restart):
					# bcast rst data
					if (_mpi.parallel): _mpi.bcast_rst(_calc, _exp)
					# if rst, print previous results
					if (do_print):
						for _exp.order in range(_exp.start_order, _exp.min_order):
							_prt.mbe_header(_calc, _exp)
							_prt.mbe_micro_results(_calc, _exp)
							_prt.mbe_end(_calc, _exp)
							_prt.mbe_results(_mol, _calc, _exp, _kernel)
							_exp.thres = self.screening.update(_calc, _exp)
							_prt.screen_header(_calc, _exp)
							_prt.screen_end(_calc, _exp)
							_rst.rst_freq = _rst.update()
					# reset restart logical and init _exp.order
					_rst.restart = False
				# now do expansion
				for _exp.order in range(_exp.min_order, _exp.max_order+1):
					#
					#** energy phase **#
					#
					if (do_print):
						# print mbe header
						_prt.mbe_header(_calc, _exp)
					# init energies
					if (len(_exp.energy['inc']) != _exp.order):
						inc = np.empty(len(_exp.tuples[-1]), dtype=np.float64)
						inc.fill(np.nan)
						_exp.energy['inc'].append(inc)
					# mbe calculations
					self.mbe.main(_mpi, _mol, _calc, _kernel, _exp, _prt, _rst)
					if (do_print):
						# print micro results
						_prt.mbe_micro_results(_calc, _exp)
						# print mbe end
						_prt.mbe_end(_calc, _exp)
						# write restart files
						_rst.write_mbe(_calc, _exp, True)
						# print mbe results
						_prt.mbe_results(_mol, _calc, _exp, _kernel)
					#
					#** screening phase **#
					#
					if (do_print):
						# print screen header
						_prt.screen_header(_calc, _exp)
					# orbital screening
					if (_exp.order < _exp.max_order):
						# start time
						if (do_print): _exp.time_screen.append(MPI.Wtime())
						# perform screening
						self.screening.main(_mpi, _mol, _calc, _exp, _rst)
						if (do_print):
							# collect time
							_exp.time_screen[-1] -= MPI.Wtime()
							_exp.time_screen[-1] *= -1.0
							# write restart files
							if (not _exp.conv_orb[-1]):
								_rst.write_screen(_exp)
							# print screen end
							_prt.screen_end(_calc, _exp)
					else:
						if (do_print):
							# print screen end
							_prt.screen_end(_calc, _exp)
							# collect time
							_exp.time_screen.append(0.0)
					# update restart frequency
					if (do_print): _rst.rst_freq = _rst.update()
					#
					#** convergence check **#
					#
					if (_exp.conv_orb[-1] or (_exp.order == _exp.max_order)):
						# recast as numpy array
						_exp.energy['tot'] = np.array(_exp.energy['tot'])
						# now break
						break
				#
				return


		def local_master(self, _mpi, _mol, _calc, _kernel, _rst):
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
						exp = ExpCls(_mol, _calc, 'occupied')
						# mark expansion as macro
						exp.level = 'macro'
						# set min order
						exp.min_order = msg['min_order']
						# receive rst data
						_rst.restart = msg['rst']
						if (_rst.restart): _mpi.bcast_rst(_calc, exp)
						# reset restart logical
						_rst.restart = False
					#
					#** energy phase **#
					#
					if (msg['task'] == 'mbe_local_master'):
						exp.order = msg['exp_order']
						self.mbe.slave(_mpi, _mol, _calc, _kernel, exp, _rst)
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'screen_local_master'):
						exp.order = msg['exp_order']
						exp.thres = msg['thres']
						self.screening.slave(_mpi, _mol, _calc, exp)
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_local_master'):
						local_master = False
				# finalize
				_mpi.final(None)
	
	
		def slave(self, _mpi, _mol, _calc, _kernel):
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
						exp = ExpCls(_mol, _calc, msg['type'])
						# mark expansion as micro
						exp.level = 'micro'
						# distinguish between occ-virt expansions and combined expansions
						if (_calc.exp_type == 'combined'):
							exp.incl_idx = msg['incl_idx']
							# receive distinct natural virtual orbitals
							if (_calc.exp_virt == 'DNO'):
								_mpi.bcast_mo_info(_mol, _calc, _mpi.local_comm)
						else:
							# receive rst data
							if (msg['rst']): _mpi.bcast_rst(_calc, exp)
					#
					#** energy phase **#
					#
					if (msg['task'] == 'mbe_slave'):
						exp.order = msg['exp_order']
						self.mbe.slave(_mpi, _mol, _calc, _kernel, exp)
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'screen_slave'):
						exp.order = msg['exp_order']
						exp.thres = msg['thres']
						self.screening.slave(_mpi, _mol, _calc, exp)
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_slave'):
						slave = False
				# finalize
				_mpi.final(None)
	
	
