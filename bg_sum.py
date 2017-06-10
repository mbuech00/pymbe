#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_sum.py: summation class for Bethe-Goldstone correlation calculations."""

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


class SumCls():
		""" summation class """
		def main(self, _mpi, _calc, _exp, _time, _rst):
				""" energy summation phase """
				# mpi parallel version
				if (_mpi.parallel):
					self.sum_par(_mpi, _calc, _exp, _time)
					_time.coll_summation_time(_mpi, _rst, _exp.order)
				else:
					# start work time
					_time.timer('work_summation', _exp.order)
					# compute energy increments at current order
					for j in range(len(_exp.tuples[-1])):
						# loop over previous orders
						for i in range(_exp.order-1, 0, -1):
							# test if tuple is a subset
							combs = _exp.tuples[-1][j,_exp.comb_index(_exp.order, i)]
							dt = np.dtype((np.void,_exp.tuples[i-1].dtype.itemsize * \
											_exp.tuples[i-1].shape[1]))
							idx = np.nonzero(np.in1d(_exp.tuples[i-1].view(dt).reshape(-1),
												combs.view(dt).reshape(-1)))[0]
							for l in idx: _exp.energy_inc[-1][j] -= _exp.energy_inc[i-1][l]
					# sum of energy increments
					e_tmp = np.sum(_exp.energy_inc[-1])
					# sum of total energy
					if (_exp.order >= 2): e_tmp += _exp.energy_tot[-1]
					# add to total energy list
					_exp.energy_tot.append(e_tmp)
					# collect work time
					_time.timer('work_summation', _exp.order, True)
					# check for convergence wrt total energy
					if ((_exp.order >= 2) and (abs(_exp.energy_tot[-1] - _exp.energy_tot[-2]) < _calc.energy_thres)):
						_exp.conv_energy.append(True)
				#
				return
		
		
		def sum_par(self, _mpi, _calc, _exp, _time):
				""" master / slave function """
				if (_mpi.master):
					# start idle time
					_time.timer('idle_summation', _exp.order)
					# wake up slaves
					msg = {'task': 'sum_par', 'order': _exp.order}
					# bcast
					_mpi.comm.bcast(msg, root=0)
					# re-init e_inc[-1] with 0.0
					_exp.energy_inc[-1].fill(0.0)
				# start work time
				_time.timer('work_summation', _exp.order)
				# compute energy increments at current order
				for j in range(len(_exp.tuples[-1])):
					# distribute jobs according to work distribution in energy kernel phases
					if (_exp.energy_inc[-1][j] != 0.0):
						# loop over previous orders
						for i in range(_exp.order-1, 0, -1):
							# test if tuple is a subset
							combs = _exp.tuples[-1][j,_exp.comb_index(_exp.order, i)]
							dt = np.dtype((np.void,_exp.tuples[i-1].dtype.itemsize * \
											_exp.tuples[i-1].shape[1]))
							idx = np.nonzero(np.in1d(_exp.tuples[i-1].view(dt).reshape(-1),
												combs.view(dt).reshape(-1)))[0]
							for l in idx: _exp.energy_inc[-1][j] -= _exp.energy_inc[i-1][l]
				# allreduce e_inc[-1]
				_mpi.allred_e_inc(_exp, _time)
				# let master calculate the total energy
				if (_mpi.master):
					# sum of energy increments 
					e_tmp = np.sum(_exp.energy_inc[-1])
					# sum of total energy
					if (_exp.order >= 2): e_tmp += _exp.energy_tot[-1]
					# add to total energy list
					_exp.energy_tot.append(e_tmp)
					# check for convergence wrt total energy
					if ((_exp.order >= 2) and (abs(_exp.energy_tot[-1] - _exp.energy_tot[-2]) < _calc.energy_thres)): 
						_exp.conv_energy.append(True)
				#
				return


