#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_ent.py: entanglement class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '1.0'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI


class EntCls():
		""" entanglement class """
		def main(self, _mpi, _mol, _calc, _exp, _time, _rst):
				""" main driver for entanglement routines """
				if (_exp.order >= 2):
					# absolute entanglement
					self.ent_abs(_mpi, _exp, _time)
					# relative entanglement
					self.ent_rel(_exp, _time)
				# relative orbital contributions
				self.orb_cont(_mol, _calc, _exp, _time)
				# collect timings
				if (_mpi.parallel and (_exp.order >= 2)):
					_time.coll_screen_time(_mpi, _rst, _exp.order, _exp.conv_energy[-1])
				#
				return
		
		
		def ent_abs(self, _mpi, _exp, _time):
				""" absolute orbital entanglement """
				if (_mpi.parallel):
					# mpi parallel version
					self.ent_abs_par(_mpi, _exp, _time)
				else:
					# start work time
					_time.timer('work_screen', _exp.order)
					# write orbital entanglement matrix (abs)
					_exp.orb_ent_abs.append(np.zeros([_exp.u_limit,_exp.u_limit],
											dtype=np.float64))
					for l in range(len(_exp.tuples[-1])):
						for i in range(_exp.l_limit,_exp.l_limit+_exp.u_limit):
							for j in range(_exp.l_limit, i):
								# add up absolute contributions from the correlation between orbs i and j at current order
								if (set([i,j]) <= set(_exp.tuples[-1][l])):
									_exp.orb_ent_abs[-1][i-_exp.l_limit,j-_exp.l_limit] += _exp.energy_inc[-1][l]
					# collect work time
					_time.timer('work_screen', _exp.order, True)
				#
				return
		
     
		def ent_abs_par(self, _mpi, _exp, _time):
				""" master / slave routine """
				if (_mpi.master):
					# start idle time
					_time.timer('idle_screen', _exp.order)
					# wake up slaves
					msg = {'task': 'ent_abs_par', 'order': _exp.order, 'conv_energy': _exp.conv_energy[-1]}
					# bcast
					_mpi.comm.bcast(msg, root=0)
					# start work time
					_time.timer('work_screen', _exp.order)
				else:
					# start work time
					_time.timer('work_screen', _exp.order)
				# init tmp array
				tmp = np.zeros([_exp.u_limit,_exp.u_limit], dtype=np.float64)
				# loop over tuple
				for l in range(len(_exp.tuples[-1])):
					# simple modulo distribution of tasks
					if ((l % _mpi.size) == _mpi.rank):
						for i in range(_exp.l_limit,_exp.l_limit+_exp.u_limit):
							for j in range(_exp.l_limit, i):
								# add up contributions from the correlation between orbs i and j at current order
								if (set([i,j]) <= set(_exp.tuples[-1][l])):
									tmp[i-_exp.l_limit,j-_exp.l_limit] += _exp.energy_inc[-1][l]
				# init recv_buff
				if (_mpi.master):
					recv_buff = np.zeros([_exp.u_limit,_exp.u_limit], dtype=np.float64)
				else:
					recv_buff = None
				# reduce tmp onto master
				_mpi.red_orb_ent(_exp, _time, tmp, recv_buff)
				# master appends results to orb_ent list
				if (_mpi.master):
					# start work time
					_time.timer('work_screen', _exp.order)
					# append results 
					_exp.orb_ent_abs.append(recv_buff)
					# collect work time
					_time.timer('work_screen', _exp.order, True)
				#
				return

		 
		def ent_rel(self, _exp, _time):
				""" relative orbital entanglement """
				# start work time
				_time.timer('work_screen', _exp.order)
				# write orbital entanglement matrix (rel)
				_exp.orb_ent_rel.append(np.zeros([_exp.u_limit,_exp.u_limit],
										dtype=np.float64))
				_exp.orb_ent_rel[-1] = (np.abs(_exp.orb_ent_abs[-1]) / \
										np.amax(np.abs(_exp.orb_ent_abs[-1]))) * 100.0
				# collect work time
				_time.timer('work_screen', _exp.order, True)
				#
				return
		

		def orb_cont(self, _mol, _calc, _exp, _time):
				""" relative orbital contributions """
				# start time
				_time.timer('work_screen', _exp.order)
				# init lists
				_exp.orb_con_abs.append([]); _exp.orb_con_rel.append([])
				# order k == 1
				if (_exp.order == 1):
					# total absolute orbital contributions
					if ((_calc.exp_type == 'occupied') and _mol.frozen):
						for _ in range(_mol.ncore):
							_exp.orb_con_abs[-1].append(0.0)
					for i in range(len(_exp.energy_inc[-1])):
						_exp.orb_con_abs[-1].append(_exp.energy_inc[-1][i])
					# total relative orbital contributions
					for i in range(len(_exp.orb_con_abs[-1])):
						_exp.orb_con_rel[-1].append(abs(_exp.orb_con_abs[-1][i]) / \
													np.abs(np.sum(_exp.energy_inc[-1])))
				else:
					# total absolute orbital contributions
					for i in range(len(_exp.orb_ent_abs[-1])):
						_exp.orb_con_abs[-1].append(_exp.orb_con_abs[-2][i] + \
													np.sum(_exp.orb_ent_abs[-1][i]))
					# total relative orbital contributions
					for i in range(len(_exp.orb_con_abs[-1])):
						if (_exp.orb_con_abs[-1][i] == 0.0):
							_exp.orb_con_rel[-1].append(0.0)
						else:
							_exp.orb_con_rel[-1].append(_exp.orb_con_abs[-1][i] / \
														sum(_exp.orb_con_abs[-1]))
				# collect time
				_time.timer('work_screen', _exp.order, True)
				#
				return


