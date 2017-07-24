#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_time.py: MPI time-related routines for Bethe-Goldstone correlation calculations."""

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


class TimeCls():
		""" time class """
		def __init__(self, _mpi, _rst):
				""" init mpi timings """
				# init tmp time and time label
				self.store_time = 0.0; self.store_key = ''
				# init timings dict
				self.timings = {}
				# mpi distribution
				if (_rst.restart):
					# 'energy kernel' timings
					self.timings['work_kernel'] = []
					self.timings['comm_kernel'] = []
					self.timings['idle_kernel'] = []
					# 'screen' timings
					self.timings['work_screen'] = []
					self.timings['comm_screen'] = []
					self.timings['idle_screen'] = []
				else:
					# 'energy kernel' timings
					self.timings['work_kernel'] = [0.0]
					self.timings['comm_kernel'] = [0.0]
					self.timings['idle_kernel'] = [0.0]
					# 'screen' timings
					self.timings['work_screen'] = [0.0]
					self.timings['comm_screen'] = [0.0]
					self.timings['idle_screen'] = [0.0]
				# collective lists
				if (_mpi.parallel and _mpi.master):
					self.time_work = [[[] for i in range(_mpi.size)] for j in range(2)]
					self.time_comm = [[[] for i in range(_mpi.size)] for j in range(2)]
					self.time_idle = [[[] for i in range(_mpi.size)] for j in range(2)]
				#
				return


		def timer(self, _key, _order, _end=False):
				""" timer function """
				# new key (wrt previous)
				if (_key != self.store_key):
					if (self.store_key != ''):
						if (len(self.timings[self.store_key]) < _order):
							self.timings[self.store_key].append(MPI.Wtime() - self.store_time)
						else:
							self.timings[self.store_key][-1] += MPI.Wtime() - self.store_time
						self.store_time = MPI.Wtime()
						self.store_key = _key
					else:
						self.store_time = MPI.Wtime()
						self.store_key = _key
				# same key as previous (i.e., collect time)
				elif ((_key == self.store_key) and _end):
					if (len(self.timings[_key]) < _order):
						self.timings[_key].append(MPI.Wtime() - self.store_time)
					else:
						self.timings[_key][-1] += MPI.Wtime() - self.store_time
					self.store_key = ''
				#
				return


		def coll_time(self, _mpi, _phase):
				""" collect timings """
				# set phase identifier
				if (_phase == 'kernel'):
					idx = 0
				elif (_phase == 'screen'):
					idx = 1
				# master collects the timings
				if (_mpi.master):
					if (_phase == 'kernel'):
						self.time_work[0][0][-1] = self.timings['work_' + str(_phase)][-1]
						self.time_comm[0][0][-1] = self.timings['comm_' + str(_phase)][-1]
						self.time_idle[0][0][-1] = self.timings['idle_' + str(_phase)][-1]
					else:
						self.time_work[1][0].append(self.timings['work_' + str(_phase)][-1])
						self.time_comm[1][0].append(self.timings['comm_' + str(_phase)][-1])
						self.time_idle[1][0].append(self.timings['idle_' + str(_phase)][-1])
					# receive individual timings (in ordered sequence)
					for i in range(1,_mpi.size):
						time_info = _mpi.comm.recv(source=i, status=_mpi.stat)
						if (_phase == 'kernel'):
							self.time_work[0][i][-1] = time_info['work']
							self.time_comm[0][i][-1] = time_info['comm']
							self.time_idle[0][i][-1] = time_info['idle']
						else:
							self.time_work[1][i].append(time_info['work'])
							self.time_comm[1][i].append(time_info['comm'])
							self.time_idle[1][i].append(time_info['idle'])
				# slaves send their timings to master
				else:
					time_info = {'work': self.timings['work_' + str(_phase)][-1],
								'comm': self.timings['comm_' + str(_phase)][-1],
								'idle': self.timings['idle_' + str(_phase)][-1]}
					_mpi.comm.send(time_info, dest=0)
				#
				return


		def calc_time(self, _mpi, _calc, _exp):
				""" calculate mpi timings """
				# check for exp_max_order
				if (_exp.conv_energy[-1] or (_exp.order == _calc.exp_max_order)):
					self.timings['work_screen'].append(0.0)
					self.timings['comm_screen'].append(0.0)
					self.timings['idle_screen'].append(0.0)
					for i in range(_mpi.size):
						self.time_work[1][i].append(0.0)
						self.time_comm[1][i].append(0.0)
						self.time_idle[1][i].append(0.0)
				# use master timings to calculate overall phase timings
				if (not _mpi.parallel):
					self.time_kernel = np.asarray(self.timings['work_kernel'] + \
													[sum(self.timings['work_kernel'])])
					self.time_screen = np.asarray(self.timings['work_screen'] + \
													[sum(self.timings['work_screen'])])
					self.time_tot = self.time_kernel + self.time_screen
				else:
					self.time_kernel = np.asarray(self.timings['work_kernel'] + \
													[sum(self.timings['work_kernel'])]) + \
													np.asarray(self.timings['comm_kernel'] + \
													[sum(self.timings['comm_kernel'])]) + \
													np.asarray(self.timings['idle_kernel'] + \
													[sum(self.timings['idle_kernel'])])
					self.time_screen = np.asarray(self.timings['work_screen'] + \
													[sum(self.timings['work_screen'])]) + \
													np.asarray(self.timings['comm_screen'] + \
													[sum(self.timings['comm_screen'])]) + \
													np.asarray(self.timings['idle_screen'] + \
													[sum(self.timings['idle_screen'])])
					# calc total timings
					self.time_tot = self.time_kernel + self.time_screen
					# init summation arrays
					self.sum_work_abs = np.empty([2,_mpi.size], dtype=np.float64)
					self.sum_comm_abs = np.empty([2,_mpi.size], dtype=np.float64)
					self.sum_idle_abs = np.empty([2,_mpi.size], dtype=np.float64)
					# sum up work/comm/idle contributions
					# from all orders for the individual mpi procs
					for i in range(2):
						for j in range(_mpi.size):
							self.sum_work_abs[i][j] = np.sum(np.asarray(self.time_work[i][j]))
							self.sum_comm_abs[i][j] = np.sum(np.asarray(self.time_comm[i][j]))
							self.sum_idle_abs[i][j] = np.sum(np.asarray(self.time_idle[i][j]))
					# mpi distribution - slave (only count slave timings)
					self.dist_kernel = np.empty([3,_mpi.size], dtype=np.float64)
					self.dist_screen = np.empty([3,_mpi.size], dtype=np.float64)
					self.dist_total = np.empty([3,_mpi.size], dtype=np.float64)
					# for each of the phases, calculate the relative distribution between work/comm/idle for the individual slaves
					for j in range(_mpi.size):
						self.dist_kernel[0][j] = (self.sum_work_abs[0][j] / (self.sum_work_abs[0][j] + \
										self.sum_comm_abs[0][j] + self.sum_idle_abs[0][j])) * 100.0
						self.dist_kernel[1][j] = (self.sum_comm_abs[0][j] / (self.sum_work_abs[0][j] + \
										self.sum_comm_abs[0][j] + self.sum_idle_abs[0][j])) * 100.0
						self.dist_kernel[2][j] = (self.sum_idle_abs[0][j] / (self.sum_work_abs[0][j] + \
										self.sum_comm_abs[0][j] + self.sum_idle_abs[0][j])) * 100.0
						self.dist_screen[0][j] = (self.sum_work_abs[1][j] / (self.sum_work_abs[1][j] + \
										self.sum_comm_abs[1][j] + self.sum_idle_abs[1][j])) * 100.0
						self.dist_screen[1][j] = (self.sum_comm_abs[1][j] / (self.sum_work_abs[1][j] + \
										self.sum_comm_abs[1][j] + self.sum_idle_abs[1][j])) * 100.0
						self.dist_screen[2][j] = (self.sum_idle_abs[1][j] / (self.sum_work_abs[1][j] + \
										self.sum_comm_abs[1][j] + self.sum_idle_abs[1][j])) * 100.0
						self.dist_total[0][j] = ((self.sum_work_abs[0][j] + self.sum_work_abs[1][j]) / \
										((self.sum_work_abs[0][j] + self.sum_work_abs[1][j]) + \
										(self.sum_comm_abs[0][j] + self.sum_comm_abs[1][j]) + \
										(self.sum_idle_abs[0][j] + self.sum_idle_abs[1][j]))) * 100.0
						self.dist_total[1][j] = ((self.sum_comm_abs[0][j] + self.sum_comm_abs[1][j]) / \
										((self.sum_work_abs[0][j] + self.sum_work_abs[1][j]) + \
										(self.sum_comm_abs[0][j] + self.sum_comm_abs[1][j]) + \
										(self.sum_idle_abs[0][j] + self.sum_idle_abs[1][j]))) * 100.0
						self.dist_total[2][j] = ((self.sum_idle_abs[0][j] + self.sum_idle_abs[1][j]) / \
										((self.sum_work_abs[0][j] + self.sum_work_abs[1][j]) + \
										(self.sum_comm_abs[0][j] + self.sum_comm_abs[1][j]) + \
										(self.sum_idle_abs[0][j] + self.sum_idle_abs[1][j]))) * 100.0
					# mpi distribution - order
					# (only count slave timings - total results are stored as the last entry)
					self.dist_order = np.zeros([3,len(_exp.energy_tot) + 1], dtype=np.float64)
					# absolute amount of work/comm/idle at each order
					for k in range(len(_exp.energy_tot)):
						for i in range(2):
							for j in range(1,_mpi.size):
								self.dist_order[0,k] += self.time_work[i][j][k]
								self.dist_order[1,k] += self.time_comm[i][j][k]
								self.dist_order[2,k] += self.time_idle[i][j][k]
					self.dist_order[0,-1] = np.sum(self.dist_order[0,:-1])
					self.dist_order[1,-1] = np.sum(self.dist_order[1,:-1]) 
					self.dist_order[2,-1] = np.sum(self.dist_order[2,:-1])
					# calculate relative results
					for k in range(len(_exp.energy_tot) + 1):
						sum_k = self.dist_order[0,k] + self.dist_order[1,k] + self.dist_order[2,k]
						self.dist_order[0,k] = (self.dist_order[0,k] / sum_k) * 100.0
						self.dist_order[1,k] = (self.dist_order[1,k] / sum_k) * 100.0
						self.dist_order[2,k] = (self.dist_order[2,k] / sum_k) * 100.0
				#
				return


		def coll_phase_time(self, _mpi, _rst, _order, _phase):
				""" collect timings for phase """
				# start idle time
				self.timer('idle_'+_phase, _order)
				# collect idle time
				_mpi.comm.Barrier()
				self.timer('idle_'+_phase, _order, True)
				self.coll_time(_mpi, _phase)
				# write restart files
				if (_mpi.master): _rst.write_time(_mpi, self, _phase)
				#
				return


