#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_time.py: MPI time-related routines for Bethe-Goldstone correlation calculations."""

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


class TimeCls():
		""" time class """
		def __init__(self, _mpi, _rst):
				""" init mpi timings """
				# init tmp time and time label
				self.store_time = 0.0; self.store_key = ''
				# mpi distribution
				if (_rst.restart):
					# 'energy kernel' timings
					self.time_work_kernel = []
					self.time_comm_kernel = []
					self.time_idle_kernel = []
					# 'energy summation' timings
					self.time_work_summation = []
					self.time_comm_summation = []
					self.time_idle_summation = []
					# 'screen' timings
					self.time_work_screen = []
					self.time_comm_screen = []
					self.time_idle_screen = []
				else:
					# 'energy kernel' timings
					self.time_work_kernel = [0.0]
					self.time_comm_kernel = [0.0]
					self.time_idle_kernel = [0.0]
					# 'energy summation' timings
					self.time_work_summation = [0.0]
					self.time_comm_summation = [0.0]
					self.time_idle_summation = [0.0]
					# 'screen' timings
					self.time_work_screen = [0.0]
					self.time_comm_screen = [0.0]
					self.time_idle_screen = [0.0]
				# collective lists
				if (_mpi.parallel and _mpi.master):
					self.time_work = [[[] for i in range(_mpi.size)] for j in range(3)]
					self.time_comm = [[[] for i in range(_mpi.size)] for j in range(3)]
					self.time_idle = [[[] for i in range(_mpi.size)] for j in range(3)]


		def timer(self, _key, _order, _end=False):
				""" timer function """
				# new key (wrt previous)
				if (_key != self.store_key):
					if (self.store_key != ''):
						if (len(self.self.store_key) < _order):
							self.self.store_key.append(MPI.Wtime() - self.store_time)
						else:
							self.self.store_key[order-1] += MPI.Wtime() - self.store_time
						self.store_time = MPI.Wtime()
						self.store_key = _key
					else:
						self.store_time = MPI.Wtime()
						self.store_key = _key
				# same key as previous (i.e., collect time)
				elif ((_key == self.store_key) and _end):
					if (len(self.key) < _order):
						self.key.append(MPI.Wtime() - self.store_time)
					else:
						self.key[order - 1] += MPI.Wtime() - self.store_time
					self.store_key = ''
				#
				return


		def coll_time(self, _mpi, _phase):
				""" collect timings """
				# set phase identifier
				if (_phase == 'kernel'):
					idx = 0
				elif (_phase == 'summation'):
					idx = 1
				elif (_phase == 'screen'):
					idx = 2
				# master collects the timings
				if (_mpi.master):
					if (idx == 0):
						self.time_work[idx][0][-1] = self.time_work_ + str(phase)[-1]
						self.time_comm[idx][0][-1] = self.time_comm_ + str(phase)[-1]
						self.time_idle[idx][0][-1] = self.time_idle_ + str(phase)[-1]
					else:
						self.time_work[idx][0].append(self.time_work_ + str(phase)[-1])
						self.time_comm[idx][0].append(self.time_comm_ + str(phase)[-1])
						self.time_idle[idx][0].append(self.time_idle_ + str(phase)[-1])
					# receive individual timings (in ordered sequence)
					for i in range(1,_mpi.size):
						time_info = _mpi.comm.recv(source=i, status=_mpi.stat)
						if (idx == 0):
							self.time_work[idx][i][-1] = time_info['work']
							self.time_comm[idx][i][-1] = time_info['comm']
							self.time_idle[idx][i][-1] = time_info['idle']
						else:
							self.time_work[idx][i].append(time_info['work'])
							self.time_comm[idx][i].append(time_info['comm'])
							self.time_idle[idx][i].append(time_info['idle'])
				# slaves send their timings to master
				else:
					time_info = {'work': self.time_work_ + str(phase)[-1],
								'comm': self.time_comm_ + str(phase)[-1],
								'idle': self.time_idle_ + str(phase)[-1]}
					_mpi.comm.send(time_info, dest=0)
				#
				return


		def calc_time(self, _mpi, _exp):
				""" calculate mpi timings """
				# use master timings to calculate overall phase timings
				if (not _mpi.parallel):
					self.time_kernel = np.asarray(self.time_work_kernel + \
													[sum(self.time_work_kernel)])
					self.time_summation = np.asarray(self.time_work_summation + \
														[sum(self.time_work_summation)])
					self.time_screen = np.asarray(self.time_work_screen + \
													[sum(self.time_work_screen)])
					self.time_tot = self.time_kernel + self.time_summation + \
									self.time_screen
				else:
					self.time_kernel = np.asarray(self.time_work_kernel + \
													[sum(self.time_work_kernel)]) + \
													np.asarray(self.time_comm_kernel + \
													[sum(self.time_comm_kernel)]) + \
													np.asarray(self.time_idle_kernel + \
													[sum(self.time_idle_kernel)])
					self.time_summation = np.asarray(self.time_work_summation + \
													[sum(self.time_work_summation)]) + \
													np.asarray(self.time_comm_summation + \
													[sum(self.time_comm_summation)]) + \
													np.asarray(self.time_idle_summation + \
													[sum(self.time_idle_summation)])
					self.time_screen = np.asarray(self.time_work_screen + \
													[sum(self.time_work_screen)]) + \
													np.asarray(self.time_comm_screen + \
													[sum(self.time_comm_screen)]) + \
													np.asarray(self.time_idle_screen + \
													[sum(self.time_idle_screen)])
					# calc total timings
					self.time_tot = self.time_kernel + self.time_summation + \
									self.time_screen
					# init summation arrays
					self.sum_work_abs = np.empty([3,_mpi.size], dtype=np.float64)
					self.sum_comm_abs = np.empty([3,_mpi.size], dtype=np.float64)
					self.sum_idle_abs = np.empty([3,_mpi.size], dtype=np.float64)
					# sum up work/comm/idle contributions
					# from all orders for the individual mpi procs
					for i in range(3):
						for j in range(_mpi.size):
							self.sum_work_abs[i][j] = np.sum(np.asarray(self.time_work[i][j]))
							self.sum_comm_abs[i][j] = np.sum(np.asarray(self.time_comm[i][j]))
							self.sum_idle_abs[i][j] = np.sum(np.asarray(self.time_idle[i][j]))
					# mpi distribution - slave (only count slave timings)
					self.dist_kernel = np.empty([3,_mpi.size], dtype=np.float64)
					self.dist_summation = np.empty([3,_mpi.size], dtype=np.float64)
					self.dist_screen = np.empty([3,_mpi.size], dtype=np.float64)
					for i in range(3):
						if (i == 0):
							dist = self.dist_kernel
						elif (i == 1):
							dist = self.dist_summation
						elif (i == 2):
							dist = self.dist_screen
						# for each of the phases, calculate the relative distribution between work/comm/idle for the individual slaves
						for j in range(_mpi.size):
							dist[0][j] = (self.sum_work_abs[i][j] / (self.sum_work_abs[i][j] + \
											self.sum_comm_abs[i][j] + self.sum_idle_abs[i][j])) * 100.0
							dist[1][j] = (self.sum_comm_abs[i][j] / (self.sum_work_abs[i][j] + \
											self.sum_comm_abs[i][j] + self.sum_idle_abs[i][j])) * 100.0
							dist[2][j] = (self.sum_idle_abs[i][j] / (self.sum_work_abs[i][j] + \
											self.sum_comm_abs[i][j] + self.sum_idle_abs[i][j])) * 100.0
					# mpi distribution - order
					# (only count slave timings - total results are stored as the last entry)
					self.dist_order = np.zeros([3,len(_exp.energy) + 1], dtype=np.float64)
					# absolute amount of work/comm/idle at each order
					for k in range(len(_exp.energy)):
						for i in range(3):
							for j in range(1,_mpi.size):
								self.dist_order[0,k] += self.time_work[i][j][k]
								self.dist_order[1,k] += self.time_comm[i][j][k]
								self.dist_order[2,k] += self.time_idle[i][j][k]
					self.dist_order[0,-1] = np.sum(self.dist_order[0,:-1])
					self.dist_order[1,-1] = np.sum(self.dist_order[1,:-1]) 
					self.dist_order[2,-1] = np.sum(self.dist_order[2,:-1])
					# calculate relative results
					for k in range(len(_exp.energy) + 1):
						sum_k = self.dist_order[0,k] + self.dist_order[1,k] + self.dist_order[2,k]
						self.dist_order[0,k] = (self.dist_order[0,k] / sum_k) * 100.0
						self.dist_order[1,k] = (self.dist_order[1,k] / sum_k) * 100.0
						self.dist_order[2,k] = (self.dist_order[2,k] / sum_k) * 100.0
				#
				return


		def coll_kernel_time(self, _mpi, _rst, _order):
				""" collect timings for energy kernel phase """
				# start idle time
				self.timer('time_idle_kernel', _order)
				# collect idle time
				_mpi.comm.Barrier()
				self.timer('time_idle_kernel', _order, True)
				self.coll_time(_mpi, 'kernel')
				# write restart files
				if (_mpi.master): _rst.write_time('kernel')
				#
				return


		def coll_summation_time(self, _mpi, _rst, _order):
				""" collect timings for energy summation phase """
				# start idle time
				self.timer('time_idle_summation', _order)
				# collect idle time
				_mpi.comm.Barrier()
				self.timer('time_idle_summation', _order, True)
				self.coll_time(_mpi, 'summation')
				# write restart files
				if (_mpi.master): _rst.write_time('summation')
				#
				return


		def coll_screen_time(self, _mpi, _rst, _order, _second_call=False):
				""" collect timings for screening phase """
				# start idle time
				self.timer('time_idle_screen', _order)
				# collect idle time
				_mpi.comm.Barrier()
				self.timer('time_idle_screen', _order, True)
				# first or second call? if second, write restart files
				if (_second_call): self.coll_time(_mpi, 'screen')
				if (_second_call and _mpi.master): _rst.write_time('screen')
				#
				return 


