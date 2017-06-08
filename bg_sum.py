#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_sum.py: summation class for Bethe-Goldstone correlation calculations."""

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


class SumCls():
		""" summation class """
		def main(molecule, tup, e_inc, energy, thres, order, level):
				""" energy summation phase """
				# mpi parallel version
				if (molecule['mpi_parallel']):
					energy_summation_par(molecule,tup,e_inc,energy,thres,order,level)
					collect_summation_mpi_time(molecule,order)
				else:
					# start work time
					timer_mpi(molecule,'mpi_time_work_summation',order)
					# compute energy increments at current order
					for j in range(0,len(tup[-1])):
						# loop over previous orders
						for i in range(order-1,0,-1):
							# test if tuple is a subset
							combs = tup[-1][j,comb_index(order,i)]
							dt = np.dtype((np.void,tup[i-1].dtype.itemsize*tup[i-1].shape[1]))
							idx = np.nonzero(np.in1d(tup[i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
							for l in idx: e_inc[-1][j] -= e_inc[i-1][l]
					# sum of energy increments
					e_tmp = np.sum(e_inc[-1])
					# sum of total energy
					if (order >= 2): e_tmp += energy[-1]
					# add to total energy list
					energy.append(e_tmp)
					# collect work time
					timer_mpi(molecule,'mpi_time_work_summation',order,True)
					# check for convergence wrt total energy
					if ((order >= 2) and (abs(energy[-1]-energy[-2]) < thres)): molecule['conv_energy'].append(True)
				#
				return
		
		
		def energy_summation_par(molecule, tup, e_inc, energy, thres, order, level):
				""" energy summation phase (mpi parallel version, master / slave function) """
				if (molecule['mpi_master']):
					# start idle time
					timer_mpi(molecule,'mpi_time_idle_summation',order)
					# wake up slaves
					msg = {'task': 'energy_summation_par', 'order': order, 'level': level}
					# bcast
					molecule['mpi_comm'].bcast(msg,root=0)
					# re-init e_inc[-1] with 0.0
					e_inc[-1].fill(0.0)
				# start work time
				timer_mpi(molecule,'mpi_time_work_summation',order)
				# compute energy increments at current order
				for j in range(0,len(tup[-1])):
					# distribute jobs according to work distribution in energy kernel phases
					if (e_inc[-1][j] != 0.0):
						# loop over previous orders
						for i in range(order-1,0,-1):
							# test if tuple is a subset
							combs = tup[-1][j,comb_index(order,i)]
							dt = np.dtype((np.void,tup[i-1].dtype.itemsize*tup[i-1].shape[1]))
							idx = np.nonzero(np.in1d(tup[i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
							for l in idx: e_inc[-1][j] -= e_inc[i-1][l]
				# allreduce e_inc[-1]
				allred_e_inc(molecule,e_inc,order)
				# let master calculate the total energy
				if (molecule['mpi_master']):
					# sum of energy increments 
					e_tmp = np.sum(e_inc[-1])
					# sum of total energy
					if (order >= 2): e_tmp += energy[-1]
					# add to total energy list
					energy.append(e_tmp)
					# check for convergence wrt total energy
					if ((order >= 2) and (abs(energy[-1]-energy[-2]) < thres)): molecule['conv_energy'].append(True)
				#
				return


