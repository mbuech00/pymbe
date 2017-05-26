#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_entanglement_mpi.py: MPI entanglement and orbital contribution routines for Bethe-Goldstone correlation calculations."""

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

from bg_mpi_time import timer_mpi


def red_orb_ent(molecule, tmp, recv_buff, order):
		""" reduce orb_ent onto master proc. """
		# start idle time
		timer_mpi(molecule,'mpi_time_idle_screen',order)
		# collect idle time
		molecule['mpi_comm'].Barrier()
		# start comm time
		timer_mpi(molecule,'mpi_time_comm_screen',order)
		# reduce tmp into recv_buff
		molecule['mpi_comm'].Reduce([tmp,MPI.DOUBLE],[recv_buff,MPI.DOUBLE],op=MPI.SUM,root=0)
		# collect comm time
		timer_mpi(molecule,'mpi_time_comm_screen',order,True)
		#
		return


def entanglement_abs_par(molecule, l_limit, u_limit, order, calc_end):
		""" master / slave routine for calculation absolute orbital entanglement """
		if (molecule['mpi_master']):
			# start idle time
			timer_mpi(molecule,'mpi_time_idle_screen',order)
			# wake up slaves
			msg = {'task': 'entanglement_abs_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'calc_end': calc_end}
			# bcast
			molecule['mpi_comm'].bcast(msg,root=0)
			# start work time
			timer_mpi(molecule,'mpi_time_work_screen',order)
		else:
			# start work time
			timer_mpi(molecule,'mpi_time_work_screen',order)
		# init tmp array
		tmp = np.zeros([u_limit,u_limit],dtype=np.float64)
		# loop over tuple
		for l in range(0,len(molecule['prim_tuple'][-1])):
			# simple modulo distribution of tasks
			if ((l % molecule['mpi_size']) == molecule['mpi_rank']):
				for i in range(l_limit,l_limit+u_limit):
					for j in range(i+1,l_limit+u_limit):
						# add up contributions from the correlation between orbs i and j at current order
						if (set([i+1,j+1]) <= set(molecule['prim_tuple'][-1][l])):
							tmp[i-l_limit,j-l_limit] += molecule['prim_energy_inc'][-1][l]
		# init recv_buff
		if (molecule['mpi_master']):
			recv_buff = np.zeros([u_limit,u_limit],dtype=np.float64)
		else:
			recv_buff = None
		# reduce tmp onto master
		red_orb_ent(molecule,tmp,recv_buff,order)
		# master appends results to orb_ent list
		if (molecule['mpi_master']):
			# start work time
			timer_mpi(molecule,'mpi_time_work_screen',order)
			# append results 
			molecule['prim_orb_ent_abs'].append(recv_buff)
			# collect work time
			timer_mpi(molecule,'mpi_time_work_screen',order,True)
		#
		return


