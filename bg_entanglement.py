#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_entanglement.py: entanglement and orbital contribution routines for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np

from bg_mpi_time import timer_mpi, collect_screen_mpi_time
from bg_entanglement_mpi import entanglement_abs_par


def entanglement_main(molecule, l_limit, u_limit, order, calc_end=False):
		""" main driver for entanglement routines """
		if (order >= 2):
			# absolute entanglement
			entanglement_abs(molecule,l_limit,u_limit,order,calc_end)
			# relative entanglement
			entanglement_rel(molecule,u_limit,order)
		# relative orbital contributions
		orb_contributions(molecule,order,(order == 1))
		# collect timings
		if (molecule['mpi_parallel'] and (order >= 2)): collect_screen_mpi_time(molecule,order,calc_end)
		#
		return molecule


def orb_contributions(molecule, order, singles=False):
		""" relative orbital contributions """
		# start time
		timer_mpi(molecule,'mpi_time_work_screen',order)
		# init lists
		molecule['prim_orb_con_abs'].append([]); molecule['prim_orb_con_rel'].append([])
		# order k == 1
		if (singles):
			# total absolute orbital contributions
			if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'])):
				for _ in range(0,molecule['ncore']):
					molecule['prim_orb_con_abs'][-1].append(0.0)
			for i in range(0,len(molecule['prim_energy_inc'][-1])):
				molecule['prim_orb_con_abs'][-1].append(molecule['prim_energy_inc'][-1][i])
			# total relative orbital contributions
			for i in range(0,len(molecule['prim_orb_con_abs'][-1])):
				molecule['prim_orb_con_rel'][-1].append(abs(molecule['prim_orb_con_abs'][-1][i]) / 
									np.abs(np.sum(molecule['prim_energy_inc'][-1])))
		else:
			# total absolute orbital contributions
			for i in range(0,len(molecule['prim_orb_ent_abs'][-1])):
				molecule['prim_orb_con_abs'][-1].append(molecule['prim_orb_con_abs'][-2][i]+np.sum(molecule['prim_orb_ent_abs'][-1][i]))
			# total relative orbital contributions
			for i in range(0,len(molecule['prim_orb_con_abs'][-1])):
				if (molecule['prim_orb_con_abs'][-1][i] == 0.0):
					molecule['prim_orb_con_rel'][-1].append(0.0)
				else:
					molecule['prim_orb_con_rel'][-1].append(molecule['prim_orb_con_abs'][-1][i]/sum(molecule['prim_orb_con_abs'][-1]))
		# collect time
		timer_mpi(molecule,'mpi_time_work_screen',order,True)
		#
		return


def entanglement_abs(molecule, l_limit, u_limit, order, calc_end):
		""" absolute orbital entanglement """
		if (molecule['mpi_parallel']):
			# mpi parallel version
			entanglement_abs_par(molecule,l_limit,u_limit,order,calc_end)
		else:
			# start work time
			timer_mpi(molecule,'mpi_time_work_screen',order)
			# write orbital entanglement matrix (abs)
			molecule['prim_orb_ent_abs'].append(np.zeros([u_limit,u_limit],dtype=np.float64))
			for l in range(0,len(molecule['prim_tuple'][-1])):
				for i in range(l_limit,l_limit+u_limit):
					for j in range(l_limit,i):
						# add up absolute contributions from the correlation between orbs i and j at current order
						if (set([i+1,j+1]) <= set(molecule['prim_tuple'][-1][l])):
							molecule['prim_orb_ent_abs'][-1][i-l_limit,j-l_limit] += molecule['prim_energy_inc'][-1][l]
			# collect work time
			timer_mpi(molecule,'mpi_time_work_screen',order,True)
		#
		return
     
 
def entanglement_rel(molecule, u_limit, order):
		""" relative orbital entanglement """
		# start work time
		timer_mpi(molecule,'mpi_time_work_screen',order)
		# write orbital entanglement matrix (rel)
		molecule['prim_orb_ent_rel'].append(np.zeros([u_limit,u_limit],dtype=np.float64))
		molecule['prim_orb_ent_rel'][-1] = (np.abs(molecule['prim_orb_ent_abs'][-1])/np.amax(np.abs(molecule['prim_orb_ent_abs'][-1])))*100.0
		# collect work time
		timer_mpi(molecule,'mpi_time_work_screen',order,True)
		#
		return molecule

