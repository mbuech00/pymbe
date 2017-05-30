#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst_write.py: restart write utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from os.path import join


def rst_write_kernel(molecule, e_inc, order):
		""" write energy kernel restart files """
		# write e_inc
		rst_write_e_inc(molecule,e_inc,order)
		# write timings
		rst_write_time(molecule,'kernel')
		#
		return


def rst_write_summation(molecule, e_inc, e_tot, order):
		""" write energy summation restart files """
		# write e_inc and e_tot
		rst_write_e_inc(molecule,e_inc,order)
		rst_write_e_tot(molecule,e_tot,order)
		# write timings
		rst_write_time(molecule,'summation')
		#
		return


def rst_write_screen(molecule, tup, orb_con_abs, orb_con_rel, orb_ent_abs, orb_ent_rel, order):
		""" write screening restart files """
		# write tuples
		rst_write_tup(molecule,tup,order)
		# write orb_con_abs and orb_con_rel
		rst_write_orb_con(molecule,orb_con_abs,orb_con_rel,order)
		# write timings
		rst_write_time(molecule,'screen')
		# write orb_ent_abs and orb_ent_rel
		if (order >= 2): rst_write_orb_ent(molecule,orb_ent_abs,orb_ent_rel,order)
		#
		return


def rst_write_tup(molecule, tup, order):
		""" write tuples """
		# write tup[k-1]
		np.save(join(molecule['rst_dir'],'tup_'+str(order+1)),tup[order])
		#
		return


def rst_write_orb_ent(molecule, orb_ent_abs, orb_ent_rel, order):
		""" write orbital entanglement arrays """
		# write orb_ent_abs[k-1]
		np.save(join(molecule['rst_dir'],'orb_ent_abs_'+str(order)),orb_ent_abs[order-2])
		# write orb_ent_rel[k-1]
		np.save(join(molecule['rst_dir'],'orb_ent_rel_'+str(order)),orb_ent_rel[order-2])
		#
		return


def rst_write_orb_con(molecule, orb_con_abs, orb_con_rel, order):
		""" write orbital contribution arrays """
		# write orb_con_abs[k-1]
		np.save(join(molecule['rst_dir'],'orb_con_abs_'+str(order)),np.asarray(orb_con_abs[order-1]))
		# write orb_con_rel[k-1]
		np.save(join(molecule['rst_dir'],'orb_con_rel_'+str(order)),np.asarray(orb_con_rel[order-1]))
		#
		return


def rst_write_e_inc(molecule, e_inc, order):
		""" write energy increments """
		# write e_inc[k-1]
		np.save(join(molecule['rst_dir'],'e_inc_'+str(order)),e_inc[order-1])
		#
		return


def rst_write_e_tot(molecule, e_tot, order):
		""" write total energy """
		# write e_tot[k-1]
		np.save(join(molecule['rst_dir'],'e_tot_'+str(order)),np.asarray(e_tot[order-1]))
		#
		return


def rst_write_time(molecule, phase):
		""" write timings for phase """
		# set phase index
		if (phase == 'kernel'):
			idx = 0
		elif (phase == 'summation'):
			idx = 1
		elif (phase == 'screen'):
			idx = 2
		# write mpi timings
		if (molecule['mpi_parallel']):
			np.save(join(molecule['rst_dir'],'mpi_time_work_'+str(phase)),np.asarray(molecule['mpi_time_work'][idx]))
			np.save(join(molecule['rst_dir'],'mpi_time_comm_'+str(phase)),np.asarray(molecule['mpi_time_comm'][idx]))
			np.save(join(molecule['rst_dir'],'mpi_time_idle_'+str(phase)),np.asarray(molecule['mpi_time_idle'][idx]))
		else:
			np.save(join(molecule['rst_dir'],'mpi_time_work_'+str(phase)),np.asarray(molecule['mpi_time_work_'+str(phase)]))
		#
		return


