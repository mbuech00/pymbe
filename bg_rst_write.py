#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst_write.py: restart write utilities for Bethe-Goldstone correlation calculations."""

import numpy as np
from os.path import join

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def rst_write_tup(molecule,order):
   #
   # write tup[k-1]
   #
   np.save(join(molecule['rst_dir'],'tup_'+str(order)),molecule['prim_tuple'][order-1])
   #
   return

def rst_write_dom(molecule,order):
   #
   # write dom[k-1]
   #
   np.save(join(molecule['rst_dir'],'dom_'+str(order)),np.asarray(molecule['prim_domain'][order-1]))
   #
   return

def rst_write_orb_con(molecule,order):
   #
   # write orb_con_abs[k-1]
   #
   np.save(join(molecule['rst_dir'],'orb_con_abs_'+str(order)),np.asarray(molecule['prim_orb_con_abs'][order-1]))
   #
   # write orb_con_rel[k-1]
   #
   np.save(join(molecule['rst_dir'],'orb_con_rel_'+str(order)),np.asarray(molecule['orb_con_rel'][order-1]))
   #
   return

def rst_write_orb_arr(molecule,order):
   #
   # write orb_arr[k-1]
   #
   np.save(join(molecule['rst_dir'],'orb_arr_'+str(order)),molecule['prim_orb_arr'][order-1])
   #
   return

def rst_write_e_inc(molecule,order):
   #
   # write e_inc[k-1]
   #
   np.save(join(molecule['rst_dir'],'e_inc_'+str(order)),molecule['prim_energy_inc'][order-1])
   #
   return

def rst_write_e_tot(molecule,order):
   #
   # write e_tot[k-1]
   #
   np.save(join(molecule['rst_dir'],'e_tot_'+str(order)),np.asarray(molecule['prim_energy'][order-1]))
   #
   return

def rst_write_time(molecule,phase):
   #
   # write mpi timings for phase
   #
   if (phase == 'init'):
      #
      idx = 0
   #
   elif (phase == 'kernel'):
      #
      idx = 1
   #
   elif (phase == 'final'):
      #
      idx = 2
   #
   np.save(join(molecule['rst_dir'],'mpi_time_work_'+str(phase)),np.asarray(molecule['mpi_time_work'][idx]))
   np.save(join(molecule['rst_dir'],'mpi_time_comm_'+str(phase)),np.asarray(molecule['mpi_time_comm'][idx]))
   np.save(join(molecule['rst_dir'],'mpi_time_idle_'+str(phase)),np.asarray(molecule['mpi_time_idle'][idx]))
   #
   return

