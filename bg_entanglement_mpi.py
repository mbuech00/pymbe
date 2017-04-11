#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_entanglement_mpi.py: MPI entanglement and orbital contribution routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import timer_mpi

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def red_orb_ent(molecule,tmp,recv_buff,order):
   #
   timer_mpi(molecule,'mpi_time_idle_screen',order)
   #
   molecule['mpi_comm'].Barrier()
   #
   # reduce tmp into recv_buff
   #
   timer_mpi(molecule,'mpi_time_comm_screen',order)
   #
   molecule['mpi_comm'].Reduce([tmp,MPI.DOUBLE],[recv_buff,MPI.DOUBLE],op=MPI.SUM,root=0)
   #
   timer_mpi(molecule,'mpi_time_comm_screen',order,True)
   #
   return recv_buff

def entanglement_abs_par(molecule,l_limit,u_limit,order):
   #
   #  ---  master/slave routine
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      timer_mpi(molecule,'mpi_time_idle_screen',order)
      #
      msg = {'task': 'entanglement_abs_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': order}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   tmp = np.zeros([u_limit,u_limit],dtype=np.float64)
   #
   for l in range(0,len(molecule['prim_tuple'][-1])):
      #
      # simple modulo distribution of tasks
      #
      if ((l % molecule['mpi_size']) == molecule['mpi_rank']):
         #
         for i in range(l_limit,l_limit+u_limit):
            #
            for j in range(i+1,l_limit+u_limit):
               #
               # add up contributions from the correlation between orbs i and j at current order
               #
               if (set([i+1,j+1]) <= set(molecule['prim_tuple'][-1][l])):
                  #
                  tmp[i-l_limit,j-l_limit] += molecule['prim_energy_inc'][-1][l]
   #
   # init recv_buff
   #
   if (molecule['mpi_master']):
      #
      recv_buff = np.zeros([u_limit,u_limit],dtype=np.float64)
   #
   else:
      #
      recv_buff = None
   #
   # reduce tmp onto master
   #
   red_orb_ent(molecule,tmp,recv_buff,order)
   #
   if (molecule['mpi_master']):
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      if (level == 'MACRO'):
         #
         molecule['prim_orb_ent'].append(recv_buff)
      #
      timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule

