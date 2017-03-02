#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_orbitals.py: MPI orbital-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import timer_mpi

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def bcast_dom_master(molecule,dom,l_limit,u_limit,k,level):
   #
   #  ---  master routine
   #
   # wake up slaves
   #
   if (k >= 2): timer_mpi(molecule,'mpi_time_comm_init',k-1)
   #
   msg = {'task': 'orb_generator_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': k, 'level': level}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # bcast orbital domains and lower/upper limits
   #
   dom_info = {'dom': dom}
   #
   molecule['mpi_comm'].bcast(dom_info,root=0)
   #
   if (k >= 2): timer_mpi(molecule,'mpi_time_comm_init',k-1,True)
   #
   dom_info.clear()
   #
   return

def bcast_dom_slave(molecule,k):
   #
   #  ---  slave routine
   #
   if (k >= 2): timer_mpi(molecule,'mpi_time_comm_init',k-1)
   #
   # receive domains
   #
   dom_info = MPI.COMM_WORLD.bcast(None,root=0)
   #
   molecule['dom'] = dom_info['dom']
   #
   if (k >= 2): timer_mpi(molecule,'mpi_time_comm_init',k-1,True)
   #
   dom_info.clear()
   #
   return molecule

def orb_entanglement_main_par(molecule,l_limit,u_limit,order,level):
   #
   #  ---  master/slave routine
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      timer_mpi(molecule,'mpi_time_comm_init',order)
      #
      if (level == 'MACRO'):
         #
         orb = molecule['prim_orb_ent']
         #
         end = len(molecule['prim_tuple'][order-1])
      #
      elif (level == 'CORRE'):
         #
         orb = molecule['corr_orb_ent']
         #
         end = len(molecule['corr_tuple'][order-1])+len(molecule['prim_tuple'][order-1])
      #
      orb.append(np.zeros([u_limit,u_limit],dtype=np.float64))
      #
      msg = {'task': 'orb_entanglement_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'level': level}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
   #
   timer_mpi(molecule,'mpi_time_work_init',order)
   #
   if (level == 'MACRO'):
      #
      end = len(molecule['prim_tuple'][order-1])
   #
   elif (level == 'CORRE'):
      #
      end = len(molecule['corr_tuple'][order-1])+len(molecule['prim_tuple'][order-1])
   #
   tmp = np.zeros([u_limit,u_limit],dtype=np.float64)
   #
   for l in range(0,end):
      #
      # simple modulo distribution of tasks
      #
      if ((l % molecule['mpi_size']) == molecule['mpi_rank']):
         #
         if ((level == 'CORRE') and (l >= len(molecule['prim_tuple'][order-1]))):
            #
            tup = molecule['corr_tuple'][order-1]
            e_inc = molecule['corr_energy_inc'][order-1]
            ldx = l-len(molecule['prim_tuple'][order-1])
         #
         else:
            #
            tup = molecule['prim_tuple'][order-1]
            e_inc = molecule['prim_energy_inc'][order-1]
            ldx = l
         #
         for i in range(l_limit,l_limit+u_limit):
            #
            for j in range(i+1,l_limit+u_limit):
               #
               # add up contributions from the correlation between orbs i and j at current order
               #
               if (set([i+1,j+1]) <= set(tup[ldx])):
                  #
                  tmp[i-l_limit][j-l_limit] += e_inc[ldx]
                  tmp[j-l_limit][i-l_limit] = tmp[i-l_limit][j-l_limit]
   #
   timer_mpi(molecule,'mpi_time_idle_init',order)
   #
   molecule['mpi_comm'].Barrier()
   #
   # reduce orb[-1]
   #
   timer_mpi(molecule,'mpi_time_comm_init',order)
   #
   if (molecule['mpi_master']):
      #
      recv_buff = orb[-1] 
   #
   else:
      #
      recv_buff = None
   #
   molecule['mpi_comm'].Reduce([tmp,MPI.DOUBLE],[recv_buff,MPI.DOUBLE],op=MPI.SUM,root=0)
   #
   timer_mpi(molecule,'mpi_time_comm_init',order,True)
   #
   return molecule

def collect_init_mpi_time(molecule,k):
   #
   #  ---  master/slave routine
   #
   timer_mpi(molecule,'mpi_time_idle_init',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_idle_init',k,True)
   #
   return molecule

