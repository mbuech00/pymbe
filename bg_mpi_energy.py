#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_energy.py: MPI energy-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import timer_mpi
from bg_utils import run_calc_corr, orb_string, comb_index
from bg_print import print_status 

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_kernel_mono_exp_par(molecule,k,tup,n_tup,e_inc,l_limit,u_limit,level):
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      msg = {'task': 'energy_kernel_mono_exp_par', 'order': k, 'level': level, 'l_limit': l_limit, 'u_limit': u_limit}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
   #
   # slaves init e_inc[k-1]
   #
   if (not molecule['mpi_master']): e_inc.append(np.zeros(len(tup[k-1]),dtype=np.float64))
   #
   string = {'drop': ''}
   #
   for i in range(0,len(tup[k-1])):
      #
      # simple modulo distribution of tasks
      #
      if ((i % molecule['mpi_size']) == molecule['mpi_rank']):
         #
         timer_mpi(molecule,'mpi_time_work_kernel',k)
         #
         # write string
         #
         orb_string(molecule,l_limit,u_limit,tup[k-1][i],string)
         #
         # run correlated calc
         #
         run_calc_corr(molecule,string['drop'],level)
         #
         # write tuple energy
         #
         e_inc[k-1][i] = molecule['e_tmp']
         #
         timer_mpi(molecule,'mpi_time_work_kernel',k,True)
         #
         # print status (master)
         #
         if (molecule['mpi_master']): print_status(float(i+1)/float(n_tup[k-1]),level)
         #
         # error check
         #
         if (molecule['error'][-1]):
            #
            return molecule, tup, e_inc
   #
   timer_mpi(molecule,'mpi_time_idle_kernel',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_idle_kernel',k,True)
   #
   # make sure that STATUS = 100.00 % has been written
   #
   if (molecule['mpi_master']): print_status(1.0,level)
   #
   return molecule, tup, e_inc

def energy_summation_par(molecule,k,tup,e_inc,energy,level):
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      msg = {'task': 'energy_summation_par', 'order': k, 'level': level}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
   #
   for j in range(0,len(tup[k-1])):
      #
      if (e_inc[k-1][j] != 0.0):
         #
         timer_mpi(molecule,'mpi_time_work_final',k)
         #
         for i in range(k-1,0,-1):
            #
            combs = tup[k-1][j,comb_index(k,i)]
            #
            if (level == 'CORRE'):
               #
               if (len(tup[i-1]) > 0):
                  #
                  dt = np.dtype((np.void,tup[i-1].dtype.itemsize*tup[i-1].shape[1]))
                  #
                  idx = np.nonzero(np.in1d(tup[i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
                  #
                  for l in idx: e_inc[k-1][j] -= e_inc[i-1][l]
               #
               dt = np.dtype((np.void,molecule['prim_tuple'][i-1].dtype.itemsize*molecule['prim_tuple'][i-1].shape[1]))
               #
               idx = np.nonzero(np.in1d(molecule['prim_tuple'][i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
               #
               for l in idx: e_inc[k-1][j] -= molecule['prim_energy_inc'][i-1][l]
            #
            elif (level == 'MACRO'):
               #
               dt = np.dtype((np.void,tup[i-1].dtype.itemsize*tup[i-1].shape[1]))
               #
               idx = np.nonzero(np.in1d(tup[i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
               #
               for l in idx: e_inc[k-1][j] -= e_inc[i-1][l]
         #
         timer_mpi(molecule,'mpi_time_work_final',k,True)
   #
   timer_mpi(molecule,'mpi_time_work_final',k)
   #
   timer_mpi(molecule,'mpi_time_idle_final',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   # allreduce e_inc[k-1] (here: do explicit Reduce+Bcast, as Allreduce has been observed to hang)
   #
   timer_mpi(molecule,'mpi_time_comm_final',k)
   #
   if (molecule['mpi_master']):
      #
      recv_buff = np.zeros(len(e_inc[k-1]),dtype=np.float64)
   #
   else:
      #
      recv_buff = None
   #
   molecule['mpi_comm'].Reduce([e_inc[k-1],MPI.DOUBLE],[recv_buff,MPI.DOUBLE],op=MPI.SUM,root=0)
   #
   if (molecule['mpi_master']):
      #
      timer_mpi(molecule,'mpi_time_work_final',k)
      #
      e_inc[k-1] = recv_buff
      #
      timer_mpi(molecule,'mpi_time_comm_final',k)
   #
   molecule['mpi_comm'].Bcast([e_inc[k-1],MPI.DOUBLE],root=0)
   #
   timer_mpi(molecule,'mpi_time_work_final',k)
   #
   # let master calculate the total energy
   #
   if (molecule['mpi_master']):
      #
      # sum of energy increment of level k
      #
      e_tmp = np.sum(e_inc[k-1])
      #
      # sum of total energy
      #
      if (k > 1):
         #
         e_tmp += energy[k-2]
      #
      energy.append(e_tmp)
   #
   timer_mpi(molecule,'mpi_time_idle_final',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_idle_final',k,True)
   #
   del recv_buff
   #
   return e_inc, energy



