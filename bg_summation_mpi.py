#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_summation_mpi.py: MPI energy summation routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import timer_mpi
from bg_utils import comb_index

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_summation_par(molecule,tup,e_inc,energy,thres,order,level):
   #
   #  ---  master/slave routine
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      timer_mpi(molecule,'mpi_time_idle_summation',order)
      #
      msg = {'task': 'energy_summation_par', 'order': order, 'level': level}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      # re-init e_inc[-1] with 0.0
      #
      e_inc[-1].fill(0.0)
   #
   timer_mpi(molecule,'mpi_time_work_summation',order)
   #
   for j in range(0,len(tup[-1])):
      #
      if (e_inc[-1][j] != 0.0):
         #
         for i in range(order-1,0,-1):
            #
            combs = tup[-1][j,comb_index(order,i)]
            #
            dt = np.dtype((np.void,tup[i-1].dtype.itemsize*tup[i-1].shape[1]))
            #
            idx = np.nonzero(np.in1d(tup[i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
            #
            for l in idx: e_inc[-1][j] -= e_inc[i-1][l]
   #
   # allreduce e_inc[-1]
   #
   allred_e_inc(molecule,e_inc,order)
   #
   # let master calculate the total energy
   #
   if (molecule['mpi_master']):
      #
      # sum of energy increments 
      #
      e_tmp = np.sum(e_inc[-1])
      #
      # sum of total energy
      #
      if (order >= 2): e_tmp += energy[-1]
      #
      energy.append(e_tmp)
      #
      # check for convergence wrt total energy
      #
      if ((order >= 2) and (abs(energy[-1]-energy[-2]) < thres)): molecule['conv_energy'].append(True)
   #
   return molecule, e_inc, energy

def allred_e_inc(molecule,e_inc,order):
   #
   # Allreduce e_inc[-1]
   #
   timer_mpi(molecule,'mpi_time_idle_summation',order)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_comm_summation',order)
   #
   # init receive buffer
   #
   recv_buff = np.zeros(len(e_inc[-1]),dtype=np.float64)
   #
   # now do Allreduce
   #
   molecule['mpi_comm'].Allreduce([e_inc[-1],MPI.DOUBLE],[recv_buff,MPI.DOUBLE],op=MPI.SUM)
   #
   # finally, overwrite e_inc[-1]
   #
   timer_mpi(molecule,'mpi_time_work_summation',order)
   #
   e_inc[-1] = recv_buff
   #
   del recv_buff
   #
   return e_inc

