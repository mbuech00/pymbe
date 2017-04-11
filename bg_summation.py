#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_summation.py: energy summation routines for Bethe-Goldstone correlation calculations."""

import numpy as np

from bg_mpi_time import timer_mpi, collect_summation_mpi_time
from bg_summation_mpi import energy_summation_par
from bg_utils import comb_index 

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_summation(molecule,order,tup,e_inc,energy):
   #
   if (molecule['mpi_parallel']):
      #
      energy_summation_par(molecule,order,tup,e_inc,energy)
      #
      collect_summation_mpi_time(molecule,order)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_summation',order)
      #
      # compute energy increments at current order
      #
      for j in range(0,len(tup[-1])):
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
      # sum of energy increments
      #
      e_tmp = np.sum(e_inc[-1])
      #
      # sum of total energy
      #
      if (order >= 2):
         #
         e_tmp += energy[-2]
      #
      energy.append(e_tmp)
      #
      timer_mpi(molecule,'mpi_time_work_summation',order,True)
      #
      # check for convergence wrt total energy
      #
      if ((order >= 2) and (abs(energy[-1]-energy[-2]) < molecule['prim_thres'])): molecule['conv_energy'].append(True)
   #
   return molecule, e_inc, energy

