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

def energy_summation(molecule,k,tup,e_inc,energy,level):
   #
   if (molecule['mpi_parallel']):
      #
      energy_summation_par(molecule,k,tup,e_inc,energy,level)
      #
      collect_summation_mpi_time(molecule,k)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_summation',k)
      #
      # compute energy increments at level k
      #
      for j in range(0,len(tup[k-1])):
         #
         for i in range(k-1,0,-1):
            #
            combs = tup[k-1][j,comb_index(k,i)]
            #
            dt = np.dtype((np.void,tup[i-1].dtype.itemsize*tup[i-1].shape[1]))
            #
            idx = np.nonzero(np.in1d(tup[i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
            #
            for l in idx: e_inc[k-1][j] -= e_inc[i-1][l]
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
      timer_mpi(molecule,'mpi_time_work_summation',k,True)
      #
      # check for convergence wrt total energy
      #
      if ((k >= 2) and (abs(energy[-1]-energy[-2]) < molecule['prim_e_thres'])): molecule['conv_energy'].append(True)
   #
   return molecule, e_inc, energy

