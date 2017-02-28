#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_energy.py: energy-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np

from bg_mpi_energy import energy_kernel_mono_exp_par, energy_summation_par
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

def energy_kernel_mono_exp(molecule,order,tup,n_tup,e_inc,l_limit,u_limit,level):
   #
   if (molecule['mpi_parallel']):
      #
      energy_kernel_mono_exp_par(molecule,order,tup,n_tup,e_inc,l_limit,u_limit,level)
   #
   else:
      #
      string = {'drop': ''}
      #
      counter = 0
      #
      for i in range(0,len(tup[order-1])):
         #
         # write string
         #
         counter += 1
         #
         orb_string(molecule,l_limit,u_limit,tup[order-1][i],string)
         #
         # run correlated calc
         #
         run_calc_corr(molecule,string['drop'],level)
         #
         # write tuple energy
         #
         e_inc[order-1][i] = molecule['e_tmp']
         #
         # print status
         #
         print_status(float(counter)/float(n_tup[order-1]),level)
         #
         # error check
         #
         if (molecule['error'][-1]):
            #
            return molecule, tup, e_inc
   #
   return molecule, tup, e_inc

def energy_summation(molecule,k,tup,e_inc,energy,level):
   #
   if (molecule['mpi_parallel']):
      #
      energy_summation_par(molecule,k,tup,e_inc,energy,level)
   #
   else:
      #
      # compute energy increments at level k
      #
      for j in range(0,len(tup[k-1])):
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
            else:
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
   return e_inc, energy


