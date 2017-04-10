#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_energy.py: energy-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np

from bg_mpi_time import timer_mpi, collect_kernel_mpi_time, collect_summation_mpi_time
from bg_mpi_energy import energy_kernel_mono_exp_master, energy_summation_par
from bg_utils import run_calc_corr, term_calc, orb_string, comb_index 
from bg_print import print_status
from bg_rst_write import rst_write_e_inc, rst_write_time

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_kernel_mono_exp(molecule,order,tup,e_inc,l_limit,u_limit,level):
   #
   if (molecule['mpi_parallel']):
      #
      energy_kernel_mono_exp_master(molecule,order,tup,e_inc,l_limit,u_limit,level)
      #
      collect_kernel_mpi_time(molecule,order)
   #
   else:
      #
      string = {'drop': ''}
      #
      if (molecule['rst'] and (order == molecule['min_order'])):
         #
         start = np.argmax(e_inc[order-1] == 0.0)
      #
      else:
         #
         start = 0
      #
      for i in range(start,len(tup[order-1])):
         #
         timer_mpi(molecule,'mpi_time_work_kernel',order)
         #
         # write string
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
         print_status(float(i+1)/float(len(tup[order-1])),level)
         #
         # error check
         #
         if (molecule['error'][-1]):
            #
            molecule['error_rank'] = 0
            #
            molecule['error_drop'] = drop['string']
            #
            term_calc(molecule)
         #
         # write e_inc restart file
         #
         timer_mpi(molecule,'mpi_time_work_kernel',order,True)
         #
         if (((i+1) % molecule['rst_freq']) == 0):
            #
            rst_write_time(molecule,'kernel')
            # 
            rst_write_e_inc(molecule,order)
   #
   return molecule, tup, e_inc

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
      rst_write_time(molecule,'summation')
   #
   return e_inc, energy

def chk_energy_conv(molecule,e_tot,k):
   #
   if ((k >= 2) and (abs(e_tot[-1]-e_tot[-2]) < molecule['prim_e_thres'])): molecule['conv_energy'].append(True)
   #
   return molecule


