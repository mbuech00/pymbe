#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_energy.py: energy-related routines for Bethe-Goldstone correlation calculations."""

from mpi4py import MPI

from bg_mpi_energy import energy_calc_mono_exp_master
from bg_utilities import run_calc_corr, orb_string 
from bg_print import print_status

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_calc_mono_exp(molecule,order,tup,n_tup,l_limit,u_limit,level):
   #
   if (molecule['mpi_parallel']):
      #
      energy_calc_mono_exp_master(molecule,order,tup,n_tup,l_limit,u_limit,level)
   #
   else:
      #
      string = {'drop': ''}
      #
      if (level == 'MACRO'):
         #
         end = n_tup[order-1]
      #
      elif (level == 'CORRE'):
         #
         end = len(tup[order-1])
      #
      counter = 0
      #
      for i in range(0,end):
         #
         # write string
         #
         if ((level == 'MACRO') or ((level == 'CORRE') and (len(tup[order-1][i]) == 1))):
            #
            counter += 1
            #
            orb_string(molecule,l_limit,u_limit,tup[order-1][i][0],string)
            #
            # run correlated calc
            #
            run_calc_corr(molecule,string['drop'],level)
            #
            # write tuple energy
            #
            tup[order-1][i].append(molecule['e_tmp'])
            #
            # print status
            #
            print_status(float(counter)/float(n_tup[order-1]),level)
            #
            # error check
            #
            if (molecule['error'][-1]):
               #
               return molecule, tup
   #
   return molecule, tup

def bg_order(molecule,k,tup,energy):
   #
   for j in range(0,len(tup[k-1])):
      #
      for i in range(k-1,0,-1):
         #
         for l in range(0,len(tup[i-1])):
            #
            if (set(tup[i-1][l][0]) < set(tup[k-1][j][0])):
               #
               tup[k-1][j][1] -= tup[i-1][l][1]
   #
   e_tmp = 0.0
   #
   for j in range(0,len(tup[k-1])):
      #
      e_tmp += tup[k-1][j][1]
   #
   if (k > 1):
      #
      e_tmp += energy[k-2]
   #
   energy.append(e_tmp)
   #
   return energy

