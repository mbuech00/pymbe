#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_screening.py: screening routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from itertools import combinations

from bg_mpi_time import timer_mpi, collect_screen_mpi_time
from bg_screening_mpi import tuple_generation_master
from bg_print import print_screening

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def screening_main(molecule,tup,e_inc,thres,l_limit,u_limit,order,level):
   #
   # generate tuples for next order
   #
   tuple_generation(molecule,tup,e_inc,thres,l_limit,u_limit,order,level)
   #
   # print screening results
   #
   print_screening(molecule,thres,tup,order,level)
   #
   if (molecule['mpi_parallel']): collect_screen_mpi_time(molecule,order,True)
   #
   # update threshold and restart frequency
   #
   update_thres_and_rst_freq(molecule,order)
   #
   return molecule, tup, e_inc

def tuple_screening(molecule,tup,e_inc,thres,order):
   #
   timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   # fill parent_tup array with non-negligible tuples only
   #
   molecule['parent_tup'] = tup[-1][np.where(np.abs(e_inc[-1]) >= thres)]
   #
   timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule, tup

def tuple_generation(molecule,tup,e_inc,thres,l_limit,u_limit,order,level):
   #
   if (molecule['mpi_parallel']):
      #
      tuple_generation_master(molecule,tup,e_inc,thres,l_limit,u_limit,order,level)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      # determine which tuples have contributions below the threshold
      #
      if (thres > 0.0): molecule['allow_tuple'] = tup[-1][np.where(np.abs(e_inc[-1]) >= thres)]
      #
      molecule['screen_count'] = 0
      #
      tmp = []
      combs = []
      #
      for i in range(0,len(tup[-1])):
         #
         if (np.abs(e_inc[-1][i]) >= thres):
            #
            # loop through possible orbitals to augment the parent tuple with
            #
            for m in range(tup[-1][i][-1]+1,(l_limit+u_limit)+1): tmp.append(tup[-1][i].tolist()+[m])
         #
         else:
            #
            # generate list with all subsets of particular tuple
            #
            combs = list(list(comb) for comb in combinations(tup[-1][i],order-1))
            #
            # loop through possible orbitals to augment the combinations with
            #
            for m in range(tup[-1][i][-1]+1,(l_limit+u_limit)+1):
               #
               screen = True
               #
               for j in range(0,len(combs)):
                  #
                  # check whether or not the particular tuple is actually allowed
                  #
                  if (np.equal(combs[j]+[m],molecule['allow_tuple']).all(axis=1).any()):
                     #
                     screen = False
                     #
                     tmp.append(tup[-1][i].tolist()+[m])
                     #
                     break
               #
               # if tuple should be screened away, then increment screen counter
               #
               if (screen): molecule['screen_count'] += 1
      #
      # write to tup list or mark expansion as converged
      #
      if (len(tmp) >= 1):
         #
         tup.append(np.array(tmp,dtype=np.int32))
      #
      else:
         #
         molecule['conv_orb'].append(True)
      #
      del combs
      del tmp
      #
      timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule, tup

def update_thres_and_rst_freq(molecule,order):
   #
   # update threshold with dampening
   #
   if (order >= 2): molecule['prim_exp_thres'] = (molecule['prim_exp_scaling'])**(order-2) * molecule['prim_exp_thres_init']
   #
   # update restart frequency by halving it
   #
   molecule['rst_freq'] /= 2.
   #
   return molecule

