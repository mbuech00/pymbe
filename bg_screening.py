#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_screening.py: screening routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from copy import deepcopy

from bg_mpi_time import timer_mpi, collect_screen_mpi_time
from bg_mpi_screening import tuple_generation_master
from bg_print import print_orb_info, print_update

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def screening_main(molecule,tup,n_tup,e_inc,thres,l_limit,u_limit,order):
   #
   # screen away tuples with energy contributions lower than thres
   #
   tuple_screening(molecule,tup,e_inc,thres,order)
   #
   if (len(e_inc[-1]) < n_tup[-1]): print_screening(molecule,n_tup,e_inc)
   #
   # now generate new tuples for following order
   #
   tuple_generation(molecule,tup,n_tup,l_limit,u_limit,order)
   #
   if (molecule['mpi_parallel']): collect_screen_mpi_time(molecule,order,True)
   #
   return molecule, tup, e_inc

def tuple_screening(molecule,tup,e_inc,thres,order):
   #
   timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   # generate list of indices for negligible tuples
   #
   indices = []
   #
   for i in range(0,len(e_inc[-1])):
      #
      if (np.abs(e_inc[-1][i]) < thres): indices.append(i)
   #
   # now screen away tuples according to indices list
   #
   tup[-1] = tup[-1][indices]
   e_inc[-1] = e_inc[-1][indices]
   #
   # check for convergence wrt orbital expansion
   #
   if (len(indices) == 0): molecule['conv_orb'].append(True)
   #
   del indices
   #
   timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule, tup, e_inc

def tuple_generation(molecule,tup,n_tup,l_limit,u_limit,order):
   #
   if (molecule['mpi_parallel']):
      #
      tuple_generation_master(molecule,tup,n_tup,l_limit,u_limit,order)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      tmp = []
      #
      for i in range(0,len(tup[-1])):
         #
         # set parent tuple
         #
         parent_tup = tup[-1][i]
         #
         # loop through possible orbitals to augment the parent tuple with
         #
         for m in range(parent_tup[-1]+1,(l_limit+u_limit)+1):
            #
            # append the child tuple to the tmp list
            #
            tmp.append(list(deepcopy(parent_tup)))
            #
            tmp[-1].append(m)
      #
      # write to tup list
      #
      tup.append(np.array(tmp,dtype=np.int32))
      #
      # update n_tup list
      #
      n_tup.append(len(tup[-1]))
      #
      del tmp
      #
      timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule, tup, n_tup

