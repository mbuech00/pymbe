#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_screening.py: screening routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from copy import deepcopy

from bg_mpi_time import timer_mpi, collect_screen_mpi_time
from bg_screening_mpi import tuple_generation_master
from bg_print import print_screening

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def screening_main(molecule,tup,e_inc,thres,l_limit,u_limit,order,level):
   #
   # screen away tuples with energy contributions lower than thres
   #
   tuple_screening(molecule,tup,e_inc,thres,order)
   #
   if (len(molecule['parent_tup']) < len(tup[-1])): print_screening(molecule,thres,tup,level)
   #
   # now generate new tuples for following order
   #
   tuple_generation(molecule,tup,l_limit,u_limit,order,level)
   #
   if (molecule['mpi_parallel']): collect_screen_mpi_time(molecule,order,True)
   #
   # update threshold and restart frequency
   #
   update_thres_and_rst_freq(molecule)
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
      if (np.abs(e_inc[-1][i]) >= thres): indices.append(i)
   #
   # now screen away tuples according to indices list
   #
   molecule['parent_tup'] = tup[-1][indices]
   #
   # check for convergence wrt orbital expansion
   #
   if (len(indices) == 0): molecule['conv_orb'].append(True)
   #
   del indices
   #
   timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule, tup

def tuple_generation(molecule,tup,l_limit,u_limit,order,level):
   #
   if (molecule['mpi_parallel']):
      #
      tuple_generation_master(molecule,tup,l_limit,u_limit,order,level)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      tmp = []
      #
      for i in range(0,len(molecule['parent_tup'])):
         #
         # loop through possible orbitals to augment the parent tuple with
         #
         for m in range(molecule['parent_tup'][i][-1]+1,(l_limit+u_limit)+1):
            #
            # append the child tuple to the tmp list
            #
            tmp.append(list(deepcopy(molecule['parent_tup'][i])))
            #
            tmp[-1].append(m)
      #
      # write to tup list
      #
      tup.append(np.array(tmp,dtype=np.int32))
      #
      del tmp
      #
      timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule, tup

def update_thres_and_rst_freq(molecule):
   #
   # update threshold by multiplying it by exp. scaling
   #
   molecule['prim_thres'] *= molecule['prim_scaling']
   #
   # update restart frequency by halving it
   #
   molecule['rst_freq'] /= 2.
   #
   return molecule

