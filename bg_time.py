#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_time.py: time-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import collect_mpi_timings, calc_mpi_timings

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def timer_phase(molecule,key,order,level):
   #
   if (level == 'MACRO'):
      #
      key = 'prim_'+key
   #
   elif (level == 'CORRE'):
      #
      key = 'corr_'+key
   #
   elif (level == 'REF'):
      #
      key = 'ref_'+key
   #
   if (len(molecule[key]) < order):
      #
      molecule[key].append(-1.0*MPI.Wtime())
   #
   else:
      #
      molecule[key][order-1] += MPI.Wtime()
   #
   return molecule

def init_phase_timings(molecule):
   #
   # program phase distribution
   #
   molecule['prim_time_tot'] = []
   molecule['prim_time_init'] = []
   molecule['prim_time_kernel'] = []
   molecule['prim_time_final'] = []
   #
   if (molecule['corr']):
      #
      molecule['corr_time_tot'] = []
      molecule['corr_time_init'] = []
      molecule['corr_time_kernel'] = []
      molecule['corr_time_final'] = []
   #
   if (molecule['ref']):
      #
      molecule['ref_time_tot'] = []
   #
   return molecule

def timings_main(molecule):
   #
   # check if *_time_tot and *_time_init lists contain contribution from order k > max_order
   #
   if (len(molecule['prim_time_tot']) > len(molecule['prim_energy'])): molecule['prim_time_tot'].pop(-1)
   if (len(molecule['prim_time_init']) > len(molecule['prim_energy'])): molecule['prim_time_init'].pop(-1)
   if (len(molecule['corr_time_tot']) > len(molecule['prim_energy'])): molecule['corr_time_tot'].pop(-1)
   if (len(molecule['corr_time_init']) > len(molecule['prim_energy'])): molecule['corr_time_init'].pop(-1)
   #
   calc_phase_timings(molecule)
   #
   if (molecule['mpi_parallel']):
      #
      collect_mpi_timings(molecule)
      #
      calc_mpi_timings(molecule)
   #
   return molecule

def calc_phase_timings(molecule):
   #
   molecule['time_tot'] = np.asarray(molecule['prim_time_tot'])+np.asarray(molecule['corr_time_tot'])
   molecule['time_init'] = np.asarray(molecule['prim_time_init'])+np.asarray(molecule['corr_time_init'])
   molecule['time_kernel'] = np.asarray(molecule['prim_time_kernel'])+np.asarray(molecule['corr_time_kernel'])
   molecule['time_final'] = np.asarray(molecule['prim_time_final'])+np.asarray(molecule['corr_time_final'])
   molecule['time_remain'] = molecule['time_tot']-(molecule['time_init']+molecule['time_kernel']+molecule['time_final'])
   #
   return molecule

