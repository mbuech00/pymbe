#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_time.py: MPI time-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

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

def timer_mpi(molecule,key,order,end=False):
   #
   if (key != molecule['store_key']):
      #
      if (molecule['store_key'] != ''):
         #
         if (len(molecule[molecule['store_key']]) < order):
            #
            molecule[molecule['store_key']].append(MPI.Wtime()-molecule['store_time'])
         #
         else:
            #
            molecule[molecule['store_key']][order-1] += MPI.Wtime()-molecule['store_time']
         #
         molecule['store_time'] = MPI.Wtime()
         #
         molecule['store_key'] = key
      #
      else:
         #
         molecule['store_time'] = MPI.Wtime()
         #
         molecule['store_key'] = key
   #
   elif ((key == molecule['store_key']) and end):
      #
      if (len(molecule[key]) < order):
         #
         molecule[key].append(MPI.Wtime()-molecule['store_time'])
      #
      else:
         #
         molecule[key][order-1] += MPI.Wtime()-molecule['store_time']
      #
      molecule['store_key'] = ''
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

def init_mpi_timings(molecule):
   #
   # init tmp time and time label
   #
   molecule['store_time'] = 0.0
   #
   molecule['store_key'] = ''
   #
   # mpi distribution
   #
   # 'init' timings
   #
   molecule['mpi_time_idle_init'] = []
   molecule['mpi_time_comm_init'] = []
   molecule['mpi_time_work_init'] = []
   #
   # 'energy kernel' timings
   #
   molecule['mpi_time_idle_kernel'] = []
   molecule['mpi_time_comm_kernel'] = []
   molecule['mpi_time_work_kernel'] = []
   #
   # 'energy summation' timings
   #
   molecule['mpi_time_idle_final'] = []
   molecule['mpi_time_comm_final'] = []
   molecule['mpi_time_work_final'] = []
   #
   return molecule

def collect_mpi_timings(molecule):
   #
   #
   #  ---  master/slave routine
   #
   # first, check if *_init lists contain contribution from order k > max_order
   #
   if (len(molecule['mpi_time_work_init']) > len(molecule['mpi_time_work_kernel'])): molecule['mpi_time_work_init'].pop(-1)
   if (len(molecule['mpi_time_comm_init']) > len(molecule['mpi_time_work_kernel'])): molecule['mpi_time_comm_init'].pop(-1)
   if (len(molecule['mpi_time_idle_init']) > len(molecule['mpi_time_work_kernel'])): molecule['mpi_time_idle_init'].pop(-1)
   #
   # next, check if mpi_time_comm_kernel is empty
   #
   if (len(molecule['mpi_time_comm_kernel']) == 0): molecule['mpi_time_comm_kernel'] = [0.0]*len(molecule['mpi_time_comm_init'])
   #
   # now, send/recv data as numpy arrays
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      msg = {'task': 'collect_mpi_timings'}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      # next, collect mpi timings
      #
      # init mpi lists with master timings
      #
      molecule['mpi_time_work'] = [[np.asarray(molecule['mpi_time_work_init'])],[np.asarray(molecule['mpi_time_work_kernel'])],[np.asarray(molecule['mpi_time_work_final'])]]
      molecule['mpi_time_idle'] = [[np.asarray(molecule['mpi_time_idle_init'])],[np.asarray(molecule['mpi_time_idle_kernel'])],[np.asarray(molecule['mpi_time_idle_final'])]]
      molecule['mpi_time_comm'] = [[np.asarray(molecule['mpi_time_comm_init'])],[np.asarray(molecule['mpi_time_comm_kernel'])],[np.asarray(molecule['mpi_time_comm_final'])]]
      #
      # receive individual timings (in ordered sequence)
      #
      time = np.empty([9,len(molecule['prim_energy'])],dtype=np.float64)
      #
      for i in range(1,molecule['mpi_size']):
         #
         molecule['mpi_comm'].Recv([time,MPI.DOUBLE],source=i,status=molecule['mpi_stat'])
         #
         for j in range(0,3):
            #
            molecule['mpi_time_work'][j].append(np.copy(time[j]))
            molecule['mpi_time_comm'][j].append(np.copy(time[j+3]))
            molecule['mpi_time_idle'][j].append(np.copy(time[j+6]))
   #
   else:
      #
      # send mpi timings to master
      #
      time = np.array([molecule['mpi_time_work_init'],molecule['mpi_time_work_kernel'],molecule['mpi_time_work_final'],\
             molecule['mpi_time_comm_init'],molecule['mpi_time_comm_kernel'],molecule['mpi_time_comm_final'],\
             molecule['mpi_time_idle_init'],molecule['mpi_time_idle_kernel'],molecule['mpi_time_idle_final']])
      #
      molecule['mpi_comm'].Send([time,MPI.DOUBLE],dest=0)
      #
      del time
      #
      return
   #
   del time
   #
   return molecule

