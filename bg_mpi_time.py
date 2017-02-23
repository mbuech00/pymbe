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

def init_mpi_timings(molecule):
   #
   # init tmp time and time label
   #
   molecule['store_time'] = 0.0
   #
   molecule['store_key'] = ''
   #
   # program phase distribution
   #
   if (molecule['mpi_master']):
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
   # mpi distribution
   #
   if (molecule['mpi_parallel']):
      #
      # init timings
      #
      molecule['mpi_time_idle_init'] = []
      molecule['mpi_time_comm_init'] = []
      molecule['mpi_time_work_init'] = []
      #
      # energy kernel timings
      #
      molecule['mpi_time_idle_kernel'] = []
      molecule['mpi_time_comm_kernel'] = []
      molecule['mpi_time_work_kernel'] = []
      #
      # energy summation timings
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
   if (molecule['mpi_parallel']):
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
         molecule['mpi_time_idle'] = [[molecule['mpi_time_idle_init']],[molecule['mpi_time_idle_kernel']],[molecule['mpi_time_idle_final']]]
         molecule['mpi_time_comm'] = [[molecule['mpi_time_comm_init']],[molecule['mpi_time_comm_kernel']],[molecule['mpi_time_comm_final']]]
         molecule['mpi_time_work'] = [[molecule['mpi_time_work_init']],[molecule['mpi_time_work_kernel']],[molecule['mpi_time_work_final']]]
         #
         # receive individual timings (in ordered sequence)
         #
         for i in range(0,molecule['mpi_size']-1):
            #
            time = molecule['mpi_comm'].recv(source=i+1,status=molecule['mpi_stat'])
            #
            for j in range(0,3):
               #
               molecule['mpi_time_idle'][j].append(time['time_idle'][j])
               molecule['mpi_time_comm'][j].append(time['time_comm'][j])
               molecule['mpi_time_work'][j].append(time['time_work'][j])
      #
      else:
         #
         # send mpi timings to master
         #
         time = {}
         #
         time['time_idle'] = [molecule['mpi_time_idle_init'],molecule['mpi_time_idle_kernel'],molecule['mpi_time_idle_final']]
         time['time_comm'] = [molecule['mpi_time_comm_init'],molecule['mpi_time_comm_kernel'],molecule['mpi_time_comm_final']]
         time['time_work'] = [molecule['mpi_time_work_init'],molecule['mpi_time_work_kernel'],molecule['mpi_time_work_final']]
         #
         molecule['mpi_comm'].send(time,dest=0)
         #
         time.clear()
         #
         return
      #
      time.clear()
   #
   return molecule

