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
   #  ---  master/slave routine
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

def calc_mpi_timings(molecule):
   #
   #  ---  master routine
   #
   molecule['sum_work_abs'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
   molecule['sum_comm_abs'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
   molecule['sum_idle_abs'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
   #
   for i in range(0,3):
      #
      for j in range(0,molecule['mpi_size']):
         #
         molecule['sum_work_abs'][i][j] = np.sum(molecule['mpi_time_work'][i][j])
         molecule['sum_comm_abs'][i][j] = np.sum(molecule['mpi_time_comm'][i][j])
         molecule['sum_idle_abs'][i][j] = np.sum(molecule['mpi_time_idle'][i][j])
   #
   molecule['dist_init'] = np.empty([3,molecule['mpi_size']-1],dtype=np.float64)
   molecule['dist_kernel'] = np.empty([3,molecule['mpi_size']-1],dtype=np.float64)
   molecule['dist_final'] = np.empty([3,molecule['mpi_size']-1],dtype=np.float64)
   #
   for i in range(0,3):
      #
      if (i == 0):
         #
         dist = molecule['dist_init']
      #
      elif (i == 1):
         #
         dist = molecule['dist_kernel']
      #
      elif (i == 2):
         #
         dist = molecule['dist_final']
      #
      for j in range(1,molecule['mpi_size']):
         #
         dist[0][j-1] = (molecule['sum_work_abs'][i][j]/(molecule['sum_work_abs'][i][j]+molecule['sum_comm_abs'][i][j]+molecule['sum_idle_abs'][i][j]))*100.0
         dist[1][j-1] = (molecule['sum_comm_abs'][i][j]/(molecule['sum_work_abs'][i][j]+molecule['sum_comm_abs'][i][j]+molecule['sum_idle_abs'][i][j]))*100.0
         dist[2][j-1] = (molecule['sum_idle_abs'][i][j]/(molecule['sum_work_abs'][i][j]+molecule['sum_comm_abs'][i][j]+molecule['sum_idle_abs'][i][j]))*100.0
   #
   return molecule

