#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_time.py: MPI time-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_rst_write import rst_write_time

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
   #  ---  master/slave routine
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
   #  ---  master/slave routine
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
   molecule['mpi_time_idle_init'] = [0.0]
   molecule['mpi_time_comm_init'] = [0.0]
   molecule['mpi_time_work_init'] = [0.0]
   #
   # 'energy kernel' timings
   #
   molecule['mpi_time_idle_kernel'] = [0.0]
   molecule['mpi_time_comm_kernel'] = [0.0]
   molecule['mpi_time_work_kernel'] = [0.0]
   #
   # 'energy summation' timings
   #
   molecule['mpi_time_idle_final'] = [0.0]
   molecule['mpi_time_comm_final'] = [0.0]
   molecule['mpi_time_work_final'] = [0.0]
   #
   # collective lists
   #
   if (molecule['mpi_parallel'] and molecule['mpi_master']):
      #
      molecule['mpi_time_work'] = [[[] for i in range(0,molecule['mpi_size'])] for j in range(0,3)]
      molecule['mpi_time_comm'] = [[[] for i in range(0,molecule['mpi_size'])] for j in range(0,3)]
      molecule['mpi_time_idle'] = [[[] for i in range(0,molecule['mpi_size'])] for j in range(0,3)]
   #
   return molecule

def collect_mpi_timings(molecule,phase):
   #
   #  ---  master/slave routine
   #
   if (phase == 'init'):
      #
      idx = 0
   #
   elif (phase == 'kernel'):
      #
      idx = 1
   #
   elif (phase == 'final'):
      #
      idx = 2
   #
   if (molecule['mpi_master']):
      #
      # write master timings
      #
      if (idx == 1):
         #
         molecule['mpi_time_work'][idx][0][-1] = molecule['mpi_time_work_'+str(phase)][-1]
         molecule['mpi_time_comm'][idx][0][-1] = molecule['mpi_time_comm_'+str(phase)][-1]
         molecule['mpi_time_idle'][idx][0][-1] = molecule['mpi_time_idle_'+str(phase)][-1]
      #
      else:
         #
         molecule['mpi_time_work'][idx][0].append(molecule['mpi_time_work_'+str(phase)][-1])
         molecule['mpi_time_comm'][idx][0].append(molecule['mpi_time_comm_'+str(phase)][-1])
         molecule['mpi_time_idle'][idx][0].append(molecule['mpi_time_idle_'+str(phase)][-1])
      #
      # receive individual timings (in ordered sequence)
      #
      for i in range(1,molecule['mpi_size']):
         #
         time = molecule['mpi_comm'].recv(source=i,status=molecule['mpi_stat'])
         #
         if (idx == 1):
            #
            molecule['mpi_time_work'][idx][i][-1] = time['work']
            molecule['mpi_time_comm'][idx][i][-1] = time['comm']
            molecule['mpi_time_idle'][idx][i][-1] = time['idle']
         #
         else:
            #
            molecule['mpi_time_work'][idx][i].append(time['work'])
            molecule['mpi_time_comm'][idx][i].append(time['comm'])
            molecule['mpi_time_idle'][idx][i].append(time['idle'])
   #
   else:
      #
      # send mpi timings to master
      #
      time = {'work': molecule['mpi_time_work_'+str(phase)][-1], 'comm': molecule['mpi_time_comm_'+str(phase)][-1], 'idle': molecule['mpi_time_idle_'+str(phase)][-1]}
      #
      molecule['mpi_comm'].send(time,dest=0)
   #
   time.clear()
   #
   return molecule

def calc_mpi_timings(molecule):
   #
   #  ---  master routine
   #
   # use master timings to calculate overall phase timings
   #
   if (not molecule['mpi_parallel']):
      #
      molecule['time_init'] = np.asarray(molecule['mpi_time_work_init']+[sum(molecule['mpi_time_work_init'])])
      molecule['time_kernel'] = np.asarray(molecule['mpi_time_work_kernel']+[sum(molecule['mpi_time_work_kernel'])])
      molecule['time_final'] = np.asarray(molecule['mpi_time_work_final']+[sum(molecule['mpi_time_work_final'])])
      molecule['time_tot'] = molecule['time_init']+molecule['time_kernel']+molecule['time_final']
   #
   else:
      #
      molecule['time_init'] = np.asarray(molecule['mpi_time_work_init']+[sum(molecule['mpi_time_work_init'])])\
                              +np.asarray(molecule['mpi_time_comm_init']+[sum(molecule['mpi_time_comm_init'])])\
                               +np.asarray(molecule['mpi_time_idle_init']+[sum(molecule['mpi_time_idle_init'])])
      molecule['time_kernel'] = np.asarray(molecule['mpi_time_work_kernel']+[sum(molecule['mpi_time_work_kernel'])])\
                                +np.asarray(molecule['mpi_time_comm_kernel']+[sum(molecule['mpi_time_comm_kernel'])])\
                                 +np.asarray(molecule['mpi_time_idle_kernel']+[sum(molecule['mpi_time_idle_kernel'])])
      molecule['time_final'] = np.asarray(molecule['mpi_time_work_final']+[sum(molecule['mpi_time_work_final'])])\
                               +np.asarray(molecule['mpi_time_comm_final']+[sum(molecule['mpi_time_comm_final'])])\
                                +np.asarray(molecule['mpi_time_idle_final']+[sum(molecule['mpi_time_idle_final'])])
      #
      molecule['time_tot'] = molecule['time_init']+molecule['time_kernel']+molecule['time_final']
      #
      # init summation arrays
      #
      molecule['sum_work_abs'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
      molecule['sum_comm_abs'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
      molecule['sum_idle_abs'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
      #
      # sum up work/comm/idle contributions from all orders for the individual mpi procs
      #
      for i in range(0,3):
         #
         for j in range(0,molecule['mpi_size']):
            #
            molecule['sum_work_abs'][i][j] = np.sum(np.asarray(molecule['mpi_time_work'][i][j]))
            molecule['sum_comm_abs'][i][j] = np.sum(np.asarray(molecule['mpi_time_comm'][i][j]))
            molecule['sum_idle_abs'][i][j] = np.sum(np.asarray(molecule['mpi_time_idle'][i][j]))
      #
      # mpi distribution - slave (only count slave timings)
      #
      molecule['dist_init'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
      molecule['dist_kernel'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
      molecule['dist_final'] = np.empty([3,molecule['mpi_size']],dtype=np.float64)
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
         # for init/kernel/final, calculate the relative distribution between work/comm/idle for the individual slaves
         #
         for j in range(0,molecule['mpi_size']):
            #
            dist[0][j] = (molecule['sum_work_abs'][i][j]/(molecule['sum_work_abs'][i][j]+molecule['sum_comm_abs'][i][j]+molecule['sum_idle_abs'][i][j]))*100.0
            dist[1][j] = (molecule['sum_comm_abs'][i][j]/(molecule['sum_work_abs'][i][j]+molecule['sum_comm_abs'][i][j]+molecule['sum_idle_abs'][i][j]))*100.0
            dist[2][j] = (molecule['sum_idle_abs'][i][j]/(molecule['sum_work_abs'][i][j]+molecule['sum_comm_abs'][i][j]+molecule['sum_idle_abs'][i][j]))*100.0
      #
      # mpi distribution - order (only count slave timings - total results are stored as the last entry)
      #
      molecule['dist_order'] = np.zeros([3,len(molecule['prim_energy'])+1],dtype=np.float64)
      #
      # absolute amount of work/comm/idle at each order
      #
      for k in range(0,len(molecule['prim_energy'])):
         #
         for i in range(0,3):
            #
            for j in range(1,molecule['mpi_size']):
               #
               molecule['dist_order'][0][k] += molecule['mpi_time_work'][i][j][k]
               molecule['dist_order'][1][k] += molecule['mpi_time_comm'][i][j][k]
               molecule['dist_order'][2][k] += molecule['mpi_time_idle'][i][j][k]
      #
      molecule['dist_order'][0][-1] = np.sum(molecule['dist_order'][0][:-1])
      molecule['dist_order'][1][-1] = np.sum(molecule['dist_order'][1][:-1]) 
      molecule['dist_order'][2][-1] = np.sum(molecule['dist_order'][2][:-1])
      #
      # calculate relative results
      #
      for k in range(0,len(molecule['prim_energy'])+1):
         #
         sum_k = molecule['dist_order'][0][k]+molecule['dist_order'][1][k]+molecule['dist_order'][2][k]
         #
         molecule['dist_order'][0][k] = (molecule['dist_order'][0][k]/sum_k)*100.0
         molecule['dist_order'][1][k] = (molecule['dist_order'][1][k]/sum_k)*100.0
         molecule['dist_order'][2][k] = (molecule['dist_order'][2][k]/sum_k)*100.0
   #
   return molecule

def collect_init_mpi_time(molecule,k,write_time=False):
   #
   #  ---  master/slave routine
   #
   timer_mpi(molecule,'mpi_time_idle_init',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_idle_init',k,True)
   #
   collect_mpi_timings(molecule,'init')
   #
   if (write_time and molecule['mpi_master']): rst_write_time(molecule,'init')
   #
   return molecule

def collect_kernel_mpi_time(molecule,k):
   #
   #  ---  master/slave routine
   #
   timer_mpi(molecule,'mpi_time_idle_kernel',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_idle_kernel',k,True)
   #
   collect_mpi_timings(molecule,'kernel')
   #
   if (molecule['mpi_master']): rst_write_time(molecule,'kernel')
   #
   return molecule

def collect_final_mpi_time(molecule,k):
   #
   #  ---  master/slave routine
   #
   timer_mpi(molecule,'mpi_time_idle_final',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_idle_final',k,True)
   #
   collect_mpi_timings(molecule,'final')
   #
   if (molecule['mpi_master']): rst_write_time(molecule,'final')
   #
   return molecule

