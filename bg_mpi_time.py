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

def init_mpi_timings(molecule):
   #
   # program phase distribution
   #
   if (molecule['mpi_master']):
      #
      molecule['time_init'] = 0.0
      molecule['time_kernel'] = 0.0
      molecule['time_final'] = 0.0
   #
   # mpi distribution
   #
   if (molecule['mpi_parallel']):
      #
      # init timings
      #
      molecule['mpi_time_idle_init'] = 0.0
      molecule['mpi_time_comm_init'] = 0.0
      molecule['mpi_time_work_init'] = 0.0
      #
      # energy kernel timings
      #
      molecule['mpi_time_idle_kernel'] = 0.0
      molecule['mpi_time_comm_kernel'] = 0.0
      molecule['mpi_time_work_kernel'] = 0.0
      #
      # energy summation timings
      #
      molecule['mpi_time_idle_final'] = 0.0
      molecule['mpi_time_comm_final'] = 0.0
      molecule['mpi_time_work_final'] = 0.0
   #
   return molecule

def collect_mpi_timings(molecule):
   #
   #
   #  ---  master/slave routine
   #
   if (molecule['mpi_master']):
      #
      # first, calculate time_remainder
      #
      molecule['time_remain'] = molecule['time_tot']-(molecule['time_init']+molecule['time_kernel']+molecule['time_final'])
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

