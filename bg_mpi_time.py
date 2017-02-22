#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_time.py: MPI time-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_utils import add_time 

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def red_mpi_timings(molecule):
   #
   #  ---  master routine
   #
   start_comm = MPI.Wtime()
   #
   # define sum operation for dicts
   #
   time_sum_op = MPI.Op.Create(add_time,commute=True)
   #
   msg = {'task': 'red_mpi_timings'}
   #
   # wake up slaves
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # receive timings
   #
   time = molecule['mpi_comm'].reduce({},op=time_sum_op,root=0)
   #
   # collect mpi_time_comm_master
   #
   molecule['mpi_time_comm_master'] += MPI.Wtime()-start_comm
   #
   sum_slave = time['time_idle_slave']+time['time_comm_slave']+time['time_work_slave']
   #
   molecule['mpi_time_idle'] = [(time['time_idle_slave']/float(molecule['mpi_size']-1)),(time['time_idle_slave']/sum_slave)*100.0]
   #
   molecule['mpi_time_comm'] = [(time['time_comm_slave']/float(molecule['mpi_size']-1)),(time['time_comm_slave']/sum_slave)*100.0]
   #
   molecule['mpi_time_work'] = [(time['time_work_slave']/float(molecule['mpi_size']-1)),(time['time_work_slave']/sum_slave)*100.0]
   #
   time.clear()
   #
   return

