#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_orbitals.py: MPI orbital-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import timer_mpi

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def bcast_tuples_master(molecule,tup,k,level):
   #
   #  ---  master routine
   #
   print('master in bcast_tuples_master')
   #
   # wake up slaves
   #
   timer_mpi(molecule,'mpi_time_comm_init',k)
   #
   msg = {'task': 'bcast_tuples', 'order': k, 'level': level}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # bcast total number of tuples
   #
   tup_info = {'tot_tup': len(tup[k-1])}
   #
   molecule['mpi_comm'].bcast(tup_info,root=0)
   #
   # bcast the tuples
   #
   molecule['mpi_comm'].Bcast([tup[k-1],MPI.INT],root=0)
   #
   timer_mpi(molecule,'mpi_time_comm_init',k,True)
   #
   msg.clear()
   tup_info.clear()
   #
   return

def bcast_tuples_slave(molecule,tup,k):
   #
   #  ---  slave routine
   #
   print('slave {0:} in bcast_tuples_slave'.format(molecule['mpi_rank']))
   #
   # receive the total number of tuples
   #
   timer_mpi(molecule,'mpi_time_idle_init',k)
   #
   tup_info = MPI.COMM_WORLD.bcast(None,root=0)
   #
   # init tup[k-1]
   #
   timer_mpi(molecule,'mpi_time_work_init',k)
   #
   tup.append(np.empty([tup_info['tot_tup'],k],dtype=np.int))
   #
   # receive the tuples
   #
   timer_mpi(molecule,'mpi_time_comm_init',k)
   #
   MPI.COMM_WORLD.Bcast([tup[k-1],MPI.INT],root=0)
   #
   timer_mpi(molecule,'mpi_time_comm_init',k,True)
   #
   tup_info.clear()
   #
   return molecule


