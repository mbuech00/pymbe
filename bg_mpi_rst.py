#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_rst.py: mpi restart utilities for Bethe-Goldstone correlation calculations."""

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

def rst_dist_master(molecule):
   # 
   #  ---  master routine
   #
   # wake up slaves
   #
   msg = {'task': 'rst_dist'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # send info
   #
   e_inc_end = np.argmax(molecule['prim_energy_inc'][-1] == 0.0)
   #
   info = {'len_tup': [len(molecule['prim_tuple'][i]) for i in range(1,len(molecule['prim_tuple']))],\
           'len_e_inc': [len(molecule['prim_energy_inc'][i]) for i in range(0,len(molecule['prim_energy_inc']))],\
           'min_order': molecule['min_order'], 'e_inc_end': e_inc_end}
   #
   molecule['mpi_comm'].bcast(info,root=0)
   #
   # bcast tuples
   #
   for i in range(1,len(molecule['prim_tuple'])):
      #
      molecule['mpi_comm'].Bcast([molecule['prim_tuple'][i],MPI.INT],root=0)
   #
   # bcast e_inc
   #
   for i in range(0,len(molecule['prim_energy_inc'])):
      #
      if (i < (len(molecule['prim_energy_inc'])-1)):
         #
         molecule['mpi_comm'].Bcast([molecule['prim_energy_inc'][i],MPI.DOUBLE],root=0)
      #
      else:
         #
         molecule['mpi_comm'].Bcast([molecule['prim_energy_inc'][i][:e_inc_end],MPI.DOUBLE],root=0)
   #
   # send timings
   #
   for i in range(1,molecule['mpi_size']):
      #
      time = {'init': [molecule['mpi_time_work'][0][i],molecule['mpi_time_comm'][0][i],molecule['mpi_time_idle'][0][i]],\
              'kernel': [molecule['mpi_time_work'][1][i],molecule['mpi_time_comm'][1][i],molecule['mpi_time_idle'][1][i]],\
              'final': [molecule['mpi_time_work'][2][i],molecule['mpi_time_comm'][2][i],molecule['mpi_time_idle'][2][i]]}
      #
      molecule['mpi_comm'].send(time,dest=i)
   #
   time.clear()
   #
   return molecule

def rst_dist_slave(molecule):
   #
   # receive info
   #
   info = molecule['mpi_comm'].bcast(None,root=0)
   #
   # set min_order
   #
   molecule['min_order'] = info['min_order']
   #
   # receive tuples
   #
   for i in range(0,len(info['len_tup'])):
      #
      buff = np.empty([info['len_tup'][i],i+2],dtype=np.int32)
      #
      molecule['mpi_comm'].Bcast([buff,MPI.INT],root=0)
      #
      molecule['prim_tuple'].append(buff)
   #
   # receive e_inc
   #
   for i in range(0,len(info['len_e_inc'])):
      #
      buff = np.zeros(info['len_e_inc'][i],dtype=np.float64)
      #
      if (i < (len(info['len_e_inc'])-1)):
         #
         molecule['mpi_comm'].Bcast([buff,MPI.DOUBLE],root=0)
      #
      else:
         #
         molecule['mpi_comm'].Bcast([buff[:info['e_inc_end']],MPI.DOUBLE],root=0)
      #
      molecule['prim_energy_inc'].append(buff)
   #
   # for e_inc[-1], make sure that this is distributed among the slaves
   #
   for i in range(0,info['e_inc_end']):
      #
      if ((i % (molecule['mpi_size']-1)) != (molecule['mpi_rank']-1)): molecule['prim_energy_inc'][-1][i] = 0.0 
   #
   # receive timings
   #
   time = molecule['mpi_comm'].recv(source=0,status=molecule['mpi_stat'])
   #
   molecule['mpi_time_work_init'] = time['init'][0]; molecule['mpi_time_comm_init'] = time['init'][1]; molecule['mpi_time_idle_init'] = time['init'][2]
   molecule['mpi_time_work_kernel'] = time['kernel'][0]; molecule['mpi_time_comm_kernel'] = time['kernel'][1]; molecule['mpi_time_idle_kernel'] = time['kernel'][2]
   molecule['mpi_time_work_final'] = time['final'][0]; molecule['mpi_time_comm_final'] = time['final'][1]; molecule['mpi_time_idle_final'] = time['final'][2]
   #
   del buff
   #
   time.clear()
   #
   return molecule


