#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_screening_mpi.py: MPI screening routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI
from copy import deepcopy

from bg_mpi_time import timer_mpi
from bg_mpi_utils import enum

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def bcast_tuples(molecule,buff,tup,order):
   #
   #  ---  master/slave routine
   #
   # bcast total number of tuples
   #
   if (molecule['mpi_master']):
      #
      timer_mpi(molecule,'mpi_time_comm_screen',order)
      #
      tup_info = {'tup_len': len(buff)}
      #
      molecule['mpi_comm'].bcast(tup_info,root=0)
   #
   timer_mpi(molecule,'mpi_time_idle_screen',order)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_comm_screen',order)
   #
   # bcast buffer
   #
   molecule['mpi_comm'].Bcast([buff,MPI.INT],root=0)
   #
   timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   # append tup[-1] with buff
   #
   if (len(buff) >= 1): tup.append(buff)
   #
   timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return tup

def tuple_generation_master(molecule,tup,l_limit,u_limit,order,level):
   #
   #  ---  master routine
   #
   # wake up slaves
   #
   timer_mpi(molecule,'mpi_time_idle_screen',order)
   #
   msg = {'task': 'tuple_generation_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'level': level}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   # init job_info dictionary
   #
   job_info = {}
   #
   # number of slaves
   #
   num_slaves = molecule['mpi_size']-1
   #
   # number of available slaves
   #
   slaves_avail = num_slaves
   #
   # define mpi message tags
   #
   tags = enum('ready','done','exit','start')
   #
   # init job index
   #
   i = 0
   #
   # init tmp list
   #
   tmp = []
   #
   while (slaves_avail >= 1):
      #
      # receive data dict
      #
      timer_mpi(molecule,'mpi_time_idle_screen',order)
      #
      data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=molecule['mpi_stat'])
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      # probe for source
      #
      source = molecule['mpi_stat'].Get_source()
      #
      # probe for tag
      #
      tag = molecule['mpi_stat'].Get_tag()
      #
      if (tag == tags.ready):
         #
         if (i <= len(molecule['parent_tup'])-1):
            #
            job_info['parent_tup'] = molecule['parent_tup'][i]
            #
            # send parent tuple index
            #
            timer_mpi(molecule,'mpi_time_comm_screen',order)
            #
            molecule['mpi_comm'].send(job_info,dest=source,tag=tags.start)
            #
            timer_mpi(molecule,'mpi_time_work_screen',order)
            #
            # increment job index
            #
            i += 1
         #
         else:
            #
            timer_mpi(molecule,'mpi_time_comm_screen',order)
            #
            molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
            #
            timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      elif (tag == tags.done):
         #
         # write tmp child tuple list
         #
         tmp += data['child_tup'] 
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   # finally we sort the tuples or mark expansion as converged 
   #
   if (len(tmp) >= 1):
      #
      tmp.sort()
   #
   else:
      #
      molecule['conv_orb'].append(True)
   #
   # make numpy array out of tmp
   #
   buff = np.array(tmp,dtype=np.int32)
   #
   # bcast buff
   #
   bcast_tuples(molecule,buff,tup,order)
   #
   del tmp
   #
   return molecule, tup

def tuple_generation_slave(molecule,tup,l_limit,u_limit,order,level):
   #
   #  ---  slave routine
   #
   timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   # define mpi message tags
   #
   tags = enum('ready','done','exit','start')
   #
   # init data dict
   #
   data = {'child_tup': []}
   #
   while True:
      #
      # ready for task
      #
      timer_mpi(molecule,'mpi_time_comm_screen',order)
      #
      molecule['mpi_comm'].send(None,dest=0,tag=tags.ready)
      #
      # receive parent tuple
      #
      job_info = molecule['mpi_comm'].recv(source=0,tag=MPI.ANY_SOURCE,status=molecule['mpi_stat'])
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      # recover tag
      #
      tag = molecule['mpi_stat'].Get_tag()
      #
      # do job or break out (exit)
      #
      if (tag == tags.start):
         #
         data['child_tup'][:] = []
         #
         # loop through possible orbitals to augment the parent tuple with
         #
         for m in range(job_info['parent_tup'][-1]+1,(l_limit+u_limit)+1):
            #
            # append the child tuple to the tup list
            #
            data['child_tup'].append(list(deepcopy(job_info['parent_tup'])))
            #
            data['child_tup'][-1].append(m)
         #
         timer_mpi(molecule,'mpi_time_comm_screen',order)
         #
         # send child tuple back to master
         #
         molecule['mpi_comm'].send(data,dest=0,tag=tags.done)
         #
         timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      elif (tag == tags.exit):
         #
         break
   #
   # exit
   #
   timer_mpi(molecule,'mpi_time_comm_screen',order)
   #
   molecule['mpi_comm'].send(None,dest=0,tag=tags.exit)
   #
   # init buffer
   #
   tup_info = molecule['mpi_comm'].bcast(None,root=0)
   #
   timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   buff = np.empty([tup_info['tup_len'],order+1],dtype=np.int32)
   #
   # receive buffer
   #
   bcast_tuples(molecule,buff,tup,order)
   #
   data.clear()
   #
   return molecule, tup

