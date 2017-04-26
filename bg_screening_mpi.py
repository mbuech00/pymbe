#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_screening_mpi.py: MPI screening routines for Bethe-Goldstone correlation calculations."""

from mpi4py import MPI
import numpy as np
from itertools import combinations

from bg_mpi_time import timer_mpi
from bg_mpi_utils import enum

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
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

def tuple_generation_master(molecule,tup,e_inc,thres,l_limit,u_limit,order,level):
   #
   #  ---  master routine
   #
   # wake up slaves
   #
   timer_mpi(molecule,'mpi_time_idle_screen',order)
   #
   msg = {'task': 'tuple_generation_par', 'thres': thres, 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'level': level}
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
   molecule['screen_count'] = 0
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
         if (i <= len(tup[-1])-1):
            #
            job_info['index'] = i
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
         tmp += data['child_tuple'] 
         #
         # increment number of screened tuples
         #
         molecule['screen_count'] += data['screen_count']
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

def tuple_generation_slave(molecule,tup,e_inc,thres,l_limit,u_limit,order,level):
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
   data = {'child_tuple': [], 'screen_count': 0}
   #
   # init combs list
   #
   combs = []
   #
   # determine which tuples have contributions larger than the threshold
   #
   molecule['allow_tuple'] = tup[-1][np.where(np.abs(e_inc[-1]) >= thres)]
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
         data['child_tuple'][:] = []
         #
         data['screen_count'] = 0
         #
         if (np.abs(e_inc[-1][job_info['index']]) >= thres):
            #
            # loop through possible orbitals to augment the parent tuple with
            #
            for m in range(tup[-1][job_info['index']][-1]+1,(l_limit+u_limit)+1): data['child_tuple'].append(tup[-1][job_info['index']].tolist()+[m])
         #
         else:
            #
            # generate list with all subsets of particular tuple
            #
            combs = list(list(comb) for comb in combinations(tup[-1][job_info['index']],order-1))
            #
            # loop through possible orbitals to augment the combinations with
            #
            for m in range(tup[-1][job_info['index']][-1]+1,(l_limit+u_limit)+1):
               #
               screen = True
               #
               for j in range(0,len(combs)):
                  #
                  # check whether or not the particular tuple is actually allowed
                  #
                  if (np.equal(combs[j]+[m],molecule['allow_tuple']).all(axis=1).any()):
                     #
                     screen = False
                     #
                     data['child_tuple'].append(tup[-1][job_info['index']].tolist()+[m])
                     #
                     break
               #
               # if tuple should be screened away, then increment screen counter
               #
               if (screen): data['screen_count'] += 1
         #
         timer_mpi(molecule,'mpi_time_comm_screen',order)
         #
         # send data back to master
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
   del combs
   #
   data.clear()
   #
   return molecule, tup

