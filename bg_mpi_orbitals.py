#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_orbitals.py: MPI orbital-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI
from itertools import combinations
from copy import deepcopy

from bg_mpi_time import timer_mpi
from bg_mpi_utils import enum

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.6'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def bcast_domains(molecule,dom,k):
   #
   #  ---  master/slave routine
   #
   timer_mpi(molecule,'mpi_time_idle_screen',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_comm_screen',k)
   #
   # bcast domains
   #
   if (molecule['mpi_master']):
      #
      dom_info = {'dom': dom[-1]}
      #
      molecule['mpi_comm'].bcast(dom_info,root=0)
   #
   else:
      #
      dom_info = molecule['mpi_comm'].bcast(None,root=0)
      #
      dom[:] = []
      #
      dom += dom_info['dom']
   #
   timer_mpi(molecule,'mpi_time_comm_screen',k,True)
   #
   dom_info.clear()
   #
   return dom

def bcast_tuples(molecule,buff,tup,k):
   #
   #  ---  master/slave routine
   #
   # bcast total number of tuples
   #
   if (molecule['mpi_master']):
      #
      timer_mpi(molecule,'mpi_time_comm_screen',k)
      #
      tup_info = {'tup_len': len(buff)}
      #
      molecule['mpi_comm'].bcast(tup_info,root=0)
   #
   timer_mpi(molecule,'mpi_time_idle_screen',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_comm_screen',k)
   #
   # bcast buffer
   #
   molecule['mpi_comm'].Bcast([buff,MPI.INT],root=0)
   #
   timer_mpi(molecule,'mpi_time_work_screen',k)
   #
   # append tup[k-1] with buff
   #
   tup.append(buff)
   #
   timer_mpi(molecule,'mpi_time_work_screen',k,True)
   #
   return tup

def orb_generator_master(molecule,dom,tup,l_limit,u_limit,k,level):
   #
   #  ---  master routine
   #
   # wake up slaves
   #
   timer_mpi(molecule,'mpi_time_idle_screen',k)
   #
   msg = {'task': 'orb_generator_slave', 'l_limit': l_limit, 'u_limit': u_limit, 'order': k, 'level': level}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   timer_mpi(molecule,'mpi_time_work_screen',k)
   #
   # bcast domains
   #
   bcast_domains(molecule,dom,k)
   #
   timer_mpi(molecule,'mpi_time_work_screen',k)
   #
   # init job_info dictionary
   #
   job_info = {}
   #
   # number of slaves
   #
   num_slaves = molecule['mpi_size'] - 1
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
      timer_mpi(molecule,'mpi_time_idle_screen',k)
      #
      data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=molecule['mpi_stat'])
      #
      timer_mpi(molecule,'mpi_time_work_screen',k)
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
         if (i <= len(tup[k-1])-1):
            #
            job_info['index'] = i
            #
            # send parent tuple index
            #
            timer_mpi(molecule,'mpi_time_comm_screen',k)
            #
            molecule['mpi_comm'].send(job_info,dest=source,tag=tags.start)
            #
            timer_mpi(molecule,'mpi_time_work_screen',k)
            #
            # increment job index
            #
            i += 1
         #
         else:
            #
            timer_mpi(molecule,'mpi_time_comm_screen',k)
            #
            molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
            #
            timer_mpi(molecule,'mpi_time_work_screen',k)
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
   # finally we sort the tuples
   #
   if (len(tmp) >= 1): tmp.sort()
   #
   # make numpy array out of tmp
   #
   buff = np.array(tmp,dtype=np.int32)
   #
   # bcast buff
   #
   bcast_tuples(molecule,buff,tup,k)
   #
   del tmp
   #
   return tup

def orb_generator_slave(molecule,dom,tup,l_limit,u_limit,k,level):
   #
   #  ---  slave routine
   #
   # receive domains
   #
   bcast_domains(molecule,dom,k)
   #
   timer_mpi(molecule,'mpi_time_work_screen',k)
   #
   # define mpi message tags
   #
   tags = enum('ready','done','exit','start')
   #
   # init data dict
   #
   data = {'child_tup': []}
   #
   # init tmp lists
   #
   tmp = []
   #
   while True:
      #
      # ready for task
      #
      timer_mpi(molecule,'mpi_time_comm_screen',k)
      #
      molecule['mpi_comm'].send(None,dest=0,tag=tags.ready)
      #
      # receive parent tuple
      #
      job_info = molecule['mpi_comm'].recv(source=0,tag=MPI.ANY_SOURCE,status=molecule['mpi_stat'])
      #
      timer_mpi(molecule,'mpi_time_work_screen',k)
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
         # generate subset of all pairs within the parent tuple
         #
         parent_tup = tup[k-1][job_info['index']]
         #
         tmp = list(list(comb) for comb in combinations(parent_tup,2))
         #
         mask = True
         #
         for j in range(0,len(tmp)):
            #
            # is the parent tuple still allowed?
            #
            if (not (set([tmp[j][1]]) < set(dom[(tmp[j][0]-l_limit)-1]))):
               #
               mask = False
               #
               break
         #
         if (mask):
            #
            # loop through possible orbitals to augment the parent tuple with
            #
            for m in range(parent_tup[-1]+1,(l_limit+u_limit)+1):
               #
               mask_2 = True
               #
               for l in parent_tup:
                  #
                  # is the new child tuple allowed?
                  #
                  if (not (set([m]) < set(dom[(l-l_limit)-1]))):
                     #
                     mask_2 = False
                     #
                     break
               #
               if (mask_2):
                  #
                  # append the child tuple to the tup list
                  #
                  data['child_tup'].append(list(deepcopy(parent_tup)))
                  #
                  data['child_tup'][-1].append(m)
         #
         timer_mpi(molecule,'mpi_time_comm_screen',k)
         #
         # send child tuple back to master
         #
         molecule['mpi_comm'].send(data,dest=0,tag=tags.done)
         #
         timer_mpi(molecule,'mpi_time_work_screen',k)
      #
      elif (tag == tags.exit):
         #
         break
   #
   # exit
   #
   timer_mpi(molecule,'mpi_time_comm_screen',k)
   #
   molecule['mpi_comm'].send(None,dest=0,tag=tags.exit)
   #
   # init buffer
   #
   tup_info = molecule['mpi_comm'].bcast(None,root=0)
   #
   timer_mpi(molecule,'mpi_time_work_screen',k)
   #
   buff = np.empty([tup_info['tup_len'],k+1],dtype=np.int32)
   #
   # receive buffer
   #
   bcast_tuples(molecule,buff,tup,k)
   #
   data.clear()
   #
   del tmp
   #
   return molecule

def red_orb_ent(molecule,tmp,recv_buff,k):
   #
   timer_mpi(molecule,'mpi_time_idle_screen',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   # reduce tmp into recv_buff
   #
   timer_mpi(molecule,'mpi_time_comm_screen',k)
   #
   molecule['mpi_comm'].Reduce([tmp,MPI.DOUBLE],[recv_buff,MPI.DOUBLE],op=MPI.SUM,root=0)
   #
   timer_mpi(molecule,'mpi_time_comm_screen',k,True)
   #
   return recv_buff

def orb_entanglement_main_par(molecule,l_limit,u_limit,order,level,calc_end):
   #
   #  ---  master/slave routine
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      timer_mpi(molecule,'mpi_time_idle_screen',order)
      #
      msg = {'task': 'orb_entanglement_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'level': level, 'calc_end': calc_end}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   tmp = np.zeros([u_limit,u_limit],dtype=np.float64)
   #
   for l in range(0,len(molecule['prim_tuple'][order-1])):
      #
      # simple modulo distribution of tasks
      #
      if ((l % molecule['mpi_size']) == molecule['mpi_rank']):
         #
         tup = molecule['prim_tuple'][order-1]
         e_inc = molecule['prim_energy_inc'][order-1]
         ldx = l
         #
         for i in range(l_limit,l_limit+u_limit):
            #
            for j in range(i+1,l_limit+u_limit):
               #
               # add up contributions from the correlation between orbs i and j at current order
               #
               if (set([i+1,j+1]) <= set(tup[ldx])):
                  #
                  tmp[i-l_limit,j-l_limit] += e_inc[ldx]
                  tmp[j-l_limit,i-l_limit] = tmp[i-l_limit,j-l_limit]
   #
   # init recv_buff
   #
   if (molecule['mpi_master']):
      #
      recv_buff = np.zeros([u_limit,u_limit],dtype=np.float64)
   #
   else:
      #
      recv_buff = None
   #
   # reduce tmp onto master
   #
   red_orb_ent(molecule,tmp,recv_buff,order)
   #
   if (molecule['mpi_master']):
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      if (level == 'MACRO'):
         #
         molecule['prim_orb_ent'].append(recv_buff)
      #
      timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule

