#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_orbitals.py: MPI orbital-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI
from itertools import combinations
from copy import deepcopy

from bg_mpi_utils import enum

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def orb_generator_master(molecule,dom,tup,l_limit,u_limit,k,level):
   #
   #  ---  master routine
   #
   if ((k <= 2) and molecule['mpi_master']):
      #
      if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']):
         #
         start = molecule['ncore']
      #
      else:
         #
         start = 0
      #
      if (k == 1):
         #
         # all singles contributions
         #
         tmp = []
         #
         for i in range(start,len(dom)):
            #
            tmp.append([(i+l_limit)+1])
         #
         tup.append(np.array(tmp,dtype=np.int))
      #
      elif (k == 2):
         #
         # generate all possible (unique) pairs
         #
         tmp = list(list(comb) for comb in combinations(range(start+(1+l_limit),(l_limit+u_limit)+1),2))
         #
         tup.append(np.array(tmp,dtype=np.int))
      #
      # wake up slaves
      #
      msg = {'task': 'bcast_tuples', 'order': k}
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
   else:
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
      # wake up slaves
      #
      msg = {'task': 'orb_generator_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': k, 'level': level}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      # bcast orbital domains and lower/upper limits
      #
      dom_info = {'dom': dom}
      #
      molecule['mpi_comm'].bcast(dom_info,root=0)
      #
      tmp = []
      #
      # init parent_tup
      #
      if (level == 'MACRO'):
         #
         parent_tup = tup[k-2]
      #
      elif (level == 'CORRE'):
         #
         if (k == molecule['min_corr_order']):
            #
            parent_tup = molecule['prim_tuple'][k-2]
         #
         else:
            #
            parent_tup = np.vstack((tup[k-2],molecule['prim_tuple'][k-2]))
      #
      while (slaves_avail >= 1):
         #
         # receive data dict
         #
         data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=molecule['mpi_stat'])
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
            if (i <= (len(parent_tup)-1)):
               #
               job_info['index'] = i
               #
               # send parent tuple
               #
               molecule['mpi_comm'].send(job_info,dest=source,tag=tags.start)
               #
               # increment job index
               #
               i += 1
            #
            else:
               #
               molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
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
      # append tup[k-1] with numpy array of tmp list
      #
      tup.append(np.array(tmp,dtype=np.int))
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
      dom_info.clear()
      #
      del parent_tup
   #
   tup_info.clear()
   #
   del tmp
   #
   return tup

def orb_generator_slave(molecule,dom,tup,l_limit,u_limit,k,level):
   #
   #  ---  slave routine
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
   # init parent_tup
   #
   if (level == 'MACRO'):
      #
      parent_tup = tup[k-2]
   #
   elif (level == 'CORRE'):
      #
      if (k == molecule['min_corr_order']):
         #
         parent_tup = molecule['prim_tuple'][k-2]
      #
      else:
         #
         parent_tup = np.vstack((tup[k-2],molecule['prim_tuple'][k-2]))
   #
   while True:
      #
      # ready for task
      #
      start_comm = MPI.Wtime()
      #
      molecule['mpi_comm'].send(None,dest=0,tag=tags.ready)
      #
      # receive parent tuple
      #
      job_info = molecule['mpi_comm'].recv(source=0,tag=MPI.ANY_SOURCE,status=molecule['mpi_stat'])
      #
      molecule['mpi_time_comm_slave'] += MPI.Wtime()-start_comm
      #
      start_work = MPI.Wtime()
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
         tmp = list(list(comb) for comb in combinations(parent_tup[job_info['index']],2))
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
            for m in range(parent_tup[job_info['index']][-1]+1,(l_limit+u_limit)+1):
               #
               mask_2 = True
               #
               for l in parent_tup[job_info['index']]:
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
                  data['child_tup'].append(list(deepcopy(parent_tup[job_info['index']])))
                  #
                  data['child_tup'][-1].append(m)
                  #
                  # check whether this tuple has already been accounted for in the primary expansion
                  #
                  if ((level == 'CORRE') and (np.equal(data['child_tup'][-1],molecule['prim_tuple'][k-1]).all(axis=1).any())):
                     #
                     data['child_tup'].pop(-1)
         #
         molecule['mpi_time_work_slave'] += MPI.Wtime()-start_work
         #
         start_comm = MPI.Wtime()
         #
         # send child tuple back to master
         #
         molecule['mpi_comm'].send(data,dest=0,tag=tags.done)
         #
         molecule['mpi_time_comm_slave'] += MPI.Wtime()-start_comm
      #
      elif (tag == tags.exit):
         #
         molecule['mpi_time_work_slave'] += MPI.Wtime()-start_work
         #
         break
   #
   # exit
   #
   start_comm = MPI.Wtime()
   #
   molecule['mpi_comm'].send(None,dest=0,tag=tags.exit)
   #
   molecule['mpi_time_comm_slave'] += MPI.Wtime()-start_comm
   #
   data.clear()
   del tmp
   del parent_tup
   #
   return molecule



