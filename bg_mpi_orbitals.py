#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_orbitals.py: MPI orbital-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI
from itertools import combinations
from copy import deepcopy

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
   if ((k <= 2)):
      #
      # wake up slaves
      #
      msg = {'task': 'bcast_tuples', 'order': k}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      start_work = MPI.Wtime() 
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
      # collect mpi_time_work_init
      #
      molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
      #
      # bcast final dummy message to collect idle slave time
      #
      start_comm = MPI.Wtime()
      #
      final_msg = {'done': None}
      #
      molecule['mpi_comm'].bcast(final_msg,root=0)
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
      # collect mpi_time_comm_init
      #
      molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
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
      tags = enum('ready','comm','done','exit','start')
      #
      # init job index
      #
      i = 0
      #
      # wake up slaves
      #
      start_comm = MPI.Wtime()
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
      # collect mpi_time_comm_init
      #
      molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
      #
      # init tmp list
      #
      tmp = []
      #
      start_work = MPI.Wtime()
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
         # collect mpi_time_work_init
         #
         molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
         #
         # receive data dict
         #
         start_idle = MPI.Wtime()
         #
         data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=molecule['mpi_stat'])
         #
         # collect mpi_time_idle_init
         #
         molecule['mpi_time_idle_init'] += MPI.Wtime()-start_idle
         #
         start_work = MPI.Wtime()
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
               # collect mpi_time_work_init
               #
               molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
               #
               # send parent tuple
               #
               start_comm = MPI.Wtime()
               #
               # store index
               #
               job_info['index'] = i
               #
               molecule['mpi_comm'].send(job_info,dest=source,tag=tags.start)
               #
               # collect mpi_time_comm_init
               #
               molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
               #
               start_work = MPI.Wtime()
               #
               # increment job index
               #
               i += 1
            #
            else:
               #
               # collect mpi_time_work_init
               #
               molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
               #
               start_comm = MPI.Wtime()
               #
               molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
               #
               # collect mpi_time_comm_init
               #
               molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
               #
               start_work = MPI.Wtime()
         #
         elif (tag == tags.comm):
            #
            # collect mpi_time_work_init
            #
            molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
            #
            # receive child_tup for source
            #
            start_comm = MPI.Wtime()
            #
            data = molecule['mpi_comm'].recv(source=source,tag=tags.done,status=molecule['mpi_stat'])
            #
            # collect mpi_time_comm_init
            #
            molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
            #
            # write tmp child tuple list
            #
            start_work = MPI.Wtime()
            #
            tmp += data['child_tup'] 
         #
         elif (tag == tags.exit):
            #
            # collect mpi_time_work_init
            #
            molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
            #
            start_work = MPI.Wtime()
            #
            slaves_avail -= 1
      #
      # collect mpi_time_work_init
      #
      molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
      #
      # bcast final dummy message to collect idle slave time
      #
      start_comm = MPI.Wtime()
      #
      final_msg = {'done': None}
      #
      molecule['mpi_comm'].bcast(final_msg,root=0)
      #
      # collect mpi_time_comm_init
      #
      molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
      #
      # finally we sort the tuples
      #
      start_work = MPI.Wtime()
      #
      if (len(tmp) >= 1): tmp.sort()
      #
      # append tup[k-1] with numpy array of tmp list
      #
      tup.append(np.array(tmp,dtype=np.int))
      #
      # collect mpi_time_work_init
      #
      molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
      #
      # bcast total number of tuples
      #
      start_comm = MPI.Wtime()
      #
      tup_info = {'tot_tup': len(tup[k-1])}
      #
      molecule['mpi_comm'].bcast(tup_info,root=0)
      #
      # bcast the tuples
      #
      molecule['mpi_comm'].Bcast([tup[k-1],MPI.INT],root=0)
      #
      # collect mpi_time_comm_init
      #
      molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
      #
      dom_info.clear()
      #
      del parent_tup
   #
   tup_info.clear()
   final_msg.clear()
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
   tags = enum('ready','comm','done','exit','start')
   #
   # prepare msg dict
   #
   msg = {'done': None}
   #
   # init data dict
   #
   data = {'child_tup': []}
   #
   # init tmp lists
   #
   tmp = []
   #
   start_work = MPI.Wtime()
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
      # collect mpi_time_work_init
      #
      molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
      #
      # ready for task
      #
      start_comm = MPI.Wtime()
      #
      molecule['mpi_comm'].send(None,dest=0,tag=tags.ready)
      #
      # receive parent tuple index
      #
      job_info = molecule['mpi_comm'].recv(source=0,tag=MPI.ANY_SOURCE,status=molecule['mpi_stat'])
      #
      # collect mpi_time_comm_init
      #
      molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
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
         # collect mpi_time_work_init
         #
         molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
         #
         start_comm = MPI.Wtime()
         #
         # prepare master for communication (to distinguish between idle and comm time on master)
         #
         molecule['mpi_comm'].send(msg,dest=0,tag=tags.comm)
         #
         # send child tuple back to master
         #
         molecule['mpi_comm'].send(data,dest=0,tag=tags.done)
         #
         # collect mpi_time_comm_init
         #
         molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
      #
      elif (tag == tags.exit):
         #
         # collect mpi_time_work_init
         #
         molecule['mpi_time_work_init'] += MPI.Wtime()-start_work
         #
         break
   #
   # exit
   #
   start_comm = MPI.Wtime()
   #
   molecule['mpi_comm'].send(None,dest=0,tag=tags.exit)
   #
   # collect mpi_time_comm_init
   #
   molecule['mpi_time_comm_init'] += MPI.Wtime()-start_comm
   #
   data.clear()
   msg.clear()
   #
   del tmp
   del parent_tup
   #
   return molecule

def enum(*sequential,**named):
   #
   # hardcoded enums
   #
   enums = dict(zip(sequential,range(len(sequential))),**named)
   #
   return type('Enum',(), enums)


