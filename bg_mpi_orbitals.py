#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_orbitals.py: MPI orbital-related routines for Bethe-Goldstone correlation calculations."""

from mpi4py import MPI
from itertools import combinations
from copy import deepcopy

from bg_mpi_utilities import enum

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def orb_generator_master(molecule,dom,tup,l_limit,u_limit,order):
   #
   #  ---  master routine
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
   msg = {'task': 'orb_generator_par'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # bcast orbital domains and lower/upper limits
   #
   dom_info = {'dom': dom, 'l_limit': l_limit, 'u_limit': u_limit}
   #
   molecule['mpi_comm'].bcast(dom_info,root=0)
   #
   dom_info.clear()
   #
   tmp = []
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
         if (i <= (len(tup[order-2])-1)):
            #
            job_info['tup_parent'] = tup[order-2][i]
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
         # write child tuple
         #
         for j in range(0,len(data['tup_child'])):
            #
            if ((level == 'MACRO') or ((level == 'CORRE') and (not (data['tup_child'][j] in molecule['prim_tuple'][order-1])))): tmp.append(data['tup_child'][j])
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   # finally we sort the tuples
   #
   if (len(tmp) >= 1): tmp.sort()
   #
   tup[order-1].append(tmp)
   #
   del tmp
   #
   return tup

def orb_generator_slave(molecule,dom,l_limit,u_limit):
   #
   #  ---  slave routine
   #
   level = 'SLAVE'
   #
   # define mpi message tags
   #
   tags = enum('ready','done','exit','start')
   #
   # init data dict
   #
   data = {}
   #
   tmp = []
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
         data['tup_child'] = []
         #
         tmp = list(list(comb) for comb in combinations(job_info['tup_parent'],2))
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
            for m in range(job_info['tup_parent'][-1]+1,(l_limit+u_limit)+1):
               #
               mask2 = True
               #
               for l in job_info['tup_parent']:
                  #
                  # is the new child tuple allowed?
                  #
                  if (not (set([m]) < set(dom[(l-l_limit)-1]))):
                     #
                     mask2 = False
                     #
                     break
               #
               if (mask2):
                  #
                  # append the child tuple to the tup list
                  #
                  data['tup_child'].append(deepcopy(job_info['tup_parent']))
                  #
                  data['tup_child'][-1].append(m)
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
   del tmp
   #
   return molecule



