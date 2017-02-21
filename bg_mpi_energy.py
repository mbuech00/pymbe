#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_energy.py: MPI energy-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_utilities import run_calc_corr, orb_string 
from bg_print import print_status 
from bg_mpi_utilities import enum, add_tup

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_kernel_mono_exp_master(molecule,order,tup,n_tup,e_inc,l_limit,u_limit,level):
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
   # init stat counter
   #
   counter = 0
   #
   # wake up slaves
   #
   msg = {'task': 'energy_kernel_mono_exp_par'}
   #
   molecule['mpi_comm'].bcast(msg,root=MPI.ROOT)
   #
   while (slaves_avail >= 1):
      #
      # write string
      #
      if (i <= (n_tup[order-1]-1)): orb_string(molecule,l_limit,u_limit,tup[order-1][i],job_info)
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
         if (i <= (n_tup[order-1]-1)):
            #
            # store job index
            #
            job_info['index'] = i
            #
            # send string dict
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
         # write tuple energy
         #
         e_inc[order-1][data['index']] = data['e_tmp']
         #
         # increment stat counter
         #
         counter += 1
         #
         # print status
         #
         print_status(float(counter)/float(n_tup[order-1]),level)
         #
         # error check
         #
         if (data['error']):
            #
            print('problem with slave '+str(source)+' in energy_kernel_mono_exp_master  ---  aborting...')
            #
            molecule['error'].append(True)
            #
            return molecule, tup, e_inc
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   return molecule, tup, e_inc

def energy_kernel_slave(molecule):
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
   while True:
      #
      # ready for task
      #
      start_comm = MPI.Wtime()
      #
      molecule['mpi_comm'].send(None,dest=0,tag=tags.ready)
      #
      # receive drop string
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
         run_calc_corr(molecule,job_info['drop'],level)
         #
         # write e_tmp
         #
         data['e_tmp'] = molecule['e_tmp']
         #
         # copy job index / indices
         #
         data['index'] = job_info['index']
         #
         # write error logical
         #
         data['error'] = molecule['error'][-1]
         #
         molecule['mpi_time_work_slave'] += MPI.Wtime()-start_work
         #
         start_comm = MPI.Wtime()
         #
         # send data back to master
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
   return molecule

def energy_summation_par(molecule,k,tup,e_inc,energy,level):
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      msg = {'task': 'energy_summation_par', 'order': k, 'level': level}
      #
      molecule['mpi_comm'].bcast(msg,root=MPI.ROOT)
   #
   for j in range(0,len(tup[k-1])):
      #
      if (e_inc[k-1][j] != 0.0):
         #
         for i in range(k-1,0,-1):
            #
            for l in range(0,len(tup[i-1])):
               #
               # is tup[i-1][l] a subset of tup[k-1][j] ?
               #
               if (all(idx in iter(loc[k-1][j]) for idx in tup[i-1][l])): e_inc[k-1][j] -= e_inc[i-1][l]
            #
            if (level == 'CORRE'):
               #
               for l in range(0,len(molecule['prim_tuple'][i-1])):
                  #
                  # is molecule['prim_tuple'][i-1][l] a subset of tup[k-1][j] ?
                  #
                  if (all(idx in iter(tup[k-1][j]) for idx in molecule['prim_tuple'][i-1][l])): e_inc[k-1][j] -= molecule['prim_energy_inc'][i-1][l]
   #
   # allreduce e_inc[k-1]
   #
   molecule['mpi_comm'].Allreduce([e_inc[k-1],MPI.DOUBLE],[e_inc[k-1],MPI.DOUBLE],op=MPI.SUM)
   #
   # let master calculate the total energy
   #
   if (molecule['mpi_master']):
      #
      # sum of energy increment of level k
      #
      e_tmp = np.sum(e_inc[k-1])
      #
      # sum of total energy
      #
      if (k > 1):
         #
         e_tmp += energy[k-2]
      #
      energy.append(e_tmp)
   #
   return e_inc, energy



