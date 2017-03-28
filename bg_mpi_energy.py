#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_energy.py: MPI energy-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import timer_mpi
from bg_mpi_utils import enum
from bg_rst_write import rst_write_e_inc, rst_write_time
from bg_utils import run_calc_corr, orb_string, comb_index
from bg_print import print_status 

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_kernel_mono_exp_master(molecule,order,tup,e_inc,l_limit,u_limit,level):
   #
   #  ---  master routine
   #
   # wake up slaves
   #
   timer_mpi(molecule,'mpi_time_idle_kernel',order)
   #
   msg = {'task': 'energy_kernel_mono_exp_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'level': level}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   timer_mpi(molecule,'mpi_time_work_kernel',order)
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
   tags = enum('ready','done','data','exit','start')
   #
   # init job index
   #
   if (molecule['rst'] and (order == molecule['min_order'])):
      #
      i = np.argmax(e_inc[order-1] == 0.0)
   #
   else:
      #
      i = 0
   #
   # init stat counter
   #
   counter = i
   #
   # init timings
   #
   if ((not molecule['rst']) or (order != molecule['min_order'])):
      #
      for j in range(0,molecule['mpi_size']):
         #
         molecule['mpi_time_work'][1][j].append(0.0)
         molecule['mpi_time_comm'][1][j].append(0.0)
         molecule['mpi_time_idle'][1][j].append(0.0)
   #
   # print 0.0 %
   #
   print_status(0.0,level)
   #
   while (slaves_avail >= 1):
      #
      # receive data dict
      #
      timer_mpi(molecule,'mpi_time_idle_kernel',order)
      #
      stat = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=molecule['mpi_stat'])
      #
      timer_mpi(molecule,'mpi_time_work_kernel',order)
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
         if (i <= (len(tup[order-1])-1)):
            #
            # store job index
            #
            job_info['index'] = i
            #
            # send string dict
            #
            timer_mpi(molecule,'mpi_time_comm_kernel',order)
            #
            molecule['mpi_comm'].send(job_info,dest=source,tag=tags.start)
            #
            timer_mpi(molecule,'mpi_time_work_kernel',order)
            #
            # increment job index
            #
            i += 1
         #
         else:
            #
            timer_mpi(molecule,'mpi_time_comm_kernel',order)
            #
            molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
            #
            timer_mpi(molecule,'mpi_time_work_kernel',order)
      #
      elif (tag == tags.done):
         #
         timer_mpi(molecule,'mpi_time_comm_kernel',order)
         #
         data = molecule['mpi_comm'].recv(source=source,tag=tags.data,status=molecule['mpi_stat'])
         #
         timer_mpi(molecule,'mpi_time_work_kernel',order)
         #
         e_inc[order-1][data['index']] = data['energy']
         #
         molecule['mpi_time_work'][1][source][-1] = data['t_work']
         molecule['mpi_time_comm'][1][source][-1] = data['t_comm']
         molecule['mpi_time_idle'][1][source][-1] = data['t_idle']
         #
         if (((data['index']+1) % int(molecule['rst_freq'])) == 0):
            #
            molecule['mpi_time_work'][1][0][-1] = molecule['mpi_time_work_kernel'][-1]
            molecule['mpi_time_comm'][1][0][-1] = molecule['mpi_time_comm_kernel'][-1]
            molecule['mpi_time_idle'][1][0][-1] = molecule['mpi_time_idle_kernel'][-1]
            #
            rst_write_time(molecule,'kernel')
            #
            rst_write_e_inc(molecule,order)
         #
         # increment stat counter
         #
         counter += 1
         #
         # print status
         #
         if (((data['index']+1) % 1000) == 0): print_status(float(counter)/float(len(tup[order-1])),level)
         #
         # error check
         #
         if (data['error']):
            #
            print('problem with slave '+str(source)+' in energy_kernel_mono_exp_master  ---  aborting...')
            #
            molecule['error'].append(True)
            #
            return molecule, tup
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   # print 100.0 %
   #
   print_status(1.0,level)
   #
   timer_mpi(molecule,'mpi_time_work_kernel',order,True)
   #
   return molecule, e_inc

def energy_kernel_mono_exp_slave(molecule,order,tup,e_inc,l_limit,u_limit,level):
   #
   #  ---  slave routine
   #
   timer_mpi(molecule,'mpi_time_work_kernel',order)
   #
   # init e_inc list
   #
   if (order != molecule['min_order']): e_inc.append(np.zeros(len(tup[order-1]),dtype=np.float64))
   #
   # define mpi message tags
   #
   tags = enum('ready','done','data','exit','start')
   #
   # init string dict
   #
   string = {'drop': ''}
   #
   # init data dict
   #
   data = {}
   #
   while True:
      #
      # ready for task
      #
      timer_mpi(molecule,'mpi_time_comm_kernel',order)
      #
      molecule['mpi_comm'].send(None,dest=0,tag=tags.ready)
      #
      # receive drop string
      #
      job_info = molecule['mpi_comm'].recv(source=0,tag=MPI.ANY_SOURCE,status=molecule['mpi_stat'])
      #
      timer_mpi(molecule,'mpi_time_work_kernel',order)
      #
      # recover tag
      #
      tag = molecule['mpi_stat'].Get_tag()
      #
      # do job or break out (exit)
      #
      if (tag == tags.start):
         #
         # write string
         #
         orb_string(molecule,l_limit,u_limit,tup[order-1][job_info['index']],string)
         #
         run_calc_corr(molecule,string['drop'],level)
         #
         # write tuple energy
         #
         e_inc[order-1][job_info['index']] = molecule['e_tmp']
         #
         # report status back to master
         #
         timer_mpi(molecule,'mpi_time_comm_kernel',order)
         #
         molecule['mpi_comm'].send(None,dest=0,tag=tags.done)
         #
         timer_mpi(molecule,'mpi_time_work_kernel',order)
         #
         # write info into data dict
         #
         data['index'] = job_info['index']
         data['energy'] = molecule['e_tmp']
         data['t_work'] = molecule['mpi_time_work_kernel'][-1]
         data['t_comm'] = molecule['mpi_time_comm_kernel'][-1]
         data['t_idle'] = molecule['mpi_time_idle_kernel'][-1]
         data['error'] = molecule['error'][-1]
         #
         # send data back to master
         #
         timer_mpi(molecule,'mpi_time_comm_kernel',order)
         #
         molecule['mpi_comm'].send(data,dest=0,tag=tags.data)
         #
         timer_mpi(molecule,'mpi_time_work_kernel',order)
      #
      elif (tag == tags.exit):
         #    
         break
   #
   # exit
   #
   timer_mpi(molecule,'mpi_time_comm_kernel',order)
   #
   molecule['mpi_comm'].send(None,dest=0,tag=tags.exit)
   #
   timer_mpi(molecule,'mpi_time_comm_kernel',order,True)
   #
   return molecule, e_inc

def energy_summation_par(molecule,k,tup,e_inc,energy,level):
   #
   #  ---  master/slave routine
   #
   if (molecule['mpi_master']):
      #
      # wake up slaves
      #
      timer_mpi(molecule,'mpi_time_idle_final',k)
      #
      msg = {'task': 'energy_summation_par', 'order': k, 'level': level}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      # re-init e_inc[-1] with 0.0
      #
      e_inc[k-1].fill(0.0)
   #
   timer_mpi(molecule,'mpi_time_work_final',k)
   #
   for j in range(0,len(tup[k-1])):
      #
      if (e_inc[k-1][j] != 0.0):
         #
         for i in range(k-1,0,-1):
            #
            combs = tup[k-1][j,comb_index(k,i)]
            #
            if (level == 'CORRE'):
               #
               if (len(tup[i-1]) > 0):
                  #
                  dt = np.dtype((np.void,tup[i-1].dtype.itemsize*tup[i-1].shape[1]))
                  #
                  idx = np.nonzero(np.in1d(tup[i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
                  #
                  for l in idx: e_inc[k-1][j] -= e_inc[i-1][l]
               #
               dt = np.dtype((np.void,molecule['prim_tuple'][i-1].dtype.itemsize*molecule['prim_tuple'][i-1].shape[1]))
               #
               idx = np.nonzero(np.in1d(molecule['prim_tuple'][i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
               #
               for l in idx: e_inc[k-1][j] -= molecule['prim_energy_inc'][i-1][l]
            #
            elif (level == 'MACRO'):
               #
               dt = np.dtype((np.void,tup[i-1].dtype.itemsize*tup[i-1].shape[1]))
               #
               idx = np.nonzero(np.in1d(tup[i-1].view(dt).reshape(-1),combs.view(dt).reshape(-1)))[0]
               #
               for l in idx: e_inc[k-1][j] -= e_inc[i-1][l]
   #
   # allreduce e_inc[-1]
   #
   allred_e_inc(molecule,e_inc,k)
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

def allred_e_inc(molecule,e_inc,k):
   #
   # Allreduce e_inc[-1] (here: do explicit Reduce+Bcast, as Allreduce has been observed to hang)
   #
   timer_mpi(molecule,'mpi_time_idle_final',k)
   #
   molecule['mpi_comm'].Barrier()
   #
   timer_mpi(molecule,'mpi_time_comm_final',k)
   #
   # init receive buffer
   #
   recv_buff = np.zeros(len(e_inc[k-1]),dtype=np.float64)
   #
   # now do Allreduce
   #
   molecule['mpi_comm'].Allreduce([e_inc[k-1],MPI.DOUBLE],[recv_buff,MPI.DOUBLE],op=MPI.SUM)
   #
   # finally, overwrite e_inc[k-1]
   #
   timer_mpi(molecule,'mpi_time_work_final',k)
   #
   e_inc[k-1] = recv_buff
   #
   del recv_buff
   #
   return e_inc


