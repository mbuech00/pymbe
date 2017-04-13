#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_kernel_mpi.py: MPI energy kernel routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import timer_mpi
from bg_mpi_utils import enum
from bg_utils import run_calc_corr, term_calc, orb_string
from bg_print import print_status 
from bg_rst_write import rst_write_kernel

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_kernel_master(molecule,tup,e_inc,l_limit,u_limit,order,level):
   #
   #  ---  master routine
   #
   # wake up slaves
   #
   timer_mpi(molecule,'mpi_time_idle_kernel',order)
   #
   msg = {'task': 'energy_kernel_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'level': level}
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
      i = np.argmax(e_inc[-1] == 0.0)
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
         molecule['mpi_time_work'][0][j].append(0.0)
         molecule['mpi_time_comm'][0][j].append(0.0)
         molecule['mpi_time_idle'][0][j].append(0.0)
   #
   # print status for START
   #
   print_status(float(counter)/float(len(tup[-1])),level)
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
         if (i <= (len(tup[-1])-1)):
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
            # send exit signal
            #
            molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
            #
            timer_mpi(molecule,'mpi_time_work_kernel',order)
      #
      elif (tag == tags.done):
         #
         timer_mpi(molecule,'mpi_time_comm_kernel',order)
         #
         # receive data
         #
         data = molecule['mpi_comm'].recv(source=source,tag=tags.data,status=molecule['mpi_stat'])
         #
         timer_mpi(molecule,'mpi_time_work_kernel',order)
         #
         # write to e_inc
         #
         e_inc[-1][data['index']] = data['energy']
         #
         # store timings
         #
         molecule['mpi_time_work'][0][source][-1] = data['t_work']
         molecule['mpi_time_comm'][0][source][-1] = data['t_comm']
         molecule['mpi_time_idle'][0][source][-1] = data['t_idle']
         #
         # write restart files
         #
         if (((data['index']+1) % int(molecule['rst_freq'])) == 0):
            #
            molecule['mpi_time_work'][0][0][-1] = molecule['mpi_time_work_kernel'][-1]
            molecule['mpi_time_comm'][0][0][-1] = molecule['mpi_time_comm_kernel'][-1]
            molecule['mpi_time_idle'][0][0][-1] = molecule['mpi_time_idle_kernel'][-1]
            #
            rst_write_kernel(molecule,e_inc,order)
         #
         # increment stat counter
         #
         counter += 1
         #
         # print status
         #
         if (((data['index']+1) % 1000) == 0): print_status(float(counter)/float(len(tup[-1])),level)
         #
         # error check
         #
         if (data['error']):
            #
            molecule['error'].append(True)
            molecule['error_code'] = data['error_code']
            molecule['error_msg'] = data['error_msg']
            #
            molecule['error_rank'] = source
            #
            string = {}
            #
            orb_string(molecule,l_limit,u_limit,tup[-1][job_info['index']],string)
            #
            molecule['error_drop'] = string['drop']
            # 
            term_calc(molecule)
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

def energy_kernel_slave(molecule,tup,e_inc,l_limit,u_limit,order,level):
   #
   #  ---  slave routine
   #
   timer_mpi(molecule,'mpi_time_work_kernel',order)
   #
   # init e_inc list
   #
   if (order != molecule['min_order']): e_inc.append(np.zeros(len(tup[-1]),dtype=np.float64))
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
         orb_string(molecule,l_limit,u_limit,tup[-1][job_info['index']],string)
         #
         run_calc_corr(molecule,string['drop'],level)
         #
         # write tuple energy
         #
         e_inc[-1][job_info['index']] = molecule['e_tmp']
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
         data['error_code'] = molecule['error_code']
         data['error_msg'] = molecule['error_msg']
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

