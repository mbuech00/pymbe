#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_kernels.py: MPI kernels for Bethe-Goldstone correlation calculations."""

from os import getcwd, mkdir, chdir
from shutil import copy, rmtree
from mpi4py import MPI

from bg_utilities import run_calc_corr, orb_string 
from bg_print import print_status 

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.3'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def main_slave_rout(molecule):
   #
   #  ---  slave routine
   #
   slave = True
   #
   while (slave):
      #
      # start time
      #
      start_idle = MPI.Wtime()
      #
      msg = MPI.COMM_WORLD.bcast(None,root=0)
      #
      if (msg['task'] == 'bcast_mol_dict'):
         #
         # receive molecule dict from master
         #
         start_comm = MPI.Wtime()
         #
         molecule = MPI.COMM_WORLD.bcast(None,root=0)
         #
         molecule['mpi_time_comm'] = MPI.Wtime()-start_comm
         #
         # overwrite wrk_dir in case this is different from the one on the master node
         #
         molecule['wrk'] = getcwd()
         #
         # update with private mpi info
         #
         molecule['mpi_comm'] = MPI.COMM_WORLD
         molecule['mpi_size'] = molecule['mpi_comm'].Get_size()
         molecule['mpi_rank'] = molecule['mpi_comm'].Get_rank()
         molecule['mpi_name'] = MPI.Get_processor_name()
         molecule['mpi_stat'] = MPI.Status()
         #
         molecule['mpi_master'] = False
      #
      elif (msg['task'] == 'print_mpi_table'):
         #
         molecule['mpi_time_idle'] += MPI.Wtime()-start_idle
         #
         print_mpi_table(molecule)
      #
      elif (msg['task'] == 'init_slave_env'):
         #
         molecule['mpi_time_idle'] = MPI.Wtime()-start_idle
         #
         start_work = MPI.Wtime()
         #
         # private scr dir
         #
         molecule['scr'] = molecule['wrk']+'/'+molecule['scr_name']+'-'+str(molecule['mpi_rank'])
         #
         # init scr env
         #
         mkdir(molecule['scr'])
         #
         chdir(molecule['scr'])
         #
         molecule['mpi_time_work'] = MPI.Wtime()-start_work
      #
      elif (msg['task'] == 'energy_calc_mono_exp'):
         #
         molecule['mpi_time_idle'] += MPI.Wtime()-start_idle
         #
         energy_calc_slave(molecule)
      #
      elif (msg['task'] == 'remove_slave_env'):
         #
         molecule['mpi_time_idle'] += MPI.Wtime()-start_idle
         #
         start_work = MPI.Wtime()
         #
         # remove scr env
         #
         chdir(molecule['wrk'])
         #
         if (molecule['error'][0][-1]):
            #
            copy(molecule['scr']+'/OUTPUT.OUT',molecule['wrk']+'/OUTPUT.OUT')
         #
         rmtree(molecule['scr'],ignore_errors=True)
         #
         molecule['mpi_time_work'] += MPI.Wtime()-start_work
      #
      elif (msg['task'] == 'red_mpi_timings'):
         #
         molecule['mpi_time_idle'] += MPI.Wtime()-start_idle
         #
         # reduce mpi timings onto master (cannot time this reduction)
         #
         # define sum operation for dicts
         #
         dict_sum_op = MPI.Op.Create(add_dict,commute=True)
         #
         time = {}
         #
         time['time_idle'] = molecule['mpi_time_idle']
         #
         time['time_comm'] = molecule['mpi_time_comm']
         #
         time['time_work'] = molecule['mpi_time_work']
         #
         molecule['mpi_comm'].reduce(time,op=dict_sum_op,root=0)
         #
      elif (msg['task'] == 'finalize_mpi'):
         #
         slave = False
   #
   return molecule

def bcast_mol_dict(molecule):
   #
   #  ---  master routine
   #
   msg = {'task': 'bcast_mol_dict'}
   #
   MPI.COMM_WORLD.bcast(msg,root=0)
   #
   # bcast molecule dict
   #
   MPI.COMM_WORLD.bcast(molecule,root=0)
   #
   # private mpi info
   #
   molecule['mpi_comm'] = MPI.COMM_WORLD
   molecule['mpi_size'] = molecule['mpi_comm'].Get_size()
   molecule['mpi_rank'] = molecule['mpi_comm'].Get_rank()
   molecule['mpi_name'] = MPI.Get_processor_name()
   molecule['mpi_stat'] = MPI.Status()
   #
   # private scr dir
   #
   molecule['scr'] = molecule['wrk']+'/'+molecule['scr_name']+'-'+str(molecule['mpi_rank'])
   #
   return molecule

def init_slave_env(molecule):
   #
   #  ---  master routine
   #
   msg = {'task': 'init_slave_env'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   return

def remove_slave_env(molecule):
   #
   #  ---  master routine
   #
   msg = {'task': 'remove_slave_env'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   return

def energy_calc_mono_exp_master(molecule,order,tup,n_tup,l_limit,u_limit,level):
   #
   string = {'drop': ''}
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
   msg = {'task': 'energy_calc_mono_exp'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   while (slaves_avail >= 1):
      #
      # write string
      #
      if (i <= (n_tup[order-1]-1)): orb_string(molecule,l_limit,u_limit,tup[order-1][i][0],string)
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
            string['index'] = i
            #
            # send string dict
            #
            molecule['mpi_comm'].send(string,dest=source,tag=tags.start)
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
         tup[order-1][data['index']].append(data['e_tmp'])
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
            print('problem with slave '+str(source)+' in energy_calc_mono_exp_par  ---  aborting...')
            #
            molecule['error'][0].append(True)
            #
            return molecule, tup
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   return molecule, tup

def energy_calc_slave(molecule):
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
      string = molecule['mpi_comm'].recv(source=0,tag=MPI.ANY_SOURCE,status=molecule['mpi_stat'])
      #
      molecule['mpi_time_comm'] += MPI.Wtime()-start_comm
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
         run_calc_corr(molecule,string['drop'],level)
         #
         # write e_tmp
         #
         data['e_tmp'] = molecule['e_tmp']
         #
         # copy job index / indices
         #
         data['index'] = string['index']
         #
         # write error logical
         #
         data['error'] = molecule['error'][0][-1]
         #
         molecule['mpi_time_work'] += MPI.Wtime()-start_work
         #
         start_comm = MPI.Wtime()
         #
         # send data back to master
         #
         molecule['mpi_comm'].send(data,dest=0,tag=tags.done)
         #
         molecule['mpi_time_comm'] += MPI.Wtime()-start_comm
      #
      elif (tag == tags.exit):
         #    
         molecule['mpi_time_work'] += MPI.Wtime()-start_work
         #
         break
   #
   # exit
   #
   start_comm = MPI.Wtime()
   #
   molecule['mpi_comm'].send(None,dest=0,tag=tags.exit)
   #
   molecule['mpi_time_comm'] += MPI.Wtime()-start_comm
   #
   return molecule

def print_mpi_table(molecule):
   #
   if (molecule['mpi_master']):
      #
      msg = {'task': 'print_mpi_table'}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      full_info = []
      #
      for i in range(0,molecule['mpi_size']-1):
         #
         info = molecule['mpi_comm'].recv(source=i+1,status=molecule['mpi_stat'])
         #
         full_info.append([info['rank'],info['name']])
   #
   else:
      #
      info = {}
      #
      info['rank'] = molecule['mpi_rank']
      #
      info['name'] = molecule['mpi_name']
      #
      molecule['mpi_comm'].send(info, dest=0)
      #
      return
   #
   print('')
   print('')
   print('                     ---------------------------------------------                ')
   print('                                  mpi rank/node info                              ')
   print('                     ---------------------------------------------                ')
   print('')
   #
   idx = 0
   #
   while True:
      #
      if ((molecule['mpi_size']-10**idx) < 0):
         #
         width_int = idx+1
         #
         break
      #
      else:
         #
         idx += 1
   #
   width_str = max(map(lambda x: len(x[1]),full_info))
   #
   print(' master  ---  proc =  {0:>{w_int}d}  ---  node =  {1:>{w_str}s}'.format(molecule['mpi_rank'],molecule['mpi_name'],w_int=width_int,w_str=width_str))
   #
   for j in range(0,len(full_info)):
      #
      print(' slave   ---  proc =  {0:>{w_int}d}  ---  node =  {1:>{w_str}s}'.format(full_info[j][0],full_info[j][1],w_int=width_int,w_str=width_str))
   #
   return

def red_mpi_timings(molecule):
   #
   #  ---  master routine
   #
   # define sum operation for dicts
   #
   dict_sum_op = MPI.Op.Create(add_dict,commute=True)
   #
   msg = {'task': 'red_mpi_timings'}
   #
   # wake up slaves
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # receive timings
   #
   time = molecule['mpi_comm'].reduce({},op=dict_sum_op,root=0)
   #
   molecule['mpi_time_idle'] = [(time['time_idle']/float(molecule['mpi_size']-1)),(time['time_idle']/(time['time_idle']+time['time_comm']+time['time_work']))*100.0]
   #
   molecule['mpi_time_comm'] = [(time['time_comm']/float(molecule['mpi_size']-1)),(time['time_comm']/(time['time_idle']+time['time_comm']+time['time_work']))*100.0]
   #
   molecule['mpi_time_work'] = [(time['time_work']/float(molecule['mpi_size']-1)),(time['time_work']/(time['time_idle']+time['time_comm']+time['time_work']))*100.0]
   #
   return

def enum(*sequential,**named):
   #
   enums = dict(zip(sequential,range(len(sequential))),**named)
   #
   return type('Enum',(), enums)

def add_dict(dict_1, dict_2, datatype):
   #
   for item in dict_2:
      #
      if (item in dict_1):
         #
         dict_1[item] += dict_2[item]
      #
      else:
         #
         dict_1[item] = dict_2[item]
   #
   return dict_1


