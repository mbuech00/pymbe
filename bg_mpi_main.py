#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_main.py: main MPI driver routined for Bethe-Goldstone correlation calculations."""

from os import getcwd, mkdir, chdir
from shutil import copy, rmtree
from mpi4py import MPI

from bg_mpi_energy import energy_kernel_slave, energy_summation_par
from bg_mpi_orbitals import orb_generator_slave 
from bg_mpi_utilities import add_time 

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def init_mpi(molecule):
   #
   #  ---  master and slave routine
   #
   if (MPI.COMM_WORLD.Get_size() > 1):
      #
      molecule['mpi_parallel'] = True
   #
   else:
      #
      molecule['mpi_parallel'] = False
   #
   # slave proceed to the main slave routine
   #
   if (MPI.COMM_WORLD.Get_rank() != 0):
      #
      main_slave_rout(molecule)
   #
   else:
      #
      molecule['mpi_master'] = True
   #
   return molecule

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
         # init mpi_time_comm_slave
         #
         molecule['mpi_time_comm_slave'] = MPI.Wtime()-start_comm
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
      elif (msg['task'] == 'init_slave_env'):
         #
         # init mpi_time_idle_slave
         #
         molecule['mpi_time_idle_slave'] = MPI.Wtime()-start_idle
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
         # init mpi_time_work_slave
         #
         molecule['mpi_time_work_slave'] = MPI.Wtime()-start_work
         #
         # init tuple lists
         #
         molecule['prim_tuple'] = []
         molecule['corr_tuple'] = []
      #
      elif (msg['task'] == 'print_mpi_table'):
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         print_mpi_table(molecule)
      #
      elif (msg['task'] == 'orb_generator_par'):
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         # receive domain information
         #
         dom_info = MPI.COMM_WORLD.bcast(None,root=0)
         #
         orb_generator_slave(molecule,dom_info['dom'],dom_info['l_limit'],dom_info['u_limit'])
         #
         dom_info.clear()
      #
      elif (msg['task'] == 'energy_kernel_mono_exp_par'):
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         energy_kernel_slave(molecule)
      #
      elif (msg['task'] == 'energy_summation_par'):
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         # receive tuple information
         #
         tup_info = MPI.COMM_WORLD.bcast(None,root=0)
         #
         if (tup_info['level'] == 'MACRO'):
            #
            molecule['prim_tuple'].append([])
            #
            molecule['prim_tuple'][tup_info['order']-1] = tup_info['tup']
            #
            energy_summation_par(molecule,tup_info['order'],molecule['prim_tuple'],tup_info['energy'],'MACRO')
         #
         elif (tup_info['level'] == 'CORRE'):
            #
            molecule['corr_tuple'].append([])
            #
            molecule['corr_tuple'][tup_info['order']-1] = tup_info['tup']
            #
            energy_summation_par(molecule,tup_info['order'],molecule['corr_tuple'],tup_info['energy'],'CORRE')
         #
         tup_info.clear()
      #
      elif (msg['task'] == 'remove_slave_env'):
         #
         # remove scr env
         #
         chdir(molecule['wrk'])
         #
         if (molecule['error'][-1]):
            #
            copy(molecule['scr']+'/OUTPUT.OUT',molecule['wrk']+'/OUTPUT.OUT')
         #
         rmtree(molecule['scr'],ignore_errors=True)
      #
      elif (msg['task'] == 'red_mpi_timings'):
         #
         molecule['mpi_time_idle_slave'] += MPI.Wtime()-start_idle
         #
         # reduce mpi timings onto master
         #
         dict_sum_op = MPI.Op.Create(add_time,commute=True)
         #
         time = {}
         #
         time['time_idle_slave'] = molecule['mpi_time_idle_slave']
         #
         time['time_comm_slave'] = molecule['mpi_time_comm_slave']
         #
         time['time_work_slave'] = molecule['mpi_time_work_slave']
         #
         molecule['mpi_comm'].reduce(time,op=dict_sum_op,root=0)
         #
         time.clear()
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

def print_mpi_table(molecule):
   #
   #  ---  master/slave routine
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
   info.clear()
   #
   return

def red_mpi_timings(molecule):
   #
   #  ---  master routine
   #
   # define sum operation for dicts
   #
   time_sum_op = MPI.Op.Create(add_time,commute=True)
   #
   msg = {'task': 'red_mpi_timings'}
   #
   # wake up slaves
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # receive timings
   #
   time = molecule['mpi_comm'].reduce({},op=time_sum_op,root=0)
   #
   sum_slave = time['time_idle_slave']+time['time_comm_slave']+time['time_work_slave']
   #
   molecule['mpi_time_idle'] = [(time['time_idle_slave']/float(molecule['mpi_size']-1)),(time['time_idle_slave']/sum_slave)*100.0]
   #
   molecule['mpi_time_comm'] = [(time['time_comm_slave']/float(molecule['mpi_size']-1)),(time['time_comm_slave']/sum_slave)*100.0]
   #
   molecule['mpi_time_work'] = [(time['time_work_slave']/float(molecule['mpi_size']-1)),(time['time_work_slave']/sum_slave)*100.0]
   #
   time.clear()
   #
   return

