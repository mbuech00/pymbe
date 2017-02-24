#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_utils.py: MPI utilities for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import init_mpi_timings
from bg_time import init_phase_timings

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

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
   # init program phase timings and mpi master timings
   #
   init_phase_timings(molecule)
   #
   init_mpi_timings(molecule)
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
      # wake up slaves
      #
      msg = {'task': 'print_mpi_table'}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
      #
      full_info = []
      #
      # receive node info (in ordered sequence)
      #
      for i in range(0,molecule['mpi_size']-1):
         #
         info = molecule['mpi_comm'].recv(source=i+1,status=molecule['mpi_stat'])
         #
         full_info.append([info['rank'],info['name']])
   #
   else:
      #
      # send mpi rank and node name to master
      #
      info = {}
      #
      info['rank'] = molecule['mpi_rank']
      #
      info['name'] = molecule['mpi_name']
      #
      molecule['mpi_comm'].send(info, dest=0)
      #
      info.clear()
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
   # determine column width in print-out below
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
   # print info - master first, then slaves (in ordered sequence)
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

def mono_exp_merge_info(molecule):
   #
   #  ---  master/slave routine
   #
   if (molecule['mpi_parallel'] and molecule['mpi_master']):
      #
      # wake up slaves
      #
      msg = {'task': 'mono_exp_merge_info', 'min_corr_order': molecule['min_corr_order']}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
   #
   for k in range(1,molecule['min_corr_order']):
      #
      molecule['corr_tuple'].append(np.array([],dtype=np.int))
      molecule['corr_energy_inc'].append(np.array([],dtype=np.float64))
   #
   if (molecule['mpi_master']):
      #
      for k in range(1,molecule['min_corr_order']-1):
         #
         molecule['corr_domain'].append(molecule['prim_domain'][k-1])
         molecule['corr_orb_con_abs'].append(molecule['prim_orb_con_abs'][k-1])
         molecule['corr_orb_con_rel'].append(molecule['prim_orb_con_rel'][k-1])
      #
      for k in range(1,molecule['min_corr_order']-2):
         #
         molecule['corr_orb_ent'].append(molecule['prim_orb_ent'][k-1])
         molecule['corr_orb_arr'].append(molecule['prim_orb_arr'][k-1])
   #
   return molecule

