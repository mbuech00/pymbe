#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_utils.py: MPI utilities for Bethe-Goldstone correlation calculations."""

import numpy as np
from copy import deepcopy
from scipy.misc import factorial
from mpi4py import MPI

from bg_mpi_time import init_mpi_timings

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def bcast_mol_dict(molecule):
   #
   #  ---  master routine
   #
   msg = {'task': 'bcast_mol_dict', 'order': 1}
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
   # init mpi master timings
   #
   init_mpi_timings(molecule)
   #
   # private scr dir
   #
   molecule['scr_dir'] = molecule['wrk_dir']+'/'+molecule['scr_name']+'-'+str(molecule['mpi_rank'])
   #
   return molecule

def init_slave_env(molecule):
   #
   #  ---  master routine
   #
   msg = {'task': 'init_slave_env', 'order': 1}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   return

def remove_slave_env(molecule):
   #
   #  ---  master routine
   #
   msg = {'task': 'remove_slave_env', 'order': len(molecule['prim_energy'])}
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
      msg = {'task': 'print_mpi_table', 'order': 1}
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
   print('')
   #
   info.clear()
   #
   return

def prepare_calc(molecule):
   #
   #  --- master/slave routine
   #
   if (molecule['mpi_parallel'] and molecule['mpi_master']):
      #
      # wake up slaves
      #
      msg = {'task': 'prepare_calc_par', 'nocc': molecule['nocc'], 'nvirt': molecule['nvirt'], 'ncore': molecule['ncore'], 'order': 1}
      #
      molecule['mpi_comm'].bcast(msg,root=0)
   #
   # set params and lists for occ expansion
   #
   if (molecule['exp'] == 'occ'):
      #
      # set lower and upper limits
      #
      molecule['l_limit'] = 0
      molecule['u_limit'] = molecule['nocc']
      #
      # init domains on master
      #
      if (molecule['mpi_master']):
         #
         molecule['prim_domain'] = deepcopy([molecule['occ_domain']])
      #
      else:
         #
         molecule['prim_domain'] = []
      #
      # init prim tuple and e_inc
      #
      molecule['prim_tuple'] = [np.array(list([i+1] for i in range(molecule['ncore'],molecule['u_limit'])),dtype=np.int32)]
      molecule['prim_energy_inc'] = [np.zeros(len(molecule['prim_tuple'][0]),dtype=np.float64)]
   #
   # set params and lists for virt expansion
   #
   elif (molecule['exp'] == 'virt'):
      #
      # set lower and upper limits
      #
      molecule['l_limit'] = molecule['nocc']
      molecule['u_limit'] = molecule['nvirt']
      #
      # init domains on master
      #
      if (molecule['mpi_master']):
         #
         molecule['prim_domain'] = deepcopy([molecule['virt_domain']])
      #
      else:
         #
         molecule['prim_domain'] = []
      #
      # init prim tuple and e_inc
      #
      molecule['prim_tuple'] = [np.array(list([i+1] for i in range(molecule['l_limit'],molecule['l_limit']+molecule['u_limit'])),dtype=np.int32)]
      #
      if (molecule['rst']):
         #
         molecule['prim_energy_inc'] = []
      #
      else:
         #
         molecule['prim_energy_inc'] = [np.zeros(len(molecule['prim_tuple'][0]),dtype=np.float64)]
   #
   # set max_order
   #
   if ((molecule['max_order'] == 0) or (molecule['max_order'] > molecule['u_limit'])):
      #
      molecule['max_order'] = molecule['u_limit']
      #
      if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): molecule['max_order'] -= molecule['ncore']
   #
   # determine max theoretical work
   #
   theo_work(molecule)
   #
   # init convergence lists
   #
   molecule['conv_orb'] = [False]
   molecule['conv_energy'] = [False]
   #
   # init orb_ent and orb_con lists
   #
   molecule['prim_orb_ent'] = []
   #
   molecule['prim_orb_arr'] = []
   #
   molecule['prim_orb_con_abs'] = []
   molecule['prim_orb_con_rel'] = []
   #
   # init total energy lists for prim exp
   #
   molecule['prim_energy'] = []
   #
   # init exclusion list and e_tmp
   #
   molecule['e_tmp'] = 0.0
   molecule['excl_list'] = []
   #
   return molecule

def theo_work(molecule):
   #
   molecule['theo_work'] = []
   #
   dim = molecule['u_limit']
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): dim -= molecule['ncore']
   #
   for k in range(0,dim):
      #
      molecule['theo_work'].append(int(factorial(dim)/(factorial(k+1)*factorial(dim-(k+1)))))
   #
   return molecule

def enum(*sequential,**named):
   #
   # hardcoded enums
   #
   enums = dict(zip(sequential,range(len(sequential))),**named)
   #
   return type('Enum',(), enums)

