#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_main.py: main MPI driver routine for Bethe-Goldstone correlation calculations."""

import numpy as np
from os import getcwd, mkdir, chdir
from shutil import copy, rmtree
from mpi4py import MPI

from bg_mpi_utils import print_mpi_table, mono_exp_merge_info
from bg_mpi_time import init_mpi_timings, collect_mpi_timings
from bg_mpi_energy import energy_kernel_mono_exp_par, energy_summation_par
from bg_mpi_orbitals import bcast_tuples, bcast_dom, orb_generator_slave 

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
      msg = MPI.COMM_WORLD.bcast(None,root=0)
      #
      # bcast_mol_dict
      #
      if (msg['task'] == 'bcast_mol_dict'):
         #
         # receive molecule dict from master
         #
         molecule = MPI.COMM_WORLD.bcast(None,root=0)
         #
         # set current mpi proc to 'slave'
         #
         molecule['mpi_master'] = False
         #
         # init slave mpi timings
         #
         init_mpi_timings(molecule)
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
      # init_slave_env
      #
      elif (msg['task'] == 'init_slave_env'):
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
         # init tuple lists
         #
         molecule['prim_tuple'] = []
         molecule['corr_tuple'] = []
         #
         # init e_inc lists
         #
         molecule['prim_energy_inc'] = []
         molecule['corr_energy_inc'] = []
      #
      # print_mpi_table
      #
      elif (msg['task'] == 'print_mpi_table'):
         #
         print_mpi_table(molecule)
      #
      # mono_exp_merge_info
      #
      elif (msg['task'] == 'mono_exp_merge_info'):
         #
         molecule['min_corr_order'] = msg['min_corr_order']
         #
         mono_exp_merge_info(molecule)
      #
      # bcast_tuples
      #
      elif (msg['task'] == 'bcast_tuples'):
         #
         # receive tuples
         #
         bcast_tuples(molecule,msg['order'])
      #
      # orb_generator_par
      #
      elif (msg['task'] == 'orb_generator_par'):
         #
         # receive domains
         #
         bcast_dom(molecule,msg['order'])
         #
         if (msg['level'] == 'MACRO'):
            #
            orb_generator_slave(molecule,molecule['dom_info']['dom'],molecule['prim_tuple'],msg['l_limit'],msg['u_limit'],msg['order'],'MACRO')
         #
         elif (msg['level'] == 'CORRE'):
            #
            orb_generator_slave(molecule,molecule['dom_info']['dom'],molecule['corr_tuple'],msg['l_limit'],msg['u_limit'],msg['order'],'CORRE')
         #
         molecule['dom_info'].clear()
      #
      # energy_kernel_mono_exp_par
      #
      elif (msg['task'] == 'energy_kernel_mono_exp_par'):
         #
         if (msg['level'] == 'MACRO'):
            #
            energy_kernel_mono_exp_par(molecule,msg['order'],molecule['prim_tuple'],None,molecule['prim_energy_inc'],msg['l_limit'],msg['u_limit'],'MACRO')
         #
         elif (msg['level'] == 'CORRE'):
            #
            energy_kernel_mono_exp_par(molecule,msg['order'],molecule['corr_tuple'],None,molecule['corr_energy_inc'],msg['l_limit'],msg['u_limit'],'CORRE')
      #
      # energy_summation_par
      #
      elif (msg['task'] == 'energy_summation_par'):
         #
         if (msg['level'] == 'MACRO'):
            #
            energy_summation_par(molecule,msg['order'],molecule['prim_tuple'],molecule['prim_energy_inc'],None,'MACRO')
         #
         elif (msg['level'] == 'CORRE'):
            #
            energy_summation_par(molecule,msg['order'],molecule['corr_tuple'],molecule['corr_energy_inc'],None,'CORRE')
      #
      # remove_slave_env
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
      # collect_mpi_timings
      #
      elif (msg['task'] == 'collect_mpi_timings'):
         #
         collect_mpi_timings(molecule)
      #
      # finalize_mpi
      #
      elif (msg['task'] == 'finalize_mpi'):
         #
         slave = False
   #
   return molecule

