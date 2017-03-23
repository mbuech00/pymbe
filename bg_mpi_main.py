#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_main.py: main MPI driver routine for Bethe-Goldstone correlation calculations."""

import numpy as np
from os import getcwd, mkdir, chdir
from shutil import copy, rmtree
from mpi4py import MPI

from bg_mpi_utils import print_mpi_table, mono_exp_merge_info, prepare_calc
from bg_mpi_rst import rst_dist_slave
from bg_mpi_time import init_mpi_timings, collect_init_mpi_time, collect_kernel_mpi_time, collect_final_mpi_time
from bg_mpi_energy import energy_kernel_mono_exp_slave, energy_summation_par
from bg_mpi_orbitals import orb_generator_slave, orb_entanglement_main_par

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
      main_slave(molecule)
   #
   else:
      #
      molecule['mpi_master'] = True
      #
      molecule['mpi_rank'] = 0
   #
   return molecule

def main_slave(molecule):
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
         molecule['wrk_dir'] = getcwd()
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
         molecule['scr_dir'] = molecule['wrk_dir']+'/'+molecule['scr_name']+'-'+str(molecule['mpi_rank'])
         #
         # init scr env
         #
         mkdir(molecule['scr_dir'])
         #
         chdir(molecule['scr_dir'])
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
      # prepare_calc_par
      #
      elif (msg['task'] == 'prepare_calc_par'):
         #
         # set mol params
         #
         molecule['nocc'] = msg['nocc']
         molecule['nvirt'] = msg['nvirt']
         molecule['ncore'] = msg['ncore']
         #
         prepare_calc(molecule)
      #
      # rst_dist_slave
      #
      elif (msg['task'] == 'rst_dist'):
         #
         rst_dist_slave(molecule) 
      #
      # mono_exp_merge_info
      #
      elif (msg['task'] == 'mono_exp_merge_info'):
         #
         molecule['min_corr_order'] = msg['min_corr_order']
         #
         mono_exp_merge_info(molecule)
      #
      # orbital entanglement
      #
      elif (msg['task'] == 'orb_entanglement_par'):
         #
         orb_entanglement_main_par(molecule,msg['l_limit'],msg['u_limit'],msg['order'],msg['level'])
         #
         collect_init_mpi_time(molecule,msg['order'])
      #
      # orb_generator_slave
      #
      elif (msg['task'] == 'orb_generator_slave'):
         #
         # generate tuples
         #
         if (msg['level'] == 'MACRO'):
            #
            orb_generator_slave(molecule,molecule['prim_domain'],molecule['prim_tuple'],msg['l_limit'],msg['u_limit'],msg['order'],msg['level'])
         #
         elif (msg['level'] == 'CORRE'):
            #
            orb_generator_slave(molecule,molecule['corr_domain'],molecule['corr_tuple'],msg['l_limit'],msg['u_limit'],msg['order'],msg['level'])
      #
      # energy_kernel_mono_exp_par
      #
      elif (msg['task'] == 'energy_kernel_mono_exp_par'):
         #
         if (msg['level'] == 'MACRO'):
            #
            energy_kernel_mono_exp_slave(molecule,msg['order'],molecule['prim_tuple'],molecule['prim_energy_inc'],msg['l_limit'],msg['u_limit'],'MACRO')
         #
         elif (msg['level'] == 'CORRE'):
            #
            energy_kernel_mono_exp_slave(molecule,msg['order'],molecule['corr_tuple'],molecule['corr_energy_inc'],msg['l_limit'],msg['u_limit'],'CORRE')
         #
         collect_kernel_mpi_time(molecule,msg['order'])
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
         collect_final_mpi_time(molecule,msg['order'])
      #
      # remove_slave_env
      #
      elif (msg['task'] == 'remove_slave_env'):
         #
         # remove scr env
         #
         chdir(molecule['wrk_dir'])
         #
         if (molecule['error'][-1]):
            #
            copy(molecule['scr_dir']+'/OUTPUT_'+str(molecule['mpi_rank'])+'.OUT',molecule['wrk_dir']+'/OUTPUT_'+str(molecule['mpi_rank'])+'.OUT')
         #
         rmtree(molecule['scr_dir'],ignore_errors=True)
      #
      # finalize_mpi
      #
      elif (msg['task'] == 'finalize_mpi'):
         #
         slave = False
   #
   return molecule

