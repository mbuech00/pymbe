#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst_read.py: restart read utilities for Bethe-Goldstone correlation calculations."""

import numpy as np
from os import listdir
from os.path import isfile, join
from re import search
from copy import deepcopy

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def rst_read_main(molecule):
   #
   # list filenames in files list
   #
   files = [f for f in listdir(molecule['rst_dir']) if isfile(join(molecule['rst_dir'],f))]
   #
   for i in range(0,len(files)):
      #
      # read tuples
      #
      if ('tup' in files[i]):
         #
         rst_read_tup(molecule,files[i])
      #
      # read domains
      #
      elif ('dom' in files[i]):
         #
         rst_read_dom(molecule,files[i])
      #
      # read orbital entanglement matrix
      #
      elif ('orb_ent' in files[i]):
         #
         rst_read_orb_ent(molecule,files[i])
      #
      # read orbital array
      #
      elif ('orb_arr' in files[i]):
         #
         rst_read_orb_arr(molecule,files[i])
      #
      # read orbital contributions
      #
      elif ('orb_con' in files[i]):
         #
         rst_read_orb_con(molecule,files[i])
      #
      # read orbital exclusion lists
      #
      elif ('excl_list' in files[i]):
         #
         rst_read_excl_list(molecule,files[i])
      #
      # read e_inc
      #
      elif ('e_inc' in files[i]):
         #
         rst_read_e_inc(molecule,files[i])
      #
      # read e_tot
      #
      elif ('e_tot' in files[i]):
         #
         rst_read_e_tot(molecule,files[i])
      #
      # read timings
      #
      elif ('time' in files[i]):
         #
         rst_read_timings(molecule,files[i])
   #
   rst_sanity_chk(molecule)
   #
   molecule['min_order'] = len(molecule['prim_tuple'])
   #
   return molecule

def rst_read_tup(molecule,inp):
   #
   molecule['prim_tuple'].append(np.load(join(molecule['rst_dir'],inp)))
   #
   return molecule

def rst_read_dom(molecule,inp):
   #
   molecule['prim_domain'].append(np.load(join(molecule['rst_dir'],inp)).tolist())
   #
   return molecule

def rst_read_orb_ent(molecule,inp):
   #
   molecule['prim_orb_ent'].append(np.load(join(molecule['rst_dir'],inp)))
   #
   return molecule

def rst_read_orb_arr(molecule,inp):
   #
   molecule['prim_orb_arr'].append(np.load(join(molecule['rst_dir'],inp)))
   #
   return molecule

def rst_read_orb_con(molecule,inp):
   #
   if ('abs' in inp):
      #
      molecule['prim_orb_con_abs'].append(np.load(join(molecule['rst_dir'],inp)).tolist())
   #
   elif ('rel' in inp):
      #
      molecule['prim_orb_con_rel'].append(np.load(join(molecule['rst_dir'],inp)).tolist())
   #
   return molecule

def rst_read_excl_list(molecule,inp):
   #
   molecule['excl_list'].append(np.load(join(molecule['rst_dir'],inp)).tolist())
   #
   return molecule

def rst_read_e_inc(molecule,inp):
   #
   molecule['prim_energy_inc'].append(np.load(join(molecule['rst_dir'],inp)))
   #
   return molecule

def rst_read_e_tot(molecule,inp):
   #
   molecule['prim_energy'].append(np.load(join(molecule['rst_dir'],inp)).tolist())
   #
   return molecule

def rst_read_timings(molecule,inp):
   #
   if ('init' in inp):
      #
      if ('work' in inp):
         #
         if (molecule['mpi_parallel']):
            #
            molecule['mpi_time_work'][0] = np.load(join(molecule['rst_dir'],inp)).tolist()
            molecule['mpi_time_work_init'] = deepcopy(molecule['mpi_time_work'][0][0])
         #
         else:
            #
            molecule['mpi_time_work_init'] = np.load(join(molecule['rst_dir'],inp)).tolist()
      #
      elif ('comm' in inp):
         #
         molecule['mpi_time_comm'][0] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_comm_init'] = deepcopy(molecule['mpi_time_comm'][0][0])
      #
      elif ('idle' in inp):
         #
         molecule['mpi_time_idle'][0] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_idle_init'] = deepcopy(molecule['mpi_time_idle'][0][0])
   #
   elif ('kernel' in inp):
      #
      if ('work' in inp):
         #
         if (molecule['mpi_parallel']):
            #
            molecule['mpi_time_work'][1] = np.load(join(molecule['rst_dir'],inp)).tolist()
            molecule['mpi_time_work_kernel'] = deepcopy(molecule['mpi_time_work'][1][0])
         #
         else:
            #
            molecule['mpi_time_work_kernel'] = np.load(join(molecule['rst_dir'],inp)).tolist()
      #
      elif ('comm' in inp):
         #
         molecule['mpi_time_comm'][1] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_comm_kernel'] = deepcopy(molecule['mpi_time_comm'][1][0])
      #
      elif ('idle' in inp):
         #
         molecule['mpi_time_idle'][1] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_idle_kernel'] = deepcopy(molecule['mpi_time_idle'][1][0])
   #
   elif ('final' in inp):
      #
      if ('work' in inp):
         #
         if (molecule['mpi_parallel']):
            #
            molecule['mpi_time_work'][2] = np.load(join(molecule['rst_dir'],inp)).tolist()
            molecule['mpi_time_work_final'] = deepcopy(molecule['mpi_time_work'][2][0])
         #
         else:
            #
            molecule['mpi_time_work_final'] = np.load(join(molecule['rst_dir'],inp)).tolist()
      #
      elif ('comm' in inp):
         #
         molecule['mpi_time_comm'][2] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_comm_final'] = deepcopy(molecule['mpi_time_comm'][2][0])
      #
      elif ('idle' in inp):
         #
         molecule['mpi_time_idle'][2] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_idle_final'] = deepcopy(molecule['mpi_time_idle'][2][0])
   #
   return molecule

def rst_sanity_chk(molecule):
   #
   fail = False
   #
   # dom
   #
   if (len(molecule['prim_domain']) != len(molecule['prim_tuple'])): fail = True
   #
   # orb_ent
   #
   if (len(molecule['prim_orb_ent']) != (len(molecule['prim_tuple'])-2)): fail = True
   #
   # orb_arr
   #
   if (len(molecule['prim_orb_arr']) != (len(molecule['prim_tuple'])-2)): fail = True
   #
   # orb_con_abs and orb_con_rel
   #
   if ((len(molecule['prim_orb_con_abs']) != (len(molecule['prim_tuple'])-1)) or (len(molecule['prim_orb_con_rel']) != (len(molecule['prim_tuple'])-1))): fail = True
   #
   # excl_list
   #
   if (len(molecule['excl_list']) != (len(molecule['prim_tuple'])-2)): fail = True
   #
   # e_inc
   #
   if (len(molecule['prim_energy_inc']) != len(molecule['prim_tuple'])): fail = True
   #
   # e_tot
   #
   if (len(molecule['prim_energy']) != (len(molecule['prim_tuple'])-1)): fail = True
   #
   # check for correct number of mpi procs
   #
   if (molecule['mpi_parallel']):
      #
      if (len(molecule['mpi_time_work'][0]) != molecule['mpi_size']): fail = True
      if (len(molecule['mpi_time_work'][1]) != molecule['mpi_size']): fail = True
      if (len(molecule['mpi_time_work'][2]) != molecule['mpi_size']): fail = True
      #
      if (len(molecule['mpi_time_comm'][0]) != molecule['mpi_size']): fail = True
      if (len(molecule['mpi_time_comm'][1]) != molecule['mpi_size']): fail = True
      if (len(molecule['mpi_time_comm'][2]) != molecule['mpi_size']): fail = True
      #
      if (len(molecule['mpi_time_idle'][0]) != molecule['mpi_size']): fail = True
      if (len(molecule['mpi_time_idle'][1]) != molecule['mpi_size']): fail = True
      if (len(molecule['mpi_time_idle'][2]) != molecule['mpi_size']): fail = True
   #
   # check for errors
   #
   if (fail):
      #
      print('init restart failed, aborting ...')
      #
      molecule['error'].append(True)
   #
   return molecule

