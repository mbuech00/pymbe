#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst_read.py: restart read utilities for Bethe-Goldstone correlation calculation."""

import numpy as np
from os import listdir
from os.path import isfile, join
from re import search

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def rst_read(molecule):
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
      if ('dom' in files[i]):
         #
         rst_read_dom(molecule,files[i])
      #
      # read orbital contributions
      #
      if ('orb_con' in files[i]):
         #
         rst_read_orb_con(molecule,files[i])
      #
      # read orbital array
      #
      if ('orb_arr' in files[i]):
         #
         rst_read_orb_arr(molecule,files[i])
      #
      # read e_inc
      #
      if ('e_inc' in files[i]):
         #
         rst_read_e_inc(molecule,files[i])
      #
      # read e_tot
      #
      if ('e_tot' in files[i]):
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

def rst_read_tup(molecule,file_num):
   #
   molecule['prim_tuple'].append(np.load(join(molecule['rst_dir'],file_num)))
   #
   return molecule

def rst_read_dom(molecule,file_num):
   #
   molecule['prim_domain'].append(np.load(join(molecule['rst_dir'],file_num))).tolist()
   #
   return molecule

def rst_read_orb_con(molecule,file_num):
   #
   if ('abs' in file_num):
      #
      molecule['prim_orb_con_abs'].append(np.load(join(molecule['rst_dir'],file_num))).tolist()
   #
   elif ('rel' in file_num):
      #
      molecule['prim_orb_con_rel'].append(np.load(join(molecule['rst_dir'],file_num))).tolist()
   #
   return molecule

def rst_read_orb_arr(molecule,file_num):
   #
   molecule['prim_orb_arr'].append(np.load(join(molecule['rst_dir'],file_num)))
   #
   return molecule

def rst_read_e_inc(molecule,file_num):
   #
   molecule['prim_energy_inc'].append(np.load(join(molecule['rst_dir'],file_num)))
   #
   return molecule

def rst_read_e_tot(molecule,file_num):
   #
   molecule['prim_energy'].append(np.load(join(molecule['rst_dir'],file_num))).tolist()
   #
   return molecule

def rst_read_timings(molecule,file_num,time_order):
   #
   if ('init' in file_num):
      #
      if ('work' in file_num):
         #
         molecule['mpi_time_work'][0] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('comm' in file_num):
         #
         molecule['mpi_time_comm'][0] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('idle' in file_num):
         #
         molecule['mpi_time_idle'][0] = np.load(join(molecule['rst_dir'],file_num)).tolist()
   #
   elif ('kernel' in file_num):
      #
      if ('work' in file_num):
         #
         molecule['mpi_time_work'][1] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('comm' in file_num):
         #
         molecule['mpi_time_comm'][1] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('idle' in file_num):
         #
         molecule['mpi_time_idle'][1] = np.load(join(molecule['rst_dir'],file_num)).tolist()
   #
   elif ('final' in file_num):
      #
      if ('work' in file_num):
         #
         molecule['mpi_time_work'][2] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('comm' in file_num):
         #
         molecule['mpi_time_comm'][2] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('idle' in file_num):
         #
         molecule['mpi_time_idle'][2] = np.load(join(molecule['rst_dir'],file_num)).tolist()
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
   # orb_con_abs and orb_con_rel
   #
   if ((len(molecule['prim_orb_con_abs']) != (len(molecule['prim_tuple'])-1)) or (len(molecule['prim_orb_con_rel']) != (len(molecule['prim_tuple'])-1))): fail = True 
   #
   # orb_arr
   #
   if (len(molecule['prim_orb_arr']) != (len(molecule['prim_tuple'])-2)): fail = True
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

