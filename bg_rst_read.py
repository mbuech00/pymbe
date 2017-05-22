#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst_read.py: restart read utilities for Bethe-Goldstone correlation calculations."""

import numpy as np
from os import listdir
from os.path import isfile, join
from re import search
from copy import deepcopy

from bg_utils import term_calc

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def rst_read_main(molecule):
   #
   # list filenames in files list
   #
   files = [f for f in listdir(molecule['rst_dir']) if isfile(join(molecule['rst_dir'],f))]
   #
   # sort the list of files
   #
   files.sort()
   #
   for i in range(0,len(files)):
      #
      # read tuples
      #
      if ('tup' in files[i]):
         #
         rst_read_tup(molecule,files[i])
      #
      # read orbital entanglement matrices
      #
      elif ('orb_ent' in files[i]):
         #
         rst_read_orb_ent(molecule,files[i])
      #
      # read orbital contributions
      #
      elif ('orb_con' in files[i]):
         #
         rst_read_orb_con(molecule,files[i])
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

def rst_read_orb_ent(molecule,inp):
   #
   if ('abs' in inp):
      #
      molecule['prim_orb_ent_abs'].append(np.load(join(molecule['rst_dir'],inp)))
   #
   elif ('rel' in inp):
      #
      molecule['prim_orb_ent_rel'].append(np.load(join(molecule['rst_dir'],inp)))
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
   if ('kernel' in inp):
      #
      if ('work' in inp):
         #
         if (molecule['mpi_parallel']):
            #
            molecule['mpi_time_work'][0] = np.load(join(molecule['rst_dir'],inp)).tolist()
            molecule['mpi_time_work_kernel'] = deepcopy(molecule['mpi_time_work'][0][0])
         #
         else:
            #
            molecule['mpi_time_work_kernel'] = np.load(join(molecule['rst_dir'],inp)).tolist()
      #
      elif ('comm' in inp):
         #
         molecule['mpi_time_comm'][0] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_comm_kernel'] = deepcopy(molecule['mpi_time_comm'][0][0])
      #
      elif ('idle' in inp):
         #
         molecule['mpi_time_idle'][0] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_idle_kernel'] = deepcopy(molecule['mpi_time_idle'][0][0])
   #
   elif ('summation' in inp):
      #
      if ('work' in inp):
         #
         if (molecule['mpi_parallel']):
            #
            molecule['mpi_time_work'][1] = np.load(join(molecule['rst_dir'],inp)).tolist()
            molecule['mpi_time_work_summation'] = deepcopy(molecule['mpi_time_work'][1][0])
         #
         else:
            #
            molecule['mpi_time_work_summation'] = np.load(join(molecule['rst_dir'],inp)).tolist()
      #
      elif ('comm' in inp):
         #
         molecule['mpi_time_comm'][1] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_comm_summation'] = deepcopy(molecule['mpi_time_comm'][1][0])
      #
      elif ('idle' in inp):
         #
         molecule['mpi_time_idle'][1] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_idle_summation'] = deepcopy(molecule['mpi_time_idle'][1][0])
   #
   elif ('screen' in inp):
      #
      if ('work' in inp):
         #
         if (molecule['mpi_parallel']):
            #
            molecule['mpi_time_work'][2] = np.load(join(molecule['rst_dir'],inp)).tolist()
            molecule['mpi_time_work_screen'] = deepcopy(molecule['mpi_time_work'][2][0])
         #
         else:
            #
            molecule['mpi_time_work_screen'] = np.load(join(molecule['rst_dir'],inp)).tolist()
      #
      elif ('comm' in inp):
         #
         molecule['mpi_time_comm'][2] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_comm_screen'] = deepcopy(molecule['mpi_time_comm'][2][0])
      #
      elif ('idle' in inp):
         #
         molecule['mpi_time_idle'][2] = np.load(join(molecule['rst_dir'],inp)).tolist()
         molecule['mpi_time_idle_screen'] = deepcopy(molecule['mpi_time_idle'][2][0])
   #
   return molecule

def rst_sanity_chk(molecule):
   #
   fail = False
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
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
      #
      term_calc(molecule)
   #
   return molecule

