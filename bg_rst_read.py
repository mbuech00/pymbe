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
   print('files = '+str(files))
   #
   tup_order = 0
   dom_order = 0
   orb_con_order = [0]*2
   orb_arr_order = 0
   e_inc_order = 0
   e_tot_order = 0
   time_order = [[0]*3,[0]*3,[0]*3]
   #
   for i in range(0,len(files)):
      #
      # read tuples
      #
      if ('tup' in files[i]):
         #
         rst_read_tup(molecule,files[i],tup_order)
      #
      # read domains
      #
      if ('dom' in files[i]):
         #
         rst_read_dom(molecule,files[i],dom_order)
      #
      # read orbital contributions
      #
      if ('orb_con' in files[i]):
         #
         rst_read_orb_con(molecule,files[i],orb_con_order)
      #
      # read orbital array
      #
      if ('orb_arr' in files[i]):
         #
         rst_read_orb_arr(molecule,files[i],orb_arr_order)
      #
      # read e_inc
      #
      if ('e_inc' in files[i]):
         #
         rst_read_e_inc(molecule,files[i],e_inc_order)
      #
      # read e_tot
      #
      if ('e_tot' in files[i]):
         #
         rst_read_e_tot(molecule,files[i],e_tot_order)
      #
      # read timings
      #
      elif ('time' in files[i]):
         #
         rst_read_timings(molecule,files[i],time_order)
   #
   rst_sanity_chk(molecule,tup_order,dom_order,orb_con_order,orb_arr_order,e_inc_order,e_tot_order,time_order)
   #
   molecule['min_order'] = tup_order
   #
   return molecule

def rst_read_tup(molecule,file_num,tup_order):
   #
   tup_order = max(int(search(r'\d+',file_num).group()),tup_order)
   #
   molecule['prim_tuple'].append(np.load(join(molecule['rst_dir'],file_num)))
   #
   return molecule

def rst_read_dom(molecule,file_num,dom_order):
   #
   dom_order = max(int(search(r'\d+',file_num).group()),tup_order)
   #
   molecule['prim_domain'].append(np.load(join(molecule['rst_dir'],file_num))).tolist()
   #
   return molecule

def rst_read_orb_con(molecule,file_num,orb_con_order):
   #
   if ('abs' in file_num):
      #
      orb_con_order[0] = max(int(search(r'\d+',file_num).group()),orb_con_order[0])
      #
      molecule['prim_orb_con_abs'].append(np.load(join(molecule['rst_dir'],file_num))).tolist()
   #
   elif ('rel' in file_num):
      #
      orb_con_order[1] = max(int(search(r'\d+',file_num).group()),orb_con_order[1])
      #
      molecule['prim_orb_con_rel'].append(np.load(join(molecule['rst_dir'],file_num))).tolist()
   #
   return molecule

def rst_read_orb_arr(molecule,file_num,orb_arr_order):
   #
   orb_arr_order = max(int(search(r'\d+',file_num).group()),orb_arr_order)
   #
   molecule['prim_orb_arr'].append(np.load(join(molecule['rst_dir'],file_num)))
   #
   return molecule

def rst_read_e_inc(molecule,file_num,e_inc_order):
   #
   e_inc_order = max(int(search(r'\d+',file_num).group()),e_inc_order)
   #
   molecule['prim_energy_inc'].append(np.load(join(molecule['rst_dir'],file_num)))
   #
   return molecule

def rst_read_e_tot(molecule,file_num,e_tot_order):
   #
   e_tot_order = max(int(search(r'\d+',file_num).group()),e_tot_order)
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
         time_order[0][0] = max(int(search(r'\d+',file_num).group()),time_order[0][0])
         #
         molecule['mpi_time_work_init'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('comm' in file_num):
         #
         time_order[0][1] = max(int(search(r'\d+',file_num).group()),time_order[0][1])
         #
         molecule['mpi_time_comm_init'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('idle' in file_num):
         #
         time_order[0][2] = max(int(search(r'\d+',file_num).group()),time_order[0][2])
         #
         molecule['mpi_time_idle_init'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
   #
   elif ('kernel' in file_num):
      #
      if ('work' in file_num):
         #
         time_order[1][0] = max(int(search(r'\d+',file_num).group()),time_order[1][0])
         #
         molecule['mpi_time_work_kernel'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('comm' in file_num):
         #
         time_order[1][1] = max(int(search(r'\d+',file_num).group()),time_order[1][1])
         #
         molecule['mpi_time_comm_kernel'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('idle' in file_num):
         #
         time_order[1][2] = max(int(search(r'\d+',file_num).group()),time_order[1][2])
         #
         molecule['mpi_time_idle_kernel'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
   #
   elif ('final' in file_num):
      #
      if ('work' in file_num):
         #
         time_order[2][0] = max(int(search(r'\d+',file_num).group()),time_order[2][0])
         #
         molecule['mpi_time_work_final'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('comm' in file_num):
         #
         time_order[2][1] = max(int(search(r'\d+',file_num).group()),time_order[2][1])
         #
         molecule['mpi_time_comm_final'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
      #
      elif ('idle' in file_num):
         #
         time_order[2][2] = max(int(search(r'\d+',file_num).group()),time_order[2][2])
         #
         molecule['mpi_time_idle_final'] = np.load(join(molecule['rst_dir'],file_num)).tolist()
   #
   return molecule

def rst_sanity_chk(molecule,tup_order,dom_order,orb_con_order,orb_arr_order,e_inc_order,e_tot_order,time_order):
   #
   fail = False
   #
   # dom
   #
   if (dom_order != tup_order): fail = True
   #
   # orb_con_abs and orb_con_rel
   #
   if ((orb_con_order[0] != orb_con_order[1]) or (orb_con_order[0] != tup_order)): fail = True
   #
   # orb_arr
   #
   if (orb_arr_order != tup_order): fail = True
   #
   # e_inc
   #
   if (e_inc_order != tup_order): fail = True
   #
   # e_tot
   #
   if (e_tot_order != tup_order): fail = True
   #
   # time_order
   #
   if (time_order[0].count(time_order[0][0]) != len(time_order[0])): fail = True
   #
   if ((time_order[0][0] != time_order[1][0]) or (time_order[1][0] != time_order[2][0])): fail = True
   #
   if (not ((time_order[0][0] == tup_order) or ((time_order[0]+1) == tup_order))): fail = True
   #
   # check for correct number of mpi procs
   #
   
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

