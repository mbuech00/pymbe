#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver routines for Bethe-Goldstone correlation calculations."""

import numpy as np

from bg_mpi_utils import prepare_calc, mono_exp_merge_info
from bg_mpi_time import timer_mpi, collect_init_mpi_time
from bg_print import print_mono_exp_header, print_status_header, print_status_end, print_result,\
                     print_init_header, print_init_end, print_final_header, print_final_end
from bg_energy import energy_kernel_mono_exp, energy_summation
from bg_orbitals import init_domains, update_domains, orb_generator,\
                        orb_screening, orb_exclusion
from bg_rst_main import rst_main
from bg_rst_write import rst_write_tup, rst_write_dom, rst_write_orb_ent, rst_write_orb_arr, rst_write_excl_list,\
                         rst_write_orb_con, rst_write_e_inc, rst_write_e_tot, rst_write_time

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def main_drv(molecule):
   #
   # initialize domains
   #
   init_domains(molecule)
   #
   # initialize variable and lists
   #
   prepare_calc(molecule)   
   #
   # run the specified calculation
   #
   if ((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')):
      #
      # run mono expansion
      #
      print_mono_exp_header(molecule)
      #
      rst_main(molecule)
      #
      mono_exp_drv(molecule,molecule['min_order'],molecule['max_order'],'MACRO')
      #
      if (molecule['corr']):
         #
         # energy correction for mono expansion
         #
         # set min and max _corr_order
         #
         set_corr_order(molecule)
         #
         # merge info from prim exp
         #
         mono_exp_merge_info(molecule)
         #
         # calculate correction (if possible)
         #
         if (molecule['corr']):
            #
            print('')
            print('                     ---------------------------------------------                ')
            print('                                   energy correction                              ')
            print('                     ---------------------------------------------                ')
            #
            mono_exp_drv(molecule,molecule['min_corr_order'],molecule['max_corr_order'],'CORRE')
   #
   elif ((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')):
      #
      # run dual expansion
      #
      dual_exp_drv(molecule)
   #
   return molecule

def mono_exp_drv(molecule,start,end,level):
   #
   for k in range(start,end+1):
      #
      # mono expansion initialization
      #
      if (k == start):
         #
         print('')
      #
      else:
         #
         mono_exp_init(molecule,k,level)
      #
      # mono expansion kernel
      #
      mono_exp_kernel(molecule,k,level)
      #
      # return if converged
      #
      if (molecule['conv_orb'][-1] or molecule['conv_energy'][-1]): break
   #
   print('')
   #
   orb_screening(molecule,molecule['l_limit'],molecule['u_limit'],k,level,True)
   #
   mono_exp_finish(molecule)
   #
   print('')
   #
   return molecule

def mono_exp_kernel(molecule,k,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      e_inc = molecule['prim_energy_inc']
      e_tot = molecule['prim_energy']
   #
   elif (level == 'CORRE'):
      #
      tup = molecule['corr_tuple']
      e_inc = molecule['corr_energy_inc']
      e_tot = molecule['corr_energy']
   #
   print_status_header(molecule,tup[-1],k,molecule['conv_orb'][-1],level)
   #
   if ((level == 'MACRO') and molecule['conv_orb'][-1]): return molecule
   #
   # run the calculations
   #
   energy_kernel_mono_exp(molecule,k,tup,e_inc,molecule['l_limit'],molecule['u_limit'],level)
   #
   # print status end
   #
   print_status_end(molecule,k,level)
   #
   print_final_header(molecule,k,level)
   #
   # calculate the energy at order k
   #
   energy_summation(molecule,k,tup,e_inc,e_tot,level)
   #
   # write e_inc and e_tot restart files
   #
   rst_write_e_inc(molecule,k)
   rst_write_e_tot(molecule,k)
   #
   if ((k >= 2) and (abs(e_tot[-1]-e_tot[-2]) < molecule['prim_e_thres'])): molecule['conv_energy'].append(True)
   #
   print_final_end(molecule,k,molecule['conv_energy'][-1],level)
   # 
   # print results
   #
   print_result(molecule,tup[k-1],e_inc[k-1],level)
   #
   return molecule

def mono_exp_init(molecule,k,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      e_inc = molecule['prim_energy_inc']
      dom = molecule['prim_domain']
      orb = molecule['prim_orb_ent']
      thres = molecule['prim_thres']
   #
   elif (level == 'CORRE'):
      #
      tup = molecule['corr_tuple']
      e_inc = molecule['corr_energy_inc']
      dom = molecule['corr_domain']
      orb = molecule['corr_orb_ent']
      thres = molecule['corr_thres']
   #
   # print init header
   #
   print_init_header(molecule,k,level)
   #
   # orbital screening (using info from order k-1)
   #
   orb_screening(molecule,molecule['l_limit'],molecule['u_limit'],k-1,level)
   #
   # write dom, orb_con_abs, orb_con_rel, and orb_arr restart files
   #
   rst_write_dom(molecule,k)
   rst_write_orb_con(molecule,k-1)
   if (k >= 3):
      #
      rst_write_orb_ent(molecule,k-2)
      rst_write_orb_arr(molecule,k-2)
      rst_write_excl_list(molecule,k-2)
   #
   # generate all tuples at order k
   #
   orb_generator(molecule,dom[k-1],tup,molecule['l_limit'],molecule['u_limit'],k,level)
   #
   # write tup restart file
   #
   rst_write_tup(molecule,k)
   #
   timer_mpi(molecule,'mpi_time_work_init',k-1)
   #
   # check for convergence
   #
   if ((level == 'MACRO') and (len(tup[k-1]) == 0)): molecule['conv_orb'].append(True)
   #
   # print init end
   #
   print_init_end(molecule,k,level)
   #
   # init e_inc list
   #
   e_inc.append(np.zeros(len(tup[k-1]),dtype=np.float64))
   #
   # if converged, pop last element of tup and e_inc lists
   #
   if ((level == 'MACRO') and molecule['conv_orb'][-1]):
      #
      tup.pop(-1)
      e_inc.pop(-1)
   #
   if (molecule['mpi_parallel']):
      #
      collect_init_mpi_time(molecule,k-1,True)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_init',k-1,True)
      #
      rst_write_time(molecule,'init')
   #
   return molecule

def mono_exp_finish(molecule):
   #
   if (not molecule['corr']):
      #
      molecule['min_corr_order'] = 0
      molecule['max_corr_order'] = 0
   #
   # make the corr_energy list of the same length as the prim_energy list
   #
   for _ in range(molecule['max_corr_order'],len(molecule['prim_energy'])):
      #
      if (molecule['corr']):
         #
         molecule['corr_energy'].append(molecule['corr_energy'][-1])
      #
      else:
         #
         molecule['corr_energy'].append(0.0)
   #
   # make corr_tuple and corr_energy_inc lists of the same length as prim_tuple and prim_energy_inc
   #
   for _ in range(len(molecule['corr_tuple']),len(molecule['prim_tuple'])):
      #
      molecule['corr_tuple'].append(np.array([],dtype=np.int32))
   #
   for _ in range(len(molecule['corr_energy_inc']),len(molecule['prim_energy_inc'])):
      #
      molecule['corr_energy_inc'].append(np.array([],dtype=np.float64))
   #
   if (molecule['corr']):
      #
      # make cor_orb_con lists of same length as orb_con lists for prim exp
      #
      for i in range(len(molecule['corr_orb_ent']),len(molecule['prim_orb_ent'])):
         #
         molecule['corr_orb_con_abs'].append([])
         molecule['corr_orb_con_rel'].append([])
         #
         for j in range(0,len(molecule['prim_orb_ent'][i])):
            #
            molecule['corr_orb_con_abs'][-1].append(molecule['corr_orb_con_abs'][-2][j]+np.sum(molecule['prim_orb_ent'][i][j]))
         #
         for j in range(0,len(molecule['corr_orb_con_abs'][-1])):
            #
            if (molecule['corr_orb_con_abs'][-1][j] == 0.0):
               #
               molecule['corr_orb_con_rel'][-1].append(0.0)
            #
            else:
               #
               molecule['corr_orb_con_rel'][-1].append(molecule['corr_orb_con_abs'][-1][j]/sum(molecule['corr_orb_con_abs'][-1]))
   #
   return molecule

def set_corr_order(molecule):
   #
   molecule['min_corr_order'] = 0
   #
   for i in range(0,len(molecule['prim_tuple'])):
      #
      if ((len(molecule['prim_tuple'][i]) < molecule['theo_work'][i]) and (len(molecule['prim_tuple'][i]) > 0)):
         #
         molecule['min_corr_order'] = i+1
         #
         break
   #
   # no energy correction possible
   #
   if (molecule['min_corr_order'] == 0):
      #
      molecule['corr'] = False
      #
      molecule['max_corr_order'] = 0
      #
      molecule['corr_order'] = 0
      #
      for _ in range(0,len(molecule['prim_energy'])):
         #
         molecule['corr_tuple'].append(np.array([],dtype=np.int32))
         molecule['corr_energy_inc'].append(np.array([],dtype=np.float64))
         #
         molecule['corr_energy'].append(0.0)
      #
      return molecule
   #
   # the input corr_order is too high, so we correct everything
   #
   elif ((molecule['min_corr_order'] + (molecule['corr_order']-1)) > len(molecule['prim_tuple'])):
      #
      molecule['max_corr_order'] = len(molecule['prim_tuple'])
      #
      molecule['corr_order'] = (molecule['max_corr_order'] - molecule['min_corr_order']) + 1
   #
   # default, set max_corr_order according to input corr_order
   #
   else:
      #
      molecule['max_corr_order'] = molecule['min_corr_order'] + (molecule['corr_order']-1)
   #
   for _ in range(0,molecule['min_corr_order']-1):
      #
      molecule['corr_energy'].append(0.0)
   #
   return molecule

