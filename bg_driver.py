#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver routines for Bethe-Goldstone correlation calculations."""

import numpy as np

from bg_mpi_utils import prepare_calc
from bg_mpi_time import timer_mpi, collect_screen_mpi_time
from bg_print import print_mono_exp_header, print_mono_exp_end, print_kernel_header, print_kernel_end, print_results,\
                     print_screen_header, print_screen_end, print_summation_header, print_summation_end
from bg_energy import energy_kernel_mono_exp, energy_summation, chk_energy_conv
from bg_orbitals import init_domains, update_domains, orb_generator,\
                        orb_screening, orb_exclusion
from bg_rst_main import rst_main
from bg_rst_write import rst_write_tup, rst_write_dom, rst_write_orb_ent, rst_write_orb_arr, rst_write_excl_list,\
                         rst_write_orb_con, rst_write_e_inc, rst_write_e_tot, rst_write_time

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
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
      # print header for mono expansion
      #
      print_mono_exp_header(molecule)
      #
      # check for restart files
      #
      rst_main(molecule)
      #
      # call mono expansion driver function 
      #
      mono_exp_drv(molecule,molecule['min_order'],molecule['max_order'],'MACRO')
      #
      # print end for mono expansion
      #
      print_mono_exp_end(molecule)
   #
#   elif ((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')):
#      #
#      # run dual expansion (not implemented yet...)
   #
   return molecule

def mono_exp_drv(molecule,start,end,level):
   #
   for k in range(start,end+1):
      #
      # mono expansion energy kernel
      #
      mono_exp_kernel(molecule,k,level)
      #
      # mono expansion energy summation
      #
      mono_exp_summation(molecule,k,level)
      #
      # mono expansion screening
      #
      mono_exp_screen(molecule,k,level)
      #
      # return if converged
      #
      if (molecule['conv_orb'][-1] or molecule['conv_energy'][-1]): break
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
   # print kernel header
   #
   print_kernel_header(molecule,tup[-1],k,level)
   #
   # init e_int list
   #
   if (k != molecule['min_order']): e_inc.append(np.zeros(len(tup[k-1]),dtype=np.float64))
   #
   # run the calculations
   #
   energy_kernel_mono_exp(molecule,k,tup,e_inc,molecule['l_limit'],molecule['u_limit'],level)
   #
   # print kernel end
   #
   print_kernel_end(molecule,tup,k,level)
   #
   return molecule

def mono_exp_summation(molecule,k,level):
   #
   print_summation_header(molecule,k,level)
   #
   # calculate the energy at order k
   #
   energy_summation(molecule,k,molecule['prim_tuple'],molecule['prim_energy_inc'],molecule['prim_energy'],level)
   #
   # write restart files
   #
   rst_write_e_inc(molecule,k)
   rst_write_e_tot(molecule,k)
   #
   chk_energy_conv(molecule,molecule['prim_energy'],k)
   #
   print_summation_end(molecule,k,level)
   # 
   # print results
   #
   print_results(molecule,molecule['prim_tuple'][k-1],molecule['prim_energy_inc'][k-1],level)
   #
   return molecule

def mono_exp_screen(molecule,k,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      dom = molecule['prim_domain']
      orb = molecule['prim_orb_ent']
      thres = molecule['prim_thres']
   #
   # print screen header
   #
   print_screen_header(molecule,k,level)
   #
   # orbital screening (using info from order k-1)
   #
   if (molecule['conv_energy'][-1]):
      #
      orb_screening(molecule,molecule['l_limit'],molecule['u_limit'],k,level,True)
   #
   else:
      #
      orb_screening(molecule,molecule['l_limit'],molecule['u_limit'],k,level)
      #
      # generate all tuples at order k+1
      #
      orb_generator(molecule,dom,tup,molecule['l_limit'],molecule['u_limit'],k,level)
      #
      timer_mpi(molecule,'mpi_time_work_screen',k)
      #
      # check for convergence wrt prim_thres
      #
      if (len(tup[k]) == 0):
         #
         tup.pop(-1)
         #
         molecule['conv_orb'].append(True)
      #
      # write restart files
      #
      if (k >= 1):
         #
         rst_write_dom(molecule,k)
         rst_write_orb_con(molecule,k-1)
         rst_write_tup(molecule,k)
      #
      if (k >= 2):
         #
         rst_write_orb_ent(molecule,k-2)
         rst_write_orb_arr(molecule,k-2)
         rst_write_excl_list(molecule,k-2)
      #
      if (molecule['mpi_parallel']):
         #
         collect_screen_mpi_time(molecule,k,True)
      #
      else:
         #
         timer_mpi(molecule,'mpi_time_work_screen',k,True)
         #
         rst_write_time(molecule,'screen')
   #
   # print screen end
   #
   print_screen_end(molecule,k,level)
   #
   return molecule

