#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver routines for Bethe-Goldstone correlation calculations."""

import numpy as np

from bg_mpi_utils import prepare_calc
from bg_mpi_time import timer_mpi
from bg_print import print_mono_exp_header, print_mono_exp_end, print_kernel_header, print_kernel_end, print_results,\
                     print_summation_header, print_summation_end, print_screen_header, print_screen_end
from bg_energy import energy_kernel_mono_exp, energy_summation, chk_energy_conv
from bg_entanglement import entanglement_main
from bg_screening import screening_main
from bg_rst_main import rst_main
from bg_rst_write import rst_write_summation, rst_write_screen

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

def mono_exp_kernel(molecule,order,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      e_inc = molecule['prim_energy_inc']
      e_tot = molecule['prim_energy']
   #
   # print kernel header
   #
   print_kernel_header(molecule,tup[-1],order,level)
   #
   # init e_int list
   #
   if (order != molecule['min_order']): e_inc.append(np.zeros(len(tup[-1]),dtype=np.float64))
   #
   # run the calculations
   #
   energy_kernel_mono_exp(molecule,order,tup,e_inc,molecule['l_limit'],molecule['u_limit'],level)
   #
   # print kernel end
   #
   print_kernel_end(molecule,tup,order,level)
   #
   return molecule

def mono_exp_summation(molecule,order,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      e_inc = molecule['prim_energy_inc']
      e_tot = molecule['prim_thres']
   #
   # print summation header
   #
   print_summation_header(molecule,order,level)
   #
   # calculate the energy at current order
   #
   energy_summation(molecule,order,tup,e_inc,e_tot,level)
   #
   # write restart files
   #
   rst_write_summation(molecule,e_inc,e_tot,order)
   #
   # print summation end
   #
   print_summation_end(molecule,order,level)
   # 
   # print results
   #
   print_results(molecule,tup,e_inc,level)
   #
   return molecule

def mono_exp_screen(molecule,order,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      n_tup = molecule['prim_n_tuples']
      e_inc = molecule['prim_energy_inc']
      thres = molecule['prim_thres']
   #
   # print screen header
   #
   print_screen_header(molecule,order,level)
   #
   # orbital entanglement
   #
   entanglement_main(molecule,molecule['l_limit'],molecule['u_limit'],order,molecule['conv_energy'][-1])
   #
   # orbital screening (using info from previous order)
   #
   if (not molecule['conv_energy'][-1]):
      #
      # perform screening
      #
      screening_main(molecule,tup,n_tup,e_inc,thres,molecule['l_limit'],molecule['u_limit'],order)
      #
      # write restart files
      #
      rst_write_screen(molecule,tup,n_tup,e_inc,order)
   #
   # print screen end
   #
   print_screen_end(molecule,k,level)
   #
   return molecule

