#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_print.py: general print utilities for Bethe-Goldstone correlation calculations."""

import sys
import numpy as np
from contextlib import redirect_stdout

from bg_mpi_utils import print_mpi_table

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def print_main_header(molecule,foo,init=False):
   #
   with open(molecule['out_dir']+'/'+foo,'a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print("   oooooooooo.                .   oooo")
         print("   `888'   `Y8b             .o8   `888")
         print("    888     888  .ooooo.  .o888oo  888 .oo.    .ooooo.")
         print("    888oooo888' d88' `88b   888    888P'Y88b  d88' `88b")
         print("    888    `88b 888ooo888   888    888   888  888ooo888  888888")
         print("    888    .88P 888    .o   888 .  888   888  888    .o")
         print("   o888bood8P'  `Y8bod8P'   '888' o888o o888o `Y8bod8P'")
         print('')
         print("     .oooooo.              oooo        .o8               .")
         print("    d8P'  `Y8b             `888       '888             .o8")
         print("   888            .ooooo.   888   .oooo888   .oooo.o .o888oo  .ooooo.  ooo. .oo.    .ooooo.")
         print("   888           d88' `88b  888  d88' `888  d88(  '8   888   d88' `88b `888P'Y88b  d88' `88b")
         print("   888     ooooo 888   888  888  888   888  `'Y88b.    888   888   888  888   888  888ooo888")
         print("   `88.    .88'  888   888  888  888   888  o.  )88b   888 . 888   888  888   888  888    .o")
         print("    `Y8bood8P'   `Y8bod8P' o888o `Y8bod88P' `Y8888P'   '888' `Y8bod8P' o888o o888o `Y8bod8P'")
         print('')
         print('')
         print('   --- an incremental Python-based electronic structure correlation program written by:')
         print('')
         print('             Janus Juul Eriksen')
         print('')
         print('       with contributions from:')
         print('')
         print('             Filippo Lipparini')
         print('               & Juergen Gauss')
         print('')
         print('                                        *****')
         print('                                   ***************')
         print('                                        *****')
   #
   # print a table with mpi information
   #
   if (molecule['mpi_parallel'] and init): print_mpi_table(molecule)
   #
   return

def print_main_end(molecule):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
   #
   return

def print_mono_exp_header(molecule):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print('                     ---------------------------------------------                ')
         print('                                   primary expansion                              ')
         print('                     ---------------------------------------------                ')
   #
   return

def print_mono_exp_end(molecule):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
   #
   print('')
   print('')
   #
   return

def print_kernel_header(molecule,tup,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-{0:}: order = {1:>d} energy kernel started  ---  {2:d} tuples in total'.format(level,order,len(tup)))
         print(' --------------------------------------------------------------------------------------------')
   #
   print('')
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} energy kernel started  ---  {2:d} tuples in total'.format(level,order,len(tup)))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_status(prog,level):
   #
   bar_length = 50
   #
   status = ""
   #
   block = int(round(bar_length * prog))
   #
   print(' STATUS-{0:}:   [{1}]   ---  {2:>6.2f} % {3}'.format(level,'#' * block + '-' * (bar_length - block), prog * 100, status))
   #
   return

def print_kernel_end(molecule,tup,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-{0:}: order = {1:>d} energy kernel done'.format(level,order))
         print(' --------------------------------------------------------------------------------------------')
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} energy kernel done'.format(level,order))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_summation_header(molecule,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-{0:}: order = {1:>d} energy summation started'.format(level,order))
         print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_results(molecule,tup,e_inc,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' RESULT-{0:}:     mean cont.    |   min. abs. cont.   |   max. abs. cont.   |    std.dev.'.format(level))
         print(' --------------------------------------------------------------------------------------------')
         #
         print(' RESULT-{0:}:  {1:>13.4e}    |  {2:>13.4e}      |  {3:>13.4e}      |   {4:<13.4e}'.\
                 format(level,np.mean(e_inc[-1]),e_inc[-1][np.argmin(np.abs(e_inc[-1]))],\
                        e_inc[-1][np.argmax(np.abs(e_inc[-1]))],np.std(e_inc[-1],ddof=1)))
         #
         print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_summation_end(molecule,e_inc,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         if (molecule['conv_energy'][-1]):
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' STATUS-{0:}: order = {1:>d} energy summation done (E = {2:.6e}) --- *** convergence ***'.format(level,order,np.sum(e_inc[-1])))
            print(' --------------------------------------------------------------------------------------------')
         #
         else:
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' STATUS-{0:}: order = {1:>d} energy summation done (E = {2:.6e})'.format(level,order,np.sum(e_inc[-1])))
            print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_screen_header(molecule,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-{0:}: order = {1:>d} screening started'.format(level,order))
         print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_screening(molecule,thres,tup,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' UPDATE-{0:}: threshold value of {1:.2e} resulted in screening of {2:.2f} % of the tuples'.\
                 format(level,thres,(1.0-(len(molecule['parent_tup'])/len(tup[-1])))*100.0))
         print(' --------------------------------------------------------------------------------------------')
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' UPDATE-{0:}: threshold value of {1:.2e} resulted in screening of {2:.2f} % of the tuples'.\
           format(level,thres,(1.0-(len(molecule['parent_tup'])/len(tup[-1])))*100.0))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_screen_end(molecule,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         if (molecule['conv_orb'][-1]):
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' STATUS-{0:}: order = {1:>d} screening done ---  *** convergence ***'.format(level,order))
            print(' --------------------------------------------------------------------------------------------')
         #
         else:
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' STATUS-{0:}: order = {1:>d} screening done'.format(level,order))
            print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_ref_header(molecule):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('                     ---------------------------------------------                ')
         print('                                reference calculation                             ')
         print('                     ---------------------------------------------                ')
         print('')
         print('')
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-REF: full reference calculation started')
         print(' --------------------------------------------------------------------------------------------')
   #
   print('')
   print('                     ---------------------------------------------                ')
   print('                                reference calculation                             ')
   print('                     ---------------------------------------------                ')
   print('')
   print('')
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-REF: full reference calculation started')
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_ref_end(molecule):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' STATUS-REF: full reference calculation done in {0:8.2e} seconds'.format(molecule['ref_time']))
         print(' --------------------------------------------------------------------------------------------')
         print('')
   #
   print(' STATUS-REF: full reference calculation done in {0:8.2e} seconds'.format(molecule['ref_time']))
   print(' --------------------------------------------------------------------------------------------')
   print('')
   #
   return

