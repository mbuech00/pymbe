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
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def print_main_header(molecule):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print("    oooooooooo.                .   oooo")
         print("    `888'   `Y8b             .o8   `888")
         print("     888     888  .ooooo.  .o888oo  888 .oo.    .ooooo.")
         print("     888oooo888' d88' `88b   888    888P'Y88b  d88' `88b")
         print("     888    `88b 888ooo888   888    888   888  888ooo888 8888888")
         print("     888    .88P 888    .o   888 .  888   888  888    .o")
         print("    o888bood8P'  `Y8bod8P'   '888' o888o o888o `Y8bod8P'")
         print('')
         print("      .oooooo.              oooo        .o8               .")
         print("     d8P'  `Y8b             `888       '888             .o8")
         print("    888            .ooooo.   888   .oooo888   .oooo.o .o888oo  .ooooo.  ooo. .oo.    .ooooo.")
         print("    888           d88' `88b  888  d88' `888  d88(  '8   888   d88' `88b `888P'Y88b  d88' `88b")
         print("    888     ooooo 888   888  888  888   888  `'Y88b.    888   888   888  888   888  888ooo888")
         print("    `88.    .88'  888   888  888  888   888  o.  )88b   888 . 888   888  888   888  888    .o")
         print("     `Y8bood8P'   `Y8bod8P' o888o `Y8bod88P' 8''888P'   '888' `Y8bod8P' o888o o888o `Y8bod8P'")
         print('')
   #
   # print a table with mpi information
   #
   if (molecule['mpi_parallel']): print_mpi_table(molecule)
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
         print('')
         print('')
   #
   return

def print_init_header(molecule,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-{0:}: order = {1:>d} initialization started'.format(level,order))
         print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_init_end(molecule,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-{0:}: order = {1:>d} initialization done'.format(level,order))
         print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_final_header(molecule,order,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-{0:}: order = {1:>d} finalization started'.format(level,order))
         print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_final_end(molecule,order,conv,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         if ((level == 'MACRO') and conv):
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' STATUS-{0:}: order = {1:>d} finalization done --- *** calculation has converged ***'.format(level,order))
            print(' --------------------------------------------------------------------------------------------')
         #
         else:
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' STATUS-{0:}: order = {1:>d} finalization done'.format(level,order))
            print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_status_header(molecule,tup,order,conv,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         if ((level == 'MACRO') and conv):
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' STATUS-{0:}: order = {1:>d} has no contributions --- *** calculation has converged ***'.format(level,order))
            print(' --------------------------------------------------------------------------------------------')
         #
         else:
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' STATUS-{0:}: order = {1:>d} energy calculation started  ---  {2:d} tuples in total'.format(level,order,len(tup)))
            print(' --------------------------------------------------------------------------------------------')
   #
   if ((level == 'MACRO') and conv):
      #
      print('')
      print(' --------------------------------------------------------------------------------------------')
      print(' STATUS-{0:}: order = {1:>d} has no contributions --- *** calculation has converged ***'.format(level,order))
      print(' --------------------------------------------------------------------------------------------')
   #
   else:
      #
      print('')
      print(' --------------------------------------------------------------------------------------------')
      print(' STATUS-{0:}: order = {1:>d} energy calculation started  ---  {2:d} tuples in total'.format(level,order,len(tup)))
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

def print_status_end(molecule,order,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
   #
   elif (level == 'CORRE'):
      #
      tup = molecule['corr_tuple']
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' STATUS-{0:}: order = {1:>d} energy calculation done'.format(level,order))
         print(' --------------------------------------------------------------------------------------------')
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} energy calculation done'.format(level,order))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_result(molecule,tup,e_inc,level):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print(' --------------------------------------------------------------------------------------------')
         print(' RESULT-{0:}:     tuple    |    energy incr.   |    corr. orbs.'.format(level))
         print(' --------------------------------------------------------------------------------------------')
         #
         for i in range(0,len(tup)):
            #
            print(' RESULT-{0:}:  {1:>6d}           {2:> 8.4e}         {3!s:<}'.format(level,i+1,e_inc[i],tup[i].tolist()))
         #
         print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_update(molecule,l_limit,u_limit,level):
   #
   if (level == 'MACRO'):
      #
      dom = molecule['prim_domain']
   #
   elif (level == 'CORRE'):
      #
      dom = molecule['corr_domain']
   #
   count = []
   #
   for j in range(0,u_limit):
      #
      if ((len(dom[-2][j]) >= 1) and (float(len(dom[-1][j]))/float(len(dom[-2][j])) != 1.0)):
         #
         count.append(True)
      #
      else:
         #
         count.append(False)
   #
   if (any(count)):
      #
      with open(molecule['out_dir']+'/bg_output.out','a') as f:
         #
         with redirect_stdout(f):
            #
            print(' --------------------------------------------------------------------------------------------')
            print(' UPDATE-{0:}:   orb. domain  |  relat. red. (in %)  |   total red. (in %)  |  screened orbs.  '.format(level))
            print(' --------------------------------------------------------------------------------------------')
            #
            for j in range(0,u_limit):
               #
               if (count[j]):
                  #
                  print(' UPDATE-{0:}:     {1!s:>5}              {2:>6.2f}                 {3:>6.2f}            {4!s:<}'.\
                                format(level,[(j+l_limit)+1],\
                                       (1.0-float(len(dom[-1][j]))/float(len(dom[-2][j])))*100.00,\
                                       (1.0-float(len(dom[-1][j]))/float(len(dom[0][j])))*100.00,\
                                       sorted(list(set(dom[-2][j])-set(dom[-1][j])))))
            #
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

def print_orb_info(molecule,l_limit,u_limit,level):
   #
   if (level == 'MACRO'):
      #
      orb = molecule['prim_orb_ent']
      orb_arr = molecule['prim_orb_arr']
      orb_con_rel = molecule['prim_orb_con_rel']
   #
   elif (level == 'CORRE'):
      #
      orb = molecule['corr_orb_ent']
      orb_arr = molecule['corr_orb_arr']
      orb_con_rel = molecule['corr_orb_con_rel']
   #
   tmp = np.empty(u_limit,dtype=object)
   #
   index = []
   #
   index_strings(l_limit,u_limit,index)
   #
   if (level == 'MACRO'):
      #
      with open(molecule['out_dir']+'/bg_output.out','a') as f:
         #
         with redirect_stdout(f):
            #
            print('')
            print('   ---------------------------------------------')
            print('      individual orbital contributions (in %)   ')
            print('   ---------------------------------------------')
            #
            print('')
            print(' * BG exp. order = 1')
            print(' -------------------')
            #
            print('')
            print(index[0])
            print('               '+str(['{0:3d}'.format(int(m*100.0)) for m in orb_con_rel[0]]))
            print('')
   #
   if (level == 'MACRO'):
      #
      start = 0
   #
   elif ((level == 'CORRE') and (molecule['min_corr_order'] > 0)):
      #
      start = molecule['min_corr_order']-2
      #
      if (start <= (len(orb)-1)):
         #
         with open(molecule['out_dir']+'/bg_output.out','w') as f:
            #
            with redirect_stdout(f):
               #
               print('')
               print('   ---------------------------------------------')
               print('      individual orbital contributions (in %)   ')
               print('   ---------------------------------------------')
   #
   for i in range(start,len(orb)):
      #
      with open(molecule['out_dir']+'/bg_output.out','w') as f:
         #
         with redirect_stdout(f):
            #
            print('')
            print(' * BG exp. order = '+str(i+2))
            print(' -------------------')
            #
            print('')
            print(index[1])
            #
            for j in range(0,len(orb_arr[i])):
               #
               for l in range(0,len(orb_arr[i][j])):
                  #
                  if (orb_arr[i][j,l] == 0.0):
                     #
                     tmp[l] = '   '
                  #
                  else:
                     #
                     tmp[l] = int(orb_arr[i][j,l]*100.0)
               #
               print('          {0:>3d}  '.format((j+l_limit)+1)+str(['{0:3}'.format(m) for m in tmp]))
            #
            print('')
            print(index[0])
            print('               '+str(['{0:3d}'.format(int(m*100.0)) for m in orb_con_rel[i+1]]))
            print('')
   #
   del tmp
   #
   return

def index_strings(l_limit,u_limit,index):
   #
   for i in range(0,2):
      #
      if (i == 0):
         #
         index.append(' tot. contrib.    ')
      #
      elif (i == 1):
         #
         index.append(' entanglement     ')
      #
      for m in range(l_limit+1,(l_limit+u_limit)+1):
         #
         if (m < 10):
            #
            index[i] += str(m)+'      '
         #
         elif ((m >= 10) and (m < 100)):
            #
            index[i] += str(m)+'     '
         #
         elif ((m >= 100)):
            #
            index[i] += str(m)+'    '
   #
   return index

