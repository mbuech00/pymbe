#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_print.py: print utilities for Bethe-Goldstone correlation calculations."""

import sys
from os import getcwd, mkdir
from os.path import isdir
from shutil import rmtree

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
   print('\n')
   print(' ** start bethe-goldstone '+molecule['model']+' calculation  ---  '+str(molecule['scheme'])+' expansion scheme **')
   #
   return

def print_main_end(molecule):
   #
   print(' ** end of bethe-goldstone '+molecule['model']+' calculation **\n')
   print('\n')
   #
   return

def print_summary(molecule):
   #
   print('')
   print('   *************************************************')
   print('   ****************                 ****************')
   print('   ***********          RESULTS          ***********')
   print('   ****************                 ****************')
   print('   *************************************************\n')
   #
   print('            bethe-goldstone {0:} expansion\n'.format(molecule['model']))
   #
   print('   ---------------------------------------------')
   print('              molecular information             ')
   print('   ---------------------------------------------')
   print('')
   print('   frozen core                  =  {0:}'.format(molecule['frozen']))
   print('   local orbitals               =  {0:}'.format(molecule['local']))
   print('   occupied orbitals            =  {0:}'.format(molecule['nocc']))
   print('   virtual orbitals             =  {0:}'.format(molecule['nvirt']))
   #
   print('')
   print('   ---------------------------------------------')
   print('              expansion information             ')
   print('   ---------------------------------------------')
   print('')
   #
   print('   type of expansion            =  {0:}'.format(molecule['scheme']))
   #
   print('   bethe-goldstone order        =  {0:}'.format(len(molecule['prim_energy'])))
   #
   print('   prim. exp. threshold         =  {0:5.3f} %'.format(molecule['prim_thres']*100.00))
   #
   if ((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')):
      #
      print('   sec. exp. threshold          =  {0:5.3f} %'.format(molecule['sec_thres']*100.00))
   #
   print('   energy correction            =  {0:}'.format(molecule['corr']))
   #
   if (molecule['corr']):
      #
      print('   energy correction order      =  {0:}'.format(molecule['max_corr_order']))
      print('   energy correction threshold  =  {0:5.3f} %'.format(molecule['corr_thres']*100.00))
   #
   else:
      #
      print('   energy correction order      =  N/A')
   #
   print('   error in calculation         =  {0:}'.format(molecule['error'][-1]))
   #
   print('')
   print('   ---------------------------------------------')
   print('                  mpi information               ')
   print('   ---------------------------------------------')
   print('')
   #
   print('   mpi parallel run             =  {0:}'.format(molecule['mpi_parallel']))
   #
   if (molecule['mpi_parallel']):
      #
      print('   number of mpi processes      =  {0:}'.format(molecule['mpi_size']))
      #
      print('   number of mpi masters        =  {0:}'.format(1))
      #
      print('   number of mpi slaves         =  {0:}'.format(molecule['mpi_size']-1))
      #
      print('   -- time (idle)               =  {0:<d} %'.format(int(molecule['mpi_time_idle'][1]+0.5)))
      #
      print('   -- time (communication)      =  {0:<d} %'.format(int(molecule['mpi_time_comm'][1]+0.5)))
      #
      print('   -- time (comp. work)         =  {0:<d} %'.format(int(molecule['mpi_time_work'][1]+0.5)))
   #
   print('')
   print('   ---------------------------------------------')
   print('                   final results                ')
   print('   ---------------------------------------------')
   print('')
   #
   print('   final energy (excl. corr.)   =  {0:>12.5e}'.format(molecule['prim_energy'][-1]))
   print('   final energy (incl. corr.)   =  {0:>12.5e}'.format(molecule['prim_energy'][-1]+molecule['corr_energy'][-1]))
   #
   print('   ---------------------------------------------')
   #
   print('   final conv. (excl. corr.)    =  {0:>12.5e}'.format(molecule['prim_energy'][-1]-molecule['prim_energy'][-2]))
   print('   final conv. (incl. corr.)    =  {0:>12.5e}'.format((molecule['prim_energy'][-1]+molecule['corr_energy'][-1])-(molecule['prim_energy'][-2]+molecule['corr_energy'][-2])))
   #
   print('   ---------------------------------------------')
   #
   if (molecule['ref'] and (not molecule['error'][-1])):
      #
      final_diff = molecule['e_ref']-molecule['prim_energy'][-1]
      final_diff_corr = molecule['e_ref']-(molecule['prim_energy'][-1]+molecule['corr_energy'][-1])
      #
      if (abs(final_diff) < 1.0e-10):
         #
         final_diff = 0.0
      #
      if (abs(final_diff_corr) < 1.0e-10):
         #
         final_diff_corr = 0.0
      #
      print('   final diff. (excl. corr.)    =  {0:>12.5e}'.format(final_diff))
      print('   final diff. (incl. corr.)    =  {0:>12.5e}'.format(final_diff_corr))
   #
   print('')
   print('')
   print('                                              ---------------------------------------------                                                 ')
   print('                                                             detailed results                                                               ')
   print('                                              ---------------------------------------------                                                 ')
   print('')
   #
   tot_n_tup = []
   #
   for i in range(0,len(molecule['prim_energy'])):
      #
      if (molecule['prim_n_tuples'][i] == molecule['theo_work'][i]):
         #
         tot_n_tup.append(molecule['prim_n_tuples'][i])
      #
      else:
         #
         tot_n_tup.append(molecule['prim_n_tuples'][i]+molecule['corr_n_tuples'][i])
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('     BG expansion order  |   # of prim. exp. tuples   |   # of corr. tuples   |   perc. of total # of tuples:   excl. corr.  |  incl. corr.  ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['prim_energy'])):
      #
      print('          {0:>4d}                     {1:>4.2e}                    {2:>4.2e}                                           {3:>6.2f} %        {4:>6.2f} %'.\
                                                                          format(i+1,molecule['prim_n_tuples'][i],molecule['corr_n_tuples'][i],\
                                                                                 (float(molecule['prim_n_tuples'][i])/float(molecule['theo_work'][i]))*100.00,\
                                                                                 (float(tot_n_tup[i])/float(molecule['theo_work'][i]))*100.00))
   #
   total_time = 0.0
   total_time_corr = 0.0
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('   |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('     BG expansion order  |   total prim. exp. energy   |    total energy incl. energy corr.   |    total time    |    total time incl. corr.')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['prim_energy'])):
      #
      total_time += molecule['prim_time'][i]
      total_time_corr += molecule['corr_time'][i]
      #
      print('          {0:>4d}                    {1:>7.5e}                      {2:>7.5e}                   {3:4.2e} s              {4:4.2e} s'.\
                                                                          format(i+1,molecule['prim_energy'][i],molecule['prim_energy'][i]+molecule['corr_energy'][i],\
                                                                                 total_time,total_time+total_time_corr))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   print('\n')
   #
   return molecule

def print_init_header(order,level):
   #
   print('')
   print('')
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} initialization started'.format(level,order))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_init_end(order,time_init,level):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} initialization done in {2:8.2e} seconds'.format(level,order,time_init))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_final_header(order,level):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} finalization started'.format(level,order))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_final_end(order,time_final,level):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} finalization done in {2:8.2e} seconds'.format(level,order,time_final))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_status_header(num,order,conv,level):
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
      print(' STATUS-{0:}: order = {1:>d} energy calculation started  ---  {2:d} tuples in total'.format(level,order,num))
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
      n_tup = molecule['prim_n_tuples']
      time = molecule['prim_time']
   #
   elif (level == 'CORRE'):
      #
      n_tup = molecule['corr_n_tuples']
      time = molecule['corr_time']
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   if (n_tup[-1] == 0):
      #
      print(' STATUS-{0:}: order = {1:>d} energy calculation done'.format(level,order))
   #
   else:
      #
      print(' STATUS-{0:}: order = {1:>d} energy calculation done in {2:8.2e} seconds'.format(level,order,time[-1]))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_result(tup,level):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' RESULT-{0:}:     tuple    |    energy incr.   |    corr. orbs.'.format(level))
   print(' --------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(tup)):
      #
      print(' RESULT-{0:}:  {1:>6d}           {2:> 8.4e}         {3!s:<}'.format(level,i+1,tup[i][1],tup[i][0]))
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_inner_result(molecule):
   #
   rel_work_in = []
   #
   for m in range(0,len(molecule['rel_work_in'])):
      #
      rel_work_in.append([])
      #
      for n in range(0,len(molecule['rel_work_in'][m])):
         #
         rel_work_in[m].append('{0:.2f}'.format(molecule['rel_work_in'][m][n]))
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' RESULT-MICRO:     tuple    |   abs. energy diff.   |    relat. no. tuples (in %)')
   print(' --------------------------------------------------------------------------------------------')
   #
   for i in range(0,molecule['n_tuples'][-1]):
      #
      print(' RESULT-MICRO:  {0:>6d}            {1:> 8.4e}            '.\
                       format(i+1,molecule['e_diff_in'][i])+'[{0!s:<}]'.format(', '.join(str(idx) for idx in rel_work_in[i])))
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

def print_ref_header():
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
   print(' STATUS-REF: full reference calculation done in {0:10.2e} seconds'.format(molecule['prim_time'][-1]))
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
   index = '          '
   #
   for m in range(l_limit+1,(l_limit+u_limit)+1):
      #
      if (m < 10):
         #
         index += str(m)+'         '
      #
      elif ((m >= 10) and (m < 100)):
         #
         index += str(m)+'        '
      #
      elif ((m >= 100)):
         #
         index += str(m)+'       '
   #
   if (level == 'MACRO'):
      #
      print('')
      print('   ---------------------------------------------')
      print('         individual orbital contributions       ')
      print('   ---------------------------------------------')
      #
      print('')
      print(' * BG exp. order = 1')
      print(' -------------------')
      print('')
      #
      print('      --- relative orbital contributions ---')
      print('')
      #
      print(index)
      #
      print('     '+str(['{0:6.3f}'.format(m) for m in orb_con_rel[0]]))
      #
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
         print('')
         print('   ---------------------------------------------')
         print('         individual orbital contributions       ')
         print('   ---------------------------------------------')
   #
   for i in range(start,len(orb)):
      #
      print('')
      print(' * BG exp. order = '+str(i+2))
      print(' -------------------')
      print('')
      #
      print('      --- entanglement matrix ---')
      print('')
      #
      print(index)
      #
      for j in range(0,len(orb_arr[i])):
         #
         print(' {0:>3d}'.format((j+l_limit)+1)+' '+str(['{0:6.3f}'.format(m) for m in orb_arr[i][j]]))
      #
      print('')
      print('      --- relative orbital contributions ---')
      print('')
      #
      print(index)
      #
      print('     '+str(['{0:6.3f}'.format(m) for m in orb_con_rel[i+1]]))
      #
      if (i == (len(orb)-1)): print('')
   #
   return

def redirect_stdout(molecule):
   #
   molecule['wrk'] = getcwd()
   #
   if (isdir(molecule['wrk']+'/output')): rmtree(molecule['wrk']+'/output',ignore_errors=True)
   #
   mkdir(molecule['wrk']+'/output')
   #
   sys.stdout = logger(molecule['wrk']+'/output/stdout.out')
   #
   return molecule

class logger(object):
   #
   def __init__(self, filename="default.log"):
      #
      self.terminal = sys.stdout
      self.log = open(filename, "a")
   #
   def write(self, message):
      #
      self.terminal.write(message)
      self.log.write(message)
   #
   def flush(self):
      #
      pass


