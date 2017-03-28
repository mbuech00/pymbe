#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_summary.py: summary print utilities for Bethe-Goldstone correlation calculations."""

import numpy as np
from contextlib import redirect_stdout

from bg_print import print_main_header

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def summary_main(molecule):
   #
   print_main_header(molecule,'bg_results.out')
   #
   summary_overall_res(molecule)
   #
   summary_detail_res(molecule)
   #
   summary_phase_time(molecule)
   #
   if (molecule['mpi_parallel']): summary_mpi_time(molecule)
   #
   summary_end(molecule)
   #
   return

def summary_overall_res(molecule):
   #
   if (molecule['local']):
      #
      occ_orbs = 'local'
   #
   else:
      #
      occ_orbs = 'canonical'
   #
   if (molecule['mp2_nat_orbs']):
      #
      virt_orbs = 'MP2 NOs'
   #
   else:
      #
      virt_orbs = 'canonical'
   #
   with open(molecule['out_dir']+'/bg_results.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print('                                              ---------------------------------------------                                                 ')
         print('                                                             overall results                                                                ')
         print('                                              ---------------------------------------------                                                 ')
         print('')
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         print('              molecular information           |            expansion information          |              calculation information            ')
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         #
         print('            basis set       =  {0:<12s}   |        expansion model    =  {1:<6s}       |       mpi parallel run       =  {2:}'.\
                 format(molecule['basis'],molecule['model'],molecule['mpi_parallel']))
         #
         print('            frozen core     =  {0:<5b}          |        expansion type     =  {1:<8s}     |       number of mpi masters  =  {2:}'.\
                 format(molecule['frozen'],molecule['exp'],1))
         #
         print('            # occ. / virt.  =  {0:<2d} / {1:<4d}      |        final BG order     =  {2:<3d}          |       number of mpi slaves   =  {3:}'.\
                 format(molecule['nocc'],molecule['nvirt'],len(molecule['prim_energy']),molecule['mpi_size']-1))
         #
         print('            occ. orbitals   =  {0:<9s}      |        exp. threshold     =  {1:<5.3f} %      |       final corr. energy     = {2:>12.5e}'.\
                 format(occ_orbs,molecule['prim_thres_init'],molecule['prim_energy'][-1]))
         #
         print('            virt. orbitals  =  {0:<9s}      |        energy threshold   =  {1:<5.2e}     |       final convergence      = {2:>12.5e}'.\
               format(virt_orbs,molecule['prim_e_thres'],molecule['prim_energy'][-1]-molecule['prim_energy'][-2]))
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_detail_res(molecule):
   #
   total_tup = 0
   #
   with open(molecule['out_dir']+'/bg_results.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print('                                              ---------------------------------------------                                                 ')
         print('                                                             detailed results                                                               ')
         print('                                              ---------------------------------------------                                                 ')
         print('')
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         print('      BG order   |       total corr. energy       |       total time (HHH : MM : SS)      |      number of calcs. (abs. / %  --  total)     ')
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         #
         for i in range(0,len(molecule['prim_energy'])):
            #
            total_time = np.sum(molecule['time_init'][:i+1])+np.sum(molecule['time_kernel'][:i+1])+np.sum(molecule['time_final'][:i+1])
            total_tup += len(molecule['prim_tuple'][i])
            #
            print('       {0:>4d}      |          {1:>7.5e}          |              {2:03d} : {3:02d} : {4:02d}            |      {5:>9d} / {6:>6.2f}   --   {7:>9d} '.\
                                              format(i+1,molecule['prim_energy'][i],\
                                                     int(total_time//3600),int((total_time-(total_time//3600)*3600.)//60),\
                                                     int(total_time-(total_time//3600)*3600.-((total_time-(total_time//3600)*3600.)//60)*60.),\
                                                     len(molecule['prim_tuple'][i]),(float(len(molecule['prim_tuple'][i]))/float(molecule['theo_work'][i]))*100.00,total_tup))
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_phase_time(molecule):
   #
   with open(molecule['out_dir']+'/bg_results.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print('                                              ---------------------------------------------                                                 ')
         print('                                                              phase timings                                                                 ')
         print('                                              ---------------------------------------------                                                 ')
         print('')
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         print('      BG order   |      time: init (HHH : MM : SS / %)     |    time: kernel (HHH : MM : SS / %)    |     time: final (HHH : MM : SS / %)   ')
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         #
         for i in range(0,len(molecule['prim_energy'])):
            #
            time_i = molecule['time_init'][i]
            time_k = molecule['time_kernel'][i]
            time_f = molecule['time_final'][i]
            time_t = molecule['time_tot'][i]
            #
            print('       {0:>4d}      |            {1:03d} : {2:02d} : {3:02d} / {4:>6.2f}\
       |          {5:03d} : {6:02d} : {7:02d} / {8:>6.2f}        |          {9:03d} : {10:02d} : {11:02d} / {12:>6.2f}'.\
                      format(i+1,int(time_i//3600),int((time_i-(time_i//3600)*3600.)//60),int(time_i-(time_i//3600)*3600.-((time_i-(time_i//3600)*3600.)//60)*60.),(time_i/time_t)*100.0,\
                             int(time_k//3600),int((time_k-(time_k//3600)*3600.)//60),int(time_k-(time_k//3600)*3600.-((time_k-(time_k//3600)*3600.)//60)*60.),(time_k/time_t)*100.0,\
                             int(time_f//3600),int((time_f-(time_f//3600)*3600.)//60),int(time_f-(time_f//3600)*3600.-((time_f-(time_f//3600)*3600.)//60)*60.),(time_f/time_t)*100.0))
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         #
         time_i = molecule['time_init'][-1]
         time_k = molecule['time_kernel'][-1]
         time_f = molecule['time_final'][-1]
         time_t = molecule['time_tot'][-1]
         #
         print('        total    |            {0:03d} : {1:02d} : {2:02d} / {3:>6.2f}\
       |          {4:03d} : {5:02d} : {6:02d} / {7:>6.2f}        |          {8:03d} : {9:02d} : {10:02d} / {11:>6.2f}'.\
                   format(int(time_i//3600),int((time_i-(time_i//3600)*3600.)//60),int(time_i-(time_i//3600)*3600.-((time_i-(time_i//3600)*3600.)//60)*60.),(time_i/time_t)*100.0,\
                          int(time_k//3600),int((time_k-(time_k//3600)*3600.)//60),int(time_k-(time_k//3600)*3600.-((time_k-(time_k//3600)*3600.)//60)*60.),(time_k/time_t)*100.0,\
                          int(time_f//3600),int((time_f-(time_f//3600)*3600.)//60),int(time_f-(time_f//3600)*3600.-((time_f-(time_f//3600)*3600.)//60)*60.),(time_f/time_t)*100.0))
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_mpi_time(molecule):
   #
   with open(molecule['out_dir']+'/bg_results.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('')
         print('                                              ---------------------------------------------                                                 ')
         print('                                                               mpi timings                                                                  ')
         print('                                              ---------------------------------------------                                                 ')
         print('')
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         print('      mpi processor   |   time: init (work/comm/idle, in %)  |  time: kernel (work/comm/idle, in %)  |  time: final (work/comm/idle, in %)  ')
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         #
         print('    master -- {0:<8d}|       {1:>6.2f} / {2:>6.2f} / {3:>6.2f}       |       {4:>6.2f} / {5:>6.2f} / {6:>6.2f}        |       {7:>6.2f} / {8:>6.2f} / {9:>6.2f}'.\
                format(0,molecule['dist_init'][0][0],molecule['dist_init'][1][0],molecule['dist_init'][2][0],\
                       molecule['dist_kernel'][0][0],molecule['dist_kernel'][1][0],molecule['dist_kernel'][2][0],\
                       molecule['dist_final'][0][0],molecule['dist_final'][1][0],molecule['dist_final'][2][0]))
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         #
         for i in range(1,molecule['mpi_size']):
            #
            print('    slave  -- {0:<8d}|       {1:>6.2f} / {2:>6.2f} / {3:>6.2f}       |       {4:>6.2f} / {5:>6.2f} / {6:>6.2f}        |       {7:>6.2f} / {8:>6.2f} / {9:>6.2f}'.\
                   format(i,molecule['dist_init'][0][i],molecule['dist_init'][1][i],molecule['dist_init'][2][i],\
                          molecule['dist_kernel'][0][i],molecule['dist_kernel'][1][i],molecule['dist_kernel'][2][i],\
                          molecule['dist_final'][0][i],molecule['dist_final'][1][i],molecule['dist_final'][2][i]))
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
         #
         print('    mean  : slaves    |       {0:>6.2f} / {1:>6.2f} / {2:>6.2f}       |       {3:>6.2f} / {4:>6.2f} / {5:>6.2f}        |       {6:>6.2f} / {7:>6.2f} / {8:>6.2f}'.\
                format(np.mean(molecule['dist_init'][0][1:]),np.mean(molecule['dist_init'][1][1:]),np.mean(molecule['dist_init'][2][1:]),\
                       np.mean(molecule['dist_kernel'][0][1:]),np.mean(molecule['dist_kernel'][1][1:]),np.mean(molecule['dist_kernel'][2][1:]),\
                       np.mean(molecule['dist_final'][0][1:]),np.mean(molecule['dist_final'][1][1:]),np.mean(molecule['dist_final'][2][1:])))
         #
         print('    stdev : slaves    |       {0:>6.2f} / {1:>6.2f} / {2:>6.2f}       |       {3:>6.2f} / {4:>6.2f} / {5:>6.2f}        |       {6:>6.2f} / {7:>6.2f} / {8:>6.2f}'.\
                format(np.std(molecule['dist_init'][0][1:],ddof=1),np.std(molecule['dist_init'][1][1:],ddof=1),np.std(molecule['dist_init'][2][1:],ddof=1),\
                       np.std(molecule['dist_kernel'][0][1:],ddof=1),np.std(molecule['dist_kernel'][1][1:],ddof=1),np.std(molecule['dist_kernel'][2][1:],ddof=1),\
                       np.std(molecule['dist_final'][0][1:],ddof=1),np.std(molecule['dist_final'][1][1:],ddof=1),np.std(molecule['dist_final'][2][1:],ddof=1)))
         #
         print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_end(molecule):
   #
   with open(molecule['out_dir']+'/bg_results.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('\n')
   #
   return


