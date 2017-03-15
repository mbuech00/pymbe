#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_summary.py: summary print utilities for Bethe-Goldstone correlation calculations."""

import numpy as np

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
   summary_header(molecule)
   #
   summary_mol(molecule)
   #
   summary_exp(molecule)
   #
   summary_mpi(molecule)
   #
   summary_res(molecule)
   #
   summary_detail_res_1(molecule)
   #
   summary_detail_res_2(molecule)
   #
   summary_phase_time(molecule)
   #
   if (molecule['mpi_parallel']): summary_mpi_time(molecule)
   #
   summary_end()
   #
   return

def summary_header(molecule):
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
   return

def summary_mol(molecule):
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
   return

def summary_exp(molecule):
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
   print('   prim. exp. threshold         =  {0:<3.1f} %'.format(molecule['prim_thres']))
   #
   if ((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')):
      #
      print('   sec. exp. threshold          =  {0:<3.1f} %'.format(molecule['sec_thres']))
   #
   print('   energy correction            =  {0:}'.format(molecule['corr']))
   #
   if (molecule['corr']):
      #
      print('   energy correction order      =  {0:}'.format(molecule['max_corr_order']))
      print('   energy correction threshold  =  {0:<3.1f} %'.format(molecule['corr_thres']))
   #
   else:
      #
      print('   energy correction order      =  N/A')
   #
   print('   error in calculation         =  {0:}'.format(molecule['error'][-1]))
   #
   return

def summary_mpi(molecule):
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
   return

def summary_res(molecule):
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
   return

def summary_detail_res_1(molecule):
   #
   print('')
   print('')
   print('                                              ---------------------------------------------                                                 ')
   print('                                                             detailed results                                                               ')
   print('                                              ---------------------------------------------                                                 ')
   print('')
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('     BG expansion order  |   total prim. exp. energy   |    total energy incl. energy corr.   |    total time incl. corr. (HHH : MM : SS)   ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['prim_energy'])):
      #
      total_time = np.sum(molecule['time_init'][:i+1])+np.sum(molecule['time_kernel'][:i+1])+np.sum(molecule['time_final'][:i+1])
      #
      print('          {0:>4d}                    {1:>7.5e}                      {2:>7.5e}                                {3:03d} : {4:02d} : {5:02d}'.\
                                        format(i+1,molecule['prim_energy'][i],molecule['prim_energy'][i]+molecule['corr_energy'][i],\
                                               int(total_time//3600),int((total_time-(total_time//3600)*3600.)//60),int(total_time-(total_time//3600)*3600.-((total_time-(total_time//3600)*3600.)//60)*60.)))
   #
   return

def summary_detail_res_2(molecule):
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('   |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('     BG expansion order  |  number of calcs. in prim. exp. (total/in %)  |  number of calcs. incl. energy corr. (total/in %)  |     total   ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   total_all = 0
   #
   for i in range(0,len(molecule['prim_energy'])):
      #
      total_order = len(molecule['prim_tuple'][i])+len(molecule['corr_tuple'][i])
      total_all += total_order
      #
      print('          {0:>4d}                        {1:>9d} / {2:>6.2f}                                {3:>9d} / {4:>6.2f}                       {5:>9d}'.\
                                                         format(i+1,len(molecule['prim_tuple'][i]),(float(len(molecule['prim_tuple'][i]))/float(molecule['theo_work'][i]))*100.00,\
                                                                total_order,(float(total_order)/float(molecule['theo_work'][i]))*100.00,total_all))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_phase_time(molecule):
   #
   print('')
   print('')
   print('                                              ---------------------------------------------                                                 ')
   print('                                                          phase and mpi timings                                                             ')
   print('                                              ---------------------------------------------                                                 ')
   print('')
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('     BG expansion order  |   time: init (HHH : MM : SS / %)   |   time: kernel (HHH : MM : SS / %)   |    time: final (HHH : MM : SS / %)   ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['prim_energy'])):
      #
      time_i = molecule['time_init'][i]
      time_k = molecule['time_kernel'][i]
      time_f = molecule['time_final'][i]
      time_t = molecule['time_tot'][i]
      #
      print('          {0:>4d}                   {1:03d} : {2:02d} : {3:02d} / {4:>6.2f}                {5:03d} : {6:02d} : {7:02d} / {8:>6.2f}                  {9:03d} : {10:02d} : {11:02d} / {12:>6.2f}'.\
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
   print('           total                 {0:03d} : {1:02d} : {2:02d} / {3:>6.2f}                {4:03d} : {5:02d} : {6:02d} / {7:>6.2f}                  {8:03d} : {9:02d} : {10:02d} / {11:>6.2f}'.\
             format(int(time_i//3600),int((time_i-(time_i//3600)*3600.)//60),int(time_i-(time_i//3600)*3600.-((time_i-(time_i//3600)*3600.)//60)*60.),(time_i/time_t)*100.0,\
                    int(time_k//3600),int((time_k-(time_k//3600)*3600.)//60),int(time_k-(time_k//3600)*3600.-((time_k-(time_k//3600)*3600.)//60)*60.),(time_k/time_t)*100.0,\
                    int(time_f//3600),int((time_f-(time_f//3600)*3600.)//60),int(time_f-(time_f//3600)*3600.-((time_f-(time_f//3600)*3600.)//60)*60.),(time_f/time_t)*100.0))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_mpi_time(molecule):
   #
   print('   |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('      mpi processor    |  time: init (work/comm/idle, in %)  |  time: kernel (work/comm/idle, in %)  |  time: final (work/comm/idle, in %)  ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   print('    master -- {0:<8d}        {1:>6.2f} / {2:>6.2f} / {3:>6.2f}               {4:>6.2f} / {5:>6.2f} / {6:>6.2f}                {7:>6.2f} / {8:>6.2f} / {9:>6.2f}'.\
          format(0,molecule['dist_init'][0][0],molecule['dist_init'][1][0],molecule['dist_init'][2][0],\
                 molecule['dist_kernel'][0][0],molecule['dist_kernel'][1][0],molecule['dist_kernel'][2][0],\
                 molecule['dist_final'][0][0],molecule['dist_final'][1][0],molecule['dist_final'][2][0]))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(1,molecule['mpi_size']):
      #
      print('    slave  -- {0:<8d}        {1:>6.2f} / {2:>6.2f} / {3:>6.2f}               {4:>6.2f} / {5:>6.2f} / {6:>6.2f}                {7:>6.2f} / {8:>6.2f} / {9:>6.2f}'.\
             format(i,molecule['dist_init'][0][i],molecule['dist_init'][1][i],molecule['dist_init'][2][i],\
                    molecule['dist_kernel'][0][i],molecule['dist_kernel'][1][i],molecule['dist_kernel'][2][i],\
                    molecule['dist_final'][0][i],molecule['dist_final'][1][i],molecule['dist_final'][2][i]))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   print('    mean  : slaves            {0:>6.2f} / {1:>6.2f} / {2:>6.2f}               {3:>6.2f} / {4:>6.2f} / {5:>6.2f}                {6:>6.2f} / {7:>6.2f} / {8:>6.2f}'.\
          format(np.mean(molecule['dist_init'][0][1:]),np.mean(molecule['dist_init'][1][1:]),np.mean(molecule['dist_init'][2][1:]),\
                 np.mean(molecule['dist_kernel'][0][1:]),np.mean(molecule['dist_kernel'][1][1:]),np.mean(molecule['dist_kernel'][2][1:]),\
                 np.mean(molecule['dist_final'][0][1:]),np.mean(molecule['dist_final'][1][1:]),np.mean(molecule['dist_final'][2][1:])))
   #
   print('    stdev : slaves            {0:>6.2f} / {1:>6.2f} / {2:>6.2f}               {3:>6.2f} / {4:>6.2f} / {5:>6.2f}                {6:>6.2f} / {7:>6.2f} / {8:>6.2f}'.\
          format(np.std(molecule['dist_init'][0][1:],ddof=1),np.std(molecule['dist_init'][1][1:],ddof=1),np.std(molecule['dist_init'][2][1:],ddof=1),\
                 np.std(molecule['dist_kernel'][0][1:],ddof=1),np.std(molecule['dist_kernel'][1][1:],ddof=1),np.std(molecule['dist_kernel'][2][1:],ddof=1),\
                 np.std(molecule['dist_final'][0][1:],ddof=1),np.std(molecule['dist_final'][1][1:],ddof=1),np.std(molecule['dist_final'][2][1:],ddof=1)))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_end():
   #
   print('\n')
   #
   return


