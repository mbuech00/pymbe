#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_summary.py: summary print utilities for Bethe-Goldstone correlation calculations."""

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
   print('     BG expansion order  |   # of prim. exp. tuples   |   # of corr. tuples   |   perc. of total # of tuples:   excl. corr.  |  incl. corr. ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['prim_energy'])):
      #
      print('          {0:>4d}                     {1:>4.2e}                    {2:>4.2e}                                           {3:>6.2f} %        {4:>6.2f} %'.\
                                                                          format(i+1,molecule['prim_n_tuples'][i],molecule['corr_n_tuples'][i],\
                                                                                 (float(molecule['prim_n_tuples'][i])/float(molecule['theo_work'][i]))*100.00,\
                                                                                 (float(tot_n_tup[i])/float(molecule['theo_work'][i]))*100.00))
   #
   return

def summary_detail_res_2(molecule):
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
      total_time += molecule['prim_time_tot'][i]
      total_time_corr += molecule['corr_time_tot'][i]
      #
      print('          {0:>4d}                    {1:>7.5e}                      {2:>7.5e}                   {3:4.2e} s              {4:4.2e} s'.\
                                                                          format(i+1,molecule['prim_energy'][i],molecule['prim_energy'][i]+molecule['corr_energy'][i],\
                                                                                 total_time,total_time+total_time_corr))
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
   print('     BG expansion order  |   time init  (in s / %)   |   time kernel  (in s / %)   |   time final  (in s / %)   |   time remain  (in s / %) ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['prim_energy'])):
      #
      time_tot = molecule['prim_time_tot'][i]+molecule['corr_time_tot'][i]
      time_init = molecule['prim_time_init'][i]+molecule['corr_time_init'][i]
      time_kernel = molecule['prim_time_kernel'][i]+molecule['corr_time_kernel'][i]
      time_final = molecule['prim_time_final'][i]+molecule['corr_time_final'][i]
      time_remain = time_tot-(time_init+time_kernel+time_final)
      #
      print('          {0:>4d}                  {1:>4.2e} / {2:>4.2f}              {3:>4.2e} / {4:>4.2f}              {5:>4.2e} / {6:>4.2f}             {7:>4.2e} / {8:>4.2f}'.\
                format(i+1,time_init,(time_init/time_tot)*100.0,time_kernel,(time_kernel/time_tot)*100.0,\
                       time_final,(time_final/time_tot)*100.0,time_remain,(time_remain/time_tot)*100.0))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_mpi_time(molecule):
   #
   print('   |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('      mpi processor    |   time init in s (work/comm/idle)   |   time kernel in s (work/comm/idle)   |   time final in s (work/comm/idle)   ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   print('    master -- {0:<8d}     {1:>4.2e} / {2:>4.2e} / {3:>4.2e}         {4:>4.2e} / {5:>4.2e} / {6:>4.2e}          {7:>4.2e} / {8:>4.2e} / {9:>4.2e}'.\
          format(0,sum(molecule['mpi_time_work'][0][0]),sum(molecule['mpi_time_comm'][0][0]),sum(molecule['mpi_time_idle'][0][0]),\
                 sum(molecule['mpi_time_work'][1][0]),sum(molecule['mpi_time_comm'][1][0]),sum(molecule['mpi_time_idle'][1][0]),\
                 sum(molecule['mpi_time_work'][2][0]),sum(molecule['mpi_time_comm'][2][0]),sum(molecule['mpi_time_idle'][2][0])))
   #
   for i in range(1,molecule['mpi_size']):
      #
      print('    slave  -- {0:<8d}     {1:>4.2e} / {2:>4.2e} / {3:>4.2e}         {4:>4.2e} / {5:>4.2e} / {6:>4.2e}          {7:>4.2e} / {8:>4.2e} / {9:>4.2e}'.\
             format(i,sum(molecule['mpi_time_work'][0][i]),sum(molecule['mpi_time_comm'][0][i]),sum(molecule['mpi_time_idle'][0][i]),\
                    sum(molecule['mpi_time_work'][1][i]),sum(molecule['mpi_time_comm'][1][i]),sum(molecule['mpi_time_idle'][1][i]),\
                    sum(molecule['mpi_time_work'][2][i]),sum(molecule['mpi_time_comm'][2][i]),sum(molecule['mpi_time_idle'][2][i])))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   return

def summary_end():
   #
   print('\n')
   #
   return


