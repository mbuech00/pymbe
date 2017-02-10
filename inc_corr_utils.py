# -*- coding: utf-8 -*
#!/usr/bin/env python

#
# generel utilities for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#
# Requires the path of the cfour basis GENBAS file ($CFOURBASIS) and bin directory ($CFOURBIN)
#

import sys
import os
import re

import inc_corr_mpi
import inc_corr_orb_rout

CFOUR_BASIS='$CFOURBASIS'
CFOUR_BIN='$CFOURBIN'

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def redirect_stdout(molecule):
   #
   molecule['wrk'] = os.getcwd()
   #
   if (os.path.isdir(molecule['wrk']+'/output')):
      #
      command = 'rm -rf '+molecule['wrk']+'/output'
      os.system(command)
   #
   mk_out_dir(molecule['wrk']+'/output')
   #
   sys.stdout = logger(molecule['wrk']+'/output/stdout.out')
   #
   return molecule

def mk_out_dir(directory):
   #
   command = 'mkdir '+directory
   os.system(command)
   #
   return

def mk_scr_dir(directory):
   #
   command = 'mkdir '+directory
   os.system(command)
   #
   return

def rm_scr_dir(directory):
   #
   command = 'rm -rf '+directory
   os.system(command)
   #
   return

def cd_dir(directory):
   #
   os.chdir(directory)
   #
   return

def save_err_out(directory):
   #
   command = 'cp '+directory+'/CFOUR.OUT .'
   os.system(command)
   #
   return

def prepare_calc():
   #
   command = 'cp '+CFOUR_BASIS+' .'
   os.system(command)
   #
   command = 'cp '+CFOUR_BIN+'/x* .'
   os.system(command)
   #
   return

def setup_calc(directory):
   #
   mk_scr_dir(directory)
   #
   cd_dir(directory)
   #
   prepare_calc()
   #
   return

def term_calc(molecule):
   #
   cd_dir(molecule['wrk'])
   #
   if (molecule['error'][0][-1]):
      #
      save_err_out(molecule['scr'])
   #
   rm_scr_dir(molecule['scr'])
   #
   if (molecule['mpi_master']):
      #
      inc_corr_mpi.remove_slave_env(molecule)
      #
      print(' ** end of bethe-goldstone '+molecule['model']+' calculation **\n')
      print('\n')
   #
   return

def init_mol(molecule):
   #
   if (not os.path.isfile('input-mol.inp')):
      #
      print('input-mol.inp not found, aborting ...')
      #
      inc_corr_mpi.abort_mpi(molecule)
   #
   else:
      #
      with open('input-mol.inp') as f:
         #
         content = f.readlines()
         #
         for i in range(0,len(content)-2):
            #
            if (i == 0):
               #
               molecule['mol'] = str(content[i])
            else:
               #
               molecule['mol'] += str(content[i])
         #
         for j in range(1,3):
            #
            molecule[content[-j].split()[0]] = int(content[-j].split()[1])
   #
   chk = ['mult','core','mol']
   #
   for k in range(0,len(chk)-1):
      #
      if (not (chk[k] in molecule.keys())):
         #
         print('any of '+str(chk[0:2])+' keywords missing in input-mol.inp, aborting ...')
         #
         inc_corr_mpi.abort_mpi(molecule)
   #
   # rename 'core' to 'ncore' internally in the code (to adapt with convention: 'nocc'/'nvirt')
   #
   molecule['ncore'] = molecule.pop('core')
   #
   return molecule

def init_param(molecule):
   #
   if (not os.path.isfile('input-param.inp')):
      #
      print('input-param.inp not found, aborting ...')
      #
      inc_corr_mpi.abort_mpi(molecule)
   #
   else:
      #
      # init keys
      #
      molecule['max_order'] = 0
      molecule['prim_thres'] = 0.0
      molecule['sec_thres'] = 0.0
      molecule['corr'] = False
      molecule['corr_model'] = ''
      molecule['corr_order'] = 0
      molecule['corr_thres'] = 0.0
      molecule['basis'] = ''
      molecule['ref'] = False
      molecule['frozen'] = False
      molecule['debug'] = False
      molecule['local'] = False
      molecule['zmat'] = False
      molecule['mem'] = 0
      molecule['scr'] = ''
      #
      with open('input-param.inp') as f:
         #
         content = f.readlines()
         #
         for i in range(0,len(content)):
            #
            if (content[i].split()[0] == 'exp'):
               #
               molecule['exp'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'max_order'):
               #
               molecule['max_order'] = int(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'prim_thres'):
               #
               molecule['prim_thres'] = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'sec_thres'):
               #
               molecule['sec_thres'] = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'corr'):
               #
               molecule['corr'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'corr_order'):
               #
               molecule['corr_order'] = int(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'corr_thres'):
               #
               molecule['corr_thres'] = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'model'):
               #
               molecule['model'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'corr_model'):
               #
               molecule['corr_model'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'basis'):
               #
               molecule['basis'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'ref'):
               #
               molecule['ref'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'frozen'):
               #
               molecule['frozen'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'local'):
               #
               molecule['local'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'zmat'):
               #
               molecule['zmat'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'units'):
               #
               molecule['units'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'mem'):
               #
               molecule['mem'] = int(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'scr'):
               #
               molecule['scr'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'debug'):
               #
               molecule['debug'] = (content[i].split()[1] == 'True')
            #
            else:
               #
               print(str(content[i].split()[0])+' keyword in input-param.inp not recognized, aborting ...')
               #
               inc_corr_mpi.abort_mpi(molecule)
   #
   set_exp(molecule)
   #
   chk = ['mol','ncore','frozen','mult','scr','exp','max_order','prim_thres','sec_thres','corr','corr_order','corr_thres','model','corr_model',\
          'basis','ref','local','zmat','units','mem','debug']
   #
   inc = 0
   #
   for k in range(0,len(chk)):
      #
      if (not (chk[k] in molecule.keys())):
         #
         print(str(chk[k])+' keyword missing in either input-mol.inp or input-param.inp, aborting ...')
         #
         inc += 1
   #
   if (inc > 0):
      #
      inc_corr_mpi.abort_mpi(molecule)
   #
   return molecule

def set_exp(molecule):
   #
   # set thresholds and scheme
   #
   if (molecule['exp'] == 'occ'):
      #
      molecule['scheme'] = 'occupied'
   #
   elif (molecule['exp'] == 'virt'):
      #
      molecule['scheme'] = 'virtual'
   #
   elif (molecule['exp'] == 'comb-ov'):
      #
      molecule['scheme'] = 'combined occupied/virtual'
   #
   elif (molecule['exp'] == 'comb-vo'):
      #
      molecule['scheme'] = 'combined virtual/occupied'
   #
   # set regex for expansion model
   #
   if (molecule['model'] == 'fci'):
      #
      molecule['regex'] = '\s+Final Correlation Energy'
   #
   elif (molecule['model'] == 'mp2'):
      #
      molecule['regex'] = '\s+E2\(TOT\)'
   #
   else: # CC
      #
      molecule['regex'] = '\s+The correlation energy is'
   #
   # set correction model and order in case of no energy correction
   #
   if (not molecule['corr']):
      #
      molecule['corr_model'] = 'N/A'
      #
      molecule['corr_order'] = 'N/A'
   #
   return molecule

def init_calc(molecule):
   #
   # init error list
   #
   molecule['error'] = [[False]]
   #
   # init molecular info
   #
   init_mol(molecule)
   #
   # init expansion parameters et al.
   #
   init_param(molecule)
   #
   # if mpi parallel run, bcast the molecular dictionary
   #
   if (molecule['mpi_parallel']):
      #
      inc_corr_mpi.bcast_mol_dict(molecule)
      #
      inc_corr_mpi.init_slave_env(molecule)
      #
      inc_corr_mpi.print_mpi_table(molecule)
   #
   return molecule

def print_header(molecule):
   #
   print('\n')
   print(' ** start bethe-goldstone '+molecule['model']+' calculation  ---  '+str(molecule['scheme'])+' expansion scheme **')
   #
   return

def sanity_chk(molecule):
   #
   # type of expansion
   #
   if ((molecule['exp'] != 'occ') and (molecule['exp'] != 'virt') and (molecule['exp'] != 'comb-ov') and (molecule['exp'] != 'comb-vo')):
      #
      print('wrong input -- valid choices for expansion scheme are occ, virt, comb-ov, or comb-vo --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # expansion model
   #
   if (not ((molecule['model'] == 'fci') or (molecule['model'] == 'mp2') or (molecule['model'] == 'cisd') or (molecule['model'] == 'ccsd') or (molecule['model'] == 'ccsdt'))):
      #
      print('wrong input -- valid expansion models are currently: fci, mp2, cisd, ccsd, and ccsdt --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # max order
   #
   if (molecule['max_order'] < 0):
      #
      print('wrong input -- wrong maximum expansion order (must be integer >= 1) --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # expansion thresholds
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')) and ((molecule['prim_thres'] == 0.0) and (molecule['max_order'] == 0))):
      #
      print('wrong input -- no expansion threshold (prim_thres) supplied and no max_order set (either or both must be set) --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')) and (molecule['prim_thres'] < 0.0)):
      #
      print('wrong input -- expansion threshold (prim_thres) must be float >= 0.0 --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   if (((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')) and ((molecule['prim_thres'] == 0.0) and (molecule['sec_thres'] == 0.0))):
      #
      print('wrong input -- expansion thresholds for both the occ and the virt expansions need be supplied (prim_thres / sec_thres) --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   if (((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')) and ((molecule['prim_thres'] < 0.0) or (molecule['prim_thres'] < 0.0))):
      #
      print('wrong input -- expansion thresholds (prim_thres / sec_thres) must be floats >= 0.0 --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # energy correction
   #
   if (molecule['corr']):
      #
      if (molecule['corr_order'] == 0):
         #
         print('wrong input -- energy correction requested, but no correction order (integer >= 1) supplied --- aborting ...')
         #
         molecule['error'][0].append(True)
      #
      if (molecule['corr_thres'] <= 0.0):
         #
         print('wrong input -- correction threshold (corr_thres, float >= 0.0) must be supplied --- aborting ...')
         #
         molecule['error'][0].append(True)
      #
      if (molecule['corr_thres'] > molecule['prim_thres']):
         #
         print('wrong input -- correction threshold (corr_thres) must be tighter than the primary expansion threshold (prim_thres) --- aborting ...')
         #
         molecule['error'][0].append(True)
   #
   # frozen core threatment
   #
   if (molecule['frozen'] and (molecule['ncore'] == 0)):
      #
      print('wrong input -- frozen core requested, but no core orbitals specified --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   if (molecule['frozen'] and molecule['local']):
      #
      print('wrong input -- comb. of frozen core and local orbitals not implemented --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # units
   #
   if ((molecule['units'] != 'angstrom') and (molecule['units'] != 'bohr')):
      #
      print('wrong input -- valid choices of units are angstrom or bohr --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # memory
   #
   if (molecule['mem'] == 0):
      #
      print('wrong input -- memory input not supplied --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # basis set
   #
   if (molecule['basis'] == ''):
      #
      print('wrong input -- basis set not supplied --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # scratch folder
   #
   if (molecule['scr'] == ''):
      #
      print('wrong input -- scratch folder not supplied --- aborting ...')
      #
      molecule['error'][0].append(True)
   #
   # quit upon error
   #
   if (molecule['error'][0][-1]):
      #
      cd_dir(molecule['wrk'])
      #
      rm_scr_dir(molecule['scr'])
      #
      inc_corr_mpi.abort_mpi(molecule)
   #
   return molecule

def inc_corr_summary(molecule):
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
   print('   bethe-goldstone order        =  {0:}'.format(len(molecule['e_tot'][0])))
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
   print('   error in calculation         =  {0:}'.format(molecule['error'][0][-1]))
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
   print('')
   print('   ---------------------------------------------')
   print('                   final results                ')
   print('   ---------------------------------------------')
   print('')
   #
   print('   final energy (excl. corr.)   =  {0:>12.5e}'.format(molecule['e_tot'][0][-1]))
   print('   final energy (incl. corr.)   =  {0:>12.5e}'.format(molecule['e_tot'][0][-1]+molecule['e_corr'][0][-1]))
   #
   print('   ---------------------------------------------')
   #
   print('   final conv. (excl. corr.)    =  {0:>12.5e}'.format(molecule['e_tot'][0][-1]-molecule['e_tot'][0][-2]))
   print('   final conv. (incl. corr.)    =  {0:>12.5e}'.format((molecule['e_tot'][0][-1]+molecule['e_corr'][0][-1])-(molecule['e_tot'][0][-2]+molecule['e_corr'][0][-2])))
   #
   print('   ---------------------------------------------')
   #
   if (molecule['ref'] and (not molecule['error'][0][-1])):
      #
      final_diff = molecule['e_ref']-molecule['e_tot'][0][-1]
      final_diff_corr = molecule['e_ref']-(molecule['e_tot'][0][-1]+molecule['e_corr'][0][-1])
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
   for i in range(0,len(molecule['e_tot'][0])):
      #
      if (molecule['prim_n_tuples'][0][i] == molecule['theo_work'][0][i]):
         #
         tot_n_tup.append(molecule['prim_n_tuples'][0][i])
      #
      else:
         #
         tot_n_tup.append(molecule['prim_n_tuples'][0][i]+molecule['corr_n_tuples'][0][i])
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('     BG expansion order  |   # of prim. exp. tuples   |   # of corr. tuples   |   perc. of total # of tuples:   excl. corr.  |  incl. corr.  ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['e_tot'][0])):
      #
      print('          {0:>4d}                     {1:>4.2e}                    {2:>4.2e}                                           {3:>6.2f} %        {4:>6.2f} %'.\
                                                                          format(i+1,molecule['prim_n_tuples'][0][i],molecule['corr_n_tuples'][0][i],\
                                                                                 (float(molecule['prim_n_tuples'][0][i])/float(molecule['theo_work'][0][i]))*100.00,\
                                                                                 (float(tot_n_tup[i])/float(molecule['theo_work'][0][i]))*100.00))
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
   for i in range(0,len(molecule['e_tot'][0])):
      #
      total_time += molecule['prim_time'][0][i]
      total_time_corr += molecule['corr_time'][0][i]
      #
      print('          {0:>4d}                    {1:>7.5e}                      {2:>7.5e}                   {3:4.2e} s              {4:4.2e} s'.\
                                                                          format(i+1,molecule['e_tot'][0][i],molecule['e_tot'][0][i]+molecule['e_corr'][0][i],\
                                                                                 total_time,total_time+total_time_corr))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   print('\n')
   #
   return molecule

def print_status_header_1(order,level):
   #
   print('')
   print('')
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} initialization started'.format(level,order))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_status_header_2(num,order,conv,time_gen,level):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-{0:}: order = {1:>d} initialization done in {2:8.2e} seconds'.format(level,order,time_gen))
   print(' --------------------------------------------------------------------------------------------')
   print(' --------------------------------------------------------------------------------------------')
   #
   if ((level == 'MACRO') and conv):
      #
      print(' STATUS-{0:}: order = {1:>d} has no contributions --- *** calculation has converged ***'.format(level,order))
      print(' --------------------------------------------------------------------------------------------')
   #
   else:
      #
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
      n_tup = molecule['prim_n_tuples'][0]
      time = molecule['prim_time'][0]
   #
   elif (level == 'CORRE'):
      #
      n_tup = molecule['corr_n_tuples'][0]
      time = molecule['corr_time'][0]
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
   for i in range(0,molecule['n_tuples'][0][-1]):
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
      dom = molecule['prim_domain'][0]
   #
   elif (level == 'CORRE'):
      #
      dom = molecule['corr_domain'][0]
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

def orb_print(molecule,l_limit,u_limit,level):
   #
   if (level == 'MACRO'):
      #
      orb = molecule['prim_orb_ent'][0]
      orb_arr = molecule['prim_orb_arr'][0]
      orb_con_rel = molecule['prim_orb_con_rel'][0]
   #
   elif (level == 'CORRE'):
      #
      orb = molecule['corr_orb_ent'][0]
      orb_arr = molecule['corr_orb_arr'][0]
      orb_con_rel = molecule['corr_orb_con_rel'][0]
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

def enum(*sequential,**named):
   #
   enums = dict(zip(sequential,range(len(sequential))),**named)
   #
   return type('Enum',(), enums)

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


