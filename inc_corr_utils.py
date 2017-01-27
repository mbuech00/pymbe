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
      command='rm -rf '+molecule['wrk']+'/output'
      os.system(command)
   #
   mk_scr_dir(molecule['wrk']+'/output')
   #
   sys.stdout = logger(molecule['wrk']+'/output/stdout.out')
   #
   return molecule

def mk_scr_dir(directive):
   #
   command='mkdir '+directive
   os.system(command)
   #
   return

def rm_scr_dir(directive):
   #
   command='rm -rf '+directive
   os.system(command)
   #
   return

def cd_dir(directive):
   #
   os.chdir(directive)
   #
   return

def save_err_out(directive):
   #
   command='cp '+directive+'/CFOUR.OUT .'
   os.system(command)
   #
   return

def prepare_calc():
   #
   command='cp '+CFOUR_BASIS+' .'
   os.system(command)
   #
   command='cp '+CFOUR_BIN+'/x* .'
   os.system(command)
   #
   return

def setup_calc(molecule):
   #
   mk_scr_dir(molecule['scr'])
   #
   cd_dir(molecule['scr'])
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
   print(' ** end of bethe-goldstone '+molecule['model']+' calculation **\n')
   print('\n')
   #
   return

def init_mol(molecule):
   #
   if (not os.path.isfile('input-mol.inp')):
      #
      print('input-mol.inp not found, aborting ...')
      sys.exit(10)
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
         sys.exit(10)
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
      sys.exit(10)
   #
   else:
      #
      # init keys
      #
      thres_occ = 0.0
      thres_virt = 0.0
      molecule['max_order'] = 0
      molecule['est'] = False
      molecule['est_model'] = ''
      molecule['est_order'] = 0
      molecule['basis'] = ''
      molecule['ref'] = False
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
            elif (content[i].split()[0] == 'thres_occ'):
               #
               thres_occ = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'thres_virt'):
               #
               thres_virt = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'est'):
               #
               molecule['est'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'est_order'):
               #
               molecule['est_order'] = int(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'model'):
               #
               molecule['model'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'est_model'):
               #
               molecule['est_model'] = content[i].split()[1]
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
               molecule['frozen'] = content[i].split()[1]
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
               print(str(content[i].split()[1])+' keyword in input-param.inp not recognized, aborting ...')
               sys.exit(10)
   #
   set_prim_exp(thres_occ,thres_virt,molecule)
   #
   set_energy_est(molecule)
   #
   chk = ['mol','ncore','frozen','mult','scr','exp','max_order','est','est_order','model','est_model',\
          'basis','ref','local','zmat','units','mem','prim_thres','debug']
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
      sys.exit(10)
   #
   set_fc_scheme(molecule)
   #
   return molecule

def set_prim_exp(thres_occ,thres_virt,molecule):
   #
   # set thresholds and scheme
   #
   molecule['prim_thres'] = []
   #
   if (molecule['exp'] == 'occ'):
      #
      molecule['prim_thres'].append(thres_occ)
      #
      molecule['scheme'] = 'occupied'
   #
   elif (molecule['exp'] == 'virt'):
      #
      molecule['prim_thres'].append(thres_virt)
      #
      molecule['scheme'] = 'virtual'
   #
   elif (molecule['exp'] == 'comb-ov'):
      #
      molecule['prim_thres'].append(thres_occ)
      molecule['prim_thres'].append(thres_virt)
      #
      molecule['scheme'] = 'combined occupied/virtual'
   #
   elif (molecule['exp'] == 'comb-vo'):
      #
      molecule['prim_thres'].append(thres_virt)
      molecule['prim_thres'].append(thres_occ)
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
   return molecule

def set_energy_est(molecule):
   #
   # set regex for energy estimation model
   #
   if (molecule['est']):
      #
      if (molecule['est_model'] == 'fci'):
         #
         molecule['est_regex'] = '\s+Final Correlation Energy'
      #
      else: # CC
         #
         molecule['est_regex'] = '\s+The correlation energy is' 
   #
   else:
      #
      molecule['est_model'] == 'N/A'
      #
      molecule['est_order'] = 'N/A'
   #
   return molecule   

def set_fc_scheme(molecule):
   #
   if (molecule['frozen'] == 'none'):
      #
      molecule['frozen_scheme'] = ''
   #
   elif (molecule['frozen'] == 'conv'):
      #
      molecule['frozen_scheme'] = '(conventional)'
   #
   elif (molecule['frozen'] == 'screen'):
      #
      molecule['frozen_scheme'] = '(screened)'
   #
   return molecule

def init_calc(molecule):
   #
   molecule['error'] = [[False]]
   #
   init_mol(molecule)
   #
   init_param(molecule)
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
      molecule['error'][0].append(True)
   #
   # expansion model
   #
   if (not ((molecule['model'] == 'fci') or (molecule['model'] == 'mp2') or (molecule['model'] == 'cisd') or (molecule['model'] == 'ccsd') or (molecule['model'] == 'ccsdt'))):
      #
      print('wrong input -- valid expansion models are currently: fci, mp2, cisd, ccsd, and ccsdt --- aborting ...')
      molecule['error'][0].append(True)
   #
   # max order
   #
   if (molecule['max_order'] < 0):
      #
      print('wrong input -- wrong maximum expansion order (must be integer >= 1) --- aborting ...')
      molecule['error'][0].append(True)
   #
   # expansion thresholds
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')) and ((molecule['prim_thres'][0] == 0.0) and (molecule['max_order'] == 0))):
      #
      print('wrong input -- no expansion threshold supplied and no max_order set (either or both must be set) --- aborting ...')
      molecule['error'][0].append(True)
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')) and (molecule['prim_thres'][0] < 0.0)):
      #
      print('wrong input -- expansion threshold must be float >= 0.0 --- aborting ...')
      molecule['error'][0].append(True)
   #
   if (((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')) and ((molecule['prim_thres'][0] == 0.0) and (molecule['prim_thres'][1] == 0.0))):
      #
      print('wrong input -- expansion thresholds for both the occ and the virt expansions need be supplied --- aborting ...')
      molecule['error'][0].append(True)
   #
   if (((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')) and ((molecule['prim_thres'][0] < 0.0) or (molecule['prim_thres'][1] < 0.0))):
      #
      print('wrong input -- expansion thresholds must be floats >= 0.0 --- aborting ...')
      molecule['error'][0].append(True)
   #
   # energy estimation
   #
   if (molecule['est']):
      #
      if (molecule['est_model'] == ''):
         #
         print('wrong input -- energy estimation requested, but no estimation model supplied --- aborting ...')
         molecule['error'][0].append(True)
      #
      elif (not ((molecule['est_model'] == 'fci') or (molecule['est_model'] == 'cisd') or (molecule['est_model'] == 'ccsd') or (molecule['est_model'] == 'ccsdt'))):
         #
         print('wrong input -- valid energy estimation models are currently: fci, cisd, ccsd, and ccsdt --- aborting ...')
         molecule['error'][0].append(True)
      #
      if (molecule['model'] == molecule['est_model']):
         #
         print('wrong input -- models for expansion and energy estimation must differ --- aborting ...')
         molecule['error'][0].append(True)
      #
      if (molecule['est_order'] == 0):
         #
         print('wrong input -- energy estimation requested, but no estimation order (integer >= 1) supplied --- aborting ...')
         molecule['error'][0].append(True)
   #
   # frozen core threatment
   #
   if ((molecule['frozen'] != 'none') and (molecule['frozen'] != 'conv') and (molecule['frozen'] != 'screen')):
      #
      print('wrong input -- valid choices for frozen core are none, conv, or screen --- aborting ...')
      molecule['error'][0].append(True)
   #
   if (((molecule['frozen'] == 'conv') or (molecule['frozen'] == 'screen')) and (molecule['ncore'] == 0)):
      #
      print('wrong input -- frozen core requested ('+molecule['frozen_scheme']+' scheme), but no core orbitals specified --- aborting ...')
      molecule['error'][0].append(True)
   #
   if ((molecule['frozen'] == 'screen') and (molecule['exp'] == 'virt')):
      #
      print('wrong input -- '+molecule['frozen_scheme']+' frozen core scheme does not make sense with the virtual expansion scheme')
      print('            -- please use the conventional frozen core scheme instead --- aborting ...')
      molecule['error'][0].append(True)
   #
   if ((molecule['frozen'] == 'conv') and molecule['local']):
      #
      print('wrong input -- comb. of frozen core and local orbitals not implemented --- aborting ...')
      molecule['error'][0].append(True)
   #
   # units
   #
   if ((molecule['units'] != 'angstrom') and (molecule['units'] != 'bohr')):
      #
      print('wrong input -- valid choices of units are angstrom or bohr --- aborting ...')
      molecule['error'][0].append(True)
   #
   # memory
   #
   if (molecule['mem'] == 0):
      #
      print('wrong input -- memory input not supplied --- aborting ...')
      molecule['error'][0].append(True)
   #
   # basis set
   #
   if (molecule['basis'] == ''):
      #
      print('wrong input -- basis set not supplied --- aborting ...')
      molecule['error'][0].append(True)
   #
   # scratch folder
   #
   if (molecule['scr'] == ''):
      #
      print('wrong input -- scratch folder not supplied --- aborting ...')
      molecule['error'][0].append(True)
   #
   # quit upon error
   #
   if (molecule['error'][0][-1]):
      #
      cd_dir(molecule['wrk'])
      rm_scr_dir(molecule['scr'])
      sys.exit(10)
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
   print('   {0:} {1:} bethe-goldstone expansion'.format(molecule['scheme'],molecule['model']))
   #
   print('   ---------------------------------------------')
   print('   frozen core               =  {0:} {1:}'.format((molecule['frozen'] != 'none'),molecule['frozen_scheme']))
   print('   local orbitals            =  {0:}'.format(molecule['local']))
   print('   occupied orbitals         =  {0:}'.format(molecule['nocc']))
   print('   virtual orbitals          =  {0:}'.format(molecule['nvirt']))
   #
   if (molecule['exp'] == 'occ'):
      #
      print('   exp. threshold (occ.)     =  {0:5.3f} %'.format(molecule['prim_thres'][0]*100.00))
      print('   exp. threshold (virt.)    =  N/A')
   #
   elif (molecule['exp'] == 'virt'):
      #
      print('   exp. threshold (occ.)     =  N/A')
      print('   exp. threshold (virt.)    =  {0:5.3f} %'.format(molecule['prim_thres'][0]*100.00))
   #
   elif (molecule['exp'] == 'comb-ov'):
      #
      print('   exp. threshold (occ.)     =  {0:5.3f} %'.format(molecule['prim_thres'][0]*100.00))
      print('   exp. threshold (virt.)    =  {0:5.3f} %'.format(molecule['prim_thres'][1]*100.00))
   #
   elif (molecule['exp'] == 'comb-vo'):
      #
      print('   exp. threshold (occ.)     =  {0:5.3f} %'.format(molecule['prim_thres'][1]*100.00))
      print('   exp. threshold (virt.)    =  {0:5.3f} %'.format(molecule['prim_thres'][0]*100.00))
   #
   print('   energy estimation         =  {0:}'.format(molecule['est']))
   #
   if (molecule['est']):
      #
      print('   energy estimation model   =  {0:}'.format(molecule['est_model']))
   #
   else:
      #
      print('   energy estimation model   =  N/A')
   #
   print('   error in calculation      =  {0:}'.format(molecule['error'][0][-1]))
   #
   print('   ---------------------------------------------')
   #
   print('   bethe-goldstone order     =  {0:}'.format(len(molecule['e_tot'][0])))
   #
   if (molecule['est']):
      #
      print('   energy estimation order   =  {0:}'.format(molecule['max_est_order']))
   #
   else:
      #
      print('   energy estimation order   =  N/A')
   #
   print('   ---------------------------------------------')
   #
   print('   final conv. (excl. est.)  =  {0:>12.5e}'.format(molecule['e_tot'][0][-1]-molecule['e_tot'][0][-2]))
   print('   final conv. (incl. est.)  =  {0:>12.5e}'.format((molecule['e_tot'][0][-1]+molecule['e_est'][0][-1])-(molecule['e_tot'][0][-2]+molecule['e_est'][0][-2])))
   #
   print('   ---------------------------------------------')
   #
   if (molecule['ref'] and (not molecule['error'][0][-1])):
      #
      final_diff = molecule['e_ref']-molecule['e_tot'][0][-1]
      final_diff_est = molecule['e_ref']-(molecule['e_tot'][0][-1]+molecule['e_est'][0][-1])
      #
      if (abs(final_diff) < 1.0e-10):
         #
         final_diff = 0.0
      #
      if (abs(final_diff_est) < 1.0e-10):
         #
         final_diff_est = 0.0
      #
      print('   final diff. (excl. est.)  =  {0:>12.5e}'.format(final_diff))
      print('   final diff. (incl. est.)  =  {0:>12.5e}'.format(final_diff_est))
   #
   print('')
   print('')
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('   correlated orbitals   |   # of correlated tuples   |   # of est. tuples   |   perc. of total # of tuples   |   perc. of total # of tuples')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['e_tot'][0])):
      #
      print('          {0:>4d}                     {1:>4.2e}                   {2:>4.2e}                    {3:>6.2f} %                         {4:>6.2f} %'.\
                                                                          format(i+1,molecule['prim_n_tuples'][0][i],molecule['sec_n_tuples'][0][i],\
                                                                                 (float(molecule['prim_n_tuples'][0][i])/float(molecule['theo_work'][0][i]))*100.00,\
                                                                                 (float(molecule['sec_n_tuples'][0][i])/float(molecule['theo_work'][0][i]))*100.00))
   #
   total_time = 0.0
   total_time_est = 0.0
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('   |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   print('   correlated orbitals   |   total correlated energy   |    total energy incl. energy est.    |    total time    |    total time incl. est. ')
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(molecule['e_tot'][0])):
      #
      total_time += molecule['prim_time'][0][i]
      total_time_est += molecule['sec_time'][0][i]
      #
      print('          {0:>4d}                    {1:>7.5e}                      {2:>7.5e}                   {3:4.2e} s              {4:4.2e} s'.\
                                                                          format(i+1,molecule['e_tot'][0][i],molecule['e_tot'][0][i]+molecule['e_est'][0][i],\
                                                                                 total_time,total_time+total_time_est))
   #
   print('   -----------------------------------------------------------------------------------------------------------------------------------------')
   #
   print('\n')
   #
   return molecule

def print_status_header(molecule,num,order):
   #
   print('')
   print('')
   print(' --------------------------------------------------------------------------------------------')
   #
   if (molecule['conv'][-1]):
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  has no contributions --- *** calculation has converged ***'.format(order,molecule['theo_work'][0][0]))
      print(' --------------------------------------------------------------------------------------------')
   #
   else:
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  started  ---  {2:d}  tuples in total'.format(order,molecule['theo_work'][0][0],num))
      print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_status_header_est(theo_order,max_order,num):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-ESTIM:  energy est. through order =  {0:>d} / {1:<d}  started  ---  {2:d}  tuples in total'.format(max_order,theo_order,num))
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
   print(' STATUS-'+level+':   [{0}]   ---  {1:>6.2f} % {2}'.format('#' * block + '-' * (bar_length - block), prog * 100, status))
   #
   return

def print_status_end(molecule,order,time,n_tup,level):
   #
   print(' --------------------------------------------------------------------------------------------')
   if (n_tup[-1] == 0):
      #
      print(' STATUS-'+level+':  order =  {0:>d} / {1:<d}  done'.format(order,n_tup[0]))
   #
   else:
      #
      print(' STATUS-'+level+':  order =  {0:>d} / {1:<d}  done in {2:10.2e} seconds'.format(order,n_tup[0],time[-1]))
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_status_end_est(theo_order,max_order,time):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' STATUS-ESTIM:  energy est. through order =  {0:>d} / {1:<d}  done in {2:10.2e} seconds'.format(max_order,theo_order,sum(time)))
   print(' --------------------------------------------------------------------------------------------')
   print('')
   print('')
   #
   return

def print_result(tup):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' --------------------------------------------------------------------------------------------')
   print(' RESULT-MACRO:     tuple    |    energy incr.   |    corr. orbs.')
   print(' --------------------------------------------------------------------------------------------')
   #
   for i in range(0,len(tup)):
      #
      print(' RESULT-MACRO:  {0:>6d}           {1:> 8.4e}         {2:<}'.format(i+1,tup[i][1],tup[i][0]))
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_result_est(molecule,sec_tup):
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' --------------------------------------------------------------------------------------------')
   print(' RESULT-ESTIM:     tuple    |    energy incr.   |    corr. orbs.')
   print(' --------------------------------------------------------------------------------------------')
   #
   counter = 0
   #
   for i in range(0,len(sec_tup)):
      #
      for j in range(0,len(sec_tup[i])):
         #
         found = False
         #
         for k in range(0,len(molecule['prim_tuple'][0][i])):
            #
            if (set(sec_tup[i][j][0]) == set(molecule['prim_tuple'][0][i][k][0])):
               #
               found = True
               #
               break
         #
         if (not found):
            #
            counter += 1
            #
            print(' RESULT-ESTIM:  {0:>6d}           {1:> 8.4e}         {2:<}'.format(counter,sec_tup[i][j][1],sec_tup[i][j][0]))
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
                       format(i+1,molecule['e_diff_in'][i])+'[{0:<}]'.format(', '.join(str(idx) for idx in rel_work_in[i])))
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_update(domain,l_limit,u_limit,level):
   #
   count = []
   #
   for j in range(0,u_limit):
      #
      if ((len(domain[j][-2]) >= 1) and (float(len(domain[j][-1]))/float(len(domain[j][-2])) != 1.0)):
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
      print(' UPDATE-'+level+':   orb. domain  |  relat. red. (in %)  |   total red. (in %)  |  screened orbs.  ')
      print(' --------------------------------------------------------------------------------------------')
      #
      for j in range(0,u_limit):
         #
         if (count[j]):
            #
            print(' UPDATE-'+level+':     {0:>5}              {1:>6.2f}                 {2:>6.2f}            {3:<}'.\
                          format([(j+l_limit)+1],\
                                 (1.0-float(len(domain[j][-1]))/float(len(domain[j][-2])))*100.00,\
                                 (1.0-float(len(domain[j][-1]))/float(len(domain[j][0])))*100.00,\
                                 sorted(list(set(domain[j][-2])-set(domain[j][-1])))))
      #
      print(' --------------------------------------------------------------------------------------------')
   #
   return

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


