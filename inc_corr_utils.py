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
   print(' ** END OF INC.-CORR. ('+molecule['model']+') CALCULATION **\n')
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
   if (molecule['core'] > 0):
      #
      molecule['fc'] = True
   #
   else:
      #
      molecule['fc'] = False
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
      with open('input-param.inp') as f:
         #
         content = f.readlines()
         #
         for i in range(0,len(content)):
            #
            if (content[i].split()[0] == 'scr'):
               #
               molecule['scr'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'exp'):
               #
               molecule['exp'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'model'):
               #
               molecule['model'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'basis'):
               #
               molecule['basis'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'ref'):
               #
               molecule['ref'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'local'):
               #
               molecule['local'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'mem'):
               #
               molecule['mem'] = int(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'thres_occ'):
               #
               thres_occ = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'thres_virt'):
               #
               thres_virt = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'order'):
               #
               molecule['order'] = int(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'screen_occ'):
               #
               screen_occ = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'screen_virt'):
               #
               screen_virt = (content[i].split()[1] == 'True')
            #
            else:
               #
               print(str(content[i].split()[1])+' keyword in input-param.inp not recognized, aborting ...')
               sys.exit(10)
   #
   molecule['thres'] = [thres_occ,thres_virt]
   #
   molecule['screen'] = [screen_occ,screen_virt]
   #
   chk = ['mol','core','fc','mult','scr','exp','model','basis','ref','local','mem','thres','order','screen']
   #
   inc = 0
   #
   for k in range(0,len(chk)):
      #
      if (not (chk[k] in molecule.keys())):
         #
         print(str(chk[k])+' keyword missing in either input-mol.inp or input-param.inp, aborting ...')
   #
   if (inc > 0):
      #
      sys.exit(10)
   #
   if ((molecule['thres'][0] > 0.0) or (molecule['thres'][1] > 0.0)):
      #
      molecule['exp_ctrl'] = True
   #
   else:
      #
      molecule['exp_ctrl'] = False
   #
   if (molecule['model'] == 'FCI'):
      #
      molecule['regex'] = '\s+Final Correlation Energy'
   #
   else:
      #
      molecule['regex'] = '\s+The correlation energy is'
   #
   return molecule

def init_calc(molecule):
   #
   molecule['error'] = [[False]]
   #
   molecule['conv'] = [[False],[False]]
   #
   init_mol(molecule)
   #
   init_param(molecule)
   #
   print('\n')
   print(' ** START INC.-CORR. ('+molecule['model']+') CALCULATION **\n')
   #
   return molecule

def sanity_chk(molecule):
   #
   if (molecule['exp'] == 'OCC'):
      #
      if (molecule['order'] >= (molecule['nocc'] - molecule['core'])):
         #
         print 'wrong input argument for total order (must be .lt. number of available occupied orbitals), aborting ...'
         molecule['error'][0].append(True)
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      if (molecule['order'] >= molecule['nvirt']):
         #
         print 'wrong input argument for total order (must be .lt. number of virtual orbitals), aborting ...'
         molecule['error'][0].append(True)
   #
   elif (molecule['exp'] == 'COMB'):
      #
      if ((molecule['thres'][0] == 0.0) or (molecule['thres'][1] == 0.0)):
         #
         print('expansion scheme "COMB" requires both an occupied and a virtual expansion threshold, aborting ...')
         molecule['error'][0].append(True)
      #
      if (not molecule['exp_ctrl']):
         #
         print('expansion scheme "COMB" is currently not implemented for fixed order expansion, aborting ...')
         molecule['error'][0].append(True)
   #
   if ((molecule['order'] > 0) and molecule['exp_ctrl']):
      #
      print('fixed order expansion requested, but expansion thresholds provided, aborting ...')
      molecule['error'][0].append(True)
   #
   if ((molecule['order'] == 0) and (not molecule['exp_ctrl'])):
      #
      print('neither fixed order nor threshold-governed expansion requested, aborting ...')
      molecule['error'][0].append(True)
   #
   if (molecule['fc'] and molecule['local']):
      #
      print 'wrong input -- comb. of frozen core and local orbitals not implemented, aborting ...'
      molecule['error'][0].append(True)
   #
   if (molecule['error'][0][-1]):
      #
      cd_dir(molecule['wrk'])
      rm_scr_dir(molecule['scr'])
      sys.exit(10)
   #
   return molecule

def init_domains(molecule):
   #
   if (molecule['screen'][0] and (not molecule['screen'][1])):
      #
      molecule['occ_domain'] = [[]]
      molecule['virt_domain'] = []
      #
      for i in range(0,molecule['nvirt']):
         #
         molecule['virt_domain'].append(range(molecule['nocc']+1,(molecule['nocc']+molecule['nvirt'])+1))
         #
         molecule['virt_domain'][i].pop(i)
      #
      inc_corr_orb_rout.init_occ_screen(molecule)
   #
   elif (molecule['screen'][1] and (not molecule['screen'][0])):
      #
      molecule['occ_domain'] = []
      molecule['virt_domain'] = [[]]
      #
      for i in range(0,molecule['nocc']):
         #
         molecule['occ_domain'].append(range(1,molecule['nocc']+1))
         #
         molecule['occ_domain'][i].pop(i)
      #
      init_virt_screen(molecule)
   #
   elif (molecule['screen'][0] and molecule['screen'][1]):
      #
      molecule['occ_domain'] = [[]]
      molecule['virt_domain'] = [[]]
      #
      inc_corr_orb_rout.init_occ_screen(molecule)
      inc_corr_orb_rout.init_virt_screen(molecule)
   #
   else:
      #
      molecule['occ_domain'] = []
      molecule['virt_domain'] = []
      #
      for i in range(0,molecule['nocc']):
         #
         molecule['occ_domain'].append(range(1,molecule['nocc']+1))
         #
         molecule['occ_domain'][i].pop(i)
      #
      for i in range(0,molecule['nvirt']):
         #
         molecule['virt_domain'].append(range(molecule['nocc']+1,(molecule['nocc']+molecule['nvirt'])+1))
         #
         molecule['virt_domain'][i].pop(i)
   #
   return molecule

def inc_corr_summary(molecule):
   #
   print('')
   print(' *******************************')
   print(' ********    RESULTS    ********')
   print(' *******************************\n')
   #
   print('   {0:} expansion'.format(molecule['exp']))
   #
   print('   -----------------------------')
   print('   frozen core        =  {0:}'.format(molecule['fc']))
   print('   local orbitals     =  {0:}'.format(molecule['local']))
   print('   occupied orbitals  =  {0:}'.format(molecule['nocc']-molecule['core']))
   print('   virtual orbitals   =  {0:}'.format(molecule['nvirt']))
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      print('   screening (occ.)   =  {0:}'.format(molecule['screen'][0]))
      print('   screening (virt.)  =  N/A')
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      print('   screening (occ.)   =  N/A')
      print('   screening (virt.)  =  {0:}'.format(molecule['screen'][1]))
   #
   if (molecule['exp_ctrl']):
      #
      if (molecule['thres'][0] > 0.0):
         #
         if (molecule['exp'] == 'VIRT'):
            #
            print('   thres. (occ.)      =  N/A')
         #
         else:
            #
            print('   thres. (occ.)      =  {0:6.1e}'.format(molecule['thres'][0]))
      #
      else:
         #
         print('   thres. (occ.)      =  N/A')
      #
      if (molecule['thres'][1] > 0.0):
         #
         if (molecule['exp'] == 'OCC'):
            #
            print('   thres. (virt.)     =  N/A')
         #
         else:
            #
            print('   thres. (virt.)     =  {0:6.1e}'.format(molecule['thres'][1]))
      #
      else:
         #
         print('   thres. (virt.)     =  N/A')
   #
   else:
      #
      print('   thres. (occ.)      =  N/A')
      print('   thres. (virt.)     =  N/A')
   #
   print('   inc.-corr. order   =  {0:}'.format(len(molecule['e_fin'])))
   #
   if (molecule['exp_ctrl']):
      #
      print('   convergence met    =  {0:}'.format(molecule['conv'][0][-1]))
   else:
      #
      print('   convergence met    =  N/A')
   #
   print('   error in calc.     =  {0:}'.format(molecule['error'][0][-1]))
   #
   print('')
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      print('{0:4d} - # orb. tuples  =  {1:}'.format(i+1,molecule['n_contrib'][i]))
   #
   print('   --------------------------------------------------------------')
   #
   total_time = 0.0
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      total_time += molecule['time'][i]
      print('{0:4d} - E (inc-corr)   = {1:13.9f}  done in {2:10.2e} seconds'.format(i+1,molecule['e_fin'][i],total_time))
   #
   print('   --------------------------------------------------------------')
   #
   if (len(molecule['e_fin']) >= 2):
      #
      print('   final convergence  =  {0:9.4e}'.format(molecule['e_fin'][-1]-molecule['e_fin'][-2]))
   #
   if (molecule['ref'] and (not molecule['error'][0][-1])):
      #
      print('   --------------------------------------------------------------')
      #
      if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
         #
         print('{0:4d} - E (ref)        = {1:13.9f}  done in {2:10.2e} seconds'.format(molecule['nocc']-molecule['core'],molecule['e_ref'],molecule['time'][-1]))
      #
      elif (molecule['exp'] == 'VIRT'):
         #
         print('{0:4d} - E (ref)        = {1:13.9f}  done in {2:10.2e} seconds'.format(molecule['nvirt'],molecule['e_ref'],molecule['time'][-1]))
      #
      print('   --------------------------------------------------------------')
      #
      print('   final difference   =  {0:9.4e}'.format(molecule['e_ref']-molecule['e_fin'][-1]))
   #
   print('\n')
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


