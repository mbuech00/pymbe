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
            elif (content[i].split()[0] == 'thres_occ'):
               #
               thres_occ = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'thres_virt'):
               #
               thres_virt = float(content[i].split()[1])
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
   molecule['thres'] = []
   #
   if (molecule['exp'] == 'OCC'):
      #
      molecule['thres'].append(thres_occ)
      #
      molecule['scheme'] = 'OCCUPIED'
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      molecule['thres'].append(thres_virt)
      #
      molecule['scheme'] = 'VIRTUAL'
   #
   elif (molecule['exp'] == 'COMB-OV'):
      #
      molecule['thres'].append(thres_occ)
      molecule['thres'].append(thres_virt)
      #
      molecule['scheme'] = 'COMBINED OCCUPIED/VIRTUAL'
   #
   elif (molecule['exp'] == 'COMB-VO'):
      #
      molecule['thres'].append(thres_virt)
      molecule['thres'].append(thres_occ)
      #
      molecule['scheme'] = 'COMBINED VIRTUAL/OCCUPIED'
   #
   chk = ['mol','core','frozen','mult','scr','exp','model','basis','ref','local','zmat','units','mem','thres','debug']
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
   if (molecule['model'] == 'FCI'):
      #
      molecule['regex'] = '\s+Final Correlation Energy'
   #
   else:
      #
      molecule['regex'] = '\s+The correlation energy is'
   #
   if (molecule['frozen'] == 'NONE'):
      #
      molecule['frozen_scheme'] = 'none'
   #
   elif (molecule['frozen'] == 'TRAD'):
      #
      molecule['frozen_scheme'] = 'traditional'
   #
   elif (molecule['frozen'] == 'SCREEN'):
      #
      molecule['frozen_scheme'] = 'screened'
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
   print(' ** START INC.-CORR. ('+molecule['model']+') CALCULATION  ---  '+str(molecule['scheme'])+' EXPANSION SCHEME **\n')
   #
   return

def sanity_chk(molecule):
   #
   if ((molecule['exp'] != 'OCC') and (molecule['exp'] != 'VIRT') and (molecule['exp'] != 'COMB-OV') and (molecule['exp'] != 'COMB-VO')):
      #
      print('wrong input -- valid choices for expansion scheme are OCC, VIRT, COMB-OV, or COMB-VO --- aborting ...')
      molecule['error'][0].append(True)
   #
   if ((molecule['frozen'] != 'NONE') and (molecule['frozen'] != 'TRAD') and (molecule['frozen'] != 'SCREEN')):
      #
      print('wrong input -- valid choices for frozen core are NONE or TRAD --- aborting ...')
      molecule['error'][0].append(True)
   #
   if (((molecule['frozen'] == 'TRAD') or (molecule['frozen'] == 'SCREEN')) and (molecule['core'] == 0)):
      #
      print('wrong input -- frozen core requested ('+molecule['frozen_scheme']+' scheme), but no core orbitals specified --- aborting ...')
      molecule['error'][0].append(True)
   #
   if ((molecule['frozen'] == 'SCREEN') and (molecule['exp'] == 'VIRT')):
      #
      print('wrong input -- '+molecule['frozen_scheme']+' frozen core scheme does not make sense with the virtual expansion scheme')
      print('            -- please use the traditional frozen core scheme instead --- aborting ...')
      molecule['error'][0].append(True)
   #
   if ((molecule['frozen'] == 'TRAD') and molecule['local']):
      #
      print('wrong input -- comb. of frozen core and local orbitals not implemented --- aborting ...')
      molecule['error'][0].append(True)
   #
   if ((molecule['units'] != 'angstrom') and (molecule['units'] != 'bohr')):
      #
      print('wrong input -- valid choices of units are angstrom or bohr --- aborting ...')
      molecule['error'][0].append(True)
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
   print(' *******************************')
   print(' ********    RESULTS    ********')
   print(' *******************************\n')
   #
   print('   {0:} expansion'.format(molecule['scheme']))
   #
   print('   -----------------------------')
   print('   frozen core        =  {0:} ({1:})'.format((molecule['frozen'] != 'NONE'),molecule['frozen_scheme']))
   print('   local orbitals     =  {0:}'.format(molecule['local']))
   print('   occupied orbitals  =  {0:}'.format(molecule['n_tuples'][0][0]))
   print('   virtual orbitals   =  {0:}'.format(molecule['nvirt']))
   #
   if (molecule['exp'] == 'OCC'):
      #
      print('   thres. (occ.)      =  {0:5.3f} %'.format(molecule['thres'][0]*100.00))
      print('   thres. (virt.)     =  N/A')
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      print('   thres. (occ.)      =  N/A')
      print('   thres. (virt.)     =  {0:5.3f} %'.format(molecule['thres'][0]*100.00))
   #
   elif (molecule['exp'] == 'COMB-OV'):
      #
      print('   thres. (occ.)      =  {0:5.3f} %'.format(molecule['thres'][0]*100.00))
      print('   thres. (virt.)     =  {0:5.3f} %'.format(molecule['thres'][1]*100.00))
   #
   elif (molecule['exp'] == 'COMB-VO'):
      #
      print('   thres. (occ.)      =  {0:5.3f} %'.format(molecule['thres'][1]*100.00))
      print('   thres. (virt.)     =  {0:5.3f} %'.format(molecule['thres'][0]*100.00))
   #
   print('   inc.-corr. order   =  {0:}'.format(len(molecule['e_fin'][0])))
   #
   print('   error in calc.     =  {0:}'.format(molecule['error'][0][-1]))
   #
   print('')
   #
   for i in range(0,len(molecule['e_fin'][0])):
      #
      print('{0:4d} - # orb. tuples  =  {1:<7d}  ---  {2:>6.2f} %'.format(i+1,molecule['n_tuples'][0][i],\
                                                                          (float(molecule['n_tuples'][0][i])/float(molecule['theo_work'][0][i]))*100.00))
   #
   print('   --------------------------------------------------------------')
   #
   total_time = 0.0
   #
   for i in range(0,len(molecule['e_fin'][0])):
      #
      total_time += molecule['time'][0][i]
      #
      print('{0:4d} - E (inc-corr)   = {1:13.9f}  done in {2:10.2e} seconds'.format(i+1,molecule['e_fin'][0][i],total_time))
   #
   print('   --------------------------------------------------------------')
   #
   if (len(molecule['e_fin'][0]) >= 2):
      #
      print('   final convergence  =  {0:9.4e}'.format(molecule['e_fin'][0][-1]-molecule['e_fin'][0][-2]))
   #
   if (molecule['ref'] and (not molecule['error'][0][-1])):
      #
      print('   --------------------------------------------------------------')
      #
      print('{0:4d} - E (ref)        = {1:13.9f}  done in {2:10.2e} seconds'.format(molecule['n_tuples'][0][0],molecule['e_ref'],molecule['time'][0][-1]))
      #
      print('   --------------------------------------------------------------')
      #
      print('   final difference   =  {0:9.4e}'.format(molecule['e_ref']-molecule['e_fin'][0][-1]))
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


