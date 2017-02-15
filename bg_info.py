#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_info.py: info handling for Bethe-Goldstone correlation calculations."""

from os.path import isfile
from shutil import which

from bg_cfour import cfour_input_hf, cfour_input_corr, cfour_get_dim, cfour_write_energy

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.3'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def init_mol(molecule):
   #
   if (not isfile('input-mol.inp')):
      #
      print('input-mol.inp not found, aborting ...')
      #
      molecule['error'].append(True)
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
         molecule['error'].append(True)
   #
   # rename 'core' to 'ncore' internally in the code (to adapt with convention: 'nocc'/'nvirt')
   #
   molecule['ncore'] = molecule.pop('core')
   #
   return molecule

def init_param(molecule):
   #
   if (not isfile('input-param.inp')):
      #
      print('input-param.inp not found, aborting ...')
      #
      molecule['error'].append(True)
   #
   else:
      #
      # init keys
      #
      molecule['backend_prog'] = ''
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
      molecule['scr_name'] = ''
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
            elif (content[i].split()[0] == 'prog'):
               #
               molecule['backend_prog'] = content[i].split()[1]
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
               molecule['scr_name'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'debug'):
               #
               molecule['debug'] = (content[i].split()[1] == 'True')
            #
            else:
               #
               print(str(content[i].split()[0])+' keyword in input-param.inp not recognized, aborting ...')
               #
               molecule['error'][0].append(True)
   #
   set_exp(molecule)
   #
   chk = ['mol','ncore','frozen','mult','scr_name','exp','backend_prog','max_order','prim_thres','sec_thres','corr','corr_order','corr_thres','model','corr_model',\
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
      molecule['error'].append(True)
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
   # set correction model and order in case of no energy correction
   #
   if (not molecule['corr']):
      #
      molecule['corr_model'] = 'N/A'
      #
      molecule['corr_order'] = 'N/A'
   #
   return molecule

def init_backend_prog(molecule):
   #
   if (molecule['backend_prog'] == 'cfour'):
      #
      if (which('xcfour') is None):
         #
         print('no xcfour executable found in PATH env, aborting ...')
         #
         molecule['error'].append(True)
      #
      else:
         #
         # set path to executable
         # 
         molecule['backend_prog_exe'] = which('xcfour')
         #
         # set backend module routines
         #
         molecule['input_hf'] = cfour_input_hf
         molecule['input_corr'] = cfour_input_corr
         molecule['get_dim'] = cfour_get_dim
         molecule['write_energy'] = cfour_write_energy
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
   else:
      #
      print('choice of backend program not recognized, aborting ...')
      #
      molecule['error'].append(True)
   #
   return molecule

def sanity_chk(molecule):
   #
   # type of expansion
   #
   if ((molecule['exp'] != 'occ') and (molecule['exp'] != 'virt') and (molecule['exp'] != 'comb-ov') and (molecule['exp'] != 'comb-vo')):
      #
      print('wrong input -- valid choices for expansion scheme are occ, virt, comb-ov, or comb-vo --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # expansion model
   #
   if (not ((molecule['model'] == 'fci') or (molecule['model'] == 'mp2') or (molecule['model'] == 'cisd') or (molecule['model'] == 'ccsd') or (molecule['model'] == 'ccsdt'))):
      #
      print('wrong input -- valid expansion models are currently: fci, mp2, cisd, ccsd, and ccsdt --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # max order
   #
   if (molecule['max_order'] < 0):
      #
      print('wrong input -- wrong maximum expansion order (must be integer >= 1) --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # expansion thresholds
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')) and ((molecule['prim_thres'] == 0.0) and (molecule['max_order'] == 0))):
      #
      print('wrong input -- no expansion threshold (prim_thres) supplied and no max_order set (either or both must be set) --- aborting ...')
      #
      molecule['error'].append(True)
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')) and (molecule['prim_thres'] < 0.0)):
      #
      print('wrong input -- expansion threshold (prim_thres) must be float >= 0.0 --- aborting ...')
      #
      molecule['error'].append(True)
   #
   if (((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')) and ((molecule['prim_thres'] == 0.0) and (molecule['sec_thres'] == 0.0))):
      #
      print('wrong input -- expansion thresholds for both the occ and the virt expansions need be supplied (prim_thres / sec_thres) --- aborting ...')
      #
      molecule['error'].append(True)
   #
   if (((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')) and ((molecule['prim_thres'] < 0.0) or (molecule['prim_thres'] < 0.0))):
      #
      print('wrong input -- expansion thresholds (prim_thres / sec_thres) must be floats >= 0.0 --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # energy correction
   #
   if (molecule['corr']):
      #
      if (molecule['corr_order'] == 0):
         #
         print('wrong input -- energy correction requested, but no correction order (integer >= 1) supplied --- aborting ...')
         #
         molecule['error'].append(True)
      #
      if (molecule['corr_thres'] < 0.0):
         #
         print('wrong input -- correction threshold (corr_thres, float >= 0.0) must be supplied --- aborting ...')
         #
         molecule['error'].append(True)
      #
      if (molecule['corr_thres'] >= molecule['prim_thres']):
         #
         print('wrong input -- correction threshold (corr_thres) must be tighter than the primary expansion threshold (prim_thres) --- aborting ...')
         #
         molecule['error'].append(True)
   #
   # frozen core threatment
   #
   if (molecule['frozen'] and (molecule['ncore'] == 0)):
      #
      print('wrong input -- frozen core requested, but no core orbitals specified --- aborting ...')
      #
      molecule['error'].append(True)
   #
   if (molecule['frozen'] and molecule['local']):
      #
      print('wrong input -- comb. of frozen core and local orbitals not implemented --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # units
   #
   if ((molecule['units'] != 'angstrom') and (molecule['units'] != 'bohr')):
      #
      print('wrong input -- valid choices of units are angstrom or bohr --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # memory
   #
   if (molecule['mem'] == 0):
      #
      print('wrong input -- memory input not supplied --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # basis set
   #
   if (molecule['basis'] == ''):
      #
      print('wrong input -- basis set not supplied --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # scratch folder
   #
   if (molecule['scr_name'] == ''):
      #
      print('wrong input -- scratch folder not supplied --- aborting ...')
      #
      molecule['error'].append(True)
   #
   # quit upon error
   #
   if (molecule['error'][-1]):
      #
      molecule['error'].append(True)
   #
   return molecule

