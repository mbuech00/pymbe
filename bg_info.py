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
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def init_mol(molecule):
   #
   if (not isfile('input-mol.inp')):
      #
      molecule['error_msg'] = 'input-mol.inp not found'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   else:
      #
      # init keys
      #
      molecule['zmat'] = False
      molecule['units'] = ''
      #
      with open('input-mol.inp') as f:
         #
         content = f.readlines()
         #
         for i in range(0,len(content)-4):
            #
            if (i == 0):
               #
               molecule['mol'] = str(content[i])
            else:
               #
               molecule['mol'] += str(content[i])
         #
         for j in range(1,5):
            #
            if (content[-j].split()[0] == 'mult'):
               #
               molecule['mult'] = int(content[-j].split()[1])
            #
            elif (content[-j].split()[0] == 'core'):
               #
               molecule['core'] = int(content[-j].split()[1])
            #
            elif (content[-j].split()[0] == 'zmat'):
               #
               molecule['zmat'] = (content[-j].split()[1] == 'True')
            #
            elif (content[-j].split()[0] == 'units'):
               #
               molecule['units'] = content[-j].split()[1]
            #
            else:
               #
               molecule['error_msg'] = str(content[-j].split()[0])+' keyword in input-mol.inp not recognized'
               #
               molecule['error_code'] = 0
               #
               molecule['error'].append(True)
   #
   chk = ['mult','core','zmat','units','mol']
   #
   for k in range(0,len(chk)):
      #
      if (not (chk[k] in molecule.keys())):
         #
         molecule['error_msg'] = 'any of '+str(chk)+' keywords missing in input-mol.inp'
         #
         molecule['error_code'] = 0
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
      molecule['error_msg'] = 'input-param.inp not found'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   else:
      #
      # init keys
      #
      molecule['backend_prog'] = ''
      molecule['max_order'] = 0
      molecule['prim_exp_thres'] = 1.0e-04 # default setting
      molecule['prim_exp_scaling'] = 2.0 # default setting
      molecule['prim_energy_thres'] = 1.0e-05 # default setting
      molecule['occ_orbs'] = ''
      molecule['virt_orbs'] = ''
      molecule['basis'] = ''
      molecule['ref'] = False
      molecule['frozen'] = False
      molecule['debug'] = False
      molecule['mem'] = 0
      molecule['scr_name'] = 'scr'
      molecule['rst'] = False
      molecule['rst_freq'] = 50000.0 # default setting
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
               molecule['backend_prog'] = content[i].split()[1].upper()
            #
            elif (content[i].split()[0] == 'occ_orbs'):
               #
               molecule['occ_orbs'] = content[i].split()[1].upper() 
            #
            elif (content[i].split()[0] == 'virt_orbs'):
               #
               molecule['virt_orbs'] = content[i].split()[1].upper()
            #
            elif (content[i].split()[0] == 'max_order'):
               #
               molecule['max_order'] = int(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'prim_exp_thres'):
               #
               molecule['prim_exp_thres'] = float(content[i].split()[1])
               molecule['prim_exp_thres_init'] = molecule['prim_exp_thres']
            #
            elif (content[i].split()[0] == 'prim_exp_scaling'):
               #
               molecule['prim_exp_scaling'] = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'prim_energy_thres'):
               #
               molecule['prim_energy_thres'] = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'model'):
               #
               molecule['model'] = content[i].split()[1].upper()
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
            elif (content[i].split()[0] == 'mem'):
               #
               molecule['mem'] = int(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'scr'):
               #
               molecule['scr_name'] = content[i].split()[1]
            #
            elif (content[i].split()[0] == 'restart'):
               #
               molecule['rst'] = (content[i].split()[1] == 'True')
            #
            elif (content[i].split()[0] == 'rst_freq'):
               #
               molecule['rst_freq'] = float(content[i].split()[1])
            #
            elif (content[i].split()[0] == 'debug'):
               #
               molecule['debug'] = (content[i].split()[1] == 'True')
            #
            else:
               #
               molecule['error_msg'] = str(content[i].split()[0])+' keyword in input-param.inp not recognized'
               #
               molecule['error_code'] = 0
               #
               molecule['error'].append(True)
   #
   set_exp(molecule)
   #
   set_fc(molecule)
   #
   chk = ['mol','ncore','frozen','mult','occ_orbs','virt_orbs',\
          'exp','model','max_order','prim_exp_thres','prim_exp_scaling','prim_energy_thres',\
          'basis','ref','mem','debug',\
          'scr_name','rst','rst_freq','backend_prog']
   #
   for k in range(0,len(chk)):
      #
      if (not (chk[k] in molecule.keys())):
         #
         molecule['error_msg'] = str(chk[k])+' keyword missing in either input-mol.inp or input-param.inp'
         #
         molecule['error_code'] = 0
         #
         molecule['error'].append(True)
         #
         break
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
   return molecule

def set_fc(molecule):
   #
   if (not molecule['frozen']): molecule['ncore'] = 0
   #
   return molecule

def init_backend_prog(molecule):
   #
   if (molecule['backend_prog'] == 'CFOUR'):
      #
      if (which('xcfour') is None):
         #
         molecule['error_msg'] = 'no xcfour executable found in PATH env'
         #
         molecule['error_code'] = 0
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
         if (molecule['model'] == 'FCI'):
            #
            molecule['regex'] = '\s+Final Correlation Energy'
         #
         elif (molecule['model'] == 'MP2'):
            #
            molecule['regex'] = '\s+E2\(TOT\)'
         #
         else: # CC
            #
            molecule['regex'] = '\s+The correlation energy is'
   #
   else:
      #
      molecule['error_msg'] = 'choice of backend program not recognized'
      #
      molecule['error_code'] = 0
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
      molecule['error_msg'] = 'wrong input -- valid choices for expansion scheme are occ, virt, comb-ov, or comb-vo'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   # expansion model
   #
   if (not ((molecule['model'] == 'FCI') or (molecule['model'] == 'MP2') or (molecule['model'] == 'CISD') or (molecule['model'] == 'CCSD') or (molecule['model'] == 'CCSDT'))):
      #
      molecule['error_msg'] = 'wrong input -- valid expansion models are currently: FCI, MP2, CISD, CCSD, and CCSDT'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   # max order
   #
   if (molecule['max_order'] < 0):
      #
      molecule['error_msg'] = 'wrong input -- wrong maximum expansion order (must be integer >= 1)'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   # expansion thresholds
   #
   if ((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')):
      #
      if ((molecule['prim_exp_thres'] == 0.0) and (molecule['max_order'] == 0)):
         #
         molecule['error_msg'] = 'wrong input -- no expansion threshold (prim_exp_thres) supplied and no max_order set (either or both must be set)'
         #
         molecule['error_code'] = 0
         #
         molecule['error'].append(True)
      #
      if (molecule['prim_exp_thres'] < 0.0):
         #
         molecule['error_msg'] = 'wrong input -- expansion threshold (prim_exp_thres) must be float >= 0.0'
         #
         molecule['error_code'] = 0
         #
         molecule['error'].append(True)
      #
      if (molecule['prim_exp_scaling'] < 0.0):
         #
         molecule['error_msg'] = 'wrong input -- expansion scaling (prim_exp_scaling) must be float >= 0.0'
         #
         molecule['error_code'] = 0
         #
         molecule['error'].append(True)
      #
      if (molecule['prim_energy_thres'] < 0.0):
         #
         molecule['error_msg'] = 'wrong input -- energy threshold (prim_energy_thres) must be float >= 0.0'
         #
         molecule['error_code'] = 0
         #
         molecule['error'].append(True)
   #
   # orbital representations
   #
   if ((molecule['occ_orbs'] == '') or (molecule['virt_orbs'] == '')):
      #
      molecule['error_msg'] = 'wrong input -- orbital representations must be chosen for occupied and virtual orbitals'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   else:
      #
      if (not ((molecule['occ_orbs'] == 'CANONICAL') or (molecule['occ_orbs'] == 'LOCAL'))):
         #
         molecule['error_msg'] = 'wrong input -- orbital representation for occupied orbitals must be either canonical or local'
         #
         molecule['error_code'] = 0
         #
         molecule['error'].append(True)
      #
      if (not ((molecule['virt_orbs'] == 'CANONICAL') or (molecule['virt_orbs'] == 'MP2'))):
         #
         molecule['error_msg'] = 'wrong input -- orbital representation for virtual orbitals must be either canonical or mp2 (natural orbitals)'
         #
         molecule['error_code'] = 0
         #
         molecule['error'].append(True)
   #
   # frozen core threatment
   #
   if (molecule['frozen'] and (molecule['ncore'] == 0)):
      #
      molecule['error_msg'] = 'wrong input -- frozen core requested, but no core orbitals specified'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   # units
   #
   if ((molecule['units'] != 'angstrom') and (molecule['units'] != 'bohr')):
      #
      molecule['error_msg'] = 'wrong input -- valid choices of units are angstrom or bohr'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   # memory
   #
   if (molecule['mem'] == 0):
      #
      molecule['error_msg'] = 'wrong input -- memory input not supplied'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   # basis set
   #
   if (molecule['basis'] == ''):
      #
      molecule['error_msg'] = 'wrong input -- basis set not supplied'
      #
      molecule['error_code'] = 0
      #
      molecule['error'].append(True)
   #
   return molecule

