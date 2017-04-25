#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_cfour.py: cfour-related routines for Bethe-Goldstone correlation calculations."""

from re import match
from shutil import copy

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def cfour_input_hf(molecule):
   #
   out = open('ZMAT','w')
   #
   out.write(molecule['mol'])
   #
   out.write('*CFOUR(CALC=HF\n')
   out.write('SCF_CONV=10\n')
   out.write('LINEQ_CONV=9\n')
   #
   if (not molecule['zmat']):
      #
      out.write('COORD=CARTESIAN\n')
   #
   if (molecule['units'] == 'bohr'):
      #
      out.write('UNITS=BOHR\n')
   #
   out.write('MULTIPLICITY='+str(molecule['mult'])+'\n')
   #
   if (molecule['mult'] == 1):
      #
      out.write('REF=RHF\n')
   #
   else:
      #
      out.write('REF=UHF\n')
   #
   out.write('BASIS='+molecule['basis']+'\n')
   #
   out.write('MEMORY='+str(molecule['mem'])+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()

def cfour_input_corr(molecule,drop_string,level):
   #
   out = open('ZMAT','w')
   #
   out.write(molecule['mol'])
   #
   model = molecule['model']
   #
   if (model == 'FCI'):
      #
      out.write('*CFOUR(CALC=FULLCI\n')
      out.write('CAS_MMAX=10\n')
      out.write('CAS_MITMAX=1000\n')
   #
   else:
      #
      out.write('*CFOUR(CALC='+model+'\n')
      out.write('CC_PROG=VCC\n')
      out.write('CC_EXPORDER=10\n')
      out.write('CC_MAXCYC=200\n')
   #
   if (molecule['virt_orbs'] == 'MP2'): out.write('VNATORB=USE\n')
   #
   if (drop_string != '\n'):
      #
      out.write(drop_string)
   #
   out.write('SCF_CONV=10\n')
   out.write('LINEQ_CONV=9\n')
   out.write('CC_CONV=9\n')
   #
   if (not molecule['zmat']):
      #
      out.write('COORD=CARTESIAN\n')
   #
   if (molecule['units'] == 'bohr'):
      #
      out.write('UNITS=BOHR\n')
   #
   if (molecule['occ_orbs'] == 'LOCAL'):
      #
      out.write('SYMMETRY=OFF\n')
      out.write('ORBITALS=LOCAL\n')
   #
   if (molecule['frozen'] and (level == 'REF')):
      #
      out.write('FROZEN_CORE=ON\n')
   #
   out.write('MULTIPLICITY='+str(molecule['mult'])+'\n')
   #
   if (molecule['mult'] == 1):
      #
      out.write('REF=RHF\n')
   #
   else:
      #
      out.write('REF=UHF\n')
   #
   out.write('BASIS='+molecule['basis']+'\n')
   #
   out.write('MEMORY='+str(molecule['mem'])+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()
   #
   return molecule

def cfour_get_dim(molecule):
   #
   inp = open('OUTPUT_'+str(molecule['mpi_rank'])+'.OUT','r')
   #
   regex = 'basis functions'
   #
   content = inp.readlines()
   #
   inp.close()
   #
   found = False
   #
   for i in range(0,len(content)):
      #
      if (regex in content[i]):
         #
         [bf] = content[i].split()[2:3]
         #
         found = True
         #
         break
   #
   if (not found):
      #
      molecule['error_msg'] = 'problem with HF calculation (# basis functions)'
      #
      molecule['error_code'] = 1
      #
      molecule['error'].append(True)
      #
      return molecule
   #
   delim_1 = 'MO #'
   delim_2 = '+++++++++++++'
   #
   start = False
   occ_mos = []
   #
   for i in range(0,len(content)):
      #
      if (delim_1 in content[i]):
         #
         start = True
      #
      elif (delim_2 in content[i]):
         #
         start = False
         #
         break
      #
      if (start): occ_mos.append(content[i])
   #
   if (len(occ_mos) == 0):
      #
      molecule['error_msg'] = 'problem with HF calculation (# occ. MOs)'
      #
      molecule['error_code'] = 1
      #
      molecule['error'].append(True)
   #
   else:
      #
      molecule['nocc'] = len(occ_mos)-2
      #
      molecule['nvirt'] = int(bf) - molecule['nocc']
   #
   del content
   #
   return molecule

def cfour_write_energy(molecule,level):
   #
   inp = open('OUTPUT_'+str(molecule['mpi_rank'])+'.OUT','r')
   #
   content = inp.readlines()
   #
   inp.close()
   #
   found = False
   #
   for i in range(0,len(content)):
      #
      if (match(molecule['regex'],content[i]) is not None):
         #
         if (molecule['model'] == 'FCI'):
            #
            [tmp] = content[i].split()[3:4]
         #
         elif (molecule['model'] == 'MP2'):
            #
            [tmp] = content[i].split()[2:3]
         #
         else: # CC
            #
            [tmp] = content[i].split()[4:5]
         #
         if (level == 'REF'):
            #
            molecule['e_ref'] = float(tmp)
         #
         else:
            #
            molecule['e_tmp'] = float(tmp)
         #
         found = True
         #
         break
   #
   if (not found):
     #
     molecule['error_msg'] = 'problem with {0:} calculation (energy)'.format(molecule['model'])
     #
     molecule['error_code'] = 2
     #
     molecule['error'].append(True)
   #
   return molecule

