#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_cfour.py: cfour-related routines for Bethe-Goldstone correlation calculations."""

from re import match

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.3'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def cfour_input_hf(molecule):
   #
   out=open('ZMAT','w')
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
   out=open('ZMAT','w')
   #
   out.write(molecule['mol'])
   #
   model = molecule['model'].upper()
   #
   if (model == 'FCI'):
      #
      out.write('*CFOUR(CALC=FULLCI\n')
      out.write('CAS_MMAX=10\n')
      out.write('CAS_MITMAX=200\n')
   #
   else:
      #
      out.write('*CFOUR(CALC='+model+'\n')
      out.write('CC_PROG=VCC\n')
      out.write('CC_EXPORDER=10\n')
      out.write('CC_MAXCYC=200\n')
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
   if (molecule['local']):
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
   inp=open('OUTPUT.OUT','r')
   #
   regex_err = '\s+ERROR ERROR'
   #
   regex = 'basis functions'
   #
   while 1:
      #
      line=inp.readline()
      #
      if regex in line:
         #
         [bf] = line.split()[2:3]
         #
         break
      #
      elif match(regex_err,line) is not None:
         #
         print('problem with HF calculation, aborting ...')
         #
         molecule['error'][0].append(True)
         #
         inp.close()
         #
         return molecule
   #
   inp.seek(0)
   #
   regex_2 = '\s+Alpha population by irrep:'
   #
   while 1:
      #
      line=inp.readline()
      #
      if match(regex_2,line) is not None:
         #
         pop = line.split()
         #
         break
   #
   tmp = 0
   #
   for i in range(4,len(pop)):
      #
      tmp += int(pop[i])
   #
   molecule['nocc'] = tmp
   #
   molecule['nvirt'] = int(bf) - molecule['nocc']
   #
   inp.close()
   #
   return molecule

def cfour_write_energy(molecule,level):
   #
   inp=open('OUTPUT.OUT','r')
   #
   regex_err = '\s+ERROR ERROR'
   #
   model = molecule['model']
   #
   regex = molecule['regex']
   #
   while 1:
      #
      line=inp.readline()
      #
      if match(regex,line) is not None:
         #
         if (model == 'fci'):
            #
            [tmp] = line.split()[3:4]
         #
         elif (model == 'mp2'):
            #
            [tmp] = line.split()[2:3]
         #
         else: # CC
            #
            [tmp] = line.split()[4:5]
         #
         if (level == 'REF'):
            #
            molecule['e_ref'] = float(tmp)
         #
         else:
            #
            molecule['e_tmp'] = float(tmp)
         #
         break
      #
      elif match(regex_err,line) is not None:
         #
         print('problem with '+model+' calculation, aborting ...')
         #
         molecule['error'][0].append(True)
         #
         inp.close()
         #
         return molecule
   #
   inp.close()
   #
   return molecule

