#!/usr/bin/env python

#
# python driver for inc.-corr. calculations using CFOUR as backend program.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), August 2016, Mainz, Germnay.
#
# Requires the path of the cfour basis GENBAS file ($CFOURBASIS) and bin directory ($CFOURBIN)
#

import sys
from sys import stdin
import os
import os.path
import re
import argparse
import math
from timeit import default_timer as timer

import inc_corr_plot
from inc_corr_plot import ic_plot

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

CFOUR_BASIS='$CFOURBASIS'
CFOUR_BIN='$CFOURBIN'

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
   if (molecule['error']):
      #
      save_err_out(molecule['scr'])
   #
   rm_scr_dir(molecule['scr'])
   #
   return

def init_calc(molecule):
   #
   parser = argparse.ArgumentParser(description='This is an CCSD/CISD/CCSDT/FCI inc.-corr. Python script (with CFOUR backend) written by Dr. Janus Juul Eriksen, JGU Mainz, Fall 2016')
   parser.add_argument('--exp', help='type of expansion ("OCC", "VIRT", or "COMB")',required=True)
   parser.add_argument('--model', help='electronic structure model ("CCSD", "CISD", "CCSDT", or "FCI")',required=True)
   parser.add_argument('--basis', help='one-electron basis set (e.g., "cc-pVTZ")', required=True)
   parser.add_argument('--mol', help='molecule ("H2O", "C2H2", "N2", "O2", "SiH4", "CO2", "C4H6", or "C6H6")', required=True)
   parser.add_argument('--frozen', help='frozen-core logical ("True" or "False")', required=True)
   parser.add_argument('--ref', help='reference calc. logical ("True" or "False")', required=True)
   parser.add_argument('--mem', help='amount of virtual memory in GB (integer number)', required=True)
   parser.add_argument('--scr', help='location of scratch folder', required=True)
   parser.add_argument('--thres_occ', help='convergence threshold for occupied expansion (real number in scientific format, e.g., "1.0e-03")', required=False)
   parser.add_argument('--thres_virt', help='convergence threshold for virtual expansion (real number in scientific format, e.g., "1.0e-03")', required=False)
   parser.add_argument('--screen_occ', help='enable screening in the occupied expansion ("True" or "False")', required=False)
   parser.add_argument('--screen_virt', help='enable screening in the virtual expansion ("True" or "False")', required=False)
   parser.add_argument('--order', help='inc.-corr. order (integer number)', required=False)
   parser.add_argument('--bond', help='bond length parameter for PES generation (real number)', required=False)
   parser.add_argument('--local', help='local orbitals logical ("True" or "False")', required=False)
   args = parser.parse_args()
   #
   molecule['model'] = args.model
   #
   if (not ((molecule['model'] == 'CCSD') or (molecule['model'] == 'CISD') or (molecule['model'] == 'CCSDT') or (molecule['model'] == 'FCI'))):
      #
      print 'wrong choice of model (CCSD, CISD, CCSDT, or FCI), aborting ...'
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
   molecule['exp'] = args.exp
   #
   if (not ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'VIRT') or (molecule['exp'] == 'COMB'))):
      #
      print 'wrong choice of expansion type (OCC, VIRT, or COMB), aborting ...'
      sys.exit(10)
   #
   molecule['basis'] = args.basis
   #
   molecule['mol_string'] = args.mol
   #
   molecule['fc'] = (args.frozen == 'True')
   #
   if ((args.frozen != 'True') and (args.frozen != 'False')):
      #
      print 'wrong input argument for frozen core (True/False), aborting ...'
      sys.exit(10)
   #
   molecule['ref'] = (args.ref == 'True')
   #
   if ((args.ref != 'True') and (args.ref != 'False')):
      #
      print 'wrong input argument for reference calc (True/False), aborting ...'
      sys.exit(10)
   #
   molecule['mem'] = args.mem
   #
   molecule['scr'] = args.scr
   #
   molecule['wrk'] = os.getcwd()
   #
   molecule['exp_ctrl'] = False
   #
   if ((args.thres_occ is None) and (args.thres_virt is None) and (args.order is None)):
      #
      print 'either the convergence threshold(s) (--thres_occ/--thres_virt) OR the inc.-corr. order (--order) must be set, aborting ...'
      sys.exit(10)
   #
   elif (args.order is None):
      #
      molecule['exp_ctrl'] = True
      molecule['order'] = 0
      #
      if (args.thres_occ is None):
         #
         molecule['thres'] = [0.0,float(args.thres_virt)]
         #
         if (molecule['exp'] == 'COMB'):
            #
            print('expansion scheme "COMB" requires both an occupied and a virtual expansion threshold, aborting ...')
            sys.exit(10)
      #
      elif (args.thres_virt is None):
         #
         molecule['thres'] = [float(args.thres_occ),0.0]
         #
         if (molecule['exp'] == 'COMB'):
            #
            print('expansion scheme "COMB" requires both an occupied and a virtual expansion threshold, aborting ...')
            sys.exit(10)
      #
      else:
         #
         molecule['thres'] = [float(args.thres_occ),float(args.thres_virt)]
   #
   elif ((args.thres_occ is None) and (args.thres_virt is None)):
      #
      molecule['order'] = int(args.order)
      #
      if (molecule['exp'] == 'COMB'):
         #
         print('expansion scheme "COMB" is currently not implemented for fixed order expansion, aborting ...')
         sys.exit(10)
   #
   if (args.local is None):
      #
      molecule['local'] = False
   #
   else:
      #
      molecule['local'] = (args.local == 'True')
   #
   if (molecule['fc'] and molecule['local']):
      #
      print 'wrong input -- comb. of frozen core and local orbitals not implemented, aborting ...'
      sys.exit(10)
   #
   if ((args.screen_occ == 'True') and (args.screen_virt is None)):
      #
      molecule['screen'] = [True,False]
   #
   elif ((args.screen_virt == 'True') and (args.screen_occ is None)):
      #
      molecule['screen'] = [False,True]
   #
   elif ((args.screen_virt == 'True') and (args.screen_occ == 'True')):
      #
      molecule['screen'] = [True,True]
   #
   else:
      #
      molecule['screen'] = [False,False]
   #
   molecule['error'] = False
   #
   init_zmat(molecule)
   #
   return molecule

def sanity_chk(molecule):
   #
   if (molecule['exp'] == 'OCC'):
      #
      if (molecule['order'] >= (molecule['nocc'] - molecule['core'])):
         #
         print 'wrong input argument for total order (must be .lt. number of available occupied orbitals), aborting ...'
         #
         cd_dir(molecule['wrk'])
         rm_scr_dir(molecule['scr'])
         sys.exit(10)
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      if (order >= nvirt[0]):
         #
         print 'wrong input argument for total order (must be .lt. number of virtual orbitals), aborting ...'
         #
         cd_dir(molecule['wrk'])
         rm_scr_dir(molecule['scr'])
         sys.exit(10)
   #
   return molecule

def init_screen(molecule):
   #
   if (molecule['screen'][0] and (not molecule['screen'][1])):
      #
      molecule['list_excl'] = [[]]
      screen_occ(molecule)
   #
   elif (molecule['screen'][1] and (not molecule['screen'][0])):
      #
      molecule['list_excl'] = [[]]
#      screen_virt(molecule)
   #
   elif (molecule['screen'][0] and molecule[screen][1]):
      #
      molecule['list_excl'] = [[]]
      screen_occ(molecule)
#      screen_virt(molecule)
   #
   else:
      #
      molecule['list_excl'] = []
   #
   return molecule

def run_calc_hf(molecule):
   #
   write_zmat_hf(molecule)
   #
   command='xcfour &> CFOUR.OUT'
   os.system(command)
   #
   get_dim(molecule)
   #
   if (not molecule['error']):
      #
      command='xclean'
      os.system(command)
   #
   return molecule

def run_calc_corr(molecule,drop_string,ref):
   # 
   write_zmat_corr(molecule,drop_string,ref)
   #
   command='xcfour &> CFOUR.OUT'
   os.system(command)
   #
   write_energy(molecule,ref)
   #
   if (not molecule['error']):
      command='xclean'
      os.system(command)
   #
   return molecule

def write_zmat_hf(molecule):
   #
   out=open('ZMAT','w')
   #
   out.write(molecule['mol'])
   #
   out.write('*CFOUR(CALC=HF\n')
   out.write('SCF_CONV=9\n')
   out.write('LINEQ_CONV=9\n')
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
   out.write('MEMORY='+molecule['mem']+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()

def write_zmat_corr(molecule,drop_string,ref):
   #
   out=open('ZMAT','w')
   #
   out.write(molecule['mol'])
   #
   if (molecule['model'] == 'FCI'):
      #
      out.write('*CFOUR(CALC=FULLCI\n')
      out.write('CAS_MMAX=10\n')
      out.write('CAS_MITMAX=200\n')
   #
   else:
      #
      out.write('*CFOUR(CALC='+molecule['model']+'\n')
      out.write('CC_PROG=VCC\n')
      out.write('CC_EXPORDER=10\n')
      out.write('CC_MAXCYC=200\n')
   #
   if (drop_string != '\n'):
      #
      out.write(drop_string)
   #
   out.write('SCF_CONV=9\n')
   out.write('LINEQ_CONV=9\n')
   out.write('CC_CONV=9\n')
   #
   if (molecule['local']):
      #
      out.write('SYMMETRY=OFF\n')
      out.write('ORBITALS=LOCAL\n')
   #
   if (molecule['fc'] and ref):
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
   out.write('MEMORY='+molecule['mem']+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()

def init_zmat(molecule):
   #
   if (molecule['mol_string'] == 'H2O'):
      #
      s = 'ZMAT file for H2O\n'
      s += 'H\n'
      s += 'O 1 ROH\n'
      s += 'H 2 ROH 1 AHOH\n'
      s += '\n'
      s += 'ROH = 0.957\n'
      s += 'AHOH = 104.2\n'
      s += '\n'
      #
      molecule['mol'] = s
      #
      molecule['mult'] = 1
      #
      if (molecule['fc']):
         #
         molecule['core'] = 1
      #
      else:
         #
         molecule['core'] = 0
#   elif (mol_string == 'C2H2'):
#      s = 'ZMAT file for acetylene\n'
#      s += 'C\n'
#      s += 'C 1 RCC\n'
#      s += 'X 1 RX 2 A90\n'
#      s += 'H 1 RCH 3 A90 2 A180\n'
#      s += 'X 2 RX 1 A90 3 A180\n'
#      s += 'H 2 RCH 5 A90 1 A180\n'
#      s += '\n'
#      s += 'RCH = 1.08\n'
#      s += 'RX = 1.0\n'
#      if (bond[0] != 0.0):
#         s += 'RCC = '+str(bond[0])+'\n'
#      else:
#         s += 'RCC = 1.2\n'
#      s += 'A90 = 90.\n'
#      s += 'A180 = 180.\n'
#      s += '\n'
#      mol.append(s)
#      #
#      mult.append(1)
#      if (frozen):
#         core.append(2)
#      else:
#         core.append(0)
#   elif (mol_string == 'N2'):
#      s = 'ZMAT file for nitrogen (N2)\n'
#      s += 'N\n'
#      s += 'N 1 RNN\n'
#      s += '\n'
#      if (bond[0] != 0.0):
#         s += 'RNN = '+str(bond[0])+'\n'
#      else:
#         s += 'RNN = 1.098\n'
#      s += '\n'
#      mol.append(s)
#      #
#      mult.append(1)
#      if (frozen):
#         core.append(2)
#      else:
#         core.append(0)
#   elif (mol_string == 'O2'):
#      s = 'ZMAT file for (triplet) oxygen (O2)\n'
#      s += 'O\n'
#      s += 'O 1 ROO\n'
#      s += '\n'
#      if (bond[0] != 0.0):
#         s += 'ROO = '+str(bond[0])+'\n'
#      else:
#         s += 'ROO = 1.205771156354447\n'
#      s += '\n'
#      mol.append(s)
#      #
#      mult.append(3)
#      if (frozen):
#         core.append(2)
#      else:
#         core.append(0)
#   elif (mol_string == 'SiH4'):
#      s = 'ZMAT file for silane\n'
#      s += 'Si\n'
#      s += 'H 1 R\n'
#      s += 'H 1 R 2 TDA\n'
#      s += 'H 1 R 2 TDA 3 D120\n'
#      s += 'H 1 R 2 TDA 4 D120\n'
#      s += '\n'
#      if (bond[0] != 0.0):
#         s += 'R = '+str(bond[0])+'\n'
#      else:
#         s += 'R = 1.48598655\n'
#      s += 'TDA = 109.471221\n'
#      s += 'D120 = 120.\n'
#      s += '\n'
#      mol.append(s)
#      #
#      mult.append(1)
#      if (frozen):
#         core.append(5)
#      else:
#         core.append(0)
#   elif (mol_string == 'CO2'):
#      s = 'ZMAT file for carbon dioxide\n'
#      s += 'C\n'
#      s += 'X 1 RCX\n'
#      s += 'O 1 RCO 2 A90\n'
#      s += 'O 1 RCO 2 A90 3 A180\n'
#      s += '\n'
#      s += 'RCX=1.0\n'
#      if (bond[0] != 0.0):
#         s += 'RCO = '+str(bond[0])+'\n'
#      else:
#         s += 'RCO = 1.16\n'
#      s += 'A90 = 90.0\n'
#      s += 'A180 = 180.0\n'
#      s += '\n'
#      mol.append(s)
#      #
#      mult.append(1)
#      if (frozen):
#         core.append(3)
#      else:
#         core.append(0)
#   elif (mol_string == 'C4H6'):
#      s = 'ZMAT file for trans-1,3-butadiene\n'
#      s += 'C\n'
#      s += 'C 1 CDC\n'
#      s += 'C 2 CSC 1 CCC\n'
#      s += 'C 3 CDC 2 CCC 1 A180\n'
#      s += 'H 1 CH1 2 CC1 3 A180\n'
#      s += 'H 1 CH2 2 CC2 3 A0\n'
#      s += 'H 2 CH3 1 CC3 3 A180\n'
#      s += 'H 3 CH3 4 CC3 2 A180\n'
#      s += 'H 4 CH2 3 CC2 2 A0\n'
#      s += 'H 4 CH1 3 CC1 2 A180\n'
#      s += '\n'
#      s += 'CDC = 1.34054111\n' 
#      s += 'CSC = 1.45747113\n'
#      s += 'CH1 = 1.08576026\n'
#      s += 'CH2 = 1.08793795\n'
#      s += 'CH3 = 1.09045613\n'
#      s += 'CCC = 124.31048212\n' 
#      s += 'CC1 = 121.82705922\n'
#      s += 'CC2 = 121.53880082\n'
#      s += 'CC3 = 119.43908311\n'
#      s += 'A0 = 0.0\n'
#      s += 'A180 = 180.0\n'
#      s += '\n'
#      mol.append(s)
#      #
#      mult.append(1)
#      if (frozen):
#         core.append(4)
#      else:
#         core.append(0)
#   elif (mol_string == 'C6H6'):
#      s = 'ZMAT file for benzene\n'
#      s += 'X\n'
#      s += 'C 1 RCC\n'
#      s += 'C 1 RCC 2 A60\n'
#      s += 'C 1 RCC 3 A60 2 D180\n' 
#      s += 'C 1 RCC 4 A60 3 D180\n'
#      s += 'C 1 RCC 5 A60 4 D180\n'
#      s += 'C 1 RCC 6 A60 5 D180\n'
#      s += 'H 1 RXH 2 A60 7 D180\n'
#      s += 'H 1 RXH 3 A60 2 D180\n'
#      s += 'H 1 RXH 4 A60 3 D180\n'
#      s += 'H 1 RXH 5 A60 4 D180\n'
#      s += 'H 1 RXH 6 A60 5 D180\n'
#      s += 'H 1 RXH 7 A60 6 D180\n'
#      s += '\n'
#      s += 'A60 = 60.0\n'
#      s += 'D180 = 180.0\n'
#      if (bond[0] != 0.0):
#         s += 'RCC = '+str(bond[0])+'\n'
#      else:
#         s += 'RCC = 1.3914\n'
#      s += 'RXH = 2.4716\n'
#      s += '\n'
#      mol.append(s)
#      #
#      mult.append(1)
#      if (frozen):
#         core.append(6)
#      else:
#         core.append(0) 
#   else:
#      print('molecular input not recognized, aborting ...')
#      sys.exit(10)
   #
   return molecule

def get_dim(molecule):
   #
   inp=open('CFOUR.OUT','r')
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
         break
      #
      elif re.match(regex_err,line) is not None:
         #
         print('problem with HF calculation, aborting ...')
         molecule['error'] = True
         inp.close()
         return nocc, nvirt, error
   #
   inp.seek(0)
   #
   regex_2 = '\s+Alpha population by irrep:'
   #
   while 1:
      #
      line=inp.readline()
      #
      if re.match(regex_2,line) is not None:
         #
         pop = line.split()
         break
   #
   tmp = 0
   #
   for i in range(4,len(pop)):
      tmp += int(pop[i])
   #
   molecule['nocc'] = tmp
   molecule['nvirt'] = int(bf) - molecule['nocc']
   #
   inp.close()
   #
   return molecule

def write_energy(molecule,ref):
   #
   inp=open('CFOUR.OUT','r')
   #
   regex_err = '\s+ERROR ERROR'
   #
   while 1:
      #
      line=inp.readline()
      #
      if re.match(molecule['regex'],line) is not None:
         #
         if (molecule['model'] == 'FCI'):
            #
            [tmp] = line.split()[3:4]
         #
         else:
            #
            [tmp] = line.split()[4:5]
         #
         if (ref):
            #
            molecule['e_ref'] = float(tmp)
         #
         else:
            #
            molecule['e_tmp'] = float(tmp)
         #
         break
      #
      elif re.match(regex_err,line) is not None:
         #
         print('problem with '+molecule['model']+' calculation, aborting ...')
         molecule['error'] = True
         inp.close()
         #
         return molecule
   #
   inp.close()
   #
   return molecule

def inc_corr_tuple_thres(molecule):
   #
   if (molecule['exp'] == 'OCC'):
      #
      u_limit = molecule['nocc']-molecule['core']
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      u_limit = molecule['nvirt']
   #
   elif (molecule['exp'] == 'COMB'):
      #
      u_limit_1 = molecule['nocc']-molecule['core']
      u_limit_2 = molecule['nvirt']
   #
   list_drop = []
   #
   for i in range(0,molecule['nocc']+molecule['nvirt']):
      #
      list_drop.append(i+1)
   #
   if (molecule['exp'] == 'OCC'):
      #
      for i in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
         #
         list_drop[i] = 0
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      for i in range(molecule['core'],molecule['nocc']):
         #
         list_drop[i] = 0
   #
   drop_string = []
   #
   incl_list = [[]]
   #
   if (molecule['exp'] == 'COMB'):
      #
      drop_string_comb = []
      n_contrib_comb = []
      #
      incl_list_comb = [[]]
      e_contrib_comb = [[[]]]
      #
      e_fin_comb = []
      #
      conv_comb = [False]
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['e_contrib'] = [[[]]]
   #
   molecule['n_contrib'] = []
   #
   molecule['e_fin'] = []
   #
   molecule['time'] = []
   #
   molecule['conv'] = False
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'VIRT')):
      #
      for k in range(1,u_limit+1):
         #
         start = timer()
         #
         drop_string[:] = []
         #
         if (molecule['exp'] == 'OCC'):
            #
            generate_drop_occ(molecule['core']+1,1,k,molecule,list_drop,drop_string,molecule['n_contrib'])
         #
         elif (molecule['exp'] == 'VIRT'):
            #
            generate_drop_virt(molecule['nocc']+1,1,k,molecule,list_drop,drop_string,molecule['n_contrib'])
         #
         if (k > 1):
            #
            molecule['e_contrib'].append([[]])
         #
         if (len(molecule['n_contrib']) < k):
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  has no contributions'.format(k,u_limit))
            print(' --------------------------------------------------------')
            print('')
            continue
         #
         print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  started'.format(k,u_limit))
         print(' -------------------------------------------')
         #
         incl_list[:] = [[]]
         #
         for i in range(0,molecule['n_contrib'][k-1]):
            #
            run_calc_corr(molecule,drop_string[i],False)
            #
            if (i > 0):
               incl_list.append([])
            #
            orbs_incl(molecule,drop_string[i],incl_list[i],False)
            #
            if (i == 0):
               #
               molecule['e_contrib'][k-1][0].append(incl_list[i])
               molecule['e_contrib'][k-1][0].append(molecule['e_tmp'])
            #
            else:
               #
               molecule['e_contrib'][k-1].append([incl_list[i],molecule['e_tmp']])
            #
            if (molecule['error']):
               #
               return molecule
         #
         inc_corr_order(k,molecule['n_contrib'],molecule['e_contrib'],molecule['e_fin'])
         #
         if (k > 1):
            #
            if (molecule['exp'] == 'OCC'):
               #
               inc_corr_chk_conv(k,molecule['thres'][0],molecule['e_fin'],molecule['conv'])
            #
            elif (molecule['exp'] == 'VIRT'):
               #
               inc_corr_chk_conv(k,molecule['thres'][1],molecule['e_fin'],molecule['conv'])
         #
         molecule['time'].append(timer()-start)
         #
         if (k == 1):
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit,molecule['time'][k-1]))
            print(' --------------------------------------------------------------')
         #
         else:
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}  ---  conv =  {4:}'.\
                             format(k,u_limit,molecule['time'][k-1],molecule['e_fin'][k-1]-molecule['e_fin'][k-2],molecule['conv']))
            print(' ------------------------------------------------------------------------------------------------------------')
         #
         for i in range(0,molecule['n_contrib'][k-1]):
            #
            print(' RESULT-MACRO:  tuple = {0:4d} / {1:4d}  ,  corr. orbs. = {2:}  ,  abs = {3:9.4e}'.\
                             format(i+1,molecule['n_contrib'][k-1],molecule['e_contrib'][k-1][i][0],molecule['e_contrib'][k-1][i][1]))
         #
         print('')
         #
         if (molecule['conv']):
            #
            return molecule
   #
   elif (molecule['exp'] == 'COMB'):
      #
      for k in range(1,u_limit_1+1):
         #
         start = timer()
         #
         drop_string[:] = []
         #
         generate_drop_occ(molecule['core']+1,1,k,molecule,list_drop,drop_string,molecule['n_contrib'])
         #
         if (k > 1):
            #
            molecule['e_contrib'].append([[]])
         #
         if (len(molecule['n_contrib']) < k):
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  has no contributions'.format(k,u_limit_1))
            print(' --------------------------------------------------------')
            print('')
            continue
         #
         print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  started'.format(k,u_limit_1))
         print(' -------------------------------------------')
         #
         incl_list[:] = [[]]
         #
         for j in range(0,molecule['n_contrib'][k-1]):
            #
            start_comb = timer()
            #
            n_contrib_comb[:] = []
            #
            e_inc_comb[:] = []
            #
            e_contrib_comb[:] = [[[]]]
            #
            conv_comb[0] = False
            #
            for l in range(1,u_limit_2+1):
               #
               for a in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
                  #
                  list_drop[a] = a+1
               #
               drop_string_comb[:] = []
               #
               generate_drop_virt(molecule['nocc']+1,1,l,molecule,list_drop,drop_string_comb,n_contrib_comb)
               #
               if (l > 1):
                  #
                  e_contrib_comb.append([[]])
               #
               incl_list_comb[:] = [[]]
               #
               for i in range(0,n_contrib_comb[l-1]):
                  #
                  if (drop_string[j] == ''):
                     #
                     string = 'DROP_MO='+drop_string_comb[i][1:]
                  #
                  else:
                     #
                     string = drop_string[j]+drop_string_comb[i]
                  #
                  run_calc_corr(molecule,string,False)
                  #
                  if (i > 0):
                     #
                     incl_list_comb.append([])
                  #
                  orbs_incl(molecule,string,incl_list_comb[i],True)
                  #
                  if (i == 0):
                     #
                     e_contrib_comb[l-1][0].append(incl_list_comb[i])
                     e_contrib_comb[l-1][0].append(molecule['e_tmp'])
                  #
                  else:
                     #
                     e_contrib_comb[l-1].append([incl_list_comb[i],molecule['e_tmp']])
                  #
                  if (molecule['error']):
                     #
                     return molecule
               #
               inc_corr_order(l,n_contrib_comb,e_contrib_comb,e_fin_comb)
               #
               if (l > 1):
                  #
                  inc_corr_chk_conv(l,molecule['thres'][1],e_fin_comb,conv_comb[0])
               #
               nv_order = l
               #
               if (conv_comb[0]):
                  #
                  if (j > 0):
                     #
                     incl_list.append([])
                  #
                  orbs_incl(molecule,drop_string[j],incl_list[j],False)
                  #
                  if (j == 0):
                     #
                     molecule['e_contrib'][k-1][0].append(incl_list[j])
                     molecule['e_contrib'][k-1][0].append(e_inc_comb[l-1])
                  #
                  else:
                     #
                     molecule['e_contrib'][k-1].append([incl_list[j],e_inc_comb[l-1]])
                  #
                  break
            #
            print('       STATUS-MICRO:  tuple = {0:4d} / {1:4d}  (order = {2:4d} / {3:4d})  done in {4:10.2e} seconds  ---  diff =  {5:9.4e}  ---  conv =  {6:}'\
                             .format(j+1,molecule['n_contrib'][k-1],nv_order,molecule['nvirt'],timer()-start_comb,e_inc_comb[l-1]-e_inc_comb[l-2],conv_comb[0]))
         #
         inc_corr_order(k,molecule['n_contrib'],molecule['e_contrib'],molecule['e_fin'])
         #
         if (k > 1):
            #
            inc_corr_chk_conv(k,molecule['thres'][0],molecule['e_fin'],molecule['conv'])
         #
         molecule['time'].append(timer()-start)
         #
         print('')
         #
         if (k == 1):
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit_1,molecule['time'][k-1]))
            print(' --------------------------------------------------------------')
         #
         else:
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}  ---  conv =  {4:}'.\
                             format(k,u_limit_1,molecule['time'][k-1],molecule['e_fin'][k-1]-molecule['e_fin'][k-2],molecule['conv']))
            print(' ------------------------------------------------------------------------------------------------------------')
         #
         for i in range(0,molecule['n_contrib'][k-1]):
            #
            print(' RESULT-MACRO:  tuple = {0:4d} / {1:4d}  ,  corr. orbs. = {2:}  ,  abs = {3:9.4e}'.\
                             format(i+1,molecule['n_contrib'][k-1],molecule['e_contrib'][k-1][i][0],molecule['e_contrib'][k-1][i][1]))
         #
         print('')
         #
         if (molecule['conv']):
            #
            return molecule
   #
   return molecule

def inc_corr_tuple_order(molecule):
   #
   if (molecule['exp'] == 'OCC'):
      #
      u_limit = molecule['nocc']-molecule['core']
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      u_limit = molecule['nvirt']
   #
   list_drop = []
   #
   for i in range(0,molecule['nocc']+molecule['nvirt']):
      #
      list_drop.append(i+1)
   #
   if (molecule['exp'] == 'OCC'):
      #
      for i in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
         #
         list_drop[i] = 0
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      for i in range(molecule['core'],molecule['nocc']):
         #
         list_drop[i] = 0
   #
   drop_string = []
   #
   incl_list = [[]]
   #
   molecule['e_contrib'] = [[[]]]
   #
   for _ in range(0,molecule['order']):
      #
      molecule['e_contrib'].append([[]])
   #
   molecule['n_contrib'] = []
   #
   for _ in range(0,u_limit):
      #
      molecule['n_contrib'].append(0)
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['e_fin'] = []
   #
   molecule['time'] = []
   #
   molecule['conv'] = False
   #
   for k in range(molecule['order'],0,-1):
      #
      start = timer()
      #
      drop_string[:] = []
      #
      if (molecule['exp'] == 'OCC'):
         #
         generate_drop_occ(molecule['core']+1,1,k,molecule,list_drop,drop_string,molecule['n_contrib'])
      #
      elif (molecule['exp'] == 'VIRT'):
         #
         generate_drop_virt(molecule['nocc']+1,1,k,molecule,list_drop,drop_string,molecule['n_contrib'])
      #
      if (molecule['n_contrib'][k-1] == 0):
         #
         print(' STATUS:  order = {0:4d} / {1:4d}  has no contributions'.format(k,u_limit))
         print(' --------------------------------------------------')
         print('')
         continue
      #
      print(' STATUS:  order = {0:4d} / {1:4d}  started'.format(k,u_limit))
      print(' -------------------------------------')
      #
      incl_list[:] = [[]]
      #
      for i in range(0,molecule['n_contrib'][k-1]):
         #
         run_calc_corr(molecule,drop_string[i],False)
         #
         if (i > 0):
            #
            incl_list.append([])
         #
         orbs_incl(molecule,drop_string[i],incl_list[i],False)
         #
         if (i == 0):
            #
            molecule['e_contrib'][k-1][0].append(incl_list[i])
            molecule['e_contrib'][k-1][0].append(molecule['e_tmp'])
         #
         else:
            #
            molecule['e_contrib'][k-1].append([incl_list[i],molecule['e_tmp']])
         #
         if (molecule['error']):
            #
            return molecule
      #
      molecule['time'].append(timer()-start)
      #
      print(' STATUS:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit,molecule['time'][-1]))
      print(' --------------------------------------------------------')
      #
      print('')
   #
   for k in range(1,molecule['order']+1):
      #
      if (molecule['n_contrib'][k-1] > 0):
         #
         inc_corr_order(k,molecule['n_contrib'],molecule['e_contrib'],molecule['e_fin'])
   #
   molecule['time'].reverse()
   #
   return molecule

def generate_drop_occ(start,order,final,molecule,list_drop,drop_string,n_contrib):
   #
   if (order > (molecule['nocc']-molecule['core'])):
      #
      return drop_string, n_contrib
   #
   else:
      #
      for i in range(start,molecule['nocc']+1): # loop over the occupied orbs
         #
         n = list_drop[molecule['core']:molecule['nocc']].count(0) # count the number of zeros
         #
         if (n == 0):
            #
            list_drop[i-1] = 0 # singles correlation
         #
         if (n > 0):
            #
            if (not molecule['list_excl']):
               #
               list_drop[i-1] = 0 # no screening
            #
            else:
               #
               if (not molecule['list_excl'][i-1]): # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
                  #
                  continue
               #
               else:
                  #
                  list_drop[i-1] = 0 # attempt to correlate orbital 'i'
                  idx = [j for j, val in enumerate(list_drop) if val == 0] # make list containing indices with zeros in list_drop
                  idx_2 = [j for j, val in enumerate(molecule['list_excl'][i-1]) if val != 0] # make list containing indices with non-zeros in list_excl
                  #
                  if ((set(idx) > set(idx_2)) and (len(idx_2) > 0)): # check whether idx_2 is a subset of idx
                     #
                     list_drop[i-1] = i # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
         #
         s = ''
         inc = 0
         #
         if ((order == final) and (list_drop[molecule['core']:molecule['nocc']].count(0) == final)): # number of zeros in list_drop must match the final order
            #
            if (molecule['fc']): # exclude core orbitals
               #
               for m in range(0,molecule['core']):
                  #
                  if (inc == 0):
                     #
                     s = 'DROP_MO='+str(list_drop[m])
                  #
                  else:
                     #
                     s += '-'+str(list_drop[m])
                  #
                  inc += 1
            #
            for m in range(molecule['core'],molecule['nocc']): # start to exclude valence occupied orbitals
               #
               if (list_drop[m] != 0):
                  #
                  if (inc == 0):
                     #
                     s = 'DROP_MO='+str(list_drop[m])
                  #
                  else:
                     #
                     s += '-'+str(list_drop[m])
                  #
                  inc += 1
            #
            if (s != ''):
               #
               if (len(n_contrib) >= order):
                  #
                  n_contrib[order-1] += 1
               #
               else:
                  #
                  n_contrib.append(1)
               #
               if (molecule['exp'] == 'OCC'):
                  #
                  drop_string.append(s+'\n')
               #
               elif (molecule['exp' == 'COMB']):
                  #
                  drop_string.append(s)
            #
            elif (order == molecule['nocc']): # full system correlation, i.e., equal to standard N-electron calculation
               #
               n_contrib.append(1)
               #
               drop_string.append(''+'\n')
         #
         generate_drop_occ(i+1,order+1,final,molecule,list_drop,drop_string,n_contrib) # recursion
         #
         list_drop[i-1] = i # include orb back into list of orbs to drop from the calculation
   #
   return drop_string, n_contrib

def generate_drop_virt(start,order,final,molecule,list_drop,drop_string,n_contrib):
   #
   if (order > molecule['nvirt']):
      #
      return drop_string, n_contrib
   #
   else:
      #
      for i in range(start,(molecule['nocc']+molecule['nvirt'])+1):
         #
         list_drop[i-1] = 0
         s = ''
         inc = 0
         #
         if (molecule['exp'] == 'COMB'):
            #
            inc += 1
         #
         if (order == final):
            #
            if (molecule['fc']): # exclude core orbitals
               #
               for m in range(0,molecule['core']):
                  #
                  if (inc == 0):
                     #
                     s = 'DROP_MO='+str(list_drop[m])
                  #
                  else:
                     #
                     s += '-'+str(list_drop[m])
                  #
                  inc += 1
            #
            for m in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
               #
               if (list_drop[m] != 0):
                  #
                  if (inc == 0):
                     #
                     s = 'DROP_MO='+str(list_drop[m])
                  #
                  else:
                     #
                     s += '-'+str(list_drop[m])
                  #
                  inc += 1
            #
            if (len(n_contrib) >= order):
               #
               n_contrib[order-1] += 1
            #
            else:
               #
               n_contrib.append(1)
            #
            drop_string.append(s+'\n')
         #
         generate_drop_virt(i+1,order+1,final,molecule,list_drop,drop_string,n_contrib)
         #
         list_drop[i-1] = i
   #
   return drop_string, n_contrib

def screen_occ(molecule):
   #
   # screen away all interactions between orb 1 and any of the other orbs --- corresponds to a minor improvement over a frozen-core calculation
   #
#   molecule['list_excl'][0]   = []
#   molecule['list_excl'].append([1,0,0,0,0])
#   molecule['list_excl'].append([1,0,0,0,0])
#   molecule['list_excl'].append([1,0,0,0,0])
#   molecule['list_excl'].append([1,0,0,0,0])
   #
   # screen away all interactions between orb 2 and any of the other orbs
   #
   molecule['list_excl'][0]   = [0,2,0,0,0]
   molecule['list_excl'].append([])
   molecule['list_excl'].append([0,2,0,0,0])
   molecule['list_excl'].append([0,2,0,0,0])
   molecule['list_excl'].append([0,2,0,0,0])
   #
   # screen away interactions between orbs 1/2 and between orbs 4/5
   #
#   molecule['list_excl'][0]   = [0,2,0,0,0]
#   molecule['list_excl'].append([1,0,0,0,0])
#   molecule['list_excl'].append([0,0,0,0,0])
#   molecule['list_excl'].append([0,0,0,0,5])
#   molecule['list_excl'].append([0,0,0,4,0])
   #
   return molecule

def orbs_incl(molecule,string_excl,string_incl,comb):
   #
   excl_list = []
   #
   sub_string = string_excl[8:] # remove the 'DROP_MO=' part of the string
   sub_list = sub_string.split("-") # remove all the hyphens
   #
   for j in range(0,len(sub_list)):
      #
      if (sub_list[j] != ''):
         #
         excl_list.append(int(sub_list[j]))
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      for l in range(1,molecule['nocc']+1):
         #
         if (not (l in excl_list)):
            #
            string_incl.append(l)
   #
   elif ((molecule['exp'] == 'VIRT') or comb):
      #
      for l in range(molecule['nocc']+1,molecule['nocc']+molecule['nvirt']+1):
         #
         if (not (l in excl_list)):
            #
            string_incl.append(l)
   #
   return string_incl

#def inc_corr_order(k,n,e_vec,e_inc):
#   #
#   e_sum = 0.0
#   #
#   for m in range(1,k+1):
#      e_sum += (-1)**(m) * (1.0 / math.factorial(m)) * prefactor(n,k,m-1) * e_vec[(k-m)-1]
#   e_inc.append(e_vec[k-1]+e_sum)
#   #
#   return e_inc
#
#def prefactor(n,order,m):
#   #
#   pre = 1
#   #
#   for i in range(m,-1,-1):
#      pre = pre * (n - order + i)
#   #
#   return pre

def inc_corr_order(k,n_contrib,e_contrib,e_fin):
   #
   for j in range(0,n_contrib[k-1]):
      #
      for i in range(k-1,0,-1):
         #
         for l in range(0,n_contrib[i-1]):
            #
            if (set(e_contrib[i-1][l][0]) < set(e_contrib[k-1][j][0])):
               #
               e_contrib[k-1][j][1] -= e_contrib[i-1][l][1]
   #
   e_tmp = 0.0
   #
   for j in range(0,n_contrib[k-1]):
      #
      e_tmp += e_contrib[k-1][j][1]
   #
   if (k > 1):
      #
      e_tmp += e_fin[k-2]
   #
   e_fin.append(e_tmp)
   #
   return e_fin

def inc_corr_chk_conv(order,thres,e_fin,conv):
   #
   e_diff = e_fin[order-1] - e_fin[order-2]
   #
   if (abs(e_diff) < thres):
      #
      conv = True
   #
   else:
      #
      conv = False
   #
   return conv

def ref_calc(molecule):
   #
   start = timer()
   #
   print(' STATUS:  Full reference calc.  started')
   print(' --------------------------------------')
   #
   run_calc_corr(molecule,'',True)
   #
   molecule['time'].append(timer()-start)
   #
   print(' STATUS:  Full reference calc.  done in {0:10.2e} seconds'.format(molecule['time'][-1]))
   print(' ---------------------------------------------------------')
   print('')
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
   print('   screening (occ.)   =  {0:}'.format(molecule['screen'][0]))
   print('   screening (virt.)  =  {0:}'.format(molecule['screen'][1]))
   #
   if (molecule['exp_ctrl']):
      #
      if (molecule['thres'][0] > 0.0):
         #
         print('   thres. (occ.)      =  {0:6.1e}'.format(molecule['thres'][0]))
      #
      else:
         #
         print('   thres. (occ.)      =  N/A')
      #
      if (molecule['thres'][1] > 0.0):
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
      print('   convergence met    =  {0:}'.format(molecule['conv']))
   #
   print('   error in calc.     =  {0:}'.format(molecule['error']))
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
   if (molecule['ref'] and (not molecule['error'])):
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

def main():
   #
   #  ---  redirect stdout to output.out - if present in wrk dir (alongside plotting output), delete these files before proceeding...  ---
   #
   if (os.path.isfile('output.out')):
      #
      command='rm output.out'
      os.system(command)
   #
   if (os.path.isfile('output.pdf')):
      #
      command='rm output.pdf'
      os.system(command)
   #
   sys.stdout = logger('output.out')
   #
   #  ---  initialize the calculation...  ---
   #
   molecule = {}
   #
   init_calc(molecule)
   #
   print('\n')
   print(' ** START INC.-CORR. ('+molecule['model']+') CALCULATION **\n')
   #
   #  ---  setup of scratch directory...  ---
   #
   setup_calc(molecule)
   #
   #  ---  run HF calc to determine problem size parameters...  ---
   #
   run_calc_hf(molecule)
   #
   #  ---  run a few sanity checks...  ---
   #
   sanity_chk(molecule)
   #
   #  ---  initialize (potential) screening...  ---
   #
   init_screen(molecule)
   #
   #  ---  initialization done - start the calculation...  ---
   #
   if (molecule['exp_ctrl']):
      #
      inc_corr_tuple_thres(molecule)
   #
   else:
      #
      inc_corr_tuple_order(molecule)
   #
   #  ---  start (potential) reference calculation...  ---
   #
   if (molecule['ref'] and (not molecule['error'])):
      #
      ref_calc(molecule)
   #
   #  ---  clean up...  ---
   #
   term_calc(molecule)
   #
   #  ---  print summary of the calculation  ---
   #
   inc_corr_summary(molecule)
   #
   #  ---  plot the results of the calculation  ---
   #
   if (not molecule['error']):
      #
      inc_corr_plot.ic_plot(molecule)
   #
   #  ---  calculation done - terminating...  ---
   #
   print(' ** END OF INC.-CORR. ('+molecule['model']+') CALCULATION **\n')
   print('\n')

if __name__ == '__main__':
   #
   main()

