#!/usr/bin/env python

#
# python driver for inc.-corr. calculations using CFOUR as backend program.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
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

def redirect_stdout(molecule):
   #
   molecule['wrk'] = os.getcwd()
   #
   if (os.path.isfile(molecule['wrk']+'/output.out')):
      #
      command='rm '+molecule['wrk']+'/output.out'
      os.system(command)
   #
   if (os.path.isfile(molecule['wrk']+'/output.pdf')):
      #
      command='rm '+molecule['wrk']+'/output.pdf'
      os.system(command)
   #
   sys.stdout = logger(molecule['wrk']+'/output.out')
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
   if (molecule['error']):
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
   molecule['error'] = False
   #
   molecule['conv'] = False
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
         molecule['error'] = True
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      if (order >= molecule['nvirt']):
         #
         print 'wrong input argument for total order (must be .lt. number of virtual orbitals), aborting ...'
         molecule['error'] = True
   #
   elif (molecule['exp'] == 'COMB'):
      #
      if ((molecule['thres'][0] == 0.0) or (molecule['thres'][1] == 0.0)):
         #
         print('expansion scheme "COMB" requires both an occupied and a virtual expansion threshold, aborting ...')
         molecule['error'] = True
      #
      if (not molecule['exp_ctrl']):
         #
         print('expansion scheme "COMB" is currently not implemented for fixed order expansion, aborting ...')
         molecule['error'] = True
   #
   if ((molecule['order'] > 0) and molecule['exp_ctrl']):
      #
      print('fixed order expansion requested, but expansion thresholds provided, aborting ...')
      molecule['error'] = True
   #
   if ((molecule['order'] == 0) and (not molecule['exp_ctrl'])):
      #
      print('neither fixed order nor threshold-governed expansion requested, aborting ...')
      molecule['error'] = True
   #
   if (molecule['fc'] and molecule['local']):
      #
      print 'wrong input -- comb. of frozen core and local orbitals not implemented, aborting ...'
      molecule['error'] = True
   #
   if (molecule['error']):
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
   out.write('MEMORY='+str(molecule['mem'])+'\n')
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
   out.write('MEMORY='+str(molecule['mem'])+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()
   #
   return

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
                  list_drop[i-1] = i
               #
               else:
                  #
                  list_drop[i-1] = 0 # attempt to correlate orbital 'i'
                  idx = [j+1 for j, val in enumerate(list_drop) if val == 0] # make list containing indices (+1) with zeros in list_drop
                  #
                  if (set(idx) > set(molecule['list_excl'][i-1])): # check whether molecule['list_excl'][i-1] is a subset of idx
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
#   molecule['list_excl'].append([1])
#   molecule['list_excl'].append([1])
#   molecule['list_excl'].append([1])
#   molecule['list_excl'].append([1])
   #
   # screen away all interactions between orb 2 and any of the other orbs
   #
#   molecule['list_excl'][0]   = [2]
#   molecule['list_excl'].append([])
#   molecule['list_excl'].append([2])
#   molecule['list_excl'].append([2])
#   molecule['list_excl'].append([2])
   #
   # screen away interactions between orbs 1/2 and between orbs 4/5
   #
   molecule['list_excl'][0]   = [2]
   molecule['list_excl'].append([1])
   molecule['list_excl'].append([0,0,0,0,0])
   molecule['list_excl'].append([5])
   molecule['list_excl'].append([4])
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
      print('   convergence met    =  {0:}'.format(molecule['conv']))
   else:
      #
      print('   convergence met    =  N/A')
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
   #  ---  init molecule dictionary... ---
   #
   molecule = {}
   #
   #  ---  redirect stdout to output.out - if present in wrk dir (alongside plotting output), delete these files before proceeding...  ---
   #
   redirect_stdout(molecule)
   #
   #  ---  initialize the calculation...  ---
   #
   init_calc(molecule)
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
   #  ---  terminate calculation and clean up...  ---
   #
   term_calc(molecule)
   #

if __name__ == '__main__':
   #
   main()

