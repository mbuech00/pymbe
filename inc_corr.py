#!/usr/bin/env python

#
# python driver for inc.-corr. calculations using CFOUR as backend program.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), August 2016, Mainz, Germnay.
#
# Requires the path of the cfour basis GENBAS file ($CFOURBASIS) and bin directory ($CFOURBIN)
#

import sys
from sys import stdin
import re
import argparse
import os
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

def rm_scr_dir(directive):
   #
   command='rm -rf '+directive
   os.system(command)

def cd_dir(directive):
   #
   os.chdir(directive)

def save_err_out(directive):
   #
   command='cp '+directive+'/CFOUR.OUT .'
   os.system(command)

def prepare_calc():
   #
   command='cp '+CFOUR_BASIS+' .'
   os.system(command)
   command='cp '+CFOUR_BIN+'/x* .'
   os.system(command)

def run_calc_hf(mult,basis,mol,nocc,nvirt,error,mem):
   #
   write_zmat_hf(mult,basis,mol,mem)
   #
   command='xcfour &> CFOUR.OUT'
   os.system(command)
   #
   get_dim(nocc,nvirt,error)
   #
   if (not error[0]):
      command='xclean'
      os.system(command)
   #
   return nocc, nvirt, error

def run_calc_corr(order,mult,fc,model,basis,regex,mol,drop_string,e_vec,e_ref,error,ref,mem,local):
   # 
   write_zmat_corr(mult,fc,model,basis,mol,drop_string,mem,local)
   #
   command='xcfour &> CFOUR.OUT'
   os.system(command)
   #
   write_energy(order,model,regex,e_vec,e_ref,error,ref)
   #
   if (not error[0]):
      command='xclean'
      os.system(command)
   #
   return e_vec, e_ref, error

def write_zmat_hf(mult,basis,mol,mem):
   #
   out=open('ZMAT','w')
   #
   out.write(mol[0])
   #
   out.write('*CFOUR(CALC=HF\n')
   out.write('SCF_CONV=9\n')
   out.write('LINEQ_CONV=9\n')
   #
   out.write('MULTIPLICITY='+str(mult[0])+'\n')
   #
   if (mult[0] == 1):
      out.write('REF=RHF\n')
   else:
      out.write('REF=UHF\n')
   #
   out.write('BASIS='+basis+'\n')
   #
   out.write('MEMORY='+mem+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()

def write_zmat_corr(mult,fc,model,basis,mol,drop_string,mem,local):
   #
   out=open('ZMAT','w')
   #
   out.write(mol[0])
   #
   if (model == 'FCI'):
      out.write('*CFOUR(CALC=FULLCI\n')
      out.write('CAS_MMAX=10\n')
      out.write('CAS_MITMAX=200\n')
   else:
      out.write('*CFOUR(CALC='+model+'\n')
      out.write('CC_PROG=VCC\n')
      out.write('CC_EXPORDER=10\n')
      out.write('CC_MAXCYC=200\n')
   #
   if (drop_string != '\n'):
      out.write(drop_string)
   #
   out.write('SCF_CONV=9\n')
   out.write('LINEQ_CONV=9\n')
   out.write('CC_CONV=9\n')
   #
   if (local):
      out.write('SYMMETRY=OFF\n')
      out.write('ORBITALS=LOCAL\n')
   #
   if (fc):
      out.write('FROZEN_CORE=ON\n')
   #
   out.write('MULTIPLICITY='+str(mult[0])+'\n')
   #
   if (mult[0] == 1):
      out.write('REF=RHF\n')
   else:
      out.write('REF=UHF\n')
   #
   out.write('BASIS='+basis+'\n')
   #
   out.write('MEMORY='+mem+'\n')
   out.write('MEM_UNIT=GB)\n')
   #
   out.write('\n')
   #
   out.close()

def init_zmat(mol_string,bond,mol,frozen,core,mult):
   #
   if (mol_string == 'H2O'):
      s = 'ZMAT file for H2O\n'
      s += 'H\n'
      s += 'O 1 ROH\n'
      s += 'H 2 ROH 1 AHOH\n'
      s += '\n'
      if (bond[0] != 0.0):
         s += 'ROH = '+str(bond[0])+'\n'
      else:
         s += 'ROH = 0.957\n'
      s += 'AHOH = 104.2\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      if (frozen):
         core.append(1)
      else:
         core.append(0)
   elif (mol_string == 'C2H2'):
      s = 'ZMAT file for acetylene\n'
      s += 'C\n'
      s += 'C 1 RCC\n'
      s += 'X 1 RX 2 A90\n'
      s += 'H 1 RCH 3 A90 2 A180\n'
      s += 'X 2 RX 1 A90 3 A180\n'
      s += 'H 2 RCH 5 A90 1 A180\n'
      s += '\n'
      s += 'RCH = 1.08\n'
      s += 'RX = 1.0\n'
      if (bond[0] != 0.0):
         s += 'RCC = '+str(bond[0])+'\n'
      else:
         s += 'RCC = 1.2\n'
      s += 'A90 = 90.\n'
      s += 'A180 = 180.\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      if (frozen):
         core.append(2)
      else:
         core.append(0)
   elif (mol_string == 'N2'):
      s = 'ZMAT file for nitrogen (N2)\n'
      s += 'N\n'
      s += 'N 1 RNN\n'
      s += '\n'
      if (bond[0] != 0.0):
         s += 'RNN = '+str(bond[0])+'\n'
      else:
         s += 'RNN = 1.098\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      if (frozen):
         core.append(2)
      else:
         core.append(0)
   elif (mol_string == 'O2'):
      s = 'ZMAT file for (triplet) oxygen (O2)\n'
      s += 'O\n'
      s += 'O 1 ROO\n'
      s += '\n'
      if (bond[0] != 0.0):
         s += 'ROO = '+str(bond[0])+'\n'
      else:
         s += 'ROO = 1.205771156354447\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(3)
      if (frozen):
         core.append(2)
      else:
         core.append(0)
   elif (mol_string == 'SiH4'):
      s = 'ZMAT file for silane\n'
      s += 'Si\n'
      s += 'H 1 R\n'
      s += 'H 1 R 2 TDA\n'
      s += 'H 1 R 2 TDA 3 D120\n'
      s += 'H 1 R 2 TDA 4 D120\n'
      s += '\n'
      if (bond[0] != 0.0):
         s += 'R = '+str(bond[0])+'\n'
      else:
         s += 'R = 1.48598655\n'
      s += 'TDA = 109.471221\n'
      s += 'D120 = 120.\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      if (frozen):
         core.append(5)
      else:
         core.append(0)
   elif (mol_string == 'CO2'):
      s = 'ZMAT file for carbon dioxide\n'
      s += 'C\n'
      s += 'X 1 RCX\n'
      s += 'O 1 RCO 2 A90\n'
      s += 'O 1 RCO 2 A90 3 A180\n'
      s += '\n'
      s += 'RCX=1.0\n'
      if (bond[0] != 0.0):
         s += 'RCO = '+str(bond[0])+'\n'
      else:
         s += 'RCO = 1.16\n'
      s += 'A90 = 90.0\n'
      s += 'A180 = 180.0\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      if (frozen):
         core.append(3)
      else:
         core.append(0)
   elif (mol_string == 'C4H6'):
      s = 'ZMAT file for trans-1,3-butadiene\n'
      s += 'C\n'
      s += 'C 1 CDC\n'
      s += 'C 2 CSC 1 CCC\n'
      s += 'C 3 CDC 2 CCC 1 A180\n'
      s += 'H 1 CH1 2 CC1 3 A180\n'
      s += 'H 1 CH2 2 CC2 3 A0\n'
      s += 'H 2 CH3 1 CC3 3 A180\n'
      s += 'H 3 CH3 4 CC3 2 A180\n'
      s += 'H 4 CH2 3 CC2 2 A0\n'
      s += 'H 4 CH1 3 CC1 2 A180\n'
      s += '\n'
      s += 'CDC = 1.34054111\n' 
      s += 'CSC = 1.45747113\n'
      s += 'CH1 = 1.08576026\n'
      s += 'CH2 = 1.08793795\n'
      s += 'CH3 = 1.09045613\n'
      s += 'CCC = 124.31048212\n' 
      s += 'CC1 = 121.82705922\n'
      s += 'CC2 = 121.53880082\n'
      s += 'CC3 = 119.43908311\n'
      s += 'A0 = 0.0\n'
      s += 'A180 = 180.0\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      if (frozen):
         core.append(4)
      else:
         core.append(0)
   elif (mol_string == 'C6H6'):
      s = 'ZMAT file for benzene\n'
      s += 'X\n'
      s += 'C 1 RCC\n'
      s += 'C 1 RCC 2 A60\n'
      s += 'C 1 RCC 3 A60 2 D180\n' 
      s += 'C 1 RCC 4 A60 3 D180\n'
      s += 'C 1 RCC 5 A60 4 D180\n'
      s += 'C 1 RCC 6 A60 5 D180\n'
      s += 'H 1 RXH 2 A60 7 D180\n'
      s += 'H 1 RXH 3 A60 2 D180\n'
      s += 'H 1 RXH 4 A60 3 D180\n'
      s += 'H 1 RXH 5 A60 4 D180\n'
      s += 'H 1 RXH 6 A60 5 D180\n'
      s += 'H 1 RXH 7 A60 6 D180\n'
      s += '\n'
      s += 'A60 = 60.0\n'
      s += 'D180 = 180.0\n'
      if (bond[0] != 0.0):
         s += 'RCC = '+str(bond[0])+'\n'
      else:
         s += 'RCC = 1.3914\n'
      s += 'RXH = 2.4716\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      if (frozen):
         core.append(6)
      else:
         core.append(0) 
   else:
      print('molecular input not recognized, aborting ...')
      sys.exit(10)
   #
   return mol, core, mult

def get_dim(nocc,nvirt,error):
   #
   inp=open('CFOUR.OUT','r')
   #
   regex_err = '\s+ERROR ERROR'
   #
   regex = 'basis functions'
   #
   while 1:
      line=inp.readline()
      if regex in line:
         [bf] = line.split()[2:3]
         break
      elif re.match(regex_err,line) is not None:
         print('problem with HF calculation, aborting ...')
         error[0] = True
         inp.close()
         return nocc, nvirt, error
   #
   inp.seek(0)
   #
   regex_2 = '\s+Alpha population by irrep:'
   #
   while 1:
      line=inp.readline()
      if re.match(regex_2,line) is not None:
         pop = line.split()
         break
   #
   tmp = 0
   #
   for i in range(4,len(pop)):
      tmp += int(pop[i])
   #
   nocc.append(tmp)
   nvirt.append(int(bf) - nocc[0])
   #
   inp.close()
   #
   return nocc, nvirt, error

def write_energy(k,model,regex,e_vec,e_ref,error,ref):
   #
   inp=open('CFOUR.OUT','r')
   #
   regex_err = '\s+ERROR ERROR'
   #
   while 1:
      line=inp.readline()
      if re.match(regex,line) is not None:
         if (model == 'FCI'):
            [energy] = line.split()[3:4]
         else:
            [energy] = line.split()[4:5]
         if (ref):
            e_ref.append(float(energy))
         else:
            e_vec[k-1] += float(energy)
         break
      elif re.match(regex_err,line) is not None:
         print('problem with '+model+' calculation, aborting ...')
         error[0] = True
         inp.close()
         return e_ref, e_vec, error
   #
   inp.close()
   #
   return e_ref, e_vec, error

def inc_corr_tuple_thres(mol_string,nocc,nvirt,core,thres,mult,fc,exp,model,basis,regex,mol,list_drop,n_tuples,time,e_vec,e_inc,e_ref,conv,error,mem,local):
   #
   drop_string = []
   #
   if (exp[0] == 1):
      u_limit = nocc[0]-core[0]
   elif (exp[0] == 2):
      u_limit = nvirt[0]
   #
   if (exp[0] == 3):
      u_limit_1 = nocc[0]-core[0]
      u_limit_2 = nvirt[0]
      #
      n_tuples_2 = []
      drop_string_2 = []
      tmp = []
      e_vec_2 = []
      e_inc_2 = []
      time_2 = []
      for _ in range(0,nvirt[0]):
         e_vec_2.append(0.0)
         time_2.append(0.0)
      #
      conv_2 = []
      conv_2.append(False)
   #
   if (exp[0] <= 2):
      #
      for k in range(1,u_limit+1):
         #
         start = timer()
         #
         drop_string[:] = []
         #
         if (exp[0] == 1):
            generate_drop_occ(core[0]+1,1,k,nocc,core,list_drop,drop_string,n_tuples,exp[0])
         elif (exp[0] == 2):
            generate_drop_virt(nocc[0]+1,1,k,nocc,nvirt,list_drop,drop_string,n_tuples,exp[0])
         #
         for i in range(0,n_tuples[k-1]):
            #
            run_calc_corr(k,mult,False,model,basis,regex,mol,drop_string[i],e_vec,e_ref,error,False,mem,local)
            #
            if (error[0]):
               return n_tuples, time, e_inc, error
         #
         inc_corr_order(k,u_limit,e_vec,e_inc)
         #
         if (k > 1):
            if (exp[0] == 1):
               inc_corr_chk_conv(k,thres[0],e_inc,conv)
            elif (exp[0] == 2):
               inc_corr_chk_conv(k,thres[1],e_inc,conv)
         #
         time[k-1] = timer() - start
         #
         if (k == 1):
            print(' STATUS:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit,time[k-1]))
         else:
            print(' STATUS:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}  ---  conv =  {4:}'.format(k,u_limit,time[k-1],e_inc[k-1]-e_inc[k-2],conv[0]))
         #
         if (conv[0]):
            return n_tuples, time, e_inc, error
   #
   elif (exp[0] == 3):
      #
      for k in range(1,u_limit_1+1):
         #
         start = timer()
         #
         drop_string[:] = []
         #
         generate_drop_occ(core[0]+1,1,k,nocc,core,list_drop,drop_string,n_tuples,exp[0])
         #
         for j in range(0,n_tuples[k-1]):
            #
            start_2 = timer()
            #
            n_tuples_2[:] = []
            #
            e_inc_2[:] = []
            #
            for a in range(0,nvirt[0]):
               e_vec_2[a] = 0.0
            #
            conv_2[0] = False
            #
            for l in range(1,u_limit_2+1):
               #
               for a in range(nocc[0],nocc[0]+nvirt[0]):
                  list_drop[a] = a+1
               #
               drop_string_2[:] = []
               #
               generate_drop_virt(nocc[0]+1,1,l,nocc,nvirt,list_drop,drop_string_2,n_tuples_2,exp[0])
               #
               for i in range(0,n_tuples_2[l-1]):
                  #
                  run_calc_corr(l,mult,False,model,basis,regex,mol,drop_string[j]+drop_string_2[i],e_vec_2,e_ref,error,False,mem,local)
                  #
                  if (error[0]):
                     return n_tuples, time, e_inc, error
               #
               inc_corr_order(l,u_limit_2,e_vec_2,e_inc_2)
               #
               if (l > 1):
                  inc_corr_chk_conv(l,thres[1],e_inc_2,conv_2)
               #
               nv_order = l
               #
               if (conv_2[0]):
                  e_vec[k-1] += e_inc_2[l-1]
                  break
            #
            time_2[j] = timer() - start_2
            #
            print('      STATUS-MICRO:  tuple = {0:4d} / {1:4d}  (order = {2:4d} / {3:4d})  done in {4:10.2e} seconds  ---  diff =  {5:9.4e}  ---  conv =  {6:}'.format(j+1,n_tuples[k-1],nv_order,nvirt[0],time_2[j],e_inc_2[l-1]-e_inc_2[l-2],conv_2[0]))
         #
         inc_corr_order(k,u_limit_1,e_vec,e_inc)
         #
         if (k > 1):
            inc_corr_chk_conv(k,thres[0],e_inc,conv)
         #
         time[k-1] = timer() - start
         #
         if (k == 1):
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit_1,time[k-1]))
         else:
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}  ---  conv =  {4:}'.format(k,u_limit_1,time[k-1],e_inc[k-1]-e_inc[k-2],conv[0]))
         #
         if (conv[0]):
            return n_tuples, time, e_inc, error
   #
   return n_tuples, time, e_inc, error

def inc_corr_tuple_order(mol_string,nocc,nvirt,core,order,mult,fc,exp,model,basis,regex,mol,list_drop,n_tuples,time,e_vec,e_inc,e_ref,conv,error,mem,local):
   #
   drop_string = []
   #
   for i in range(0,order):
      n_tuples.append(0)
   #
   if (exp[0] == 1):
      u_limit = nocc[0]-core[0]
   elif (exp[0] == 2):
      u_limit = nvirt[0]
   #
   for k in range(order,0,-1):
      #
      start = timer()
      #
      if (exp[0] == 1):
         generate_drop_occ(core[0]+1,1,k,nocc,core,list_drop,drop_string,n_tuples,exp[0])
      elif (exp[0] == 2):
         generate_drop_virt(nocc[0]+1,1,k,nocc,nvirt,list_drop,drop_string,n_tuples,exp[0])
      #
      for i in range(0,n_tuples[k-1]):
         #
         run_calc_corr(k,mult,False,model,basis,regex,mol,drop_string[i],e_vec,e_ref,error,False,mem,local)
         #
         if (error[0]):
            return n_tuples, time, e_inc, error
      #
      time[k] = timer() - start
      #
      print(' STATUS:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit,time[k]))
   #
   for k in range(1,order+1):
      #
      inc_corr_order(k,u_limit,e_vec,e_inc)
   #
   time += [time.pop(0)] # permute all elements one time to the left in the list
   #
   return n_tuples, time, e_inc, error

def generate_drop_occ(start,order,final,nocc,core,list_drop,drop_string,n_tuples,exp):
   #
   if (order > (nocc[0]-core[0])):
      return drop_string, n_tuples
   else:
      for i in range(start,nocc[0]+1):
         list_drop[i-1] = 0
         s = ''
         inc = 0
         #
         if (order == final):
            if (core[0] > 0):
               for m in range(0,core[0]):
                  if (inc == 0):
                     s='DROP_MO='+str(list_drop[m])
                  else:
                     s+='-'+str(list_drop[m])
                  inc += 1
            for m in range(core[0],nocc[0]):
               if (list_drop[m] != 0):
                  if (inc == 0):
                     s='DROP_MO='+str(list_drop[m])
                  else:
                     s+='-'+str(list_drop[m])
                  inc += 1
            #
            if (len(n_tuples) >= order):
               n_tuples[order-1] += 1
            else:
               n_tuples.append(1)
            #
            if (exp == 1):
               drop_string.append(s+'\n')
            elif (exp == 3):
               drop_string.append(s)
         #
         generate_drop_occ(i+1,order+1,final,nocc,core,list_drop,drop_string,n_tuples,exp)
         #
         list_drop[i-1] = i
   #
   return drop_string, n_tuples

def generate_drop_virt(start,order,final,nocc,nvirt,list_drop,drop_string,n_tuples,exp):
   #
   if (order > nvirt[0]):
      return drop_string, n_tuples
   else:
      for i in range(start,(nocc[0]+nvirt[0])+1):
         list_drop[i-1] = 0
         s = ''
         inc = 0
         if (exp == 3):
            inc += 1
         #
         if (order == final):
            for m in range(nocc[0],nocc[0]+nvirt[0]):
               if (list_drop[m] != 0):
                  if (inc == 0):
                     s='DROP_MO='+str(list_drop[m])
                  else:
                     s+='-'+str(list_drop[m])
                  inc += 1
            #
            if (len(n_tuples) >= order):
               n_tuples[order-1] += 1
            else:
               n_tuples.append(1)
            #
            drop_string.append(s+'\n')
         #
         generate_drop_virt(i+1,order+1,final,nocc,nvirt,list_drop,drop_string,n_tuples,exp)
         #
         list_drop[i-1] = i
   #
   return drop_string, n_tuples

def inc_corr_order(k,n,e_vec,e_inc):
   #
   e_sum = 0.0
   #
   for m in range(1,k+1):
      e_sum += (-1)**(m) * (1.0 / math.factorial(m)) * prefactor(n,k,m-1) * e_vec[(k-m)-1]
   e_inc.append(e_vec[k-1]+e_sum)
   #
   return e_inc

def prefactor(n,order,m):
   #
   pre = 1
   #
   for i in range(m,-1,-1):
      pre = pre * (n - order + i)
   #
   return pre

def inc_corr_chk_conv(order,thres,e_inc,conv):
   #
   e_diff = e_inc[order-1] - e_inc[order-2]
   #
   if (abs(e_diff) < thres):
      conv[0] = True
   else:
      conv[0] = False
   #
   return conv

def inc_corr_summary(nocc,nvirt,core,exp,thres,order,n_tuples,time,e_inc,e_ref,conv,ref,error,local):
   print('\n')
   print(' ** RESULTS **\n')
   if (exp[0] == 1):
      print('   OCCUPIED expansion')
   elif (exp[0] == 2):
      print('   VIRTUAL expansion')
   elif (exp[0] == 3):
      print('   Combined OCCUPIED/VIRTUAL expansion')
   print('   frozen core        =  {0:}'.format(core[0] > 0))
   print('   local orbitals     =  {0:}'.format(local))
   print('   occupied orbitals  =  {0:}'.format(nocc[0]-core[0]))
   print('   virtual orbitals   =  {0:}'.format(nvirt[0]))
   if (thres[0] > 0.0):
      print('   thres. (occ.)      =  {0:6.1e}'.format(thres[0]))
   if (thres[1] > 0.0):
      print('   thres. (virt.)     =  {0:6.1e}'.format(thres[1]))
   if ((thres[0] == 0.0) and (thres[1] == 0.0)):
      print('   conv. thres.       =  NONE')
   print('   inc.-corr. order   =  {0:}'.format(len(e_inc)))
   if ((thres[0] > 0.0) or (thres[1] > 0.0)):
      print('   convergence met    =  {0:}'.format(conv[0]))
   print('   error in calc.     =  {0:}'.format(error[0]))
   print('')
   for i in range(0,len(e_inc)):
      print('{0:4d} - # orb. tuples  =  {1:}'.format(i+1,n_tuples[i]))
   print('   --------------------------------------------------------------')
   total_time = 0.0
   for i in range(0,len(e_inc)):
      total_time += time[i]
      print('{0:4d} - E (inc-corr)   = {1:13.9f}  done in {2:10.2e} seconds'.format(i+1,e_inc[i],total_time))
   print('   --------------------------------------------------------------')
   if (len(e_inc) >= 2):
      print('   final convergence  =  {0:9.4e}'.format(e_inc[-1]-e_inc[-2]))
   if (ref[0] and (not error[0])):
      print('   --------------------------------------------------------------')
      if ((exp[0] == 1) or (exp[0] == 3)):
         print('{0:4d} - E (ref)        = {1:13.9f}  done in {2:10.2e} seconds'.format(nocc[0]-core[0],e_ref[0],time[-1]))
      elif (exp[0] == 2):
         print('{0:4d} - E (ref)        = {1:13.9f}  done in {2:10.2e} seconds'.format(nvirt[0],e_ref[0],time[-1]))
      print('   --------------------------------------------------------------')
      print('   final difference   =  {0:9.4e}'.format(e_ref[0]-e_inc[-1]))
   print('\n')

def main():
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
   parser.add_argument('--order', help='inc.-corr. order (integer number)', required=False)
   parser.add_argument('--bond', help='bond length parameter for PES generation (real number)', required=False)
   parser.add_argument('--local', help='local orbitals logical ("True" or "False")', required=False)
   args = parser.parse_args()
   #
   model = args.model
   if (not ((model == 'CCSD') or (model == 'CISD') or (model == 'CCSDT') or (model == 'FCI'))):
      print 'wrong choice of model (CCSD, CISD, CCSDT, or FCI), aborting ...'
      sys.exit(10)
   if (model == 'FCI'):
      regex = '\s+Final Correlation Energy'
   else:
      regex = '\s+The correlation energy is'
   #
   exp_string = args.exp
   if (not ((exp_string == 'OCC') or (exp_string == 'VIRT') or (exp_string == 'COMB'))):
      print 'wrong choice of expansion type (OCC, VIRT, or COMB), aborting ...'
      sys.exit(10)
   exp = []
   if ((exp_string == 'OCC')):
      exp.append(1)
   elif ((exp_string == 'VIRT')):
      exp.append(2)
   elif ((exp_string == 'COMB')):
      exp.append(3)
   #
   basis = args.basis
   #
   mol_string = args.mol
   #
   fc_string = args.frozen
   fc = []
   if (fc_string == 'True'):
      fc.append(True)
   elif (fc_string == 'False'):
      fc.append(False)
   else:
      print 'wrong input argument for frozen core (True/False), aborting ...'
      sys.exit(10)   
   #
   ref_string = args.ref
   ref = []
   if (ref_string == 'True'):
      ref.append(True)
   elif (ref_string == 'False'):
      ref.append(False)
   else:
      print 'wrong input argument for reference calc (True/False), aborting ...'
      sys.exit(10)
   #
   mem = args.mem
   #
   scr_dir = args.scr
   wrk_dir = os.getcwd()
   #
   exp_ctrl = False
   thres = []
   if ((args.thres_occ is None) and (args.thres_virt is None) and (args.order is None)):
      print 'either the convergence threshold(s) (--thres_occ/--thres_virt) OR the inc.-corr. order (--order) must be set, aborting ...'
      sys.exit(10)
   elif (args.order is None):
      exp_ctrl = True
      order = 0
      if (args.thres_occ is None):
         thres.append(0.0)
         thres.append(float(args.thres_virt))
         if (exp[0] == 3):
            print('expansion scheme "COMB" requires both an occupied and a virtual expansion threshold, aborting ...')
            sys.exit(10)
      elif (args.thres_virt is None):
         thres.append(float(args.thres_occ))
         thres.append(0.0)
         if (exp[0] == 3):
            print('expansion scheme "COMB" requires both an occupied and a virtual expansion threshold, aborting ...')
            sys.exit(10)
      else:
         thres.append(float(args.thres_occ))
         thres.append(float(args.thres_virt))
   elif ((args.thres_occ is None) and (args.thres_virt is None)):
      order = int(args.order)
      thres.append(0.0)
      thres.append(0.0)
      if (exp[0] == 3):
         print('expansion scheme "COMB" is currently not implemented for fixed order expansion, aborting ...')
         sys.exit(10)
   #
   bond = []
   if (args.bond is None):
      bond.append(0.0)
   else:
      bond.append(float(args.bond))
   #
   local = []
   if (args.local is None):
      local.append(False)
   else:
      local.append(args.local)
   if (fc[0] and local[0]):
      print 'wrong input -- comb. of frozen core and local orbitals not implemented, aborting ...'
      sys.exit(10)
   #
   print('\n')
   print(' ** START INC.-CORR. ('+model+') CALCULATION **\n')
   #
   error = []
   error.append(False)
   # 
   mol = []
   core = []
   mult = []
   #
   init_zmat(mol_string,bond,mol,fc[0],core,mult)
   #
   mk_scr_dir(scr_dir)
   #
   cd_dir(scr_dir)
   #
   prepare_calc()
   #
   nocc = []
   nvirt = []
   #
   run_calc_hf(mult,basis,mol,nocc,nvirt,error,mem)
   #
   if (exp[0] == 1):
      if (order >= (nocc[0] - core[0])):
         print 'wrong input argument for total order (must be .lt. number of available occupied orbitals), aborting ...'
         #
         cd_dir(wrk_dir)
         #
         rm_scr_dir(scr_dir)
         #
         sys.exit(10)
   #
   elif (exp[0] == 2):
      if (order >= nvirt[0]):
         print 'wrong input argument for total order (must be .lt. number of virtual orbitals), aborting ...'
         #
         cd_dir(wrk_dir)
         #
         rm_scr_dir(scr_dir)
         #
         sys.exit(10)
   #
   n_tuples = []
   list_drop = []
   time = []
   e_vec = []
   e_inc = []
   e_ref = []
   conv = []
   conv.append(False)
   #
   if ((exp[0] == 1) or (exp[0] == 3)):
      for _ in range(0,nocc[0]):
         time.append(0.0)
         e_vec.append(0.0)
   elif (exp[0] == 2):
      for _ in range(0,nvirt[0]):
         time.append(0.0)
         e_vec.append(0.0)
   #
   for i in range(0,nocc[0]+nvirt[0]):
      list_drop.append(i+1)
   if (exp[0] == 1):
      for i in range(nocc[0],nocc[0]+nvirt[0]):
         list_drop[i] = 0
   elif (exp[0] == 2):
      for i in range(core[0]-1,nocc[0]):
         list_drop[i] = 0
   #
   if (exp_ctrl):
      inc_corr_tuple_thres(mol_string,nocc,nvirt,core,thres,mult,fc,exp,model,basis,regex,mol,list_drop,n_tuples,time,e_vec,e_inc,e_ref,conv,error,mem,local[0])
   else:
      inc_corr_tuple_order(mol_string,nocc,nvirt,core,order,mult,fc,exp,model,basis,regex,mol,list_drop,n_tuples,time,e_vec,e_inc,e_ref,conv,error,mem,local[0])
   #
   if (ref[0] and (not error[0])):
      start = timer()
      run_calc_corr(nocc,mult,fc[0],model,basis,regex,mol,'',e_vec,e_ref,error,True,mem,local[0])
      time.append(timer()-start)
   #
   cd_dir(wrk_dir)
   #
   if (error[0]):
      save_err_out(scr_dir)
   #
   rm_scr_dir(scr_dir)
   #
   inc_corr_summary(nocc,nvirt,core,exp,thres,order,n_tuples,time,e_inc,e_ref,conv,ref,error,local[0])
   #
   if (not error[0]):
      if ((exp[0] == 1) or (exp[0] == 3)):
         inc_corr_plot.ic_plot(mol_string,nocc,core,exp,thres,order,n_tuples,model,basis,e_inc,e_ref,(ref[0] and (not error[0])),local[0])
      elif (exp[0] == 2):
         inc_corr_plot.ic_plot(mol_string,nvirt,core,exp,thres,order,n_tuples,model,basis,e_inc,e_ref,(ref[0] and (not error[0])),local[0])
   #
   print(' ** END OF INC.-CORR. ('+model+') CALCULATION **\n')
   print('\n')

if __name__ == '__main__':
   #
   main()

