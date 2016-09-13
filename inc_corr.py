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

def prepare_calc():
   #
   command='cp '+CFOUR_BASIS+' .'
   os.system(command)
   command='cp '+CFOUR_BIN+'/x* .'
   os.system(command)

def end_calc():
   #
   command='rm *'
   os.system(command)

def run_calc(order,mult,fc,model,basis,regex,mol,drop_string,e_vec,e_ref,error,ref,mem):
   # 
   write_zmat(mult,fc,model,basis,mol,drop_string,mem)
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

def write_zmat(mult,fc,model,basis,mol,drop_string,mem):
   #
   out=open('ZMAT','w')
   #
   out.write(mol[0])
   #
   if (model == 'FCI'):
      out.write('*CFOUR(CALC=FULLCI\n')
      out.write('CAS_MMAX=10\n')
   else:
      out.write('*CFOUR(CALC='+model+'\n')
      out.write('CC_PROG=VCC\n')
   #
   if (drop_string != '\n'):
      out.write(drop_string)
   #
   out.write('SCF_CONV=9\n')
   out.write('LINEQ_CONV=9\n')
   out.write('CC_CONV=9\n')
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

def init_zmat(mol_string,mol,nocc,frozen,core,mult):
   #
   if (mol_string == 'water'):
      s = 'ZMAT file for H2O\n'
      s += 'H\n'
      s += 'O 1 ROH\n'
      s += 'H 2 ROH 1 AHOH\n'
      s += '\n'
      s += 'ROH = 0.957\n'
      s += 'AHOH = 104.2\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      nocc.append(5)
      if (frozen):
         core.append(1)
      else:
         core.append(0)
   elif (mol_string == 'acetylene'):
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
      s += 'RCC = 1.2\n'
      s += 'A90 = 90.\n'
      s += 'A180 = 180.\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      nocc.append(7)
      if (frozen):
         core.append(2)
      else:
         core.append(0)
   elif (mol_string == 'nitrogen'):
      s = 'ZMAT file for nitrogen (N2)\n'
      s += 'N\n'
      s += 'N 1 RNN\n'
      s += '\n'
      s += 'RNN = 1.098\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      nocc.append(7)
      if (frozen):
         core.append(2)
      else:
         core.append(0)
   elif (mol_string == 'oxygen'):
      s = 'ZMAT file for (triplet) oxygen (O2)\n'
      s += 'O\n'
      s += 'O 1 ROO\n'
      s += '\n'
      s += 'ROO = 1.205771156354447\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(3)
      nocc.append(7)
      if (frozen):
         core.append(2)
      else:
         core.append(0)
   elif (mol_string == 'silane'):
      s = 'ZMAT file for silane\n'
      s += 'Si\n'
      s += 'H 1 R\n'
      s += 'H 1 R 2 TDA\n'
      s += 'H 1 R 2 TDA 3 D120\n'
      s += 'H 1 R 2 TDA 4 D120\n'
      s += '\n'
      s += 'R=1.48598655\n'
      s += 'TDA=109.471221\n'
      s += 'D120=120.\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      nocc.append(9)
      if (frozen):
         core.append(5)
      else:
         core.append(0)
   elif (mol_string == 'carbondioxide'):
      s = 'ZMAT file for carbon dioxide\n'
      s += 'C\n'
      s += 'X 1 RCX\n'
      s += 'O 1 RCO 2 A90\n'
      s += 'O 1 RCO 2 A90 3 A180\n'
      s += '\n'
      s += 'RCX=1.0\n'
      s += 'RCO=1.16\n'
      s += 'A90=90.0\n'
      s += 'A180=180.0\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      nocc.append(11)
      if (frozen):
         core.append(3)
      else:
         core.append(0)
   elif (mol_string == 'butadiene'):
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
      s += 'CDC=1.34054111\n' 
      s += 'CSC=1.45747113\n'
      s += 'CH1=1.08576026\n'
      s += 'CH2=1.08793795\n'
      s += 'CH3=1.09045613\n'
      s += 'CCC=124.31048212\n' 
      s += 'CC1=121.82705922\n'
      s += 'CC2=121.53880082\n'
      s += 'CC3=119.43908311\n'
      s += 'A0 = 0.\n'
      s += 'A180 = 180.\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      nocc.append(15)
      if (frozen):
         core.append(4)
      else:
         core.append(0)
   elif (mol_string == 'benzene'):
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
      s += 'A60 = 60.\n'
      s += 'D180 = 180.\n'
      s += 'RCC = 1.3914\n'
      s += 'RXH = 2.4716\n'
      s += '\n'
      mol.append(s)
      #
      mult.append(1)
      nocc.append(21)
      if (frozen):
         core.append(6)
      else:
         core.append(0) 
   else:
      print('molecular input not recognized, aborting ...')
      sys.exit(10)
   #
   return mol, nocc, core, mult

def write_energy(order,model,regex,e_vec,e_ref,error,ref):
   #
   inp=open('CFOUR.OUT','r')
   #
   regex_2 = '\s+ERROR ERROR'
   #
   while 1:
      line=inp.readline()
      if re.match(regex,line) is not None:
         if (model == 'FCI'):
            [energy]=line.split()[3:4]
         else:
            [energy]=line.split()[4:5]
         if (ref):
            e_ref.append(float(energy))
         else:
            e_vec[order-1] += float(energy)
         break
      elif re.match(regex_2,line) is not None:
         print('problem with '+model+' calculation, aborting ...')
         error[0] = True
   #
   inp.close()
   #
   return e_ref, e_vec

def inc_corr_tuple_thres(mol_string,nocc,core,thres,mult,fc,model,basis,regex,mol,list_drop,n_tuples,time,e_vec,e_inc,e_ref,conv,error,mem):
   #
   drop_string = [[]]
   #
   print(' STATUS:  start preparing DROP_MO list...')
   #
   for k in range(0,(nocc[0]-core[0])-1):
      #
      drop_string.append([])
   #
   generate_drop(core[0]+1,1,nocc,core,list_drop,drop_string,n_tuples)
   #
   print(' STATUS:  done preparing DROP_MO list\n')
   #
   for k in range(0,nocc[0]-core[0]):
      #
      start = timer()
      #
      for i in range(0,len(drop_string[k])):
         #
         run_calc(k+1,mult,False,model,basis,regex,mol,drop_string[k][i],e_vec,e_ref,error,False,mem)
         #
         if (error[0]):
            return n_tuples, time, e_inc, error
      #
      inc_corr_order(k+1,nocc[0]-core[0],e_vec,e_inc)
      #
      if (k > 0):
         inc_corr_chk_conv(k,thres,e_inc,conv)
      #
      time[k] = timer() - start
      #
      if (k == 0):
         print(' STATUS:  order = {0:4d} / nocc = {1:4d}  done in {2:10.2e} seconds'.format(k+1,nocc[0]-core[0],time[k]))
      else:
         print(' STATUS:  order = {0:4d} / nocc = {1:4d}  done in {2:10.2e} seconds  ---  convergence =  {3:}'.format(k+1,nocc[0]-core[0],time[k],conv[0]))
      #
      if (conv[0]):
         return n_tuples, time, e_inc, error
   #
   return n_tuples, time, e_inc, error

def inc_corr_tuple_order(mol_string,nocc,core,order,mult,fc,model,basis,regex,mol,list_drop,n_tuples,time,e_vec,e_inc,e_ref,conv,error,mem):
   #
   drop_string = [[]]
   #
   print(' STATUS:  start preparing DROP_MO list...')
   #
   for k in range(0,(nocc[0]-core[0])-1):
      #
      drop_string.append([])
   #
   generate_drop(core[0]+1,1,nocc,core,list_drop,drop_string,n_tuples)
   #
   print(' STATUS:  done preparing DROP_MO list\n')
   #
   for k in range(0,nocc[0]-core[0]):
      #
      start = timer()
      #
      for i in range(0,len(drop_string[k])):
         #
         run_calc(k+1,mult,False,model,basis,regex,mol,drop_string[k][i],e_vec,e_ref,error,False,mem)
         #
         if (error[0]):
            return n_tuples, time, e_inc, error
      #
      inc_corr_order(k+1,nocc[0]-core[0],e_vec,e_inc)
      #
      time[k] = timer() - start
      #
      if (k == 0):
         print(' STATUS:  order = {0:4d} / nocc = {1:4d}  done in {2:10.2e} seconds'.format(k+1,nocc[0]-core[0],time[k]))
      else:
         print(' STATUS:  order = {0:4d} / nocc = {1:4d}  done in {2:10.2e} seconds  ---  energy diff =  {3:9.4e}'.format(k+1,nocc[0]-core[0],time[k],e_inc[k]-e_inc[k-1]))
      #
      if (k == order-1):
         return n_tuples, time, e_inc, error
   #
   return n_tuples, time, e_inc, error

def generate_drop(start,order,nocc,core,list_drop,drop_string,n_tuples):
   #
   if (order > (nocc[0]-core[0])):
      return drop_string, n_tuples
   else:
      for i in range(start,nocc[0]+1):
         list_drop[i-1] = 0
         s = ''
         inc = 0
         #
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
         drop_string[order-1].append(s+'\n')
         #
         n_tuples[order-1] += 1
         #
         generate_drop(i+1,order+1,nocc,core,list_drop,drop_string,n_tuples)
         #
         list_drop[i-1] = i
   #
   return drop_string, n_tuples

def inc_corr_order(order,n,e_vec,e_inc):
   #
   e_sum = 0.0
   #
   for m in range(1,order+1):
      e_sum += (-1)**(m) * (1.0 / math.factorial(m)) * prefactor(n,order,m-1) * e_vec[(order-m)-1]
   e_inc.append(e_vec[order-1]+e_sum)
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
   e_diff = e_inc[order] - e_inc[order-1]
   #
   if (abs(e_diff) < thres):
      conv[0] = True
   else:
      conv[0] = False
   #
   return conv

def inc_corr_summary(nocc,core,thres,order,n_tuples,time,e_inc,e_ref,conv,ref,error):
   print('\n')
   print(' ** RESULTS **\n')
   print('   frozen core        =  {0:}'.format(core[0] > 0))
   print('   occupied orbitals  =  {0:}'.format(nocc[0]-core[0]))
   if (thres > 0.0):
      print('   conv. thres.       =  {0:6.1e}'.format(thres))
   else:
      print('   conv. thres.       =  NONE')
   print('   inc.-corr. order   =  {0:}'.format(len(e_inc)))
   if (thres > 0.0):
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
   print('   final convergence  =  {0:9.4e}'.format(e_inc[-1]-e_inc[-2]))
   if (ref[0] and (not error[0])):
      print('   --------------------------------------------------------------')
      print('{0:4d} - E (ref)        = {1:13.9f}  done in {2:10.2e} seconds'.format(nocc[0]-core[0],e_ref[0],time[-1]))
      print('   --------------------------------------------------------------')
      print('   final difference   =  {0:9.4e}'.format(e_ref[0]-e_inc[-1]))
   print('\n')

def main():
   #
   parser = argparse.ArgumentParser(description='This is an CCSD/CISD/CCSDT/FCI inc.-corr. Python script (with CFOUR backend) written by Dr. Janus Juul Eriksen, JGU Mainz, September 2016')
   parser.add_argument('--model', help='electronic structure model (CCSD, CISD, CCSDT, or FCI)',required=True)
   parser.add_argument('--basis', help='one-electron basis set', required=True)
   parser.add_argument('--mol', help='molecule (H2O, C2H2, N2, O2, SiH4, CO2, C4H6, or C6H6)', required=True)
   parser.add_argument('--frozen', help='frozen-core logical', required=True)
   parser.add_argument('--ref', help='reference calc. logical', required=True)
   parser.add_argument('--mem', help='amount of virtual memory', required=True)
   parser.add_argument('--scr', help='location of scratch folder', required=True)
   parser.add_argument('--thres', help='convergence threshold', required=False)
   parser.add_argument('--order', help='inc.-corr. order', required=False)
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
   wrk_dir=os.getcwd()
   #
   exp_ctrl = False
   if ((args.thres is None) and (args.order is None)):
      print 'either the convergence threshold (--thres) OR the inc.-corr. order (--order) must be set, aborting ...'
      sys.exit(10)
   elif (args.order is None):
      exp_ctrl = True
      thres = float(args.thres)
      order = 0
   elif (args.thres is None):
      order = int(args.order)
      thres = 0.0
   #
   print('\n')
   print(' ** START INC.-CORR. ('+model+') CALCULATION **\n')
   # 
   mol = []
   nocc = []
   core = []
   mult = []
   init_zmat(mol_string,mol,nocc,fc[0],core,mult)
   #
   mk_scr_dir(scr_dir)
   #
   cd_dir(scr_dir)
   #
   prepare_calc()
   #
   n_tuples = []
   time = []
   e_vec = []
   e_inc = []
   e_ref = []
   error = []
   error.append(False)
   conv = []
   conv.append(False)
   #
   for i in range(0,nocc[0]-core[0]):
      n_tuples.append(0)
      time.append(0.0)
      e_vec.append(0.0)
   #
   list_drop = []
   for i in range(0,nocc[0]):
      list_drop.append(i+1)
   #
   if (exp_ctrl):
      inc_corr_tuple_thres(mol_string,nocc,core,thres,mult,fc,model,basis,regex,mol,list_drop,n_tuples,time,e_vec,e_inc,e_ref,conv,error,mem)
   else:
      inc_corr_tuple_order(mol_string,nocc,core,order,mult,fc,model,basis,regex,mol,list_drop,n_tuples,time,e_vec,e_inc,e_ref,conv,error,mem)
   #
   if (ref[0] and (not error[0])):
      start = timer()
      run_calc(nocc,mult,fc[0],model,basis,regex,mol,'',e_vec,e_ref,error,True,mem)
      time.append(timer()-start)
   #
   end_calc()
   #
   cd_dir(wrk_dir)
   #
   rm_scr_dir(scr_dir)
   #
   inc_corr_summary(nocc,core,thres,order,n_tuples,time,e_inc,e_ref,conv,ref,error)
   #
   inc_corr_plot.ic_plot(mol_string,nocc,core,thres,order,n_tuples,model,basis,e_inc,e_ref,(ref[0] and (not error[0])))
   #
   print(' ** END OF INC.-CORR. ('+model+') CALCULATION **\n')
   print('\n')

if __name__ == '__main__':
   #
   main()

