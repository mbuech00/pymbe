#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_utils.py: general utilities for Bethe-Goldstone correlation calculation."""

import numpy as np
from itertools import combinations, chain
from scipy.misc import comb
from math import factorial
from os import listdir, unlink
from os.path import join, isfile
from subprocess import call

from bg_time import timer_phase
from bg_print import print_ref_header, print_ref_end

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def run_calc_hf(molecule):
   #
   molecule['input_hf'](molecule)
   #
   call(molecule['backend_prog_exe']+' &> OUTPUT.OUT',shell=True) 
   #
   molecule['get_dim'](molecule)
   #
   if (not molecule['error'][-1]): rm_dir_content(molecule)
   #
   return molecule

def run_calc_corr(molecule,drop_string,level):
   #
   molecule['input_corr'](molecule,drop_string,level)
   #
   call(molecule['backend_prog_exe']+' &> OUTPUT.OUT',shell=True)
   #
   molecule['write_energy'](molecule,level)
   #
   if (not molecule['error'][-1]): rm_dir_content(molecule)
   #
   return molecule

def rm_dir_content(molecule):
   #
   for the_file in listdir(molecule['scr']):
      #
      file_path = join(molecule['scr'],the_file)
      #
      try:
         #
         if isfile(file_path):
            #
            unlink(file_path)
      #
      except Exception as e:
         #
         print(e)
   #
   return

def ref_calc(molecule):
   #
   print_ref_header()
   #
   timer_phase(molecule,'time_tot',1,'REF')
   #
   run_calc_corr(molecule,'','REF')
   #
   timer_phase(molecule,'time_tot',1,'REF')
   #
   print_ref_end(molecule)
   #
   return molecule

def orb_string(molecule,l_limit,u_limit,tup,string):
   #
   # generate list with all occ/virt orbitals
   #
   dim = range(l_limit+1,(l_limit+u_limit)+1)
   #
   # generate list with all orbs the should be dropped (not part of the current tuple)
   #
   drop = sorted(list(set(dim)-set(tup.tolist())))
   #
   # for virt scheme, explicitly drop the core orbitals for frozen core
   #
   if ((molecule['exp'] == 'virt') and molecule['frozen']):
      #
      for i in range(molecule['ncore'],0,-1):
         #
         drop.insert(0,i)
   #
   # now write the string
   #
   inc = 0
   #
   string['drop'] = ''
   #
   for i in range(0,len(drop)):
      #
      if (inc == 0):
         #
         string['drop'] += 'DROP_MO='+str(drop[i])
      #
      else:
         #
         if (drop[i] == (drop[i-1]+1)):
            #
            if (i < (len(drop)-1)):
               #
               if (drop[i] != (drop[i+1]-1)):
                  #
                  string['drop'] += '>'+str(drop[i])
            #
            else:
               #
               string['drop'] += '>'+str(drop[i])
         #
         else:
            #
            string['drop'] += '-'+str(drop[i])
      #
      inc += 1
   #
   if (string['drop'] != ''):
      #
      string['drop'] += '\n'
   #
   return string

def theo_work(molecule):
   #
   molecule['theo_work'] = []
   #
   dim = molecule['u_limit']
   #
   for k in range(0,dim):
      #
      molecule['theo_work'].append(int(factorial(dim)/(factorial(k+1)*factorial(dim-(k+1)))))
   #
   return molecule

def comb_index(n,k):
   #
   count = comb(n,k,exact=True)
   #
   index = np.fromiter(chain.from_iterable(combinations(range(n),k)),int,count=count*k)
   #
   return index.reshape(-1,k)



