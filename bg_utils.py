#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_utils.py: general utilities for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI
from itertools import combinations, chain
from scipy.misc import comb
from os import listdir, unlink, chdir
from os.path import join, isfile
from subprocess import call
from shutil import rmtree, copy, move
from glob import glob

from bg_mpi_wrapper import abort_rout
from bg_mpi_utils import remove_slave_env
from bg_print import print_ref_header, print_ref_end

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def run_calc_hf(molecule):
   #
   molecule['input_hf'](molecule)
   #
   with open('OUTPUT_'+str(molecule['mpi_rank'])+'.OUT','w') as output: call(molecule['backend_prog_exe'],stdout=output,stderr=output)
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
   with open('OUTPUT_'+str(molecule['mpi_rank'])+'.OUT','w') as output: call(molecule['backend_prog_exe'],stdout=output,stderr=output)
   #
   molecule['write_energy'](molecule,level)
   #
   if (not molecule['error'][-1]): rm_dir_content(molecule)
   #
   return molecule

def rm_dir_content(molecule):
   #
   for the_file in listdir(molecule['scr_dir']):
      #
      file_path = join(molecule['scr_dir'],the_file)
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

def term_calc(molecule,final=False):
   #
   chdir(molecule['wrk_dir'])
   #
   if (final):
      #
      rmtree(molecule['scr_dir'],ignore_errors=True)
      #
      rmtree(molecule['rst_dir'],ignore_errors=True)
      #
      if (molecule['mpi_parallel']): remove_slave_env(molecule)
   #
   if (molecule['error'][-1]): abort_rout(molecule)
   #
   return

def ref_calc(molecule):
   #
   print_ref_header(molecule)
   #
   start_time = MPI.Wtime()
   #
   run_calc_corr(molecule,'','REF')
   #
   molecule['ref_time'] = MPI.Wtime()-start_time
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

def comb_index(n,k):
   #
   count = comb(n,k,exact=True)
   #
   index = np.fromiter(chain.from_iterable(combinations(range(n),k)),int,count=count*k)
   #
   return index.reshape(-1,k)



