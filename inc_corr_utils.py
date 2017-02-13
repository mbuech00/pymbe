# -*- coding: utf-8 -*
#!/usr/bin/env python

#
# general utilities for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import os
import subprocess
import shutil
from timeit import default_timer as timer

import inc_corr_info
import inc_corr_mpi
import inc_corr_print

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def init_calc(molecule):
   #
   # redirect stdout to output.out
   #
   inc_corr_print.redirect_stdout(molecule)
   #
   # init error list
   #
   molecule['error'] = [[False]]
   #
   # init molecular info
   #
   inc_corr_info.init_mol(molecule)
   #
   # init expansion parameters et al.
   #
   inc_corr_info.init_param(molecule)
   #
   # init backend program
   #
   inc_corr_info.init_backend_prog(molecule)
   #
   # if mpi parallel run, bcast the molecular dictionary
   #
   if (molecule['mpi_parallel']):
      #
      inc_corr_mpi.bcast_mol_dict(molecule)
      #
      inc_corr_mpi.init_slave_env(molecule)
      #
      inc_corr_mpi.print_mpi_table(molecule)
   #
   else:
      #
      molecule['scr'] = molecule['wrk']+'/'+molecule['scr_name']
   #
   # init scr env
   #
   os.mkdir(molecule['scr'])
   #
   os.chdir(molecule['scr']) 
   #
   # run hf calc to determine dimensions
   #
   run_calc_hf(molecule)
   #
   # perform a few sanity checks
   #
   inc_corr_info.sanity_chk(molecule)
   #
   return molecule

def term_calc(molecule):
   #
   os.chdir(molecule['wrk'])
   #
   if (molecule['error'][0][-1]):
      #
      shutil.copy(molecule['scr']+'/OUTPUT.OUT',molecule['wrk']+'/OUTPUT.OUT')
   #
   shutil.rmtree(molecule['scr'],ignore_errors=True)
   #
   if (molecule['mpi_master']):
      #
      inc_corr_mpi.remove_slave_env(molecule)
   #
   return

def run_calc_hf(molecule):
   #
   molecule['input_hf'](molecule)
   #
   subprocess.call(molecule['backend_prog_exe']+' &> OUTPUT.OUT',shell=True) 
   #
   molecule['get_dim'](molecule)
   #
   if (not molecule['error'][0][-1]): rm_dir_content(molecule)
   #
   return molecule

def run_calc_corr(molecule,drop_string,level):
   #
   molecule['input_corr'](molecule,drop_string,level)
   #
   subprocess.call(molecule['backend_prog_exe']+' &> OUTPUT.OUT',shell=True)
   #
   molecule['write_energy'](molecule,level)
   #
   if (not molecule['error'][0][-1]): rm_dir_content(molecule)
   #
   return molecule

def rm_dir_content(molecule):
   #
   for the_file in os.listdir(molecule['scr']):
      #
      file_path = os.path.join(molecule['scr'],the_file)
      #
      try:
         #
         if os.path.isfile(file_path):
            #
            os.unlink(file_path)
      #
      except Exception as e:
         #
         print(e)
   #
   return

def ref_calc(molecule):
   #
   inc_corr_print.print_ref_header()
   #
   start = timer()
   #
   run_calc_corr(molecule,'','REF')
   #
   molecule['prim_time'][0].append(timer()-start)
   #
   inc_corr_print.print_ref_end(molecule)
   #
   return molecule

