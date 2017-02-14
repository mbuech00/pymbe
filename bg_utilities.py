#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_utilities.py: general utilities for Bethe-Goldstone correlation calculation."""

from os import mkdir, chdir, listdir, unlink
from os.path import join, isfile
from subprocess import call
from shutil import copy, rmtree 
from timeit import default_timer

import inc_corr_info
import inc_corr_mpi
from bg_print import redirect_stdout, print_ref_header, print_ref_end

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.3'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def init_calc(molecule):
   #
   # redirect stdout to output.out
   #
   redirect_stdout(molecule)
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
   mkdir(molecule['scr'])
   #
   chdir(molecule['scr']) 
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
   chdir(molecule['wrk'])
   #
   if (molecule['error'][0][-1]):
      #
      copy(molecule['scr']+'/OUTPUT.OUT',molecule['wrk']+'/OUTPUT.OUT')
   #
   rmtree(molecule['scr'],ignore_errors=True)
   #
   if (molecule['mpi_master'] and molecule['mpi_parallel']):
      #
      inc_corr_mpi.remove_slave_env(molecule)
   #
   return

def run_calc_hf(molecule):
   #
   molecule['input_hf'](molecule)
   #
   call(molecule['backend_prog_exe']+' &> OUTPUT.OUT',shell=True) 
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
   call(molecule['backend_prog_exe']+' &> OUTPUT.OUT',shell=True)
   #
   molecule['write_energy'](molecule,level)
   #
   if (not molecule['error'][0][-1]): rm_dir_content(molecule)
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
   start = default_timer()
   #
   run_calc_corr(molecule,'','REF')
   #
   molecule['prim_time'][0].append(default_timer()-start)
   #
   print_ref_end(molecule)
   #
   return molecule

