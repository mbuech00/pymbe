#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_setup.py: setup utilities for Bethe-Goldstone correlation calculation."""

from os import mkdir, chdir
from shutil import copy, rmtree 

from bg_info import init_mol, init_param, init_backend_prog, sanity_chk
from bg_utilities import run_calc_hf
from bg_mpi_kernels import bcast_mol_dict, init_slave_env, print_mpi_table, remove_slave_env 
from bg_print import redirect_stdout

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
   molecule['error'] = [False]
   #
   # init molecular info
   #
   init_mol(molecule)
   #
   # init expansion parameters et al.
   #
   init_param(molecule)
   #
   # init backend program
   #
   init_backend_prog(molecule)
   #
   # if mpi parallel run, bcast the molecular dictionary
   #
   if (molecule['mpi_parallel']):
      #
      bcast_mol_dict(molecule)
      #
      init_slave_env(molecule)
      #
      print_mpi_table(molecule)
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
   sanity_chk(molecule)
   #
   return molecule

def term_calc(molecule):
   #
   chdir(molecule['wrk'])
   #
   if (molecule['error'][-1]):
      #
      copy(molecule['scr']+'/OUTPUT.OUT',molecule['wrk']+'/OUTPUT.OUT')
   #
   rmtree(molecule['scr'],ignore_errors=True)
   #
   if (molecule['mpi_master'] and molecule['mpi_parallel']):
      #
      remove_slave_env(molecule)
   #
   return

