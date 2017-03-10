#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_setup.py: setup utilities for Bethe-Goldstone correlation calculation."""

from os import mkdir, chdir
from shutil import copy, rmtree 

from bg_mpi_wrapper import abort_mpi
from bg_mpi_utils import bcast_mol_dict, init_slave_env, remove_slave_env
from bg_mpi_time import init_mpi_timings
from bg_info import init_mol, init_param, init_backend_prog, sanity_chk
from bg_utils import run_calc_hf
from bg_print import redirect_stdout

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
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
      # bcast mol dict
      #
      bcast_mol_dict(molecule)
      #
      # init the prog env on the slaves
      #
      init_slave_env(molecule)
   #
   else:
      #
      # init timings
      #
      init_mpi_timings(molecule)
      #
      # init private scr dir
      #
      molecule['scr_dir'] = molecule['wrk_dir']+'/'+molecule['scr_name']+'-'+str(molecule['mpi_rank'])
   #
   # init scr env
   #
   mkdir(molecule['scr_dir'])
   #
   chdir(molecule['scr_dir']) 
   #
   # run hf calc to determine dimensions
   #
   run_calc_hf(molecule)
   #
   # perform a few sanity checks
   #
   sanity_chk(molecule)
   #
   if (molecule['error'][-1]): abort_mpi(molecule)
   #
   return molecule

def term_calc(molecule):
   #
   chdir(molecule['wrk_dir'])
   #
   if (molecule['error'][-1]):
      #
      copy(molecule['scr_dir']+'/OUTPUT_'+str(molecule['mpi_rank'])+'.OUT',molecule['wrk_dir']+'/OUTPUT_'+str(molecule['mpi_rank'])+'.OUT')
   #
   rmtree(molecule['scr_dir'],ignore_errors=True)
   #
   if (molecule['mpi_master'] and molecule['mpi_parallel']):
      #
      remove_slave_env(molecule)
   #
   if (molecule['error'][-1]): abort_mpi(molecule)
   #
   return

