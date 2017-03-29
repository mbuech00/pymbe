#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_setup.py: setup utilities for Bethe-Goldstone correlation calculations."""

from os import getcwd, mkdir, chdir
from os.path import isdir
from shutil import rmtree 

from bg_mpi_wrapper import set_exception_hook
from bg_mpi_utils import bcast_mol_dict, init_slave_env
from bg_mpi_time import init_mpi_timings
from bg_info import init_mol, init_param, init_backend_prog, sanity_chk
from bg_utils import run_calc_hf, term_calc
from bg_print import redirect_stdout
from bg_rst_main import rst_init_env

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
   # init output dir
   #
   init_output(molecule)
   #
   # init error handling
   #
   molecule['error'] = [False]
   molecule['error_msg'] = ''
   molecule['error_code'] = -1
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
      # set exception hook
      #
      set_exception_hook(molecule)
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
   # init restart env
   #
   rst_init_env(molecule)
   #
   # init scr env and change into this
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
   if (molecule['error'][-1]): term_calc(molecule)
   #
   return molecule

def init_output(molecule):
   #
   molecule['wrk_dir'] = getcwd()
   #
   molecule['out_dir'] = molecule['wrk_dir']+'/output'
   #
   if (isdir(molecule['out_dir'])): rmtree(molecule['out_dir'],ignore_errors=True)
   #
   mkdir(molecule['out_dir'])
   #
   return molecule

