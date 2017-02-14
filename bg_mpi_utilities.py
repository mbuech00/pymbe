#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_utilities.py: MPI utilities for Bethe-Goldstone correlation calculations."""

from os import chdir
from mpi4py import MPI

from bg_mpi_kernels import main_slave_rout

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.3'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def init_mpi(molecule):
   #
   #  ---  master and slave routine
   #
   if (MPI.COMM_WORLD.Get_size() > 1):
      #
      molecule['mpi_parallel'] = True
   #
   else:
      #
      molecule['mpi_parallel'] = False
   #
   # slave proceed to the main slave routine
   #
   if (MPI.COMM_WORLD.Get_rank() != 0):
      #
      main_slave_rout(molecule)
   #
   else:
      #
      molecule['mpi_master'] = True
   #
   return molecule

def abort_mpi(molecule):
   #
   #  ---  master routine
   #
   chdir(molecule['wrk'])
   #
   molecule['mpi_comm'].Abort()
   #
   return

def finalize_mpi(molecule):
   #
   #  ---  master and slave routine
   #
   if (MPI.COMM_WORLD.Get_rank() == 0):
      #
      msg = {'task': 'finalize_mpi'}
      #
      MPI.COMM_WORLD.bcast(msg,root=0)
   #
   MPI.COMM_WORLD.Barrier()
   #
   MPI.Finalize()
   #
   return

def add_dict(dict_1, dict_2, datatype):
   #
   for item in dict_2:
      # 
      if (item in dict_1):
         #
         dict_1[item] += dict_2[item]
      #
      else:
         #
         dict_1[item] = dict_2[item]
   #
   return dict_1


