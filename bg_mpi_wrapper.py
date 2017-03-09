#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_wrapper.py: MPI wrapper routines for Bethe-Goldstone correlation calculations."""

from os import chdir
from mpi4py import MPI

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def abort_mpi(molecule):
   #
   #  ---  master routine
   #
   chdir(molecule['wrk_dir'])
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
      msg = {'task': 'finalize_mpi', 'order': len(molecule['prim_energy'])}
      #
      MPI.COMM_WORLD.bcast(msg,root=0)
   #
   MPI.COMM_WORLD.Barrier()
   #
   MPI.Finalize()
   #
   return

