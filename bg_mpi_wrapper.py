#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_wrapper.py: MPI wrapper routines for Bethe-Goldstone correlation calculations."""

import sys
import traceback
from os import chdir
from contextlib import redirect_stdout
from mpi4py import MPI

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def set_exception_hook(molecule):
   #
   sys_excepthook = sys.excepthook 
   #
   def mpi_excepthook(t,v,tb):
      #
      sys_excepthook(t,v,tb)
      #
      traceback.print_last(file=sys.stdout)
      #
      abort_mpi(molecule)
   #
   sys.excepthook = mpi_excepthook
   #
   return

def abort_rout(molecule):
   #
   with open(molecule['out_dir']+'/bg_output.out','a') as f:
      #
      with redirect_stdout(f):
         #
         print('')
         print('!!!!!!!!!!!!!')
         print('ERROR')
         #
         if (molecule['error_code'] <= 1):
            #
            if (molecule['error_code'] == 0):
               #
               print(' - master quits with input error:')
               print(molecule['error_msg'])
            #
            else:
               #
               print(' - master quits with HF error:')
               print(molecule['error_msg'])
         #
         else:
            #
            print(' - mpi proc. # {0:} quits with correlated calc. error:'.format(molecule['error_rank']))
            print(molecule['error_msg'])
            print('print of the string of dropped MOs:')
            print(molecule['error_drop'])
         #
         print('ERROR')
         print('!!!!!!!!!!!!!')
         print('')
   #
   if (molecule['mpi_parallel']):
      #
      abort_mpi(molecule) 
   #
   else:
      #
      sys.exit()
   #
   return

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

