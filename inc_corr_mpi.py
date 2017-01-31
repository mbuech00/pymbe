#!/usr/bin/env python

#
# MPI-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall/Winter 2016 + Winter/Spring 2017, Mainz, Germnay.
#

from mpi4py import MPI

import inc_corr_e_rout

def init_mpi_info(molecule):
   #
   #  ---  master / slave
   #
   molecule['comm'] = MPI.COMM_WORLD
   molecule['size'] = molecule['comm'].Get_size()
   molecule['rank'] = molecule['comm'].Get_rank()
   molecule['name'] = MPI.Get_processor_name()
   molecule['stat'] = MPI.Status()
   #
   if (molecule['rank'] == 0):
      #
      molecule['master'] = True
   #
   else:
      #
      molecule['master'] = False
      #
      main_slave_rout(molecule)
   #
   if (molecule['size'] > 1):
      #
      molecule['parallel'] = True
   #
   return molecule

def main_slave_rout(molecule):
   #
   slave = True
   #
   while (slave):
      #
      msg = molecule['comm'].bcast(None,root=0)
      #
      if (msg['task'] == 'energy_calc'):
         #
         inc_corr_e_rout.energy_calc_mpi()
      #
      elif (msg['task'] == 'finalize'):
         #
         slave = False
         #
         finalize_mpi(molecule)
   #
   return

def finalize_mpi(molecule):
   #
   if (molecule['master']):
      #
      msg = {'task': 'finalize'}
      #
      molecule['comm'].bcast(msg,root=0)
   #
   MPI.Finalize()
   #
   return

