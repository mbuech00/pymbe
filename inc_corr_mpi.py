#!/usr/bin/env python

#
# MPI-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall/Winter 2016 + Winter/Spring 2017, Mainz, Germnay.
#

import os
from mpi4py import MPI

import inc_corr_utils
import inc_corr_gen_rout
import inc_corr_e_rout

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
      # init basic mpi info
      #
      molecule['mpi_comm'] = MPI.COMM_WORLD
      molecule['mpi_size'] = molecule['mpi_comm'].Get_size()
      #
      molecule['mpi_master'] = True
   #
   return molecule

def main_slave_rout(molecule):
   #
   #  ---  slave routine
   #
   slave = True
   #
   while (slave):
      #
      msg = MPI.COMM_WORLD.bcast(None,root=0)
      #
      if (msg['task'] == 'bcast_mol_dict'):
         #
         # receive molecule dict from master
         #
         molecule = MPI.COMM_WORLD.bcast(None,root=0)
         #
         # overwrite wrk_dir in case this is different from the one on the master node
         #
         molecule['wrk'] = os.getcwd()
         #
         # update with private mpi info
         #
         molecule['mpi_comm'] = MPI.COMM_WORLD
         molecule['mpi_rank'] = molecule['mpi_comm'].Get_rank()
         molecule['mpi_name'] = MPI.Get_processor_name()
         molecule['mpi_stat'] = MPI.Status()
         molecule['mpi_master'] = False
         #
         # private scr dir
         #
         molecule['scr'] += '-'+str(molecule['mpi_rank'])
      #
      elif (msg['task'] == 'energy_calc_mono_exp'):
         #
         level = 'MACRO'
         #
         energy_calc_mono_exp_slave(molecule,level)
      #
      elif (msg['task'] == 'finalize'):
         #
         slave = False
   #
   return molecule

def bcast_mol_dict(molecule):
   #
   #  ---  master routine
   #
   msg = {'task': 'bcast_mol_dict'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   molecule['mpi_comm'].bcast(molecule,root=0)
   #
   molecule['mpi_rank'] = molecule['mpi_comm'].Get_rank()
   molecule['mpi_name'] = MPI.Get_processor_name()
   molecule['mpi_stat'] = MPI.Status()
   #
   return molecule

def energy_calc_mono_exp_slave(molecule,level):
   #
   #  ---  slave routine
   #
   # init scr env
   #
   inc_corr_utils.setup_calc(molecule['scr'])
   #
   # define mpi message tags
   #
   tags = inc_corr_utils.enum('ready','done','exit','start')
   #
   # init data dict
   #
   data = {}
   #
   if (molecule['debug']):
      #
      print('slave no. '+str(molecule['mpi_rank'])+' on node '+str(molecule['mpi_name'])+' with scr dir = '+str(molecule['scr']))
   #
   while True:
      #
      # ready for task
      #
      molecule['mpi_comm'].send(None, dest=0, tag=tags.ready)
      #
      # receive drop string
      #
      string = molecule['mpi_comm'].recv(source=0, tag=MPI.ANY_SOURCE, status=molecule['mpi_stat'])
      #
      if (molecule['debug']):
         #
         print('slave no. '+str(molecule['mpi_rank'])+' on node '+str(molecule['mpi_name'])+' with string = '+str(string))
      #
      # recover tag
      #
      tag = molecule['mpi_stat'].Get_tag()
      #
      # do job or break out (exit)
      #
      if (tag == tags.start):
         #
         inc_corr_gen_rout.run_calc_corr(molecule,string['drop'],level)
         #
         # copy e_tmp
         #
         data['e_tmp'] = molecule['e_tmp']
         #
         # copy job index
         #
         data['index'] = string['index']
         #
         # copy error logical
         #
         data['error'] = molecule['error'][0][-1]
         #
         # send data back to master
         #
         molecule['mpi_comm'].send(data, dest=0, tag=tags.done)
      #
      elif (tag == tags.exit):
         #    
         break
   #
   # exit
   #
   molecule['mpi_comm'].send(None, dest=0, tag=tags.exit)
   #
   # remove scr env
   #
   inc_corr_utils.term_calc(molecule)
   #
   return molecule

def finalize_mpi(molecule):
   #
   #  ---  master and slave routine
   #
   if (MPI.COMM_WORLD.Get_rank() == 0):
      #
      msg = {'task': 'finalize'}
      #
      MPI.COMM_WORLD.bcast(msg,root=0)
   #
   MPI.COMM_WORLD.Barrier()
   #
   MPI.Finalize()
   #
   return

