#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_energy.py: energy-related routines for Bethe-Goldstone correlation calculations."""

from mpi4py import MPI

import inc_corr_orb_rout
from bg_utilities import run_calc_corr 
from bg_print import print_status
import inc_corr_mpi

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.3'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def energy_calc_mono_exp_ser(molecule,order,tup,n_tup,l_limit,u_limit,level):
   #
   string = {'drop': ''}
   #
   if (level == 'MACRO'):
      #
      end = n_tup[order-1]
   #
   elif (level == 'CORRE'):
      #
      end = len(tup[order-1])
   #
   counter = 0
   #
   for i in range(0,end):
      #
      # write string
      #
      if ((level == 'MACRO') or ((level == 'CORRE') and (len(tup[order-1][i]) == 1))):
         #
         counter += 1
         #
         inc_corr_orb_rout.orb_string(molecule,l_limit,u_limit,tup[order-1][i][0],string)
         #
         # run correlated calc
         #
         run_calc_corr(molecule,string['drop'],level)
         #
         # write tuple energy
         #
         tup[order-1][i].append(molecule['e_tmp'])
         #
         # print status
         #
         print_status(float(counter)/float(n_tup[order-1]),level)
         #
         # error check
         #
         if (molecule['error'][0][-1]):
            #
            return molecule, tup
   #
   return molecule, tup

def energy_calc_mono_exp_par(molecule,order,tup,n_tup,l_limit,u_limit,level):
   #
   string = {'drop': ''}
   #
   # number of slaves
   #
   num_slaves = molecule['mpi_size'] - 1
   #
   # number of available slaves
   #
   slaves_avail = num_slaves
   #
   # define mpi message tags
   #
   tags = inc_corr_mpi.enum('ready','done','exit','start')
   #
   # init job index
   #
   i = 0
   #
   # init stat counter
   #
   counter = 0
   #
   # wake up slaves
   #
   msg = {'task': 'energy_calc_mono_exp'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   while (slaves_avail >= 1):
      #
      # write string
      #
      if (i <= (n_tup[order-1]-1)): inc_corr_orb_rout.orb_string(molecule,l_limit,u_limit,tup[order-1][i][0],string)
      #
      # receive data dict
      #
      data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=molecule['mpi_stat'])
      #
      # probe for source
      #
      source = molecule['mpi_stat'].Get_source()
      #
      # probe for tag
      #
      tag = molecule['mpi_stat'].Get_tag()
      #
      if (tag == tags.ready):
         #
         if (i <= (n_tup[order-1]-1)):
            #
            # store job index
            #
            string['index'] = i
            #
            # send string dict
            #
            molecule['mpi_comm'].send(string,dest=source,tag=tags.start)
            #
            # increment job index
            #
            i += 1
         #
         else:
            #
            molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
      #
      elif (tag == tags.done):
         #
         # write tuple energy
         #
         tup[order-1][data['index']].append(data['e_tmp'])
         #
         # increment stat counter
         #
         counter += 1
         #
         # print status
         #
         print_status(float(counter)/float(n_tup[order-1]),level)
         #
         # error check
         #
         if (data['error']):
            #
            print('problem with slave '+str(source)+' in energy_calc_mono_exp_par  ---  aborting...')
            #
            molecule['error'][0].append(True)
            #
            return molecule, tup
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   return molecule, tup

def bg_order(molecule,k,tup,energy):
   #
   for j in range(0,len(tup[k-1])):
      #
      for i in range(k-1,0,-1):
         #
         for l in range(0,len(tup[i-1])):
            #
            if (set(tup[i-1][l][0]) < set(tup[k-1][j][0])):
               #
               tup[k-1][j][1] -= tup[i-1][l][1]
   #
   e_tmp = 0.0
   #
   for j in range(0,len(tup[k-1])):
      #
      e_tmp += tup[k-1][j][1]
   #
   if (k > 1):
      #
      e_tmp += energy[k-2]
   #
   energy.append(e_tmp)
   #
   return energy

