#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_entanglement.py: entanglement and orbital contribution routines for Bethe-Goldstone correlation calculations."""

import numpy as np

from bg_mpi_time import timer_mpi, collect_screen_mpi_time
from bg_entanglement_mpi import entanglement_abs_par

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def entanglement_main(molecule,l_limit,u_limit,order,calc_end=False):
   #
   # calculate orbital entanglement
   #
   if (order >= 2):
      #
      entanglement_abs(molecule,l_limit,u_limit,order,calc_end)
      #
      entanglement_rel(molecule,u_limit,order)
   #
   # calculate relative orbital contributions
   #
   orb_contributions(molecule,order,(order == 1))
   #
   if (molecule['mpi_parallel'] and (order >= 2)): collect_screen_mpi_time(molecule,order,calc_end)
   #
   return molecule

def orb_contributions(molecule,order,singles=False):
   #
   timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   molecule['prim_orb_con_abs'].append([])
   molecule['prim_orb_con_rel'].append([])
   #
   if (singles):
      #
      # total absolute orbital contributions
      #
      if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'])):
         #
         for _ in range(0,molecule['ncore']):
            #
            molecule['prim_orb_con_abs'][-1].append(0.0)
      #
      for i in range(0,len(molecule['prim_energy_inc'][-1])):
         #
         molecule['prim_orb_con_abs'][-1].append(molecule['prim_energy_inc'][-1][i])
      #
      # total relative orbital contributions
      #
      for i in range(0,len(molecule['prim_orb_con_abs'][-1])):
         #
         molecule['prim_orb_con_rel'][-1].append(abs(molecule['prim_orb_con_abs'][-1][i])/np.abs(np.sum(molecule['prim_energy_inc'][-1])))
   #
   else:
      #
      # total absolute orbital contributions
      #
      for i in range(0,len(molecule['prim_orb_ent_abs'][-1])):
         #
         molecule['prim_orb_con_abs'][-1].append(molecule['prim_orb_con_abs'][-2][i]+np.sum(molecule['prim_orb_ent_abs'][-1][i]))
      #
      # total relative orbital contributions
      #
      for i in range(0,len(molecule['prim_orb_con_abs'][-1])):
         #
         if (molecule['prim_orb_con_abs'][-1][i] == 0.0):
            #
            molecule['prim_orb_con_rel'][-1].append(0.0)
         #
         else:
            #
            molecule['prim_orb_con_rel'][-1].append(molecule['prim_orb_con_abs'][-1][i]/sum(molecule['prim_orb_con_abs'][-1]))
   #
   timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule

def entanglement_abs(molecule,l_limit,u_limit,order,calc_end):
   #
   if (molecule['mpi_parallel']):
      #
      entanglement_abs_par(molecule,l_limit,u_limit,order,calc_end)
   #
   else:
      #
      timer_mpi(molecule,'mpi_time_work_screen',order)
      #
      # write orbital entanglement matrix (abs)
      #
      molecule['prim_orb_ent_abs'].append(np.zeros([u_limit,u_limit],dtype=np.float64))
      #
      for l in range(0,len(molecule['prim_tuple'][-1])):
         #
         for i in range(l_limit,l_limit+u_limit):
            #
            for j in range(l_limit,i):
               #
               # add up absolute contributions from the correlation between orbs i and j at current order
               #
               if (set([i+1,j+1]) <= set(molecule['prim_tuple'][-1][l])):
                  #
                  molecule['prim_orb_ent_abs'][-1][i-l_limit,j-l_limit] += molecule['prim_energy_inc'][-1][l]
      #
      timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule
      
def entanglement_rel(molecule,u_limit,order):
   #
   timer_mpi(molecule,'mpi_time_work_screen',order)
   #
   # write orbital entanglement matrix (rel)
   #
   molecule['prim_orb_ent_rel'].append(np.zeros([u_limit,u_limit],dtype=np.float64))
   #
   molecule['prim_orb_ent_rel'][-1] = (np.abs(molecule['prim_orb_ent_abs'][-1])/np.amax(np.abs(molecule['prim_orb_ent_abs'][-1])))*100.0
   #
   timer_mpi(molecule,'mpi_time_work_screen',order,True)
   #
   return molecule

