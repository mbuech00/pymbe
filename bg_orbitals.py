#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_orbitals.py: orbital-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI
from itertools import combinations 
from copy import deepcopy

from bg_mpi_time import timer_mpi
from bg_mpi_orbitals import bcast_dom_master, orb_entanglement_main_par, collect_init_mpi_time
from bg_print import print_orb_info, print_update

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def orb_generator(molecule,dom,tup,l_limit,u_limit,k,level):
   #
   if (molecule['mpi_parallel'] and molecule['mpi_master']): bcast_dom_master(molecule,dom,l_limit,u_limit,k,level)
   #
   if (molecule['mpi_parallel'] and (k >= 2)): timer_mpi(molecule,'mpi_time_work_init',k-1) 
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']):
      #
      start = molecule['ncore']
   #
   else:
      #
      start = 0
   #
   if (k == 1):
      #
      # all singles contributions 
      #
      tmp = []
      #
      for i in range(start,len(dom)):
         #
         tmp.append([(i+l_limit)+1])
      #
      tup.append(np.array(tmp,dtype=np.int))
   #
   elif (k == 2):
      #
      # generate all possible (unique) pairs
      #
      tmp = list(list(comb) for comb in combinations(range(start+(1+l_limit),(l_limit+u_limit)+1),2))
      #
      tup.append(np.array(tmp,dtype=np.int))
   #
   else:
      #
      tmp_2 = []
      #
      if (level == 'MACRO'):
         #
         end = len(tup[k-2])
      #
      elif (level == 'CORRE'):
         #
         end = len(tup[k-2])+len(molecule['prim_tuple'][k-2])
      #
      for i in range(0,end):
         #
         # generate subset of all pairs within the parent tuple
         #
         if (level == 'MACRO'):
            #
            parent_tup = tup[k-2][i]
         #
         elif (level == 'CORRE'):
            #
            if (i <= (len(tup[k-2])-1)):
               #
               parent_tup = tup[k-2][i]
            #
            else:
               #
               parent_tup = molecule['prim_tuple'][k-2][i-len(tup[k-2])]
         #
         tmp = list(list(comb) for comb in combinations(parent_tup,2))
         #
         mask = True
         #
         for j in range(0,len(tmp)):
            #
            # is the parent tuple still allowed?
            #
            if (not (set([tmp[j][1]]) < set(dom[(tmp[j][0]-l_limit)-1]))):
               #
               mask = False
               #
               break
         #
         if (mask):
            #
            # loop through possible orbitals to augment the parent tuple with
            #
            for m in range(parent_tup[-1]+1,(l_limit+u_limit)+1):
               #
               mask_2 = True
               #
               for l in parent_tup:
                  #
                  # is the new child tuple allowed?
                  #
                  if (not (set([m]) < set(dom[(l-l_limit)-1]))):
                     #
                     mask_2 = False
                     #
                     break
               #
               if (mask_2):
                  #
                  # append the child tuple to the tup list
                  #
                  tmp_2.append(list(deepcopy(parent_tup)))
                  #
                  tmp_2[-1].append(m)
                  #
                  # check whether this tuple has already been accounted for in the primary expansion
                  #
                  if ((level == 'CORRE') and (np.equal(tmp_2[-1],molecule['prim_tuple'][k-1]).all(axis=1).any())):
                     #
                     tmp_2.pop(-1)
      #
      if (len(tmp_2) > 1): tmp_2.sort()
      #
      tup.append(np.array(tmp_2,dtype=np.int))
      #
      del tmp_2
   #
   if (molecule['mpi_parallel'] and (k >= 2)): timer_mpi(molecule,'mpi_time_work_init',k-1,True)
   #
   del tmp
   #
   return tup

def orb_screening(molecule,order,l_limit,u_limit,level,calc_end=False):
   #
   if (order == 1):
      #
      # add singles contributions to orb_con list
      #
      orb_contributions(molecule,order,level,True)
      #
      # print orb info
      #
      if (molecule['debug']): print_orb_info(molecule,l_limit,u_limit,level)
      #
      # update domains
      #
      update_domains(molecule,l_limit,level,True)
   #
   else:
      #
      # set up entanglement and exclusion lists
      #
      if (molecule['mpi_parallel']):
         #
         orb_entanglement_main_par(molecule,l_limit,u_limit,order,level)
      #
      else:
         #
         orb_entanglement_main(molecule,l_limit,u_limit,order,level)
      #
      timer_mpi(molecule,'mpi_time_work_init',order)
      #
      orb_entanglement_arr(molecule,l_limit,u_limit,level)
      #
      orb_contributions(molecule,order,level)
      #
      if (calc_end):
         #
         timer_mpi(molecule,'mpi_time_work_init',order,True)
         #
         collect_init_mpi_time(molecule,order)
      #
      else:
         #
         # print orb info
         #
         if (molecule['debug']): print_orb_info(molecule,l_limit,u_limit,level)
         #
         # construct exclusion list
         #
         orb_exclusion(molecule,l_limit,level)
         #
         # update domains
         #
         update_domains(molecule,l_limit,level)
         #
         # print domain updates
         #
         print_update(molecule,l_limit,u_limit,level)
         #
         timer_mpi(molecule,'mpi_time_work_init',order,True)
         #
         collect_init_mpi_time(molecule,order)
   #
   return molecule

def orb_entanglement_main(molecule,l_limit,u_limit,order,level):
   #
   if (level == 'MACRO'):
      #
      orb = molecule['prim_orb_ent']
      #
      end = len(molecule['prim_tuple'][order-1])
   #
   elif (level == 'CORRE'):
      #
      orb = molecule['corr_orb_ent']
      #
      end = len(molecule['corr_tuple'][order-1])+len(molecule['prim_tuple'][order-1])
   #
   orb.append(np.zeros([u_limit,u_limit],dtype=np.float64))
   #
   for l in range(0,end):
      #
      if ((level == 'CORRE') and (l >= len(molecule['prim_tuple'][order-1]))):
         #
         tup = molecule['corr_tuple'][order-1]
         e_inc = molecule['corr_energy_inc'][order-1]
         ldx = l-len(molecule['prim_tuple'][order-1])
      #
      else:
         #
         tup = molecule['prim_tuple'][order-1]           
         e_inc = molecule['prim_energy_inc'][order-1]
         ldx = l
      # 
      for i in range(l_limit,l_limit+u_limit):
         #
         for j in range(i+1,l_limit+u_limit):
            #
            # add up contributions from the correlation between orbs i and j at current order
            #
            if (set([i+1,j+1]) <= set(tup[ldx])):
               #
               orb[-1][i-l_limit][j-l_limit] += e_inc[ldx]
               orb[-1][j-l_limit][i-l_limit] = orb[-1][i-l_limit][j-l_limit]
   #
   return molecule
      
def orb_entanglement_arr(molecule,l_limit,u_limit,level):
   #
   if (level == 'MACRO'):
      #
      orb = molecule['prim_orb_ent']
      orb_arr = molecule['prim_orb_arr']
   #
   elif (level == 'CORRE'):
      #
      orb = molecule['corr_orb_ent']
      orb_arr = molecule['corr_orb_arr']
   #
   orb_arr.append(np.empty([u_limit,u_limit],dtype=np.float64))
   #
   for i in range(l_limit,l_limit+u_limit):
      #
      e_sum = 0.0
      #
      # calculate sum of contributions from all orbitals to orb i
      #
      for m in range(0,len(orb)):
         #
         for j in range(l_limit,l_limit+u_limit):
            #
            e_sum += orb[m][i-l_limit][j-l_limit]
      #
      # calculate relative contributions
      #
      for m in range(0,len(orb)):
         #
         for j in range(l_limit,l_limit+u_limit):
            #
            if (orb[m][i-l_limit][j-l_limit] != 0.0):
               #
               orb_arr[m][i-l_limit][j-l_limit] = orb[m][i-l_limit][j-l_limit]/e_sum
            #
            else:
               #
               orb_arr[m][i-l_limit][j-l_limit] = 0.0
   #
   return molecule

def orb_contributions(molecule,order,level,singles=False):
   #
   if (level == 'MACRO'):
      #
      orb = molecule['prim_orb_ent']
      orb_con_abs = molecule['prim_orb_con_abs']
      orb_con_rel = molecule['prim_orb_con_rel']
   #
   elif (level == 'CORRE'):
      #
      orb = molecule['corr_orb_ent']
      orb_con_abs = molecule['corr_orb_con_abs']
      orb_con_rel = molecule['corr_orb_con_rel']
   #
   if (singles):
      #
      e_inc = molecule['prim_energy_inc'][order-1]
      #
      # total orbital contribution
      #
      orb_con_abs.append([])
      orb_con_rel.append([])
      #
      e_sum = np.sum(e_inc)
      #
      if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'])):
         #
         for _ in range(0,molecule['ncore']):
            #
            orb_con_abs[-1].append(0.0)
            #
            orb_con_rel[-1].append(0.0)
      #
      for i in range(0,len(e_inc)):
         #
         orb_con_abs[-1].append(e_inc[i])
         #
         orb_con_rel[-1].append(orb_con_abs[-1][-1]/e_sum)
   #
   else:
      #
      orb_con_abs.append([])
      orb_con_rel.append([])
      #
      # total orbital contribution
      #
      tmp = []
      #
      for j in range(0,len(orb[-1])):
         #
         e_sum = 0.0
         #
         for k in range(0,len(orb[-1][j])):
            #
            e_sum += orb[-1][j][k]
         #
         tmp.append(e_sum)
      #
      for j in range(0,len(tmp)):
         #
         orb_con_abs[-1].append(orb_con_abs[-2][j]+tmp[j])
      #
      e_sum = sum(orb_con_abs[-1])
      #
      for j in range(0,len(orb_con_abs[-1])):
         #
         if (orb_con_abs[-1][j] == 0.0):
            #
            orb_con_rel[-1].append(0.0)
         #
         else:
            #
            orb_con_rel[-1].append(orb_con_abs[-1][j]/e_sum)
      #
      del tmp
   #
   return molecule

def init_domains(molecule):
   #
   molecule['occ_domain'] = []
   molecule['virt_domain'] = []
   #
   for i in range(0,molecule['nocc']):
      #
      molecule['occ_domain'].append(list(range(1,molecule['nocc']+1)))
      #
      molecule['occ_domain'][i].pop(i)
   #
   if (molecule['frozen']):
      #
      for i in range(0,molecule['ncore']):
         #
         molecule['occ_domain'][i][:] = []
      #
      for j in range(molecule['ncore'],molecule['nocc']):
         #
         for _ in range(0,molecule['ncore']):
            #
            molecule['occ_domain'][j].pop(0)
   #
   for i in range(0,molecule['nvirt']):
      #
      molecule['virt_domain'].append(list(range(molecule['nocc']+1,(molecule['nocc']+molecule['nvirt'])+1)))
      #
      molecule['virt_domain'][i].pop(i)
   #
   return molecule

def orb_exclusion(molecule,l_limit,level):
   #
   if (level == 'MACRO'):
      #
      orb = molecule['prim_orb_ent']
      orb_arr = molecule['prim_orb_arr']
      orb_con_rel = molecule['prim_orb_con_rel']
      thres = molecule['prim_thres']
   #
   else:
      #
      orb = molecule['corr_orb_ent']
      orb_arr = molecule['corr_orb_arr']
      orb_con_rel = molecule['corr_orb_con_rel']
      thres = molecule['corr_thres']
   #
   molecule['excl_list'][:] = []
   #
   # screening in individual domains based on orbital entanglement 
   #
   for i in range(0,len(orb[-1])):
      #
      molecule['excl_list'].append([])
      #
      for j in range(0,len(orb[-1][i])):
         #
         if ((abs(orb_arr[-1][i][j]) < thres) and (abs(orb_arr[-1][i][j]) != 0.0)):
            #
            molecule['excl_list'][i].append((j+l_limit)+1)
   #
   # screening in all domains based on total orbital contributions
   #
   for i in range(0,len(orb_con_rel[-1])):
      #
      if ((orb_con_rel[-1][i] < thres) and (sum(orb_arr[-1][i]) != 0.0)):
         #
         for j in range(0,len(orb_con_rel[-1])):
            #
            if (i != j):
               #
               if (not (set([(j+l_limit)+1]) <= set(molecule['excl_list'][i]))):
                  #
                  molecule['excl_list'][i].append((j+l_limit)+1)
               #
               if (not (set([(i+l_limit)+1]) <= set(molecule['excl_list'][j]))):
                  #
                  molecule['excl_list'][j].append((i+l_limit)+1)
   #
   for i in range(0,len(molecule['excl_list'])):
      #
      molecule['excl_list'][i].sort()
   #
   return molecule

def update_domains(molecule,l_limit,level,singles=False):
   #
   if (level == 'MACRO'):
      #
      dom = molecule['prim_domain']
   #
   elif (level == 'CORRE'):
      #
      dom = molecule['corr_domain']
   #
   dom.append([])
   #
   for l in range(0,len(dom[0])):
      #
      dom[-1].append(list(dom[-2][l]))
   #
   if (not singles):
      #
      for i in range(0,len(molecule['excl_list'])):
         #
         for j in range(0,len(molecule['excl_list'][i])):
            #
            if ((i+l_limit)+1 in molecule['excl_list'][(molecule['excl_list'][i][j]-l_limit)-1]):
               #
               dom[-1][i].remove(molecule['excl_list'][i][j])
               dom[-1][(molecule['excl_list'][i][j]-l_limit)-1].remove((i+l_limit)+1)
               #
               molecule['excl_list'][(molecule['excl_list'][i][j]-l_limit)-1].remove((i+l_limit)+1)
   #
   return molecule


