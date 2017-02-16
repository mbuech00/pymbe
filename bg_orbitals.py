#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_orbitals.py: orbital-related routines for Bethe-Goldstone correlation calculations."""

from itertools import combinations 
from copy import deepcopy

from bg_mpi_orbitals import orb_generator_master
from bg_print import print_orb_info, print_update

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def orb_generator(molecule,dom,tup,l_limit,u_limit,k):
   #
   if (molecule['mpi_parallel'] and (k >= 3)):
      #
      orb_generator_master(molecule,dom,tup,l_limit,u_limit,k)
   #
   else:
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
         for i in range(start,len(dom)):
            #
            # all singles contributions 
            #
            tup[k-1].append([[(i+l_limit)+1]])
      #
      elif (k == 2):
         #
         # generate all possible (unique) pairs
         #
         tmp = list(list(comb) for comb in combinations(range(start+(1+l_limit),(l_limit+u_limit)+1),2))
         #
         for i in range(0,len(tmp)):
            #
            tup[k-1].append([tmp[i]])
         #
         del tmp
      #
      else:
         #
         for i in range(0,len(tup[k-2])):
            #
            # generate subset of all pairs within the parent tuple
            #
            tmp = list(list(comb) for comb in combinations(tup[k-2][i][0],2))
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
               for m in range(tup[k-2][i][0][-1]+1,(l_limit+u_limit)+1):
                  #
                  mask2 = True
                  #
                  for l in tup[k-2][i][0]:
                     #
                     # is the new child tuple allowed?
                     #
                     if (not (set([m]) < set(dom[(l-l_limit)-1]))):
                        #
                        mask2 = False
                        #
                        break
                  #
                  if (mask2):
                     #
                     # append the child tuple to the tup list
                     #
                     tup[k-1].append([deepcopy(tup[k-2][i][0])])
                     #
                     tup[k-1][-1][0].append(m)
         #
         del tmp
   #
   return tup

def orb_screening(molecule,order,l_limit,u_limit,level):
   #
   if (order == 1):
      #
      # add singles contributions to orb_con list
      #
      orb_entanglement(molecule,l_limit,u_limit,level,True)
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
      orb_entanglement(molecule,l_limit,u_limit,level)
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
   return molecule

def orb_entanglement(molecule,l_limit,u_limit,level,singles=False):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      orb = molecule['prim_orb_ent']
      orb_arr = molecule['prim_orb_arr']
      orb_con_abs = molecule['prim_orb_con_abs']
      orb_con_rel = molecule['prim_orb_con_rel']
   #
   elif (level == 'CORRE'):
      #
      tup = molecule['corr_tuple']
      orb = molecule['corr_orb_ent']
      orb_arr = molecule['corr_orb_arr']
      orb_con_abs = molecule['corr_orb_con_abs']
      orb_con_rel = molecule['corr_orb_con_rel']
   #
   if (singles):
      #
      # total orbital contribution
      #
      orb_con_abs.append([])
      orb_con_rel.append([])
      #
      e_sum = 0.0
      #
      for i in range(0,len(tup[-1])):
         #
         e_sum += tup[-1][i][1]
      #
      if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'])):
         #
         for _ in range(0,molecule['ncore']):
            #
            orb_con_abs[-1].append(0.0)
            #
            orb_con_rel[-1].append(0.0)
      #
      for i in range(0,len(tup[-1])):
         #
         orb_con_abs[-1].append(tup[-1][i][1])
         #
         orb_con_rel[-1].append(orb_con_abs[-1][-1]/e_sum)
   #
   else:
      #
      orb.append([])
      #
      orb_con_abs.append([])
      orb_con_rel.append([])
      #
      orb_arr[:] = []
      #
      for i in range(l_limit,l_limit+u_limit):
         #
         orb[-1].append([])
         #
         for j in range(l_limit,l_limit+u_limit):
            #
            orb[-1][i-l_limit].append([])
            #
            e_abs = 0.0
            #
            if (i != j):
               #
               # add up contributions from the correlation between orbs i and j at current order
               #
               for l in range(0,len(tup[-1])):
                  #
                  if ((set([i+1]) <= set(tup[-1][l][0])) and (set([j+1]) <= set(tup[-1][l][0]))):
                     #
                     e_abs += tup[-1][l][1] 
            #
            # write to orb list
            #
            orb[-1][i-l_limit][j-l_limit].append(e_abs)
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
               e_sum += orb[m][i-l_limit][j-l_limit][0]
         #
         # calculate relative contributions
         #
         for m in range(0,len(orb)):
            #
            for j in range(l_limit,l_limit+u_limit):
               #
               if (len(orb[m][i-l_limit][j-l_limit]) == 2):
                  #
                  if (orb[m][i-l_limit][j-l_limit][0] != 0.0):
                     #
                     orb[m][i-l_limit][j-l_limit][1] = orb[m][i-l_limit][j-l_limit][0]/e_sum
                  #
                  else:
                     #
                     orb[m][i-l_limit][j-l_limit][1] = 0.0
               #
               else:
                  #
                  if (orb[m][i-l_limit][j-l_limit][0] != 0.0):
                     #
                     orb[m][i-l_limit][j-l_limit].append(orb[m][i-l_limit][j-l_limit][0]/e_sum)
                  #
                  else:
                     #
                     orb[m][i-l_limit][j-l_limit].append(0.0)
      #
      # orbital entanglement matrices for orders: 2 <= k <= current
      #
      for i in range(0,len(orb)):
         #
         orb_arr.append([])
         #
         for j in range(0,len(orb[i])):
            #
            orb_arr[i].append([])
            #
            for k in range(0,len(orb[i][j])):
               #
               orb_arr[i][j].append(orb[i][j][k][1])
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
            e_sum += orb[-1][j][k][0]
         #
         tmp.append(e_sum)
      #
      for j in range(0,len(tmp)):
         #
         orb_con_abs[-1].append(orb_con_abs[-2][j]+tmp[j])
      #
      e_sum = 0.0
      #
      for j in range(0,len(orb_con_abs[-1])):
         #
         e_sum += orb_con_abs[-1][j]
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

def select_corr_tuples(prim_tup,corr_tup,k):
   #
   pop_list = []
   #
   for i in range(0,len(corr_tup[k-1])):
      #
      found = False
      #
      for j in range(0,len(prim_tup[k-1])):
         #
         if (set(corr_tup[k-1][i][0]) <= set(prim_tup[k-1][j][0])):
            #
            found = True
            #
            break
      #
      if (found):
         #
         pop_list.append(i)
   #
   for l in range(0,len(pop_list)):
      #
      corr_tup[k-1].pop(pop_list[l]-l)
   #
   del pop_list
   #
   return corr_tup

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

def reinit_domains(molecule,dom):
   #
   dom[:] = [[]]
   #
   if (molecule['exp'] == 'comb-ov'):
      #
      for i in range(0,molecule['nvirt']):
         #
         dom.append(list(range(molecule['nocc']+1,(molecule['nocc']+molecule['nvirt'])+1)))
         #
         dom[i].pop(i)  
   #
   elif (molecule['exp'] == 'comb-vo'):
      #
      for i in range(0,molecule['nocc']):
         #
         dom.append(list(range(1,molecule['nocc']+1)))
         #
         dom[i].pop(i)
      #
      if (molecule['frozen']):
         #
         for i in range(0,molecule['ncore']):
            #
            dom[i][:] = []
         #
         for j in range(molecule['ncore'],molecule['nocc']):
            #
            for i in range(0,molecule['ncore']):
               #
               dom[j].pop(i)
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
         if ((abs(orb[-1][i][j][1]) < thres) and (abs(orb[-1][i][j][1]) != 0.0)):
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


