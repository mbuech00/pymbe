#!/usr/bin/env python

#
# orbital-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

import itertools
import math

import inc_corr_utils

def orb_generator(molecule,dom,tup,l_limit,k):
   #
   incl = []
   incl_2 = []
   #
   full_space = []
   #
   for i in range(0,len(dom)):
      #
      # construct union space of all orbitals in i-th domain + the i-th orbital itself (if not conventional frozen core scheme)
      #
      if (not ((molecule['frozen'] == 'CONV') and (((i+l_limit)+1) <= molecule['ncore']))):
         #
         full_space = sorted(list(set(dom[i][-1]).union(set([(l_limit+i)+1]))))
         #
         # generate all k-combinations in the space
         #
         incl_tmp = list(list(comb) for comb in itertools.combinations(full_space,k))
         #
         for j in range(0,len(incl_tmp)):
            #
            # is the i-th orbital in the given combination?
            #
            if (set([(l_limit+i)+1]) <= set(incl_tmp[j])):
               #
               mask = True
               #
               # loop through the given combination to check whether all orbitals are part of each other domains
               #
               for l in range(0,len(incl_tmp[j])):
                  #
                  for m in range(0,len(incl_tmp[j])):
                     #
                     if (m != l):
                        #
                        # domain check
                        #
                        if (not (set([incl_tmp[j][m]]) <= set(dom[(incl_tmp[j][l]-l_limit)-1][-1]))):
                           #
                           mask = False
                           #
                           break
                  #
                  if (not mask):
                     #
                     break
               #
               if (mask):
                  #
                  # the given tuple is allowed
                  #
                  incl.append(incl_tmp[j])
   #
   # remove duplicates
   #
   for i in range(0,len(incl)):
      #
      if (incl[i] not in incl_2):
         #
         incl_2.append(incl[i])
   #
   # write to molecule['tuple']
   #
   for i in range(0,len(incl_2)):
      #
      tup.append([incl_2[i]])
   #
   return tup

def orb_string(molecule,l_limit,u_limit,tup):
   #
   # generate list with all occ/virt orbitals
   #
   dim = range(l_limit+1,(l_limit+u_limit)+1)
   #
   # generate list with all orbs the should be dropped (not part of the current tuple)
   #
   drop = sorted(list(set(dim)-set(tup)))
   #
   # for VIRT scheme, explicitly drop the core orbitals for conventional frozen core scheme
   #
   if ((molecule['exp'] == 'VIRT') and (molecule['frozen'] == 'CONV')):
      #
      for i in range(molecule['ncore'],0,-1):
         #
         drop.insert(0,i)
   #
   # now write the string
   #
   inc = 0
   molecule['string'] = ''
   #
   for i in range(0,len(drop)):
      #
      if (inc == 0):
         #
         molecule['string'] += 'DROP_MO='+str(drop[i])
      #
      else:
         #
         if (drop[i] == (drop[i-1]+1)):
            #
            if (i < (len(drop)-1)):
               #
               if (drop[i] != (drop[i+1]-1)):
                  #
                  molecule['string'] += '>'+str(drop[i])
            #
            else:
               #
               molecule['string'] += '>'+str(drop[i])
         #
         else:
            #
            molecule['string'] += '-'+str(drop[i])
      #
      inc += 1
   #
   if (molecule['string'] != ''):
      #
      molecule['string'] += '\n'
   #
   return molecule

def orb_screen_rout(molecule,tup,orb,dom,thres,l_limit,u_limit,level):
   #
   # set up entanglement and exclusion lists
   #
   orb.append([])
   #
   orb_entang_rout(molecule,tup,orb,l_limit,u_limit)
   #
   molecule['excl_list'][0][:] = []
   #
   excl_rout(molecule,tup,orb,thres,molecule['excl_list'][0],level)
   #
   # update domains
   #
   update_domains(dom,l_limit,molecule['excl_list'][0])
   #
   # print domain updates
   #
   inc_corr_utils.print_update(dom,l_limit,u_limit,level)
   #
   return molecule, dom

def orb_entang_rout(molecule,tup,orb,l_limit,u_limit):
   #
   for i in range(l_limit,l_limit+u_limit):
      #
      orb[-1].append([[i+1]])
      #
      for j in range(l_limit,l_limit+u_limit):
         #
         if (j != i):
            #
            e_abs = 0.0
            #
            for k in range(0,len(tup[-1])):
               #
               if ((set([i+1]) <= set(tup[-1][k][0])) and (set([j+1]) <= set(tup[-1][k][0]))):
                  #
                  e_abs += tup[-1][k][1]
            #
            orb[-1][i-l_limit].append([[j+1],[e_abs]])
   #
   for i in range(l_limit,l_limit+u_limit):
      #
      e_sum = 0.0
      #
      for j in range(0,len(orb)):
         #
         for k in range(l_limit,(l_limit+u_limit)-1):
            #
            e_sum += orb[j][i-l_limit][(k-l_limit)+1][1][0]
      #
      for j in range(0,len(orb)):
         #
         for k in range(l_limit,(l_limit+u_limit)-1):
            #
            if (orb[j][i-l_limit][(k-l_limit)+1][1][0] != 0.0):
               #
               orb[j][i-l_limit][(k-l_limit)+1][1].append(orb[j][i-l_limit][(k-l_limit)+1][1][0] / e_sum)
            #
            else:
               #
               orb[j][i-l_limit][(k-l_limit)+1][1].append(0.0)
   #
   if (molecule['debug']):
      #
      print('')
      print(' --- relative contributions ---')
      #
      for i in range(0,len(orb)):
         #
         print('')
         print(' * order = '+str(i+2))
         print('')
         #
         tmp = []
         #
         for j in range(0,len(orb[i])):
            #
            tmp.append([])
            #
            for k in range(0,len(orb[i][j])-1):
               #
               tmp[j].append(orb[i][j][k+1][1][-1])
            #
            print(' {0:}'.format(j+1)+' : '+str(['{0:6.3f}'.format(m) for m in tmp[-1]]))
      #
      print('')
   #
   return orb

def select_est_tuples(prim_tup,sec_tup,k):
   #
   pop_list = []
   #
   for i in range(0,len(sec_tup[k-1])):
      #
      found = False
      #
      for j in range(0,len(prim_tup[k-1])):
         #
         if (set(sec_tup[k-1][i][0]) <= set(prim_tup[k-1][j][0])):
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
      sec_tup[k-1].pop(pop_list[l]-l)
   #
   return sec_tup

def init_domains(molecule):
   #
   molecule['occ_domain'] = []
   molecule['virt_domain'] = []
   #
   for i in range(0,molecule['nocc']):
      #
      molecule['occ_domain'].append([range(1,molecule['nocc']+1)])
      #
      molecule['occ_domain'][i][-1].pop(i)
   #
   if (molecule['frozen'] == 'CONV'):
      #
      for i in range(0,molecule['ncore']):
         #
         molecule['occ_domain'][i][-1][:] = []
      #
      for j in range(molecule['ncore'],molecule['nocc']):
         #
         for _ in range(0,molecule['ncore']):
            #
            molecule['occ_domain'][j][-1].pop(0)
   #
   for i in range(0,molecule['nvirt']):
      #
      molecule['virt_domain'].append([range(molecule['nocc']+1,(molecule['nocc']+molecule['nvirt'])+1)])
      #
      molecule['virt_domain'][i][-1].pop(i)
   #
   return molecule

def reinit_domains(molecule,domain):
   #
   domain[:] = []
   #
   if (molecule['exp'] == 'COMB-OV'):
      #
      for i in range(0,molecule['nvirt']):
         #
         domain.append([range(molecule['nocc']+1,(molecule['nocc']+molecule['nvirt'])+1)])
         #
         domain[i][-1].pop(i)  
   #
   elif (molecule['exp'] == 'COMB-VO'):
      #
      for i in range(0,molecule['nocc']):
         #
         domain.append([range(1,molecule['nocc']+1)])
         #
         domain[i][-1].pop(i)
      #
      if (molecule['frozen'] == 'CONV'):
         #
         for i in range(0,molecule['ncore']):
            #
            domain[i][-1][:] = []
         #
         for j in range(molecule['ncore'],molecule['nocc']):
            #
            for i in range(0,molecule['ncore']):
               #
               domain[j][-1].pop(i)
   #
   return molecule

def excl_rout(molecule,tup,orb,thres,excl,level):
   #
   for i in range(0,len(orb[-1])):
      #
      excl.append([])
      #
      for j in range(0,len(orb[-1][i])-1):
         #
         if ((abs(orb[-1][i][j+1][1][-1]) < thres) and (abs(orb[-1][i][j+1][1][-1]) != 0.0)):
            #
            excl[i].append(orb[-1][i][j+1][0][0])
   #
   if ((len(tup) == 2) and (len(tup[0]) == molecule['nocc']) and (molecule['frozen'] == 'SCREEN') and (level != 'ESTIM')):
      #
      for i in range(0,len(excl)):
         #
         if (i < molecule['ncore']):
            #
            for j in range(i+1,len(excl)):
               #
               if (not (j+1 in excl[i])):
                  #
                  excl[i].append(j+1)
         #
         else:
            #
            for j in range(0,molecule['ncore']):
               #
               if (not (j+1 in excl[i])):
                  #
                  excl[i].append(j+1)
   #
   for i in range(0,len(excl)):
      #
      excl[i].sort()
   #
   return excl

def update_domains(domain,l_limit,excl):
   #
   for l in range(0,len(domain)):
      #
      domain[l].append(list(domain[l][-1]))
   #
   for i in range(0,len(excl)):
      #
      for j in range(0,len(excl[i])):
         #
         if ((i+l_limit)+1 in excl[(excl[i][j]-l_limit)-1]):
            #
            domain[i][-1].remove(excl[i][j])
            domain[(excl[i][j]-l_limit)-1][-1].remove((i+l_limit)+1)
            #
            excl[(excl[i][j]-l_limit)-1].remove((i+l_limit)+1)
   #
   return domain

def n_theo_tuples(dim,k,theo_work):
   #
   theo_work.append(math.factorial(dim)/(math.factorial(k)*math.factorial(dim-k)))
   #
   return theo_work

