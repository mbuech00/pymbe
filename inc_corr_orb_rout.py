#!/usr/bin/env python

#
# orbital-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

import itertools
import math

def orb_generator(molecule,dom,tup,l_limit,k):
   #
   incl = []
   incl_2 = []
   #
   full_space = []
   #
   for i in range(0,len(dom)):
      #
      # construct union space of all orbitals in i-th domain + the i-th orbital itself (if not traditional frozen core scheme)
      #
      if (not ((molecule['frozen'] == 'TRAD') and (((i+l_limit)+1) <= molecule['ncore']))):
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
   # for VIRT scheme, explicitly drop the core orbitals for traditional frozen core scheme
   #
   if ((molecule['exp'] == 'VIRT') and (molecule['frozen'] == 'TRAD')):
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

def merge_tuples(prim_tup,sec_tup,k):
   #
   incl_list = []
   #
   for i in range(0,len(prim_tup[k-2])):
      #
      found = False
      #
      for j in range(0,len(sec_tup[k-2])):
         #
         if (set(prim_tup[k-2][i][0]) == set(sec_tup[k-2][j][0])):
            #
            found = True
            #
            break
      #
      if (not found):
         #
         incl_list.append(i)
   #
   for l in range(0,len(incl_list)):
      #
      sec_tup[k-2].append(prim_tup[k-2][incl_list[l]])
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
   if (molecule['frozen'] == 'TRAD'):
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
      if (molecule['frozen'] == 'TRAD'):
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

def excl_rout(molecule,tup,orb,thres,excl):
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
   if ((len(tup) == 2) and (len(tup[0]) == molecule['nocc']) and (molecule['frozen'] == 'SCREEN')):
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

def orbs_incl(string_excl,string_incl,l_limit,u_limit):
   #
   excl_list = []
   #
   if (string_excl[0] == 'D'):
      #
      sub_string = string_excl[8:] # remove the 'DROP_MO=' part of the string
   #
   else:
      #
      sub_string = string_excl
   #
   sub_list = sub_string.split("-") # remove all the hyphens
   #
   for j in range(0,len(sub_list)):
      #
      if ((sub_list[j] != '') and (sub_list[j] != '\n')):
         #
         excl_list.append(int(sub_list[j]))
   #
   for l in range(l_limit+1,(l_limit+u_limit)+1):
      #
      if (not (l in excl_list)):
         #
         string_incl.append(l)
   #
   return string_incl

def n_theo_tuples(dim,k,theo_work):
   #
   theo_work.append(math.factorial(dim)/(math.factorial(k)*math.factorial(dim-k)))
   #
   return theo_work

