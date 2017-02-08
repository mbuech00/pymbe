#!/usr/bin/env python

#
# orbital-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

import itertools
import math

import inc_corr_utils

def orb_generator(molecule,dom,tup,l_limit,u_limit,k):
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'] == 'conv')):
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
      incl = list(list(comb) for comb in itertools.combinations(range(start+(1+l_limit),(l_limit+u_limit)+1),2))
      #
      for i in range(0,len(incl)):
         #
         tup[k-1].append([incl[i]])
   #
   else:
      #
      select = []
      #
      for i in range(0,len(dom)-1):
         #
         # generate list of indices where val is greater than orb index = (i+l_limit)+1
         #
         idx = [x for x in range(0,len(dom[i])) if dom[i][x] > ((i+l_limit)+1)]
         #
         if (len(idx) > 0):
            #
            # generate complete set of (k-1)-combinations
            #
            tmp = list(list(comb) for comb in itertools.combinations(dom[i][idx[0]:],k-1))
            #
            select[:] = []
            #
            for j in range(0,len(tmp)):
               #
               # generate subset of all pairs within the given (k-1)-combination
               #
               tmp_sub = list(list(comb) for comb in itertools.combinations(tmp[j],2))
               #
               select.append(True)
               #
               for l in range(0,len(tmp_sub)):
                  #
                  # is the specific tuple in tmp allowed?
                  #
                  if (tmp_sub[l][1] not in dom[(tmp_sub[l][0]-l_limit)-1]):
                     #
                     select[-1] = False
                     #
                     break
            #
            for m in range(0,len(tmp)):
               #
               if (select[m]):
                  #
                  # complete k-combination by appending orb index = (i+l_limit)+1
                  #
                  tmp[m].append((i+l_limit)+1)
                  #
                  # finally, add the ordered tuple to the tuple list
                  #
                  tup[k-1].append([sorted(tmp[m])])
   #
   return tup

def orb_string(molecule,l_limit,u_limit,tup,string):
   #
   # generate list with all occ/virt orbitals
   #
   dim = range(l_limit+1,(l_limit+u_limit)+1)
   #
   # generate list with all orbs the should be dropped (not part of the current tuple)
   #
   drop = sorted(list(set(dim)-set(tup)))
   #
   # for virt scheme, explicitly drop the core orbitals for conventional frozen core scheme
   #
   if ((molecule['exp'] == 'virt') and (molecule['frozen'] == 'conv')):
      #
      for i in range(molecule['ncore'],0,-1):
         #
         drop.insert(0,i)
   #
   # now write the string
   #
   inc = 0
   #
   string['drop'] = ''
   #
   for i in range(0,len(drop)):
      #
      if (inc == 0):
         #
         string['drop'] += 'DROP_MO='+str(drop[i])
      #
      else:
         #
         if (drop[i] == (drop[i-1]+1)):
            #
            if (i < (len(drop)-1)):
               #
               if (drop[i] != (drop[i+1]-1)):
                  #
                  string['drop'] += '>'+str(drop[i])
            #
            else:
               #
               string['drop'] += '>'+str(drop[i])
         #
         else:
            #
            string['drop'] += '-'+str(drop[i])
      #
      inc += 1
   #
   if (string['drop'] != ''):
      #
      string['drop'] += '\n'
   #
   return string

def orb_screen_rout(molecule,order,tup,orb,dom,thres,l_limit,u_limit,level):
   #
   if ((order == 1) or (thres == 0.0)):
      #
      update_domains(dom,l_limit,[])
   #
   else:
      #
      # set up entanglement and exclusion lists
      #
      orb.append([])
      #
      orb_entang_rout(molecule,tup,orb,l_limit,u_limit)
      #
      molecule['excl_list'][0][:] = []
      #
      excl_rout(molecule,tup,orb,thres,l_limit,molecule['excl_list'][0],level)
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
   orb_arr = molecule['prim_orb_arr']
   orb_con = molecule['prim_orb_con']
   #
   orb_arr[:] = []
   orb_con[:] = []
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
      for j in range(l_limit,l_limit+u_limit):
         #
         e_sum += orb[-1][i-l_limit][j-l_limit][0]
         #
         # add the contributions from lower orders
         #
         for m in range(0,len(orb)-1):
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
   # write orbital entanglement matrices and total orbital contributions
   #
   tmp = []
   #
   for i in range(0,len(orb)):
      #
      orb_arr.append([])
      orb_con.append([])
      #
      tmp[:] = []
      #
      for j in range(0,len(orb[i])):
         #
         orb_arr[i].append([])
         #
         for k in range(0,len(orb[i][j])):
            #
            orb_arr[i][j].append(orb[i][j][k][1])
      #
      for k in range(0,len(orb[i][j])):
         #
         e_sum = 0.0
         #
         for j in range(0,len(orb[i])):
            #
            e_sum += orb_arr[i][j][k]
         #
         tmp.append(e_sum)
      #
      e_sum = sum(tmp)
      #
      for k in range(0,len(tmp)):
         #
         if (tmp[k] == 0.0):
            #
            orb_con[i].append(0.0)
         #
         else:
            #
            orb_con[i].append(tmp[k]/e_sum)
   #
   if (molecule['debug']):
      #
      print('')
      print('   ---------------------------------------------')
      print('           relative orb. contributions          ')
      print('   ---------------------------------------------')
      #
      index = '          '
      #
      for m in range(l_limit+1,(l_limit+u_limit)+1):
         #
         if (m < 10):
            #
            index += str(m)+'         '
         #
         elif ((m >= 10) and (m < 100)):
            #
            index += str(m)+'        '
         #
         elif ((m >= 100)):
            #
            index += str(m)+'       '
      #
      for i in range(0,len(orb)):
         #
         print('')
         print(' * BG exp. order = '+str(i+2))
         print(' -------------------')
         print('')
         #
         print('      --- entanglement matrix ---')
         print('')
         #
         print(index)
         #
         for j in range(0,len(orb_arr[i])):
            #
            print(' {0:>3d}'.format((j+l_limit)+1)+' '+str(['{0:6.3f}'.format(m) for m in orb_arr[i][j]]))
         #
         print('')
         print('      --- total orbital contributions ---')
         print('')
         #
         print(index)
         #
         print('     '+str(['{0:6.3f}'.format(m) for m in orb_con[i]]))
      #
      print('')
   #
   return orb

def select_corr_tuples(prim_tup,sec_tup,k):
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
   molecule['occ_domain'] = [[]]
   molecule['virt_domain'] = [[]]
   #
   for i in range(0,molecule['nocc']):
      #
      molecule['occ_domain'][0].append(list(range(1,molecule['nocc']+1)))
      #
      molecule['occ_domain'][0][i].pop(i)
   #
   if (molecule['frozen'] == 'conv'):
      #
      for i in range(0,molecule['ncore']):
         #
         molecule['occ_domain'][0][i][:] = []
      #
      for j in range(molecule['ncore'],molecule['nocc']):
         #
         for _ in range(0,molecule['ncore']):
            #
            molecule['occ_domain'][0][j].pop(0)
   #
   for i in range(0,molecule['nvirt']):
      #
      molecule['virt_domain'][0].append(list(range(molecule['nocc']+1,(molecule['nocc']+molecule['nvirt'])+1)))
      #
      molecule['virt_domain'][0][i].pop(i)
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
         dom[0].append(list(range(molecule['nocc']+1,(molecule['nocc']+molecule['nvirt'])+1)))
         #
         dom[0][i].pop(i)  
   #
   elif (molecule['exp'] == 'comb-vo'):
      #
      for i in range(0,molecule['nocc']):
         #
         dom[0].append(list(range(1,molecule['nocc']+1)))
         #
         dom[0][i].pop(i)
      #
      if (molecule['frozen'] == 'conv'):
         #
         for i in range(0,molecule['ncore']):
            #
            dom[0][i][:] = []
         #
         for j in range(molecule['ncore'],molecule['nocc']):
            #
            for i in range(0,molecule['ncore']):
               #
               dom[0][j].pop(i)
   #
   return molecule

def excl_rout(molecule,tup,orb,thres,l_limit,excl,level):
   #
   for i in range(0,len(orb[-1])):
      #
      excl.append([])
      #
      for j in range(0,len(orb[-1][i])):
         #
         if ((abs(orb[-1][i][j][1]) < thres) and (abs(orb[-1][i][j][1]) != 0.0)):
            #
            excl[i].append((j+l_limit)+1)
   #
   if ((len(tup) == 2) and (len(tup[0]) == molecule['nocc']) and (molecule['frozen'] == 'screen') and (level != 'CORRE')):
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

def update_domains(dom,l_limit,excl):
   #
   dom.append([])
   #
   for l in range(0,len(dom[0])):
      #
      dom[-1].append(list(dom[-2][l]))
   #
   for i in range(0,len(excl)):
      #
      for j in range(0,len(excl[i])):
         #
         if ((i+l_limit)+1 in excl[(excl[i][j]-l_limit)-1]):
            #
            dom[-1][i].remove(excl[i][j])
            dom[-1][(excl[i][j]-l_limit)-1].remove((i+l_limit)+1)
            #
            excl[(excl[i][j]-l_limit)-1].remove((i+l_limit)+1)
   #
   return dom

def n_theo_tuples(dim,k,theo_work):
   #
   theo_work.append(math.factorial(dim)/(math.factorial(k)*math.factorial(dim-k)))
   #
   return theo_work

