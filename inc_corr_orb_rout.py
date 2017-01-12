#!/usr/bin/env python

#
# orbital-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

import math

def generate_drop_occ(start,order,final,molecule,list_drop,drop_string,n_tuples):
   #
   if (order > molecule['u_limit'][0]):
      #
      return drop_string, n_tuples
   #
   else:
      #
      for i in range(start,molecule['nocc']+1): # loop over the occupied orbs
         #
         n = list_drop[0:molecule['nocc']].count(0) # count the number of zeros
         #
         if ((n == 0) and not (not molecule['occ_domain'][i-1][-1])):
            #
            list_drop[i-1] = 0 # singles correlation
         #
         if (n > 0):
            #
            if (not molecule['occ_domain'][i-1][-1]): # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
               #
               list_drop[i-1] = i
            #
            else:
               #
               list_drop[i-1] = 0 # attempt to correlate orbital 'i'
               idx = [j+1 for j, val in enumerate(list_drop[molecule['l_limit'][0]:molecule['nocc']]) if val == 0] # make list containing indices (+1) with zeros in list_drop
               #
               for k in range(0,len(idx)):
                  #
                  if (not (set(idx[:k]+idx[k+1:]) <= set(molecule['occ_domain'][idx[k]-1][-1]))): # check whether the combinations of orbs are included in the domains for each of the orbs
                     #
                     list_drop[i-1] = i # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
                     break
         #
         s = ''
         inc = 0
         #
         if ((order == final) and (list_drop[molecule['l_limit'][0]:molecule['nocc']].count(0) == final)): # number of zeros in list_drop must match the final order
            #
            for m in range(molecule['l_limit'][0],molecule['nocc']): # start to exclude valence occupied orbitals
               #
               if (list_drop[m] != 0):
                  #
                  if (inc == 0):
                     #
                     s = 'DROP_MO='+str(list_drop[m])
                  #
                  else:
                     #
                     s += '-'+str(list_drop[m])
                  #
                  inc += 1
            #
            if (s != ''):
               #
               if (len(n_tuples) >= order):
                  #
                  n_tuples[order-1] += 1
               #
               else:
                  #
                  n_tuples.append(1)
               #
               if (molecule['exp'] == 'OCC'):
                  #
                  drop_string.append(s+'\n')
               #
               elif ((molecule['exp'] == 'COMB-OV') or (molecule['exp'] == 'COMB-VO')):
                  #
                  drop_string.append(s)
            #
            elif (order == molecule['nocc']): # full system correlation, i.e., equal to standard N-electron calculation
               #
               n_tuples.append(1)
               #
               drop_string.append('\n')
         #
         generate_drop_occ(i+1,order+1,final,molecule,list_drop,drop_string,n_tuples) # recursion
         #
         list_drop[i-1] = i # include orb back into list of orbs to drop from the calculation
   #
   return drop_string, n_tuples

def generate_drop_virt(start,order,final,molecule,list_drop,drop_string,n_tuples):
   #
   if (order > molecule['nvirt']):
      #
      return drop_string, n_tuples
   #
   else:
      #
      for i in range(start,(molecule['nocc']+molecule['nvirt'])+1):
         #
         n = list_drop[molecule['nocc']:(molecule['nocc']+molecule['nvirt'])].count(0) # count the number of zeros
         #
         if (n == 0):
            #
            list_drop[i-1] = 0 # singles correlation
         #
         if (n > 0):
            #
            if (not molecule['virt_domain'][(i-molecule['nocc'])-1][-1]): # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
               #
               list_drop[i-1] = i
            #
            else:
               #
               list_drop[i-1] = 0 # attempt to correlate orbital 'i'
               idx = [(j+molecule['nocc'])+1 for j, val in enumerate(list_drop[molecule['nocc']:(molecule['nocc']+molecule['nvirt'])]) if val == 0] # make list containing indices (+1) with zeros in list_drop
               #
               for k in range(0,len(idx)):
                  #
                  if (not (set(idx[:k]+idx[k+1:]) <= set(molecule['virt_domain'][(idx[k]-molecule['nocc'])-1][-1]))): # check whether the combinations of orbs are included in the domains for each of the orbs
                     #
                     list_drop[i-1] = i # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
                     break
         #
         s = ''
         inc = 0
         #
         if ((molecule['exp'] == 'COMB-OV') or (molecule['exp'] == 'COMB-VO')):
            #
            inc += 1
         #
         if ((order == final) and (list_drop[molecule['nocc']:(molecule['nocc']+molecule['nvirt'])].count(0) == final)): # number of zeros in list_drop must match the final order
            #
            if ((molecule['frozen'] == 'TRAD') and (not ((molecule['exp'] == 'COMB-OV') or (molecule['exp'] == 'COMB-VO')))): # exclude core orbitals
               #
               for m in range(0,molecule['core']):
                  #
                  if (inc == 0):
                     #
                     s = 'DROP_MO='+str(list_drop[m])
                  #
                  else:
                     #
                     s += '-'+str(list_drop[m])
                  #
                  inc += 1
            #
            for m in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
               #
               if (list_drop[m] != 0):
                  #
                  if (inc == 0):
                     #
                     s = 'DROP_MO='+str(list_drop[m])
                  #
                  else:
                     #
                     s += '-'+str(list_drop[m])
                  #
                  inc += 1
            #
            if (s != ''):
               #
               if (len(n_tuples) >= order):
                  #
                  n_tuples[order-1] += 1
               #
               else:
                  #
                  n_tuples.append(1)
               #
               drop_string.append(s+'\n')
            #
            elif (order == molecule['nvirt']): # full system correlation, i.e., equal to standard N-electron calculation
               #
               n_tuples.append(1)
               #
               drop_string.append('\n')
         #
         generate_drop_virt(i+1,order+1,final,molecule,list_drop,drop_string,n_tuples)
         #
         list_drop[i-1] = i
   #
   return drop_string, n_tuples

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
      for i in range(0,molecule['core']):
         #
         molecule['occ_domain'][i][-1][:] = []
      #
      for j in range(molecule['core'],molecule['nocc']):
         #
         for i in range(0,molecule['core']):
            #
            molecule['occ_domain'][j][-1].pop(i)
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
         for i in range(0,molecule['core']):
            #
            molecule['occ_domain'][i][-1][:] = []
         #
         for j in range(molecule['core'],molecule['nocc']):
            #
            for i in range(0,molecule['core']):
               #
               molecule['occ_domain'][j][-1].pop(i)
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
   if ((len(tup) == 2) and (molecule['frozen'] == 'SCREEN')):
      #
      for i in range(0,len(excl)):
         #
         if (i < molecule['core']):
            #
            for j in range(i+1,len(excl)):
               #
               if (not (j+1 in excl[i])):
                  #
                  excl[i].append(j+1)
         #
         else:
            #
            for j in range(0,molecule['core']):
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

