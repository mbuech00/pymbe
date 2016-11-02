#!/usr/bin/env python

#
# orbital-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def generate_drop_occ(start,order,final,molecule,list_drop,drop_string,n_contrib):
   #
   if (order > (molecule['nocc']-molecule['core'])):
      #
      return drop_string, n_contrib
   #
   else:
      #
      for i in range(start,molecule['nocc']+1): # loop over the occupied orbs
         #
         n = list_drop[molecule['core']:molecule['nocc']].count(0) # count the number of zeros
         #
         if (n == 0):
            #
            list_drop[i-1] = 0 # singles correlation
         #
         if (n > 0):
            #
            if (not molecule['occ_domain']):
               #
               list_drop[i-1] = 0 # no screening
            #
            else:
               #
               if (not molecule['occ_domain'][i-1]): # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
                  #
                  list_drop[i-1] = i
               #
               else:
                  #
                  list_drop[i-1] = 0 # attempt to correlate orbital 'i'
                  idx = [j+1 for j, val in enumerate(list_drop[molecule['core']:molecule['nocc']]) if val == 0] # make list containing indices (+1) with zeros in list_drop
                  #
                  for k in range(0,len(idx)):
                     #
                     if (not (set(idx[:k]+idx[k+1:]) <= set(molecule['occ_domain'][idx[k]-1]))): # check whether the combinations of orbs are included in the domains for each of the orbs
                        #
                        list_drop[i-1] = i # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
                        break
         #
         s = ''
         inc = 0
         #
         if ((order == final) and (list_drop[molecule['core']:molecule['nocc']].count(0) == final)): # number of zeros in list_drop must match the final order
            #
            if (molecule['fc']): # exclude core orbitals
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
            for m in range(molecule['core'],molecule['nocc']): # start to exclude valence occupied orbitals
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
               if (len(n_contrib) >= order):
                  #
                  n_contrib[order-1] += 1
               #
               else:
                  #
                  n_contrib.append(1)
               #
               if (molecule['exp'] == 'OCC'):
                  #
                  drop_string.append(s+'\n')
               #
               elif (molecule['exp'] == 'COMB'):
                  #
                  drop_string.append(s)
            #
            elif (order == molecule['nocc']): # full system correlation, i.e., equal to standard N-electron calculation
               #
               n_contrib.append(1)
               #
               drop_string.append('\n')
         #
         generate_drop_occ(i+1,order+1,final,molecule,list_drop,drop_string,n_contrib) # recursion
         #
         list_drop[i-1] = i # include orb back into list of orbs to drop from the calculation
   #
   return drop_string, n_contrib

def generate_drop_virt(start,order,final,molecule,list_drop,drop_string,n_contrib):
   #
   if (order > molecule['nvirt']):
      #
      return drop_string, n_contrib
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
            if (not molecule['virt_domain']):
               #
               list_drop[i-1] = 0 # no screening
            #
            else:
               #
               if (not molecule['virt_domain'][(i-molecule['nocc'])-1]): # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
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
                     if (not (set(idx[:k]+idx[k+1:]) <= set(molecule['virt_domain'][(idx[k]-molecule['nocc'])-1]))): # check whether the combinations of orbs are included in the domains for each of the orbs
                        #
                        list_drop[i-1] = i # this contribution (tuple) should be screened away, i.e., do not correlate orbital 'i' in the current tuple
                        break
         #
         s = ''
         inc = 0
         #
         if (molecule['exp'] == 'COMB'):
            #
            inc += 1
         #
         if ((order == final) and (list_drop[molecule['nocc']:(molecule['nocc']+molecule['nvirt'])].count(0) == final)): # number of zeros in list_drop must match the final order
            #
            if (molecule['fc']): # exclude core orbitals
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
            if (len(n_contrib) >= order):
               #
               n_contrib[order-1] += 1
            #
            else:
               #
               n_contrib.append(1)
            #
            drop_string.append(s+'\n')
         #
         generate_drop_virt(i+1,order+1,final,molecule,list_drop,drop_string,n_contrib)
         #
         list_drop[i-1] = i
   #
   return drop_string, n_contrib

def init_occ_domains(molecule):
   #
   # define occupied domains (currently only for the water case)
   #
   # screen away all interactions between orb 1 and any of the other occupied orbs --- corresponds to a minor improvement over a frozen-core calculation
   #
#   molecule['occ_domain']      = [[]]
#   molecule['occ_domain'].append([3,4,5])
#   molecule['occ_domain'].append([2,4,5])
#   molecule['occ_domain'].append([2,3,5])
#   molecule['occ_domain'].append([2,3,4])
   #
   # screen away all interactions between orb 2 and any of the other occupied orbs
   #
   molecule['occ_domain']      = [[3,4,5]]
   molecule['occ_domain'].append([])
   molecule['occ_domain'].append([1,4,5])
   molecule['occ_domain'].append([1,3,5])
   molecule['occ_domain'].append([1,3,4])
   #
   # screen away all interactions between orb 5 (HOMO) and any of the other occupied orbs
   #
#   molecule['occ_domain']      = [[2,3,4]]
#   molecule['occ_domain'].append([1,3,4])
#   molecule['occ_domain'].append([1,2,4])
#   molecule['occ_domain'].append([1,2,3])
#   molecule['occ_domain'].append([])
   #
   # screen away interactions between orbs 1/2 and between orbs 4/5
   #
#   molecule['occ_domain']     = [[3,4,5]]
#   molecule['occ_domain'].append([3,4,5])
#   molecule['occ_domain'].append([1,2,4,5])
#   molecule['occ_domain'].append([1,2,3])
#   molecule['occ_domain'].append([1,2,3])
   #
   return molecule

def init_virt_domains(molecule):
   #
   # define virtual domains (currently only for the water case)
   #
   # screen away all interactions between orb 6 (LUMO) and any of the other virtual orbs
   #
   molecule['virt_domain']     = [[]]
   molecule['virt_domain'].append([8,9,10,11,12,13])
   molecule['virt_domain'].append([7,9,10,11,12,13])
   molecule['virt_domain'].append([7,8,10,11,12,13])
   molecule['virt_domain'].append([7,8,9,11,12,13])
   molecule['virt_domain'].append([7,8,9,10,12,13])
   molecule['virt_domain'].append([7,8,9,10,11,13])
   molecule['virt_domain'].append([7,8,9,10,11,12])
   #
   # screen away all interactions between orb 13 and any of the other virtual orbs
   #
#   molecule['virt_domain']     = [[7,8,9,10,11,12]]
#   molecule['virt_domain'].append([6,8,9,10,11,12])
#   molecule['virt_domain'].append([6,7,9,10,11,12])
#   molecule['virt_domain'].append([6,7,8,10,11,12])
#   molecule['virt_domain'].append([6,7,8,9,11,12])
#   molecule['virt_domain'].append([6,7,8,9,10,12])
#   molecule['virt_domain'].append([6,7,8,9,10,11])
#   molecule['virt_domain'].append([])
   #
   return molecule

def orbs_incl(molecule,string_excl,string_incl,comb):
   #
   excl_list = []
   #
   sub_string = string_excl[8:] # remove the 'DROP_MO=' part of the string
   sub_list = sub_string.split("-") # remove all the hyphens
   #
   for j in range(0,len(sub_list)):
      #
      if (sub_list[j] != ''):
         #
         excl_list.append(int(sub_list[j]))
   #
   if ((molecule['exp'] == 'OCC') or ((molecule['exp'] == 'COMB') and (not comb))):
      #
      for l in range(1,molecule['nocc']+1):
         #
         if (not (l in excl_list)):
            #
            string_incl.append(l)
   #
   elif ((molecule['exp'] == 'VIRT') or comb):
      #
      for l in range(molecule['nocc']+1,molecule['nocc']+molecule['nvirt']+1):
         #
         if (not (l in excl_list)):
            #
            string_incl.append(l)
   #
   return string_incl


