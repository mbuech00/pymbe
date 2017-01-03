#!/usr/bin/env python

#
# energy-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

from timeit import default_timer as timer

import inc_corr_gen_rout
import inc_corr_orb_rout
import inc_corr_plot

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def inc_corr_tuple_thres(molecule):
   #
   if (molecule['exp'] == 'OCC'):
      #
      u_limit = molecule['nocc']-molecule['core']
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      u_limit = molecule['nvirt']
   #
   elif (molecule['exp'] == 'COMB'):
      #
      u_limit_1 = molecule['nocc']-molecule['core']
      u_limit_2 = molecule['nvirt']
   #
   list_drop = list(range(1,(molecule['nocc']+molecule['nvirt'])+1))
   #
   if (molecule['exp'] == 'OCC'):
      #
      for i in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
         #
         list_drop[i] = 0
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      for i in range(molecule['core'],molecule['nocc']):
         #
         list_drop[i] = 0
   #
   drop_string = []
   #
   incl_list = []
   #
   if (molecule['exp'] == 'COMB'):
      #
      drop_string_comb = []
      n_tuples_comb = []
      #
      incl_list_comb = []
      tuple_comb = []
      #
      e_fin_comb = []
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['tuple'] = []
   #
   molecule['n_tuples'] = []
   #
   molecule['orb_contrib'] = []
   #
   molecule['e_fin'] = []
   #
   molecule['time'] = []
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'VIRT')):
      #
      for k in range(1,u_limit+1):
         #
         start = timer()
         #
         drop_string[:] = []
         #
         if (molecule['exp'] == 'OCC'):
            #
            inc_corr_orb_rout.generate_drop_occ(molecule['core']+1,1,k,molecule,list_drop,drop_string,molecule['n_tuples'])
         #
         elif (molecule['exp'] == 'VIRT'):
            #
            inc_corr_orb_rout.generate_drop_virt(molecule['nocc']+1,1,k,molecule,list_drop,drop_string,molecule['n_tuples'])
         #
         molecule['tuple'].append([])
         #
         if (len(molecule['n_tuples']) < k):
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  has no contributions'.format(k,u_limit))
            print(' --------------------------------------------------------')
            print('')
            continue
         #
         print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  started'.format(k,u_limit))
         print(' -------------------------------------------')
         #
         incl_list[:] = []
         #
         for i in range(0,molecule['n_tuples'][k-1]):
            #
            inc_corr_gen_rout.run_calc_corr(molecule,drop_string[i],False)
            #
            incl_list.append([])
            #
            inc_corr_orb_rout.orbs_incl(molecule,drop_string[i],incl_list[i],False)
            #
            molecule['tuple'][k-1].append([incl_list[i],molecule['e_tmp']])
            #
            if (molecule['error'][0][-1]):
               #
               return molecule
         #
         inc_corr_order(k,molecule['n_tuples'],molecule['tuple'],molecule['e_fin'])
         #
         molecule['orb_contrib'].append([])
         #
         orb_contrib_rout(molecule,molecule['tuple'][k-1],molecule['orb_contrib'][k-1])
         #
#         inc_corr_plot.e_contrib_plot(molecule)
         #
         if (k > 1):
            #
            if (molecule['exp'] == 'OCC'):
               #
               inc_corr_chk_conv(k,molecule['thres'][0],molecule['e_fin'],molecule,False)
            #
            elif (molecule['exp'] == 'VIRT'):
               #
               inc_corr_chk_conv(k,molecule['thres'][1],molecule['e_fin'],molecule,False)
         #
         molecule['time'].append(timer()-start)
         #
         if (k == 1):
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit,molecule['time'][k-1]))
            print(' --------------------------------------------------------------')
         #
         else:
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}  ---  conv =  {4:}'.\
                             format(k,u_limit,molecule['time'][k-1],molecule['e_fin'][k-1]-molecule['e_fin'][k-2],molecule['conv'][0][-1]))
            print(' ------------------------------------------------------------------------------------------------------------')
         #
         for i in range(0,molecule['n_tuples'][k-1]):
            #
            print(' RESULT-MACRO:  tuple = {0:4d} / {1:4d}  ,  corr. orbs. = {2:}  ,  abs = {3:9.4e}'.\
                             format(i+1,molecule['n_tuples'][k-1],molecule['tuple'][k-1][i][0],molecule['tuple'][k-1][i][1]))
         #
         print('')
         #
         if (molecule['conv'][0][-1]):
            #
            return molecule
   #
   elif (molecule['exp'] == 'COMB'):
      #
      for k in range(1,u_limit_1+1):
         #
         start = timer()
         #
         drop_string[:] = []
         #
         inc_corr_orb_rout.generate_drop_occ(molecule['core']+1,1,k,molecule,list_drop,drop_string,molecule['n_tuples'])
         #
         molecule['tuple'].append([])
         #
         if (len(molecule['n_tuples']) < k):
            #
            print('       STATUS-MACRO:  order = {0:4d} / {1:4d}  has no contributions'.format(k,u_limit_1))
            print('       --------------------------------------------------------')
            print('')
            continue
         #
         print('       STATUS-MACRO:  order = {0:4d} / {1:4d}  started'.format(k,u_limit_1))
         print('       -------------------------------------------')
         #
         incl_list[:] = []
         #
         for j in range(0,molecule['n_tuples'][k-1]):
            #
            start_comb = timer()
            #
            n_tuples_comb[:] = []
            #
            e_fin_comb[:] = []
            #
            tuple_comb[:] = []
            #
            molecule['conv'][1].append(False)
            #
            for l in range(1,u_limit_2+1):
               #
               for a in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
                  #
                  list_drop[a] = a+1
               #
               drop_string_comb[:] = []
               #
               inc_corr_orb_rout.generate_drop_virt(molecule['nocc']+1,1,l,molecule,list_drop,drop_string_comb,n_tuples_comb)
               #
               tuple_comb.append([])
               #
               incl_list_comb[:] = []
               #
               for i in range(0,n_tuples_comb[l-1]):
                  #
                  if (drop_string[j] == '\n'):
                     #
                     string = 'DROP_MO='+drop_string_comb[i][1:]
                  #
                  else:
                     #
                     string = drop_string[j]+drop_string_comb[i]
                  #
                  inc_corr_gen_rout.run_calc_corr(molecule,string,False)
                  #
                  incl_list_comb.append([])
                  #
                  inc_corr_orb_rout.orbs_incl(molecule,string,incl_list_comb[i],True)
                  #
                  tuple_comb[l-1].append([incl_list_comb[i],molecule['e_tmp']])
                  #
                  if (molecule['error'][0][-1]):
                     #
                     return molecule
               #
               inc_corr_order(l,n_tuples_comb,tuple_comb,e_fin_comb)
               #
               if (l > 1):
                  #
                  inc_corr_chk_conv(l,molecule['thres'][1],e_fin_comb,molecule,True)
               #
               nv_order = l
               #
               if (molecule['conv'][1][-1]):
                  #
                  incl_list.append([])
                  #
                  inc_corr_orb_rout.orbs_incl(molecule,drop_string[j],incl_list[j],False)
                  #
                  molecule['tuple'][k-1].append([incl_list[j],e_fin_comb[l-1],[]])
                  #
#                  orb_contrib_rout(molecule,tuple_comb,True)
                  #
                  break
            #
            print('       STATUS-MICRO:  tuple = {0:4d} / {1:4d}  (order = {2:4d} / {3:4d})  done in {4:10.2e} seconds  ---  diff =  {5:9.4e}  ---  conv =  {6:}'\
                             .format(j+1,molecule['n_tuples'][k-1],nv_order,molecule['nvirt'],timer()-start_comb,e_fin_comb[l-1]-e_fin_comb[l-2],molecule['conv'][1][-1]))
         #
         inc_corr_order(k,molecule['n_tuples'],molecule['tuple'],molecule['e_fin'])
         #
         if (k > 1):
            #
            inc_corr_chk_conv(k,molecule['thres'][0],molecule['e_fin'],molecule,False)
         #
         molecule['time'].append(timer()-start)
         #
         print('')
         #
         if (k == 1):
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit_1,molecule['time'][k-1]))
            print(' --------------------------------------------------------------')
         #
         else:
            #
            print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}  ---  conv =  {4:}'.\
                             format(k,u_limit_1,molecule['time'][k-1],molecule['e_fin'][k-1]-molecule['e_fin'][k-2],molecule['conv'][0][-1]))
            print(' ------------------------------------------------------------------------------------------------------------')
         #
         for i in range(0,molecule['n_tuples'][k-1]):
            #
            print(' RESULT-MACRO:  tuple = {0:4d} / {1:4d}  ,  corr. orbs. = {2:}  ,  abs = {3:9.4e}'.\
                             format(i+1,molecule['n_tuples'][k-1],molecule['tuple'][k-1][i][0],molecule['tuple'][k-1][i][1]))
         #
         print('')
         #
         if (molecule['conv'][0][-1]):
            #
#            molecule['orb_contrib'] = []
#            orb_contrib_rout(molecule,molecule['tuple'],False)
            #
            return molecule
   #
   return molecule

def inc_corr_tuple_order(molecule):
   #
   if (molecule['exp'] == 'OCC'):
      #
      u_limit = molecule['nocc']-molecule['core']
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      u_limit = molecule['nvirt']
   #
   list_drop = list(range(1,(molecule['nocc']+molecule['nvirt'])+1))
   #
   if (molecule['exp'] == 'OCC'):
      #
      for i in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
         #
         list_drop[i] = 0
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      for i in range(molecule['core'],molecule['nocc']):
         #
         list_drop[i] = 0
   #
   drop_string = []
   #
   incl_list = []
   #
   molecule['tuple'] = []
   #
   for _ in range(0,molecule['order']):
      #
      molecule['tuple'].append([])
   #
   molecule['n_tuples'] = [0] * u_limit
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['e_fin'] = []
   #
   molecule['time'] = []
   #
   for k in range(molecule['order'],0,-1):
      #
      start = timer()
      #
      drop_string[:] = []
      #
      if (molecule['exp'] == 'OCC'):
         #
         inc_corr_orb_rout.generate_drop_occ(molecule['core']+1,1,k,molecule,list_drop,drop_string,molecule['n_tuples'])
      #
      elif (molecule['exp'] == 'VIRT'):
         #
         inc_corr_orb_rout.generate_drop_virt(molecule['nocc']+1,1,k,molecule,list_drop,drop_string,molecule['n_tuples'])
      #
      if (molecule['n_tuples'][k-1] == 0):
         #
         print(' STATUS:  order = {0:4d} / {1:4d}  has no contributions'.format(k,u_limit))
         print(' --------------------------------------------------')
         print('')
         continue
      #
      print(' STATUS:  order = {0:4d} / {1:4d}  started'.format(k,u_limit))
      print(' -------------------------------------')
      #
      incl_list[:] = []
      #
      for i in range(0,molecule['n_tuples'][k-1]):
         #
         inc_corr_gen_rout.run_calc_corr(molecule,drop_string[i],False)
         #
         incl_list.append([])
         #
         inc_corr_orb_rout.orbs_incl(molecule,drop_string[i],incl_list[i],False)
         #
         molecule['tuple'][k-1].append([incl_list[i],molecule['e_tmp']])
         #
         if (molecule['error'][0][-1]):
            #
            return molecule
      #
      molecule['time'].append(timer()-start)
      #
      print(' STATUS:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(k,u_limit,molecule['time'][-1]))
      print(' --------------------------------------------------------')
      #
      print('')
   #
   for k in range(1,molecule['order']+1):
      #
      if (molecule['n_tuples'][k-1] > 0):
         #
         inc_corr_order(k,molecule['n_tuples'],molecule['tuple'],molecule['e_fin'])
   #
   molecule['time'].reverse()
   #
#   molecule['orb_contrib'] = []
#   orb_contrib_rout(molecule,molecule['tuple'],False)
   #
   return molecule

def inc_corr_order(k,n_tuples,tup,e_fin):
   #
   for j in range(0,n_tuples[k-1]):
      #
      for i in range(k-1,0,-1):
         #
         for l in range(0,n_tuples[i-1]):
            #
            if (set(tup[i-1][l][0]) < set(tup[k-1][j][0])):
               #
               tup[k-1][j][1] -= tup[i-1][l][1]
   #
   e_tmp = 0.0
   #
   for j in range(0,n_tuples[k-1]):
      #
      e_tmp += tup[k-1][j][1]
   #
   if (k > 1):
      #
      e_tmp += e_fin[k-2]
   #
   e_fin.append(e_tmp)
   #
   return e_fin

def inc_corr_chk_conv(order,thres,e_fin,molecule,comb):
   #
   e_diff = e_fin[order-1] - e_fin[order-2]
   #
   if (abs(e_diff) < thres):
      #
      if (comb):
         #
         molecule['conv'][1].append(True)
      #
      else:
         #
         molecule['conv'][0].append(True)
   #
   else:
      #
      if (comb):
         #
         molecule['conv'][1].append(False)
      #
      else:
         #
         molecule['conv'][0].append(False)
   #
   return molecule

def orb_contrib_rout(molecule,tup,orb_contrib):
   #
   if ((molecule['exp'] == 'OCC') or ((molecule['exp'] == 'COMB') and (not comb))):
      #
      l_limit = molecule['core']
      u_limit = molecule['nocc']
   #
   elif ((molecule['exp'] == 'VIRT') or ((molecule['exp'] == 'COMB') and comb)):
      #
      l_limit = molecule['nocc']
      u_limit = molecule['nocc']+molecule['nvirt']
   #
   for j in range(l_limit,u_limit):
      #
      tmp = 0.0
      #
      for l in range(0,len(tup)):
         #
         if (set([j+1]) <= set(tup[l][0])):
            #
            tmp += tup[l][1]
      #
      orb_contrib.append(tmp)
   #
   return molecule

#def inc_corr_order(k,n,e_vec,e_inc):
#   #
#   e_sum = 0.0
#   #
#   for m in range(1,k+1):
#      e_sum += (-1)**(m) * (1.0 / math.factorial(m)) * prefactor(n,k,m-1) * e_vec[(k-m)-1]
#   e_inc.append(e_vec[k-1]+e_sum)
#   #
#   return e_inc
#
#def prefactor(n,order,m):
#   #
#   pre = 1
#   #
#   for i in range(m,-1,-1):
#      pre = pre * (n - order + i)
#   #
#   return pre

