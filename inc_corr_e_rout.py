# -*- coding: utf-8 -*
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

def inc_corr_main(molecule):
   #
   inc_corr_prepare(molecule)   
   #
   if (molecule['exp'] == 'COMB'):
      #
      u_limit_1 = molecule['nocc']-molecule['core']
      u_limit_2 = molecule['nvirt']
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
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'VIRT')):
      #
      for k in range(1,molecule['u_limit']+1):
         #
         # generate the tuples at order k
         #
         molecule['drop_string'][:] = []
         #
         molecule['generate_drop'](molecule['l_limit']+1,1,k,molecule,molecule['list_drop'],molecule['drop_string'],molecule['n_tuples'])
         #
         # print status header
         #
         print_status_header(molecule,k,molecule['u_limit'])
         #
         # check for convergence
         #
         if (len(molecule['n_tuples']) < k):
            #
            return molecule
         #
         # init/append lists and start time
         #
         molecule['tuple'].append([])
         #
         molecule['incl_list'][:] = []
         #
         start = timer()
         #
         # run the calculations
         #
         for i in range(0,molecule['n_tuples'][k-1]):
            #
            inc_corr_gen_rout.run_calc_corr(molecule,molecule['drop_string'][i],False)
            #
            molecule['incl_list'].append([])
            #
            inc_corr_orb_rout.orbs_incl(molecule,molecule['drop_string'][i],molecule['incl_list'][i],False)
            #
            molecule['tuple'][k-1].append([molecule['incl_list'][i],molecule['e_tmp']])
            #
            if (molecule['error'][0][-1]):
               #
               return molecule
         #
         # calculate the energy at order k
         #
         inc_corr_order(k,molecule['n_tuples'],molecule['tuple'],molecule['e_fin'])
         #
         # set up entanglement and exclusion lists
         #
         if (k >= 2):
            #
            molecule['orbital'].append([])
            #
            orbital_rout(molecule,molecule['tuple'],molecule['orbital'])
            #
            molecule['excl_list'][:] = []
            #
            inc_corr_orb_rout.excl_rout(molecule['orbital'],molecule['thres'][0],molecule['excl_list'])
         #
         # update domains
         #
         inc_corr_orb_rout.update_domains(molecule,molecule['domain'],molecule['thres'][0],molecule['excl_list'])
         #
         # calculate theoretical number of tuples at order k
         #
         inc_corr_orb_rout.n_theo_tuples(molecule['u_limit'],k,molecule['theo_work'])
         #
         # collect time
         #
         molecule['time'].append(timer()-start)
         #
         # print status end, results, and domain updates
         #
         print_status_end(molecule,k,molecule['u_limit'])
         #
         print_result(molecule,k)
         #
         print_update(molecule,molecule['domain'],k,molecule['l_limit'],molecule['u_limit'])
   #
   elif (molecule['exp'] == 'COMB'):
      #
      for k in range(1,u_limit_1+1):
         #
         start = timer()
         #
         molecule['drop_string'][:] = []
         #
         inc_corr_orb_rout.generate_drop_occ(molecule['core']+1,1,k,molecule,molecule['list_drop'],molecule['drop_string'],molecule['n_tuples'])
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
         molecule['incl_list'][:] = []
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
                  molecule['list_drop'][a] = a+1
               #
               drop_string_comb[:] = []
               #
               inc_corr_orb_rout.generate_drop_virt(molecule['nocc']+1,1,l,molecule,molecule['list_drop'],drop_string_comb,n_tuples_comb)
               #
               tuple_comb.append([])
               #
               incl_list_comb[:] = []
               #
               for i in range(0,n_tuples_comb[l-1]):
                  #
                  if (molecule['drop_string'][j] == '\n'):
                     #
                     string = 'DROP_MO='+drop_string_comb[i][1:]
                  #
                  else:
                     #
                     string = molecule['drop_string'][j]+drop_string_comb[i]
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
                  molecule['incl_list'].append([])
                  #
                  inc_corr_orb_rout.orbs_incl(molecule,molecule['drop_string'][j],molecule['incl_list'][j],False)
                  #
                  molecule['tuple'][k-1].append([molecule['incl_list'][j],e_fin_comb[l-1],[]])
                  #
#                  orbital_rout(molecule,tuple_comb,True)
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
#            molecule['orbital'] = []
#            orbital_rout(molecule,molecule['tuple'],False)
            #
            return molecule
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

def orbital_rout(molecule,tup,orb):
   #
   if (molecule['exp'] == 'OCC'):
      #
      l_limit = molecule['core']
      u_limit = molecule['nocc']
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      l_limit = molecule['nocc']
      u_limit = molecule['nocc']+molecule['nvirt']
   #
   for i in range(l_limit,u_limit):
      #
      orb[-1].append([[i+1]])
      #
      for j in range(l_limit,u_limit):
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
   for i in range(l_limit,u_limit):
      #
      e_sum = 0.0
      #
      for j in range(0,len(orb)):
         #
         for k in range(l_limit,u_limit-1):
            #
            e_sum += orb[j][i-l_limit][(k-l_limit)+1][1][0]
      #
      for j in range(0,len(orb)):
         #
         for k in range(l_limit,u_limit-1):
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

def inc_corr_prepare(molecule):
   #
   molecule['list_drop'] = list(range(1,(molecule['nocc']+molecule['nvirt'])+1))
   #
   if (molecule['exp'] == 'OCC'):
      #
      molecule['l_limit'] = molecule['core']
      molecule['u_limit'] = molecule['nocc']-molecule['core']
      #
      molecule['domain'] = molecule['occ_domain']
      #
      molecule['generate_drop'] = inc_corr_orb_rout.generate_drop_occ
      #
      for i in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
         #
         molecule['list_drop'][i] = 0
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      molecule['l_limit'] = molecule['nocc']
      molecule['u_limit'] = molecule['nvirt']
      #
      molecule['domain'] = molecule['virt_domain']
      #
      molecule['generate_drop'] = inc_corr_orb_rout.generate_drop_virt
      #
      for i in range(molecule['core'],molecule['nocc']):
         #
         molecule['list_drop'][i] = 0
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['tuple'] = []
   #
   molecule['n_tuples'] = []
   #
   molecule['orbital'] = []
   #
   molecule['e_fin'] = []
   #
   molecule['drop_string'] = []
   #
   molecule['incl_list'] = []
   #
   molecule['excl_list'] = []
   #
   molecule['theo_work'] = []
   #
   molecule['time'] = []
   #
   return molecule

def print_status_header(molecule,order,u_limit):
   #
   if (len(molecule['n_tuples']) < order):
      #
      print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  has no contributions --- calculation has converged'.format(order,u_limit))
      print(' --------------------------------------------------------------------------------------')
      print('')
   #
   print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  started'.format(order,u_limit))
   print(' -------------------------------------------')
   #
   return

def print_status_end(molecule,order,u_limit):
   #
   if (order == 1):
      #
      print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds'.format(order,u_limit,molecule['time'][order-1]))
      print(' --------------------------------------------------------------')
   #
   else:
      #
      print(' STATUS-MACRO:  order = {0:4d} / {1:4d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}'.\
                       format(order,u_limit,molecule['time'][order-1],molecule['e_fin'][order-1]-molecule['e_fin'][order-2]))
      print(' ----------------------------------------------------------------------------------------')
      print(' ----------------------------------------------------------------------------------------')
   #
   return

def print_result(molecule,order):
   #
   print(' RESULT-MACRO:     tuple    |    abs. energy    |    corr. orbs.')
   print(' ----------------------------------------------------------------------------------------')
   #
   for i in range(0,molecule['n_tuples'][order-1]):
      #
      print(' RESULT-MACRO:  {0:3d} / {1:3d}        {2:9.4e}          {3:}'.\
                       format(i+1,molecule['n_tuples'][order-1],molecule['tuple'][order-1][i][1],molecule['tuple'][order-1][i][0]))
   #
   print(' ----------------------------------------------------------------------------------------')
   print(' ----------------------------------------------------------------------------------------')
   #
   return

def print_update(molecule,domain,order,l_limit,u_limit):
   #
   print(' UPDATE-MACRO:   orb. domain  |  relat. red.  |   total red.  |  screened orbs. ')
   print(' ----------------------------------------------------------------------------------------')
   #
   for j in range(0,u_limit):
      #
      cont = False
      #
      for l in range(0,molecule['n_tuples'][order-1]):
         #
         if (set([(j+l_limit)+1]) < set(molecule['tuple'][order-1][l][0])):
            #
            cont = True
      #
      if (cont or (order == 1)):
         #
         print(' UPDATE-MACRO:     {0:}             {1:5.2f}            {2:5.2f}         {3:}'.\
                       format([(j+l_limit)+1],\
                              (1.0-float(len(domain[j][-1]))/float(len(domain[j][-2]))),\
                              (1.0-float(len(domain[j][-1]))/float(len(domain[j][0]))),\
                              sorted(list(set(domain[j][-2])-set(domain[j][-1])))))
   #
   print('')
   print('')
   #
   return

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

