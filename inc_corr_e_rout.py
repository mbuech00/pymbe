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
   # initialize domains
   #
   inc_corr_orb_rout.init_domains(molecule)
   #
   # initialize variable and lists
   #
   inc_corr_prepare(molecule)   
   #
   # run the specified calculation
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'VIRT')):
      #
      inc_corr_mono_exp(molecule)
   #
   elif ((molecule['exp'] == 'COMB-OV') or (molecule['exp'] == 'COMB-VO')):
      #
      inc_corr_dual_exp(molecule)
   #
   return molecule

def inc_corr_mono_exp(molecule):
   #
   for k in range(1,molecule['u_limit'][0]+1):
      #
      # generate the tuples at order k
      #
      molecule['drop_string'][0][:] = []
      #
      molecule['generate_drop'][0](molecule['l_limit'][0]+1,1,k,molecule,molecule['list_drop'][0],molecule['drop_string'][0],molecule['n_tuples'][0])
      #
      # print status header
      #
      print_status_header(molecule,molecule['n_tuples'][0],k,molecule['u_limit'][0])
      #
      # check for convergence
      #
      if (len(molecule['n_tuples'][0]) < k):
         #
         return molecule
      #
      # init/append lists and start time
      #
      molecule['tuple'][0].append([])
      #
      molecule['incl_list'][0][:] = []
      #
      start = timer()
      #
      # print result header
      #
      print_result_header()
      #
      # run the calculations
      #
      for i in range(0,molecule['n_tuples'][0][k-1]):
         #
         inc_corr_gen_rout.run_calc_corr(molecule,molecule['drop_string'][0][i],False)
         #
         molecule['incl_list'][0].append([])
         #
         inc_corr_orb_rout.orbs_incl(molecule['drop_string'][0][i],molecule['incl_list'][0][i],molecule['l_limit'][0],molecule['u_limit'][0])
         #
         molecule['tuple'][0][k-1].append([molecule['incl_list'][0][i],molecule['e_tmp']])
         #
         print_result(i,molecule['tuple'][0][k-1][i])
         #
         if (molecule['error'][0][-1]):
            #
            return molecule
      #
      # print result end
      #
      print_result_end()
      #
      # calculate the energy at order k
      #
      inc_corr_order(k,molecule['n_tuples'][0],molecule['tuple'][0],molecule['e_fin'][0])
      #
      # set up entanglement and exclusion lists
      #
      if (k >= 2):
         #
         molecule['orbital'][0].append([])
         #
         orbital_rout(molecule,molecule['tuple'][0],molecule['orbital'][0],molecule['l_limit'][0],molecule['u_limit'][0])
         #
         molecule['excl_list'][0][:] = []
         #
         inc_corr_orb_rout.excl_rout(molecule['orbital'][0],molecule['thres'][0],molecule['excl_list'][0])
      #
      # update domains
      #
      inc_corr_orb_rout.update_domains(molecule,molecule['tuple'][0],molecule['domain'][0],molecule['thres'][0],molecule['l_limit'][0],molecule['excl_list'][0])
      #
      # calculate theoretical number of tuples at order k
      #
      inc_corr_orb_rout.n_theo_tuples(molecule['u_limit'][0],k,molecule['theo_work'][0])
      #
      # collect time
      #
      molecule['time'][0].append(timer()-start)
      #
      # print status end and domain updates
      #
      print_status_end(molecule,k,molecule['e_fin'][0],molecule['time'][0],molecule['u_limit'][0])
      #
      print_update(molecule,molecule['tuple'][0],molecule['n_tuples'][0],molecule['domain'][0],k,molecule['l_limit'][0],molecule['u_limit'][0])
   #
   return molecule

def inc_corr_dual_exp(molecule):
   #
   for k in range(1,molecule['u_limit'][0]+1):
      #
      # generate the tuples at order k (for outer expansion)
      #
      molecule['drop_string'][0][:] = []
      #
      molecule['generate_drop'][0](molecule['l_limit'][0]+1,1,k,molecule,molecule['list_drop'][0],molecule['drop_string'][0],molecule['n_tuples'][0])
      #
      # print status header (for outer expansion)
      #
      print_status_header(molecule,molecule['n_tuples'][0],k,molecule['u_limit'][0])
      #
      # check for convergence (for outer expansion)
      #
      if (len(molecule['n_tuples'][0]) < k):
         #
         return molecule
      #
      # init/append lists and start time (for outer expansion)
      #
      molecule['tuple'][0].append([])
      #
      molecule['incl_list'][0][:] = []
      #
      # init time, energy diff, and relative work (for inner expansion)
      #
      molecule['time'][1][:] = []
      #
      molecule['e_diff_in'][:] = []
      #
      molecule['rel_work_in'][:] = []
      #
      # start time (for outer expansion)
      #
      start_out = timer()
      #
      # print result header (for outer expansion)
      #
      print_result_header()
      #
      # run the calculations (for outer expansion)
      #
      for i in range(0,molecule['n_tuples'][0][k-1]):
         #
         molecule['e_fin'][1][:] = []
         #
         molecule['tuple'][1][:] = []
         #
         molecule['n_tuples'][1][:] = []
         #
         molecule['theo_work'][1][:] = []
         #
         # re-initialize the inner domain
         #
         inc_corr_orb_rout.reinit_domains(molecule,molecule['domain'][1])
         #
         # start time (for inner expansion)
         #
         start_in = timer()
         #
         for l in range(1,molecule['u_limit'][1]+1):
            #
            for a in range(molecule['l_limit'][1],molecule['l_limit'][1]+molecule['u_limit'][1]):
               #
               molecule['list_drop'][1][a] = a+1
            #
            molecule['drop_string'][1][:] = []
            #
            molecule['generate_drop'][1](molecule['l_limit'][1]+1,1,l,molecule,molecule['list_drop'][1],molecule['drop_string'][1],molecule['n_tuples'][1])
            #
            # check for convergence (for inner expansion)
            #
            if (len(molecule['n_tuples'][1]) < l):
               #
               molecule['incl_list'][0].append([])
               #
               inc_corr_orb_rout.orbs_incl(molecule['drop_string'][0][i],molecule['incl_list'][0][i],molecule['l_limit'][0],molecule['u_limit'][0])
               #
               molecule['tuple'][0][k-1].append([molecule['incl_list'][0][i],molecule['e_fin'][1][-1]])
               #
               print_result(i,molecule['tuple'][0][k-1][i])
               #
               break
            # 
            # init/append lists (for inner expansion)
            #
            molecule['tuple'][1].append([])
            #
            molecule['incl_list'][1][:] = []
            #
            # run the calculations (for inner expansion)
            #
            for j in range(0,molecule['n_tuples'][1][l-1]):
               #
               collect_drop_strings(molecule,i,j)
               #
               inc_corr_gen_rout.run_calc_corr(molecule,molecule['collect_string'],False)
               #
               molecule['incl_list'][1].append([])
               #
               inc_corr_orb_rout.orbs_incl(molecule['collect_string'],molecule['incl_list'][1][j],molecule['l_limit'][1],molecule['u_limit'][1])
               #
               molecule['tuple'][1][l-1].append([molecule['incl_list'][1][j],molecule['e_tmp']])
               #
               if (molecule['error'][0][-1]):
                  #
                  return molecule
#            print('tuple = '+str(molecule['tuple'][1][l-1]))
#            print('')
            #
            # calculate the energy at order l (for inner expansion)
            #
            inc_corr_order(l,molecule['n_tuples'][1],molecule['tuple'][1],molecule['e_fin'][1])
            #
            # set up entanglement and exclusion lists (for inner expansion)
            #
            if (l >= 2):
               #
               molecule['orbital'][1].append([])
               #
               orbital_rout(molecule,molecule['tuple'][1],molecule['orbital'][1],molecule['l_limit'][1],molecule['u_limit'][1])
               #
               molecule['excl_list'][1][:] = []
               #
               inc_corr_orb_rout.excl_rout(molecule['orbital'][1],molecule['thres'][1],molecule['excl_list'][1])
            #
            # update domains (for inner expansion)
            #
            inc_corr_orb_rout.update_domains(molecule,molecule['tuple'][1],molecule['domain'][1],molecule['thres'][1],molecule['l_limit'][1],molecule['excl_list'][1])
            #
            # calculate theoretical number of tuples at order l (for inner expansion)
            #
            inc_corr_orb_rout.n_theo_tuples(molecule['u_limit'][1],l,molecule['theo_work'][1])
            #
            # check for maximum order (for inner expansion)
            #
            if (l == molecule['u_limit'][1]):
               #
               molecule['incl_list'][0].append([])
               #
               inc_corr_orb_rout.orbs_incl(molecule['drop_string'][0][i],molecule['incl_list'][0][i],molecule['l_limit'][0],molecule['u_limit'][0])
               #
               molecule['tuple'][0][k-1].append([molecule['incl_list'][0][i],molecule['e_fin'][1][-1]])
               #
               print_result(i,molecule['tuple'][0][k-1][i])
               #
               break
         #
         # collect time, energy diff, and relative work (for inner expansion)
         #
         molecule['time'][1].append(timer()-start_in)
         #
         molecule['e_diff_in'].append(molecule['e_fin'][1][-1]-molecule['e_fin'][1][-2])
         #
         molecule['rel_work_in'].append([])
         #
         for m in range(0,len(molecule['n_tuples'][1])):
            #
            molecule['rel_work_in'][-1].append((float(molecule['n_tuples'][1][m])/float(molecule['theo_work'][1][m]))*100.00)
            #
#         print('e_fin = '+str(molecule['e_fin'][1]))
#         print('')
      #
      # print result end (for outer expansion)
      #
      print_result_end()
      #
      # calculate the energy at order k (for outer expansion)
      #
      inc_corr_order(k,molecule['n_tuples'][0],molecule['tuple'][0],molecule['e_fin'][0])
      #
      # set up entanglement and exclusion lists (for outer expansion)
      #
      if (k >= 2):
         #
         molecule['orbital'][0].append([])
         #
         orbital_rout(molecule,molecule['tuple'][0],molecule['orbital'][0],molecule['l_limit'][0],molecule['u_limit'][0])
         #
         molecule['excl_list'][0][:] = []
         #
         inc_corr_orb_rout.excl_rout(molecule['orbital'][0],molecule['thres'][0],molecule['excl_list'][0])
      #
      # update domains (for outer expansion)
      #
      inc_corr_orb_rout.update_domains(molecule,molecule['tuple'][0],molecule['domain'][0],molecule['thres'][0],molecule['l_limit'][0],molecule['excl_list'][0])
      #
      # calculate theoretical number of tuples at order k (for outer expansion)
      #
      inc_corr_orb_rout.n_theo_tuples(molecule['u_limit'][0],k,molecule['theo_work'][0])
      #
      # collect time (for outer expansion)
      #
      molecule['time'][0].append(timer()-start_out)
      #
      # print status end (for outer expansion)
      #
      print_status_end(molecule,k,molecule['e_fin'][0],molecule['time'][0],molecule['u_limit'][0])
      #
      # print results (for inner expansion)
      #
      print_inner_result(molecule)
      #
      # print domain updates (for outer expansion)
      #
      print_update(molecule,molecule['tuple'][0],molecule['n_tuples'][0],molecule['domain'][0],k,molecule['l_limit'][0],molecule['u_limit'][0])
   #
   return molecule

def collect_drop_strings(molecule,i,j):
   #
   if (molecule['exp'] == 'COMB-OV'):
      #
      if (molecule['drop_string'][0][i] == '\n'):
         #
         molecule['collect_string'] = 'DROP_MO='+molecule['drop_string'][1][j][1:]
      #
      else:
         #
         molecule['collect_string'] = molecule['drop_string'][0][i]+molecule['drop_string'][1][j]
   #
   elif (molecule['exp'] == 'COMB-VO'):
      #
      if (molecule['drop_string'][1][j] == '\n'):
         #
         molecule['collect_string'] = 'DROP_MO='+molecule['drop_string'][0][i][1:]
      #
      else:
         #
         molecule['collect_string'] = molecule['drop_string'][1][j]+molecule['drop_string'][0][i]
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

def orbital_rout(molecule,tup,orb,l_limit,u_limit):
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

def inc_corr_prepare(molecule):
   #
   molecule['list_drop'] = [list(range(1,(molecule['nocc']+molecule['nvirt'])+1)),list(range(1,(molecule['nocc']+molecule['nvirt'])+1))]
   #
   if (molecule['exp'] == 'OCC'):
      #
      molecule['l_limit'] = [molecule['core']]
      molecule['u_limit'] = [molecule['nocc']-molecule['core']]
      #
      molecule['domain'] = [molecule['occ_domain']]
      #
      molecule['generate_drop'] = [inc_corr_orb_rout.generate_drop_occ]
      #
      for i in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
         #
         molecule['list_drop'][0][i] = 0
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      molecule['l_limit'] = [molecule['nocc']]
      molecule['u_limit'] = [molecule['nvirt']]
      #
      molecule['domain'] = [molecule['virt_domain']]
      #
      molecule['generate_drop'] = [inc_corr_orb_rout.generate_drop_virt]
      #
      for i in range(molecule['core'],molecule['nocc']):
         #
         molecule['list_drop'][0][i] = 0
   #
   elif (molecule['exp'] == 'COMB-OV'):
      #
      molecule['l_limit'] = [molecule['core'],molecule['nocc']]
      molecule['u_limit'] = [molecule['nocc']-molecule['core'],molecule['nvirt']]
      #
      molecule['domain'] = [molecule['occ_domain'],molecule['virt_domain']]
      #
      molecule['generate_drop'] = [inc_corr_orb_rout.generate_drop_occ,inc_corr_orb_rout.generate_drop_virt]
      #
      for i in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
         #
         molecule['list_drop'][0][i] = 0
      #
      for i in range(molecule['core'],molecule['nocc']):
         #
         molecule['list_drop'][1][i] = 0
      #
      molecule['e_diff_in'] = []
      #
      molecule['rel_work_in'] = []
   #
   elif (molecule['exp'] == 'COMB-VO'):
      #
      molecule['l_limit'] = [molecule['nocc'],molecule['core']]
      molecule['u_limit'] = [molecule['nvirt'],molecule['nocc']-molecule['core']]
      #
      molecule['domain'] = [molecule['virt_domain'],molecule['occ_domain']]
      #
      molecule['generate_drop'] = [inc_corr_orb_rout.generate_drop_virt,inc_corr_orb_rout.generate_drop_occ]
      #
      for i in range(molecule['core'],molecule['nocc']):
         #
         molecule['list_drop'][0][i] = 0
      #
      for i in range(molecule['nocc'],molecule['nocc']+molecule['nvirt']):
         #
         molecule['list_drop'][1][i] = 0
      #
      molecule['e_diff_in'] = []
      #
      molecule['rel_work_in'] = []
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['tuple'] = [[],[]]
   #
   molecule['n_tuples'] = [[],[]]
   #
   molecule['orbital'] = [[],[]]
   #
   molecule['e_fin'] = [[],[]]
   #
   molecule['drop_string'] = [[],[]]
   #
   molecule['incl_list'] = [[],[]]
   #
   molecule['excl_list'] = [[],[]]
   #
   molecule['theo_work'] = [[],[]]
   #
   molecule['time'] = [[],[]]
   #
   return molecule

def print_status_header(molecule,n_tup,order,u_limit):
   #
   print('')
   print('')
   print(' --------------------------------------------------------------------------------------------')
   #
   if (len(n_tup) < order):
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  has no contributions --- *** calculation has converged ***'.format(order,u_limit))
      print(' --------------------------------------------------------------------------------------------')
      print('')
      print('')
   #
   else:
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  started  ---  {2:d}  correlated tuples in total'.format(order,u_limit,n_tup[-1]))
      print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_status_end(molecule,order,e_fin,time,u_limit):
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   if (order == 1):
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  done in {2:10.2e} seconds'.format(order,u_limit,time[order-1]))
   #
   else:
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}'.\
                       format(order,u_limit,time[order-1],e_fin[order-1]-e_fin[order-2]))
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_result_header():
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' RESULT-MACRO:     tuple    |    abs. energy    |    corr. orbs.')
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_result(tup_num,tup):
   #
   print(' RESULT-MACRO:  {0:>6d}           {1:> 8.4e}         {2:<}'.\
                    format(tup_num+1,tup[1],tup[0]))
   #
   return

def print_result_end():
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_inner_result(molecule):
   #
   rel_work_in = []
   #
   for m in range(0,len(molecule['rel_work_in'])):
      #
      rel_work_in.append([])
      #
      for n in range(0,len(molecule['rel_work_in'][m])):
         #
         rel_work_in[m].append('{0:.2f}'.format(molecule['rel_work_in'][m][n]))
   #
   print(' --------------------------------------------------------------------------------------------')
   print(' RESULT-MICRO:     tuple    |   abs. energy diff.   |    relat. no. tuples (in %)')
   print(' --------------------------------------------------------------------------------------------')
   #
   for i in range(0,molecule['n_tuples'][0][-1]):
      #
      print(' RESULT-MICRO:  {0:>6d}            {1:> 8.4e}            '.\
                       format(i+1,molecule['e_diff_in'][i])+'[{0:<}]'.format(', '.join(str(idx) for idx in rel_work_in[i])))
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_update(molecule,tup,n_tup,domain,order,l_limit,u_limit):
   #
   count = []
   #
   for j in range(0,u_limit):
      #
      if ((len(domain[j][-2]) >= 1) and (float(len(domain[j][-1]))/float(len(domain[j][-2])) != 1.0)):
         #
         count.append(True)
      #
      else:
         #
         count.append(False)
   #
   if (any(count)):
      #
      print(' --------------------------------------------------------------------------------------------')
      print(' UPDATE-MACRO:   orb. domain  |  relat. red. (in %)  |   total red. (in %)  |  screened orbs. ')
      print(' --------------------------------------------------------------------------------------------')
      #
      for j in range(0,u_limit):
         #
         if (count[j]):
            #
            print(' UPDATE-MACRO:     {0:>5}              {1:>6.2f}                 {2:>6.2f}            {3:<}'.\
                          format([(j+l_limit)+1],\
                                 (1.0-float(len(domain[j][-1]))/float(len(domain[j][-2])))*100.00,\
                                 (1.0-float(len(domain[j][-1]))/float(len(domain[j][0])))*100.00,\
                                 sorted(list(set(domain[j][-2])-set(domain[j][-1])))))
      #
      print(' --------------------------------------------------------------------------------------------')
   #
   return

