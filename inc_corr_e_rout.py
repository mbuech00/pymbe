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
      # append tuple list and generate all tuples at order k
      #
      molecule['tuple'][0].append([])
      #
      inc_corr_orb_rout.orb_generator(molecule,molecule['domain'][0],molecule['tuple'][0][k-1],molecule['l_limit'][0],k)
      #
      # determine number of tuples at order k
      #
      molecule['n_tuples'][0].append(len(molecule['tuple'][0][k-1]))
      #
      # print status header
      #
      print_status_header(molecule,molecule['n_tuples'][0],k)
      #
      # check for convergence
      #
      if (molecule['n_tuples'][0][k-1] == 0):
         #
         return molecule
      #
      # start time
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
         # write string
         #
         inc_corr_orb_rout.orb_string(molecule,molecule['l_limit'][0],molecule['u_limit'][0],molecule['tuple'][0][k-1][i][0])
         #
         # run correlated calc
         #
         print(str(molecule['string']))
         inc_corr_gen_rout.run_calc_corr(molecule,molecule['string'],False)
         #
         # write tuple energy
         #
         molecule['tuple'][0][k-1][i].append(molecule['e_tmp'])
         #
         # print result
         #
         print_result(i,molecule['tuple'][0][k-1][i])
         #
         # error check
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
         e_orb_rout(molecule,molecule['tuple'][0],molecule['orbital'][0],molecule['l_limit'][0],molecule['u_limit'][0])
         #
         molecule['excl_list'][0][:] = []
         #
         inc_corr_orb_rout.excl_rout(molecule,molecule['tuple'][0],molecule['orbital'][0],molecule['thres'][0],molecule['excl_list'][0])
         #
         # update domains
         #
         inc_corr_orb_rout.update_domains(molecule['domain'][0],molecule['l_limit'][0],molecule['excl_list'][0])
      #
      # calculate theoretical number of tuples at order k
      #
      inc_corr_orb_rout.n_theo_tuples(molecule['n_tuples'][0][0],k,molecule['theo_work'][0])
      #
      # collect time
      #
      molecule['time'][0].append(timer()-start)
      #
      # print status end and domain updates
      #
      print_status_end(molecule,k,molecule['e_fin'][0],molecule['time'][0],molecule['n_tuples'][0])
      #
      if (k >= 2):
         #
         print_update(molecule,molecule['tuple'][0],molecule['n_tuples'][0],molecule['domain'][0],k,molecule['l_limit'][0],molecule['u_limit'][0])
      #
      # check for maximum order
      #
      if (k == molecule['n_tuples'][0][0]):
         #
         print('')
         print('')
         #
         break
   #
   return molecule

def inc_corr_dual_exp(molecule):
   #
   for k in range(1,molecule['u_limit'][0]+1):
      #
      # append tuple list and generate all tuples at order k
      #
      molecule['tuple'][0].append([])
      #
      inc_corr_orb_rout.orb_generator(molecule,molecule['domain'][0],molecule['tuple'][0][k-1],molecule['l_limit'][0],k)
      #
      # determine number of tuples at order k
      #
      molecule['n_tuples'][0].append(len(molecule['tuple'][0][k-1]))
      #
      # print status header (for outer expansion)
      #
      print_status_header(molecule,molecule['n_tuples'][0],k)
      #
      # check for convergence (for outer expansion)
      #
      if (molecule['n_tuples'][0][k-1] == 0):
         #
         return molecule
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
            # append tuple list and generate all tuples at order l
            #
            molecule['tuple'][1].append([])
            #
            inc_corr_orb_rout.orb_generator(molecule,molecule['domain'][1],molecule['tuple'][1][l-1],molecule['l_limit'][1],l)
            #
            # determine number of tuples at order l
            #
            molecule['n_tuples'][1].append(len(molecule['tuple'][1][l-1]))
            #
            # check for convergence (for inner expansion)
            #
            if (molecule['n_tuples'][1][l-1] == 0):
               #
               molecule['tuple'][0][k-1][i].append(molecule['e_fin'][1][-1])
               #
               print_result(i,molecule['tuple'][0][k-1][i])
               #
               molecule['n_tuples'][1].pop()
               #
               break
            # 
            # run the calculations (for inner expansion)
            #
            for j in range(0,molecule['n_tuples'][1][l-1]):
               #
               # write string
               #
               inc_corr_orb_rout.orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][0][k-1][i][0]+molecule['tuple'][1][l-1][j][0])
               #
               # run correlated calc
               #
               print(str(molecule['string']))
               inc_corr_gen_rout.run_calc_corr(molecule,molecule['string'],False)
               #
               # write tuple energy
               #
               molecule['tuple'][1][l-1][j].append(molecule['e_tmp'])
               #
               # error check
               #
               if (molecule['error'][0][-1]):
                  #
                  return molecule
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
               e_orb_rout(molecule,molecule['tuple'][1],molecule['orbital'][1],molecule['l_limit'][1],molecule['u_limit'][1])
               #
               molecule['excl_list'][1][:] = []
               #
               inc_corr_orb_rout.excl_rout(molecule,molecule['tuple'][1],molecule['orbital'][1],molecule['thres'][1],molecule['excl_list'][1])
               #
               # update domains (for inner expansion)
               #
               inc_corr_orb_rout.update_domains(molecule['domain'][1],molecule['l_limit'][1],molecule['excl_list'][1])
            #
            # calculate theoretical number of tuples at order l (for inner expansion)
            #
            inc_corr_orb_rout.n_theo_tuples(molecule['n_tuples'][1][0],l,molecule['theo_work'][1])
            #
            # check for maximum order (for inner expansion)
            #
            if (l == molecule['u_limit'][1]):
               #
               molecule['tuple'][0][k-1][i].append(molecule['e_fin'][1][-1])
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
         e_orb_rout(molecule,molecule['tuple'][0],molecule['orbital'][0],molecule['l_limit'][0],molecule['u_limit'][0])
         #
         molecule['excl_list'][0][:] = []
         #
         inc_corr_orb_rout.excl_rout(molecule,molecule['tuple'][0],molecule['orbital'][0],molecule['thres'][0],molecule['excl_list'][0])
         #
         # update domains (for outer expansion)
         #
         inc_corr_orb_rout.update_domains(molecule['domain'][0],molecule['l_limit'][0],molecule['excl_list'][0])
      #
      # calculate theoretical number of tuples at order k (for outer expansion)
      #
      inc_corr_orb_rout.n_theo_tuples(molecule['n_tuples'][0][0],k,molecule['theo_work'][0])
      #
      # collect time (for outer expansion)
      #
      molecule['time'][0].append(timer()-start_out)
      #
      # print status end (for outer expansion)
      #
      print_status_end(molecule,k,molecule['e_fin'][0],molecule['time'][0],molecule['n_tuples'][0])
      #
      # print results (for inner expansion)
      #
      print_inner_result(molecule)
      #
      # print domain updates (for outer expansion)
      #
      if (k >= 2):
         #
         print_update(molecule,molecule['tuple'][0],molecule['n_tuples'][0],molecule['domain'][0],k,molecule['l_limit'][0],molecule['u_limit'][0])
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

def e_orb_rout(molecule,tup,orb,l_limit,u_limit):
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
   if (molecule['exp'] == 'OCC'):
      #
      molecule['l_limit'] = [0]
      molecule['u_limit'] = [molecule['nocc']]
      #
      molecule['domain'] = [molecule['occ_domain']]
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      molecule['l_limit'] = [molecule['nocc']]
      molecule['u_limit'] = [molecule['nvirt']]
      #
      molecule['domain'] = [molecule['virt_domain']]
      #
   #
   elif (molecule['exp'] == 'COMB-OV'):
      #
      molecule['l_limit'] = [0,molecule['nocc']]
      molecule['u_limit'] = [molecule['nocc'],molecule['nvirt']]
      #
      molecule['domain'] = [molecule['occ_domain'],molecule['virt_domain']]
      #
      molecule['e_diff_in'] = []
      #
      molecule['rel_work_in'] = []
   #
   elif (molecule['exp'] == 'COMB-VO'):
      #
      molecule['l_limit'] = [molecule['nocc'],0]
      molecule['u_limit'] = [molecule['nvirt'],molecule['nocc']]
      #
      molecule['domain'] = [molecule['virt_domain'],molecule['occ_domain']]
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
   molecule['excl_list'] = [[],[]]
   #
   molecule['theo_work'] = [[],[]]
   #
   molecule['time'] = [[],[]]
   #
   return molecule

def print_status_header(molecule,n_tup,order):
   #
   print('')
   print('')
   print(' --------------------------------------------------------------------------------------------')
   #
   if (n_tup[order-1] == 0):
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  has no contributions --- *** calculation has converged ***'.format(order,n_tup[0]))
      print(' --------------------------------------------------------------------------------------------')
      print('')
      print('')
   #
   else:
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  started  ---  {2:d}  correlated tuples in total'.format(order,n_tup[0],n_tup[-1]))
      print(' --------------------------------------------------------------------------------------------')
   #
   return

def print_status_end(molecule,order,e_fin,time,n_tup):
   #
   print(' --------------------------------------------------------------------------------------------')
   #
   if (order == 1):
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  done in {2:10.2e} seconds'.format(order,n_tup[0],time[order-1]))
   #
   else:
      #
      print(' STATUS-MACRO:  order =  {0:>d} / {1:<d}  done in {2:10.2e} seconds  ---  diff =  {3:9.4e}'.\
                       format(order,n_tup[0],time[order-1],e_fin[order-1]-e_fin[order-2]))
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

