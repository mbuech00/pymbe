# -*- coding: utf-8 -*
#!/usr/bin/env python

#
# energy-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import sys
import copy
from timeit import default_timer as timer

import inc_corr_gen_rout
import inc_corr_orb_rout
import inc_corr_utils
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
      # call mono expansion kernel
      #
      inc_corr_mono_exp_kernel(molecule,molecule['prim_tuple'][0],molecule['prim_domain'],molecule['prim_n_tuples'][0],molecule['prim_time'][0],k)
      #
      # check for convergence
      #
      if (molecule['conv'][-1] or (k == molecule['u_limit'][0])):
         #
         print('')
         print('')
         #
         return molecule
      #
      # orbital screening
      #
      if (k >= 2):
         #
         inc_corr_orb_rout.orb_screen_rout(molecule,molecule['e_inc'][0],molecule['prim_orbital'][0],molecule['prim_domain'],\
                                           molecule['prim_thres'][0],molecule['l_limit'][0],molecule['u_limit'][0],'MACRO')
      #
      # print status end
      #
      inc_corr_utils.print_status_end(molecule,k,molecule['e_tot'][0],molecule['prim_time'][0],molecule['prim_n_tuples'][0],'MACRO')
   #
   return molecule
 
def inc_corr_mono_exp_kernel(molecule,tup,dom,n_tup,time,k):
   #
   # define level
   #
   level = 'MACRO'
   #
   # generate all tuples at order k
   #
   tup.append([])
   #
   inc_corr_orb_rout.orb_generator(molecule,dom,tup[k-1],molecule['l_limit'][0],k)
   #
   # determine number of tuples at order k
   #
   n_tup.append(len(tup[k-1]))
   #
   # calculate theoretical number of tuples at order k
   #
   inc_corr_orb_rout.n_theo_tuples(n_tup[0],k,molecule['theo_work'][0])
   #
   # print status header
   #
   inc_corr_utils.print_status_header(molecule,n_tup,k,level)
   #
   # check for convergence
   #
   if (molecule['conv'][-1]):
      #
      return molecule
   #
   # start time
   #
   start = timer()
   #
   # run the calculations
   #
   for i in range(0,n_tup[k-1]):
      #
      # write string
      #
      inc_corr_orb_rout.orb_string(molecule,molecule['l_limit'][0],molecule['u_limit'][0],tup[k-1][i][0])
      #
      # run correlated calc
      #
      inc_corr_gen_rout.run_calc_corr(molecule,molecule['string'],level)
      #
      # write tuple energy
      #
      tup[k-1][i].append(molecule['e_tmp'])
      #
      # print status
      #
      inc_corr_utils.print_status(float(i+1)/float(n_tup[k-1]),level)
      #
      # error check
      #
      if (molecule['error'][0][-1]):
         #
         return molecule
   #
   # calculate the energy at order k
   #
   inc_corr_order(k,tup,molecule['e_inc'][0],molecule['e_tot'][0],level)
   #
   # collect time
   #
   time.append(timer()-start)
   #
   # print results
   #
   inc_corr_utils.print_result(molecule['e_inc'][0][k-1],level)
   #
   return molecule

def inc_corr_mono_exp_est_kernel(molecule,tup,dom,n_tup,orb,thres,time,k):
   #
   # define level
   #
   level = 'ESTIM'
   #
   # generate all tuples at order k
   #
   tup.append([])
   #
   inc_corr_orb_rout.orb_generator(molecule,dom,tup[k-1],molecule['l_limit'][0],k)
   #
   # select tuples for energy estimation
   #
   if (level == 'ESTIM'):
      #
      inc_corr_orb_rout.select_est_tuples(molecule['prim_tuple'][0],molecule['sec_tuple'][0],k)
   #
   # determine number of tuples at order k
   #
   n_tup.append(len(tup[k-1]))
   #
   if (level == 'ESTIM'):
      #
      # merge energy contributions from MACRO level
      #
      if (k >= 2):
         #
         inc_corr_orb_rout.merge_tuples(molecule['prim_tuple'][0],molecule['sec_tuple'][0],k)
      #
      if (n_tup[k-1] == 0):
         #
         return molecule
   #
   # calculate theoretical number of tuples at order k
   #
   if (level != 'ESTIM'):
      #
      inc_corr_orb_rout.n_theo_tuples(n_tup[0],k,molecule['theo_work'][0])
   #
   # print status header
   #
   inc_corr_utils.print_status_header(molecule,n_tup,k,level)
   #
   # check for convergence
   #
   if (level != 'ESTIM'):
      #
      if (molecule['conv'][-1]):
         #
         return molecule
   #
   # start time
   #
   start = timer()
   #
   # run the calculations
   #
   for i in range(0,n_tup[k-1]):
      #
      # write string
      #
      inc_corr_orb_rout.orb_string(molecule,molecule['l_limit'][0],molecule['u_limit'][0],tup[k-1][i][0])
      #
      # run correlated calc
      #
      inc_corr_gen_rout.run_calc_corr(molecule,molecule['string'],level)
      #
      # write tuple energy
      #
      tup[k-1][i].append(molecule['e_tmp'])
      #
      # print status
      #
      inc_corr_utils.print_status(float(i+1)/float(n_tup[k-1]),level)
      #
      # error check
      #
      if (molecule['error'][0][-1]):
         #
         return molecule
   #
   # calculate the energy at order k
   #
   inc_corr_order(k,tup,molecule['e_inc'][0],molecule['e_tot'][0],level)
   #
   # print results
   #
   inc_corr_utils.print_result(molecule['e_inc'][0][k-1],level)
   #
   # set up entanglement and exclusion lists
   #
   if (k >= 2):
      #
      orb.append([])
      #
      e_orb_rout(molecule,molecule['e_inc'][0],orb,molecule['l_limit'][0],molecule['u_limit'][0])
      #
      molecule['excl_list'][0][:] = []
      #
      inc_corr_orb_rout.excl_rout(molecule,molecule['e_inc'][0],orb,thres,molecule['excl_list'][0])
      #
      # update domains
      #
      inc_corr_orb_rout.update_domains(dom,molecule['l_limit'][0],molecule['excl_list'][0])
   #
   # collect time
   #
   time.append(timer()-start)
   #
   # print domain updates
   #
   if (k >= 2):
      #
      inc_corr_utils.print_update(molecule,tup,n_tup,dom,k,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   # energy estimation
   #
   if (molecule['est'] and (level == 'MACRO')):
      #
      inc_corr_mono_exp_kernel(molecule,molecule['sec_tuple'][0],molecule['sec_domain'],molecule['sec_n_tuples'][0],\
                               molecule['sec_orbital'][0],0.1*molecule['prim_thres'][0],molecule['sec_time'][0],k,'ESTIM')
   #
   # print status end
   #
   if (level == 'MACRO'):
      #
      inc_corr_utils.print_status_end(molecule,k,molecule['e_tot'][0],time,n_tup,level)
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
      inc_corr_utils.print_status_header(molecule,molecule['n_tuples'][0],k)
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
      inc_corr_utils.print_result_header()
      #
      # run the calculations (for outer expansion)
      #
      for i in range(0,molecule['n_tuples'][0][k-1]):
         #
         molecule['e_tot'][1][:] = []
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
               molecule['tuple'][0][k-1][i].append(molecule['e_tot'][1][-1])
               #
               inc_corr_utils.print_result(i,molecule['tuple'][0][k-1][i])
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
               if (molecule['exp'] == 'COMB-OV'):
                  #
                  inc_corr_orb_rout.orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][0][k-1][i][0]+molecule['tuple'][1][l-1][j][0])
               #
               elif (molecule['exp'] == 'COMB-VO'):
                  #
                  inc_corr_orb_rout.orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][1][l-1][j][0]+molecule['tuple'][0][k-1][i][0])
               #
               # run correlated calc
               #
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
            inc_corr_order(l,molecule['tuple'][1],molecule['e_tot'][1])
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
               molecule['tuple'][0][k-1][i].append(molecule['e_tot'][1][-1])
               #
               inc_corr_utils.print_result(i,molecule['tuple'][0][k-1][i])
               #
               break
         #
         # collect time, energy diff, and relative work (for inner expansion)
         #
         molecule['time'][1].append(timer()-start_in)
         #
         molecule['e_diff_in'].append(molecule['e_tot'][1][-1]-molecule['e_tot'][1][-2])
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
      inc_corr_utils.print_result_end()
      #
      # calculate the energy at order k (for outer expansion)
      #
      inc_corr_order(k,molecule['tuple'][0],molecule['e_tot'][0])
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
      inc_corr_utils.print_status_end(molecule,k,molecule['e_tot'][0],molecule['time'][0],molecule['n_tuples'][0])
      #
      # print results (for inner expansion)
      #
      inc_corr_utils.print_inner_result(molecule)
      #
      # print domain updates (for outer expansion)
      #
      if (k >= 2):
         #
         inc_corr_utils.print_update(molecule,molecule['tuple'][0],molecule['n_tuples'][0],molecule['domain'][0],k,molecule['l_limit'][0],molecule['u_limit'][0])
   #
   return molecule

def inc_corr_order(k,tup,e_inc,e_tot,level):
   #
   e_inc.append([])
   #
   for j in range(0,len(tup[k-1])):
      #
      e_inc[k-1].append([tup[k-1][j][0],copy.deepcopy(tup[k-1][j][1])])
      #
      for i in range(k-1,0,-1):
         #
         e_tmp = 0.0
         #
         for l in range(0,len(tup[i-1])):
            #
            if (set(e_inc[i-1][l][0]) < set(e_inc[k-1][j][0])):
               #
               if (i == (k-1)):
                  #
                  e_tmp -= tup[i-1][l][1]
               #
               else:
                  #
                  e_tmp += float((k-i)-1) * e_inc[i-1][l][1]
         #
         e_inc[k-1][j][1] += e_tmp
   #
   e_tmp = 0.0
   #
   for j in range(0,len(tup[k-1])):
      #
      e_tmp += e_inc[k-1][j][1]
   #
   if ((k >= 2) and (level == 'MACRO')):
      #
      e_tmp += e_tot[k-2][0]
   #
   if (level == 'MACRO'):
      #
      e_tot.append([e_tmp])
   #
   else:
      #
      e_tot[k-1] += e_tmp
   #
   return e_tot

def inc_corr_prepare(molecule):
   #
   if (molecule['exp'] == 'OCC'):
      #
      molecule['l_limit'] = [0]
      molecule['u_limit'] = [molecule['nocc']]
      #
      molecule['prim_domain'] = copy.deepcopy(molecule['occ_domain'])
      molecule['sec_domain'] = copy.deepcopy(molecule['occ_domain'])
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      molecule['l_limit'] = [molecule['nocc']]
      molecule['u_limit'] = [molecule['nvirt']]
      #
      molecule['prim_domain'] = copy.deepcopy(molecule['virt_domain'])
      molecule['sec_domain'] = copy.deepcopy(molecule['virt_domain'])
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
   molecule['conv'] = [False]
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['prim_tuple'] = [[],[]]
   molecule['sec_tuple'] = [[],[]]
   #
   molecule['prim_n_tuples'] = [[],[]]
   molecule['sec_n_tuples'] = [[],[]]
   #
   molecule['prim_orbital'] = [[],[]]
   molecule['sec_orbital'] = [[],[]]
   #
   molecule['e_inc'] = [[],[]]
   molecule['e_tot'] = [[],[]]
   #
   molecule['excl_list'] = [[],[]]
   #
   molecule['theo_work'] = [[],[]]
   #
   molecule['prim_time'] = [[],[]]
   molecule['sec_time'] = [[],[]]
   #
   return molecule

