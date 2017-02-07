# -*- coding: utf-8 -*
#!/usr/bin/env python

#
# energy driver routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import copy
from timeit import default_timer as timer

import inc_corr_gen_rout
import inc_corr_orb_rout
import inc_corr_utils
import inc_corr_e_rout

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def main_drv(molecule):
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
   if ((molecule['exp'] == 'occ') or (molecule['exp'] == 'virt')):
      #
      # run mono expansion
      #
      print('')
      print('')
      print('                     ---------------------------------------------                ')
      print('                                   primary expansion                              ')
      print('                     ---------------------------------------------                ')
      #
      mono_exp_drv(molecule,1,molecule['max_order'],'MACRO')
      #
      # set min and max _corr_order
      #
      set_corr_order(molecule)
      #
      if (molecule['corr']):
         #
         # energy correction for mono expansion
         #
         print('')
         print('                     ---------------------------------------------                ')
         print('                                   energy correction                              ')
         print('                     ---------------------------------------------                ')
         #
         merge_info(molecule)
         #
         mono_exp_drv(molecule,molecule['min_corr_order'],molecule['max_corr_order'],'CORRE')
   #
   elif ((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')):
      #
      # run dual expansion
      #
      inc_corr_dual_exp(molecule)
   #
   return molecule

def mono_exp_drv(molecule,start,end,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple'][0]
      dom = molecule['prim_domain'][0]
      n_tup = molecule['prim_n_tuples'][0]
      time = molecule['prim_time'][0]
      orb = molecule['prim_orbital'][0]
      thres = molecule['prim_thres'][0]
   #
   else:
      #
      tup = molecule['sec_tuple'][0]
      dom = molecule['sec_domain'][0]
      n_tup = molecule['sec_n_tuples'][0]
      time = molecule['sec_time'][0]
      orb = molecule['sec_orbital'][0]
      thres = 0.0
   #
   for k in range(start,end+1):
      #
      # call mono expansion kernel
      #
      mono_exp_kernel(molecule,tup,dom,n_tup,time,k,level)
      #
      # print status end
      #
      inc_corr_utils.print_status_end(k,time,n_tup,level)
      #
      # return if converged
      #
      if (((level == 'MACRO') and molecule['conv'][-1]) or ((level == 'CORRE') and (k == end))):
         #
         print('')
         #
         if ((level == 'MACRO') and (not molecule['corr'])): print('')
         #
         break
      #
      # orbital screening
      #
      inc_corr_orb_rout.orb_screen_rout(molecule,k,tup,orb,dom,thres,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   # make the e_corr and sec_time lists of the same length as the e_tot list
   # 
   if (level == 'CORRE'):
      #
      for _ in range(molecule['max_corr_order'],len(molecule['e_tot'][0])):
         #
         molecule['e_corr'][0].append(molecule['e_corr'][0][-1])
         #
         molecule['sec_time'][0].append(0.0)
      #
      # make molecule['sec_n_tuples'] of the same length as molecule['prim_n_tuples']
      #
      for _ in range(molecule['max_corr_order'],len(molecule['prim_n_tuples'][0])):
         #
         molecule['sec_n_tuples'][0].append(0)
   #
   return molecule

def mono_exp_kernel(molecule,tup,dom,n_tup,time,k,level):
   #
   # generate all tuples at order k
   #
   tup.append([])
   #
   # print status header-1
   #
   inc_corr_utils.print_status_header_1(k,level)
   #
   # start time
   #
   start = timer()
   #
   inc_corr_orb_rout.orb_generator(molecule,dom[k-1],tup,molecule['l_limit'][0],molecule['u_limit'][0],k)
   #
   if (level == 'CORRE'):
      #
      inc_corr_orb_rout.select_corr_tuples(molecule['prim_tuple'][0],tup,k)
   #
   # collect time_gen
   #
   time_gen = timer() - start
   #
   # determine number of tuples at order k
   #
   n_tup.append(len(tup[k-1]))
   #
   if (level == 'MACRO'):
      #
      # check for convergence
      #
      if (n_tup[k-1] == 0):
         #
         molecule['conv'].append(True)
      #
      # calculate theoretical number of tuples at order k
      #
      inc_corr_orb_rout.n_theo_tuples(n_tup[0],k,molecule['theo_work'][0])
   #
   # print status header-2
   #
   inc_corr_utils.print_status_header_2(n_tup[k-1],k,molecule['conv'][-1],time_gen,level)
   #
   # return if converged
   #
   if ((level == 'MACRO') and molecule['conv'][-1]):
      #
      for l in range(k+1,molecule['u_limit'][0]+1):
         #
         n_tup.append(0)
         #
         inc_corr_orb_rout.n_theo_tuples(n_tup[0],l,molecule['theo_work'][0])
      #
      return molecule
   #
   # start time
   #
   start = timer()
   #
   # run the calculations
   #
   if (molecule['mpi_parallel']):
      #
      inc_corr_e_rout.energy_calc_mono_exp_par(molecule,k,tup,n_tup,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   else:
      #
      inc_corr_e_rout.energy_calc_mono_exp_ser(molecule,k,tup,n_tup,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   # calculate the energy at order k
   #
   if (level == 'MACRO'):
      #
      inc_corr_e_rout.inc_corr_order(molecule,k,tup,molecule['e_tot'][0])
   #
   elif (level == 'CORRE'):
      #
      inc_corr_e_rout.inc_corr_order(molecule,k,tup,molecule['e_corr'][0])
   #
   # collect time
   #
   time.append(timer()-start)
   #
   # print results
   #
   inc_corr_utils.print_result(tup[-1],level)
   #
   # merge tuples from primary exp. into molecule['sec_tuple']
   #
   if (level == 'CORRE'):
      #
      for i in range(0,len(molecule['prim_tuple'][0][k-1])):
         #
         tup[k-1].append(molecule['prim_tuple'][0][k-1][i])
   #
   # check for convergence
   #
   if ((k == n_tup[0]) or (k == molecule['max_order'])):
      #
      tup.append([])
      #
      molecule['conv'].append(True)
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
      inc_corr_orb_rout.orb_generator(molecule,molecule['domain'][0],molecule['tuple'][0],molecule['l_limit'][0],molecule['u_limit'][0],k)
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
            inc_corr_orb_rout.orb_generator(molecule,molecule['domain'][1],molecule['tuple'][1],molecule['l_limit'][1],molecule['u_limit'][1],l)
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
            string = ''
            #
            for j in range(0,molecule['n_tuples'][1][l-1]):
               #
               # write string
               #
               if (molecule['exp'] == 'comb-ov'):
                  #
                  inc_corr_orb_rout.orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][0][k-1][i][0]+molecule['tuple'][1][l-1][j][0],string)
               #
               elif (molecule['exp'] == 'comb-vo'):
                  #
                  inc_corr_orb_rout.orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][1][l-1][j][0]+molecule['tuple'][0][k-1][i][0],string)
               #
               # run correlated calc
               #
               inc_corr_gen_rout.run_calc_corr(molecule,string,False)
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
            inc_corr_e_rout.inc_corr_order(molecule,l,molecule['tuple'][1],molecule['e_tot'][1])
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
      inc_corr_e_rout.inc_corr_order(molecule,k,molecule['tuple'][0],molecule['e_tot'][0])
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
      inc_corr_utils.print_status_end(molecule,k,molecule['time'][0],molecule['n_tuples'][0])
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

def inc_corr_prepare(molecule):
   #
   if (molecule['exp'] == 'occ'):
      #
      molecule['l_limit'] = [0]
      molecule['u_limit'] = [molecule['nocc']]
      #
      molecule['prim_domain'] = copy.deepcopy([molecule['occ_domain']])
      molecule['sec_domain'] = copy.deepcopy([molecule['occ_domain']])
   #
   elif (molecule['exp'] == 'virt'):
      #
      molecule['l_limit'] = [molecule['nocc']]
      molecule['u_limit'] = [molecule['nvirt']]
      #
      molecule['prim_domain'] = copy.deepcopy([molecule['virt_domain']])
      molecule['sec_domain'] = copy.deepcopy([molecule['virt_domain']])
      #
   #
   elif (molecule['exp'] == 'comb-ov'):
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
   elif (molecule['exp'] == 'comb-vo'):
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
   if ((molecule['max_order'] == 0) or (molecule['max_order'] > molecule['u_limit'][0])):
      #
      molecule['max_order'] = molecule['u_limit'][0]
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
   molecule['e_tot'] = [[],[]]
   #
   molecule['e_corr'] = [[],[]]
   #
   molecule['excl_list'] = [[],[]]
   #
   molecule['theo_work'] = [[],[]]
   #
   molecule['prim_time'] = [[],[]]
   molecule['sec_time'] = [[],[]]
   #
   return molecule

def set_corr_order(molecule):
   #
   molecule['min_corr_order'] = 0
   #
   if (not molecule['corr']):
      #
      molecule['max_corr_order'] = 0
      #
      for _ in range(0,len(molecule['e_tot'][0])):
         #
         molecule['sec_n_tuples'][0].append(0)
         #
         molecule['e_corr'][0].append(0.0)
         #
         molecule['sec_time'][0].append(0.0)
      #
      return molecule
   #
   else:
      #
      for i in range(0,len(molecule['prim_n_tuples'][0])):
         #
         if ((molecule['prim_n_tuples'][0][i] < molecule['theo_work'][0][i]) and (molecule['prim_n_tuples'][0][i] > 0)):
            #
            molecule['min_corr_order'] = i+1
            #
            break
   #
   if (molecule['min_corr_order'] == 0):
      #
      molecule['corr'] = False
      #
      molecule['max_corr_order'] = 0
      #
      molecule['corr_order'] = 0
      #
      for _ in range(0,len(molecule['e_tot'][0])):
         #
         molecule['e_corr'][0].append(0.0)
         #
         molecule['sec_time'][0].append(0.0)
      #
      for _ in range(0,len(molecule['prim_n_tuples'][0])):
         #
         molecule['sec_n_tuples'][0].append(0)
      #
      return molecule
   #
   elif ((molecule['min_corr_order'] + (molecule['corr_order']-1)) > (len(molecule['prim_tuple'][0])-1)):
      #
      molecule['max_corr_order'] = len(molecule['prim_tuple'][0])-1
      #
      molecule['corr_order'] = (len(molecule['prim_tuple'][0])-1) - molecule['min_corr_order']
   #
   else:
      #
      molecule['max_corr_order'] = molecule['min_corr_order'] + (molecule['corr_order']-1)
   #
   for _ in range(1,molecule['min_corr_order']):
      #
      molecule['sec_n_tuples'][0].append(0)
      #
      molecule['e_corr'][0].append(0.0)
      #
      molecule['sec_time'][0].append(0.0)
   #
   return molecule

def merge_info(molecule):
   #
   for k in range(1,molecule['min_corr_order']):
      #
      molecule['sec_tuple'][0].append(molecule['prim_tuple'][0][k-1])
      molecule['sec_orbital'][0].append(molecule['prim_orbital'][0][k-1])
      molecule['sec_domain'][0].append(molecule['prim_domain'][0][k-1])
   #
   return molecule




