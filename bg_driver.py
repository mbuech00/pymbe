#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver routines for Bethe-Goldstone correlation calculations."""

from copy import deepcopy
from timeit import default_timer

from bg_utilities import run_calc_corr, orb_string, n_theo_tuples
from bg_print import print_status_header, print_status_end, print_result, print_init_header, print_init_end
from bg_energy import energy_calc_mono_exp_ser, bg_order
from bg_orbitals import init_domains, update_domains, orb_generator,\
                        orb_screening, orb_entanglement, orb_exclusion, select_corr_tuples
from bg_mpi_kernels import energy_calc_mono_exp_master

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.3'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def main_drv(molecule):
   #
   # initialize domains
   #
   init_domains(molecule)
   #
   # initialize variable and lists
   #
   prepare_calc(molecule)   
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
         mono_exp_merge_info(molecule)
         #
         mono_exp_drv(molecule,molecule['min_corr_order'],molecule['max_corr_order'],'CORRE')
   #
   elif ((molecule['exp'] == 'comb-ov') or (molecule['exp'] == 'comb-vo')):
      #
      # run dual expansion
      #
      dual_exp_drv(molecule)
   #
   return molecule

def mono_exp_drv(molecule,start,end,level):
   #
   for k in range(start,end+1):
      #
      # mono expansion initialization
      #
      mono_exp_init(molecule,k,level)
      #
      # mono expansion kernel
      #
      mono_exp_kernel(molecule,k,level)
      #
      # print status end
      #
      print_status_end(molecule,k,level)
      #
      # return if converged
      #
      if (((level == 'MACRO') and molecule['conv'][-1]) or ((level == 'CORRE') and (k == end))):
         #
         print('')
         #
         if ((level == 'MACRO') and (not molecule['corr'])): print('')
         #
         if (level == 'CORRE'):
            #
            orb_entanglement(molecule,molecule['l_limit'][0],molecule['u_limit'][0],level)
         #
         break
   #
   if (level == 'CORRE'): mono_exp_finish(molecule)
   #
   return molecule

def mono_exp_kernel(molecule,k,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple'][0]
      n_tup = molecule['prim_n_tuples'][0]
      time = molecule['prim_time'][0]
   #
   elif (level == 'CORRE'):
      #
      tup = molecule['corr_tuple'][0]
      n_tup = molecule['corr_n_tuples'][0]
      time = molecule['corr_time'][0]
   #
   print_status_header(n_tup[k-1],k,molecule['conv'][-1],level)
   #
   if ((level == 'MACRO') and molecule['conv'][-1]):
      #
      return molecule
   #
   # start time
   #
   start = default_timer()
   #
   # run the calculations
   #
   if (molecule['mpi_parallel']):
      #
      energy_calc_mono_exp_master(molecule,k,tup,n_tup,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   else:
      #
      energy_calc_mono_exp_ser(molecule,k,tup,n_tup,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   # calculate the energy at order k
   #
   if (level == 'MACRO'):
      #
      bg_order(molecule,k,tup,molecule['e_tot'][0])
   #
   elif (level == 'CORRE'):
      #
      bg_order(molecule,k,tup,molecule['e_corr'][0])
   #
   # collect time
   #
   time.append(default_timer()-start)
   #
   # print results
   #
   print_result(tup[-1],level)
   #
   # merge tuples from primary exp. into molecule['corr_tuple']
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

def dual_exp_drv(molecule):
   #
   for k in range(1,molecule['u_limit'][0]+1):
      #
      # append tuple list and generate all tuples at order k
      #
      molecule['tuple'][0].append([])
      #
      orb_generator(molecule,molecule['domain'][0],molecule['tuple'][0],molecule['l_limit'][0],molecule['u_limit'][0],k)
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
      start_out = default_timer()
      #
      # print result header (for outer expansion)
      #
#      print_result_header()
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
#         reinit_domains(molecule,molecule['domain'][1])
         #
         # start time (for inner expansion)
         #
         start_in = default_timer()
         #
         for l in range(1,molecule['u_limit'][1]+1):
            #
            # append tuple list and generate all tuples at order l
            #
            molecule['tuple'][1].append([])
            #
            orb_generator(molecule,molecule['domain'][1],molecule['tuple'][1],molecule['l_limit'][1],molecule['u_limit'][1],l)
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
               print_result(i,molecule['tuple'][0][k-1][i])
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
                  orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][0][k-1][i][0]+molecule['tuple'][1][l-1][j][0],string)
               #
               elif (molecule['exp'] == 'comb-vo'):
                  #
                  orb_string(molecule,0,molecule['nocc']+molecule['nvirt'],molecule['tuple'][1][l-1][j][0]+molecule['tuple'][0][k-1][i][0],string)
               #
               # run correlated calc
               #
               run_calc_corr(molecule,string,False)
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
            bg_order(molecule,l,molecule['tuple'][1],molecule['e_tot'][1])
            #
            # set up entanglement and exclusion lists (for inner expansion)
            #
            if (l >= 2):
               #
               molecule['sec_orb_ent'][1].append([])
               #
               e_orb_rout(molecule,molecule['tuple'][1],molecule['sec_orb_ent'][1],molecule['l_limit'][1],molecule['u_limit'][1])
               #
               molecule['excl_list'][1][:] = []
               #
               orb_exclusion(molecule,molecule['tuple'][1],molecule['sec_orb_ent'][1],molecule['corr_thres'],molecule['excl_list'][1])
               #
               # update domains (for inner expansion)
               #
               update_domains(molecule['domain'][1],molecule['l_limit'][1],molecule['excl_list'][1])
            #
            # calculate theoretical number of tuples at order l (for inner expansion)
            #
            n_theo_tuples(molecule['n_tuples'][1][0],l,molecule['theo_work'][1])
            #
            # check for maximum order (for inner expansion)
            #
            if (l == molecule['u_limit'][1]):
               #
               molecule['tuple'][0][k-1][i].append(molecule['e_tot'][1][-1])
               #
               print_result(i,molecule['tuple'][0][k-1][i])
               #
               break
         #
         # collect time, energy diff, and relative work (for inner expansion)
         #
         molecule['time'][1].append(default_timer()-start_in)
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
#      print_result_end()
      #
      # calculate the energy at order k (for outer expansion)
      #
      bg_order(molecule,k,molecule['tuple'][0],molecule['e_tot'][0])
      #
      # set up entanglement and exclusion lists (for outer expansion)
      #
      if (k >= 2):
         #
         molecule['prim_orb_ent'][0].append([])
         #
         e_orb_rout(molecule,molecule['tuple'][0],molecule['prim_orb_ent'][0],molecule['l_limit'][0],molecule['u_limit'][0])
         #
         molecule['excl_list'][0][:] = []
         #
         orb_exclusion(molecule,molecule['tuple'][0],molecule['prim_orb_ent'][0],molecule['prim_thres'],molecule['excl_list'][0])
         #
         # update domains (for outer expansion)
         #
         update_domains(molecule['domain'][0],molecule['l_limit'][0],molecule['excl_list'][0])
      #
      # calculate theoretical number of tuples at order k (for outer expansion)
      #
      n_theo_tuples(molecule['n_tuples'][0][0],k,molecule['theo_work'][0])
      #
      # collect time (for outer expansion)
      #
      molecule['time'][0].append(default_timer()-start_out)
      #
      # print status end (for outer expansion)
      #
      print_status_end(molecule,k,molecule['time'][0],molecule['n_tuples'][0])
   #
   return molecule

def mono_exp_init(molecule,k,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple'][0]
      dom = molecule['prim_domain'][0]
      n_tup = molecule['prim_n_tuples'][0]
      orb = molecule['prim_orb_ent'][0]
      thres = molecule['prim_thres']
   #
   elif (level == 'CORRE'):
      #
      tup = molecule['corr_tuple'][0]
      dom = molecule['corr_domain'][0]
      n_tup = molecule['corr_n_tuples'][0]
      orb = molecule['corr_orb_ent'][0]
      thres = molecule['corr_thres']
   #
   # print init header
   #
   print_init_header(k,level)
   #
   # start time
   #
   start = default_timer()
   #
   if (k >= 2):
      #
      # orbital screening
      #
      orb_screening(molecule,k-1,molecule['l_limit'][0],molecule['u_limit'][0],level)
   #
   # generate all tuples at order k
   #
   tup.append([])
   #
   orb_generator(molecule,dom[k-1],tup,molecule['l_limit'][0],molecule['u_limit'][0],k)
   #
   if (level == 'CORRE'):
      #
      select_corr_tuples(molecule['prim_tuple'][0],tup,k)
   #
   # collect time_gen
   #
   time_gen = default_timer() - start
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
      n_theo_tuples(n_tup[0],k,molecule['theo_work'][0])
   #
   # print init end
   #
   print_init_end(k,time_gen,level)
   #
   # if converged, pop last element of tup list and append to n_tup list
   #
   if ((level == 'MACRO') and molecule['conv'][-1]):
      #
      tup.pop(-1)
      #
      for l in range(k+1,molecule['u_limit'][0]+1):
         #
         n_tup.append(0)
         #
         n_theo_tuples(n_tup[0],l,molecule['theo_work'][0])
   #
   return molecule

def mono_exp_finish(molecule):
   #
   # make the e_corr and corr_time lists of the same length as the e_tot list
   #
   for _ in range(molecule['max_corr_order'],len(molecule['e_tot'][0])):
      #
      molecule['e_corr'][0].append(molecule['e_corr'][0][-1])
      #
      molecule['corr_time'][0].append(0.0)
   #
   # make corr_n_tuples of the same length as prim_n_tuples
   #
   for _ in range(molecule['max_corr_order'],len(molecule['prim_n_tuples'][0])):
      #
      molecule['corr_n_tuples'][0].append(0)
   #
   # make cor_orb_con lists of same length as orb_con lists for prim exp
   #
   tmp = []
   #
   for i in range(len(molecule['corr_orb_ent'][0]),len(molecule['prim_orb_ent'][0])):
      #
      molecule['corr_orb_con_abs'][0].append([])
      molecule['corr_orb_con_rel'][0].append([])
      #
      tmp[:] = []
      #
      for j in range(0,len(molecule['prim_orb_ent'][0][i])):
         #
         e_sum = 0.0
         #
         for k in range(0,len(molecule['prim_orb_ent'][0][i][j])):
            #
            e_sum += molecule['prim_orb_ent'][0][i][j][k][0]
         #
         tmp.append(e_sum)
      #
      for j in range(0,len(tmp)):
         #
         molecule['corr_orb_con_abs'][0][-1].append(molecule['corr_orb_con_abs'][0][-2][j]+tmp[j])
      #
      e_sum = 0.0
      #
      for j in range(0,len(molecule['corr_orb_con_abs'][0][-1])):
         #
         e_sum += molecule['corr_orb_con_abs'][0][-1][j]
      #
      for j in range(0,len(molecule['corr_orb_con_abs'][0][-1])):
         #
         molecule['corr_orb_con_rel'][0][-1].append(molecule['corr_orb_con_abs'][0][-1][j]/e_sum)
   #
   return molecule

def prepare_calc(molecule):
   #
   if (molecule['exp'] == 'occ'):
      #
      molecule['l_limit'] = [0]
      molecule['u_limit'] = [molecule['nocc']]
      #
      molecule['prim_domain'] = deepcopy([molecule['occ_domain']])
      molecule['corr_domain'] = deepcopy([molecule['occ_domain']])
   #
   elif (molecule['exp'] == 'virt'):
      #
      molecule['l_limit'] = [molecule['nocc']]
      molecule['u_limit'] = [molecule['nvirt']]
      #
      molecule['prim_domain'] = deepcopy([molecule['virt_domain']])
      molecule['corr_domain'] = deepcopy([molecule['virt_domain']])
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
   molecule['corr_tuple'] = [[],[]]
   #
   molecule['prim_n_tuples'] = [[],[]]
   molecule['corr_n_tuples'] = [[],[]]
   #
   molecule['prim_orb_ent'] = [[],[]]
   molecule['corr_orb_ent'] = [[],[]]
   #
   molecule['prim_orb_arr'] = [[],[]]
   molecule['corr_orb_arr'] = [[],[]]
   #
   molecule['prim_orb_con_abs'] = [[],[]]
   molecule['prim_orb_con_rel'] = [[],[]]
   molecule['corr_orb_con_abs'] = [[],[]]
   molecule['corr_orb_con_rel'] = [[],[]]
   #
   molecule['e_tot'] = [[],[]]
   #
   molecule['e_corr'] = [[],[]]
   #
   molecule['excl_list'] = []
   #
   molecule['theo_work'] = [[],[]]
   #
   molecule['prim_time'] = [[],[]]
   molecule['corr_time'] = [[],[]]
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
         molecule['corr_n_tuples'][0].append(0)
         #
         molecule['e_corr'][0].append(0.0)
         #
         molecule['corr_time'][0].append(0.0)
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
   # no energy correction possible
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
         molecule['corr_time'][0].append(0.0)
      #
      for _ in range(0,len(molecule['prim_n_tuples'][0])):
         #
         molecule['corr_n_tuples'][0].append(0)
      #
      return molecule
   #
   # the input corr_order is too high, so we correct everything
   #
   elif ((molecule['min_corr_order'] + (molecule['corr_order']-1)) > len(molecule['prim_tuple'][0])):
      #
      molecule['max_corr_order'] = len(molecule['prim_tuple'][0])
      #
      molecule['corr_order'] = (molecule['max_corr_order'] - molecule['min_corr_order']) + 1
   #
   # default, set max_corr_order according to input corr_order
   #
   else:
      #
      molecule['max_corr_order'] = molecule['min_corr_order'] + (molecule['corr_order']-1)
   #
   for _ in range(1,molecule['min_corr_order']):
      #
      molecule['corr_n_tuples'][0].append(0)
      #
      molecule['e_corr'][0].append(0.0)
      #
      molecule['corr_time'][0].append(0.0)
   #
   return molecule

def mono_exp_merge_info(molecule):
   #
   for k in range(1,molecule['min_corr_order']):
      #
      molecule['corr_tuple'][0].append(molecule['prim_tuple'][0][k-1])
   #
   for k in range(1,molecule['min_corr_order']-1):
      #
      molecule['corr_domain'][0].append(molecule['prim_domain'][0][k-1])
      molecule['corr_orb_con_abs'][0].append(molecule['prim_orb_con_abs'][0][k-1])
      molecule['corr_orb_con_rel'][0].append(molecule['prim_orb_con_rel'][0][k-1])
   #
   for k in range(1,molecule['min_corr_order']-2):
      #
      molecule['corr_orb_ent'][0].append(molecule['prim_orb_ent'][0][k-1])
      molecule['corr_orb_arr'][0].append(molecule['prim_orb_arr'][0][k-1])
   #
   return molecule

