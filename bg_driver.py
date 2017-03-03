#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from copy import deepcopy

from bg_mpi_utils import mono_exp_merge_info
from bg_time import timer_phase
from bg_utils import theo_work
from bg_print import print_status_header, print_status_end, print_result,\
                     print_init_header, print_init_end, print_final_header, print_final_end
from bg_energy import energy_kernel_mono_exp, energy_summation
from bg_orbitals import init_domains, update_domains, orb_generator,\
                        orb_screening, orb_exclusion

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
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
            timer_phase(molecule,'time_init',k+1,level)
            #
            orb_screening(molecule,k,molecule['l_limit'],molecule['u_limit'],level,True)
            #
            mono_exp_finish(molecule)
            #
            timer_phase(molecule,'time_init',k+1,level)
         #
         break
   #
   return molecule

def mono_exp_kernel(molecule,k,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      e_inc = molecule['prim_energy_inc']
      e_tot = molecule['prim_energy']
   #
   elif (level == 'CORRE'):
      #
      tup = molecule['corr_tuple']
      e_inc = molecule['corr_energy_inc']
      e_tot = molecule['corr_energy']
   #
   print_status_header(tup[-1],k,molecule['conv'][-1],level)
   #
   if ((level == 'MACRO') and molecule['conv'][-1]):
      #
      return molecule
   #
   # run the calculations
   #
   timer_phase(molecule,'time_kernel',k,level)
   #
   energy_kernel_mono_exp(molecule,k,tup,e_inc,molecule['l_limit'],molecule['u_limit'],level)
   #
   timer_phase(molecule,'time_kernel',k,level)
   #
   # print status end
   #
   print_status_end(molecule,k,level)
   #
   print_final_header(k,level)
   #
   # calculate the energy at order k
   #
   timer_phase(molecule,'time_final',k,level)
   #
   energy_summation(molecule,k,tup,e_inc,e_tot,level)
   #
   timer_phase(molecule,'time_final',k,level)
   #
   print_final_end(molecule,k,level)
   # 
   # print results
   #
   print_result(tup[k-1],e_inc[k-1],level)
   #
   # check for convergence
   #
   if ((k == len(tup[0])) or (k == molecule['max_order'])):
      #
      tup.append(np.array([],dtype=np.int))
      e_inc.append(np.array([],dtype=np.float64))
      #
      molecule['conv'].append(True)
   #
   return molecule

def mono_exp_init(molecule,k,level):
   #
   if (level == 'MACRO'):
      #
      tup = molecule['prim_tuple']
      e_inc = molecule['prim_energy_inc']
      dom = molecule['prim_domain']
      orb = molecule['prim_orb_ent']
      thres = molecule['prim_thres']
   #
   elif (level == 'CORRE'):
      #
      tup = molecule['corr_tuple']
      e_inc = molecule['corr_energy_inc']
      dom = molecule['corr_domain']
      orb = molecule['corr_orb_ent']
      thres = molecule['corr_thres']
   #
   # print init header
   #
   print_init_header(k,level)
   #
   timer_phase(molecule,'time_init',k,level)
   #
   if (k >= 2):
      #
      # orbital screening
      #
      orb_screening(molecule,k-1,molecule['l_limit'],molecule['u_limit'],level)
   #
   # generate all tuples at order k
   #
   orb_generator(molecule,dom[k-1],tup,molecule['l_limit'],molecule['u_limit'],k,level)
   #
   # check for convergence
   #
   if ((level == 'MACRO') and (len(tup[k-1]) == 0)): molecule['conv'].append(True)
   #
   # print init end
   #
   timer_phase(molecule,'time_init',k,level)
   #
   print_init_end(molecule,k,level)
   #
   # init e_inc list
   #
   e_inc.append(np.zeros(len(tup[k-1]),dtype=np.float64))
   #
   # if converged, pop last element of tup list
   #
   if ((level == 'MACRO') and molecule['conv'][-1]): tup.pop(-1)
   #
   return molecule

def mono_exp_finish(molecule):
   #
   # make the corr_energy and corr_time lists of the same length as the prim_energy list
   #
   for _ in range(molecule['max_corr_order'],len(molecule['prim_energy'])):
      #
      molecule['corr_energy'].append(molecule['corr_energy'][-1])
      #
      molecule['corr_time_kernel'].append(0.0)
      molecule['corr_time_final'].append(0.0)
   #
   if (molecule['corr'] and (molecule['min_corr_order'] != 0)):
      #
      for _ in range(molecule['max_corr_order'],len(molecule['prim_energy'])-1):
         #
         molecule['corr_time_init'].append(0.0)
   #
   else:
      #
      for _ in range(molecule['max_corr_order'],len(molecule['prim_energy'])):
         #
         molecule['corr_time_init'].append(0.0)
   #
   # make cor_orb_con lists of same length as orb_con lists for prim exp
   #
   tmp = []
   #
   for i in range(len(molecule['corr_orb_ent']),len(molecule['prim_orb_ent'])):
      #
      molecule['corr_orb_con_abs'].append([])
      molecule['corr_orb_con_rel'].append([])
      #
      tmp[:] = []
      #
      for j in range(0,len(molecule['prim_orb_ent'][i])):
         #
         e_sum = 0.0
         #
         for k in range(0,len(molecule['prim_orb_ent'][i][j])):
            #
            e_sum += molecule['prim_orb_ent'][i][j][k][0]
         #
         tmp.append(e_sum)
      #
      for j in range(0,len(tmp)):
         #
         molecule['corr_orb_con_abs'][-1].append(molecule['corr_orb_con_abs'][-2][j]+tmp[j])
      #
      e_sum = 0.0
      #
      for j in range(0,len(molecule['corr_orb_con_abs'][-1])):
         #
         e_sum += molecule['corr_orb_con_abs'][-1][j]
      #
      for j in range(0,len(molecule['corr_orb_con_abs'][-1])):
         #
         molecule['corr_orb_con_rel'][-1].append(molecule['corr_orb_con_abs'][-1][j]/e_sum)
   #
   del tmp
   #
   return molecule

def prepare_calc(molecule):
   #
   if (molecule['exp'] == 'occ'):
      #
      molecule['l_limit'] = 0
      molecule['u_limit'] = molecule['nocc']
      #
      molecule['prim_domain'] = deepcopy([molecule['occ_domain']])
      molecule['corr_domain'] = deepcopy([molecule['occ_domain']])
   #
   elif (molecule['exp'] == 'virt'):
      #
      molecule['l_limit'] = molecule['nocc']
      molecule['u_limit'] = molecule['nvirt']
      #
      molecule['prim_domain'] = deepcopy([molecule['virt_domain']])
      molecule['corr_domain'] = deepcopy([molecule['virt_domain']])
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
   if ((molecule['max_order'] == 0) or (molecule['max_order'] > molecule['u_limit'])):
      #
      molecule['max_order'] = molecule['u_limit']
   #
   # determine max theoretical work
   #
   theo_work(molecule)
   #
   molecule['conv'] = [False]
   #
   molecule['e_tmp'] = 0.0
   #
   molecule['prim_tuple'] = []
   molecule['corr_tuple'] = []
   #
   molecule['prim_energy_inc'] = []
   molecule['corr_energy_inc'] = []
   #
   molecule['prim_orb_ent'] = []
   molecule['corr_orb_ent'] = []
   #
   molecule['prim_orb_arr'] = []
   molecule['corr_orb_arr'] = []
   #
   molecule['prim_orb_con_abs'] = []
   molecule['prim_orb_con_rel'] = []
   molecule['corr_orb_con_abs'] = []
   molecule['corr_orb_con_rel'] = []
   #
   molecule['prim_energy'] = []
   #
   molecule['corr_energy'] = []
   #
   molecule['excl_list'] = []
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
      for _ in range(0,len(molecule['prim_energy'])):
         #
         molecule['corr_energy'].append(0.0)
         #
         molecule['corr_time_init'].append(0.0)
         molecule['corr_time_kernel'].append(0.0)
         molecule['corr_time_final'].append(0.0)
      #
      return molecule
   #
   else:
      #
      for i in range(0,len(molecule['prim_tuple'])):
         #
         if ((len(molecule['prim_tuple'][i]) < molecule['theo_work'][i]) and (len(molecule['prim_tuple'][i]) > 0)):
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
      for _ in range(0,len(molecule['prim_energy'])):
         #
         molecule['corr_energy'].append(0.0)
         #
         molecule['corr_time_init'].append(0.0)
         molecule['corr_time_kernel'].append(0.0)
         molecule['corr_time_final'].append(0.0)
      #
      return molecule
   #
   # the input corr_order is too high, so we correct everything
   #
   elif ((molecule['min_corr_order'] + (molecule['corr_order']-1)) > len(molecule['prim_tuple'])):
      #
      molecule['max_corr_order'] = len(molecule['prim_tuple'])
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
      molecule['corr_energy'].append(0.0)
      #
      molecule['corr_time_init'].append(0.0)
      molecule['corr_time_kernel'].append(0.0)
      molecule['corr_time_final'].append(0.0)
   #
   return molecule

