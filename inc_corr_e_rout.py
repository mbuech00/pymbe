# -*- coding: utf-8 -*
#!/usr/bin/env python

#
# energy-related routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

from timeit import default_timer as timer
from mpi4py import MPI

import inc_corr_gen_rout
import inc_corr_orb_rout
import inc_corr_utils
import inc_corr_mpi

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def energy_calc_mono_exp_ser(molecule,order,tup,n_tup,l_limit,u_limit,level):
   #
   string = {'drop': ''}
   #
   for i in range(0,n_tup[order-1]):
      #
      # write string
      #
      inc_corr_orb_rout.orb_string(molecule,l_limit,u_limit,tup[order-1][i][0],string)
      #
      # run correlated calc
      #
      inc_corr_gen_rout.run_calc_corr(molecule,string['drop'],level)
      #
      # write tuple energy
      #
      tup[order-1][i].append(molecule['e_tmp'])
      #
      # print status
      #
      inc_corr_utils.print_status(float(i+1)/float(n_tup[order-1]),level)
      #
      # error check
      #
      if (molecule['error'][0][-1]):
         #
         return molecule, tup
   #
   return molecule, tup

def energy_calc_mono_exp_par(molecule,order,tup,n_tup,l_limit,u_limit,level):
   #
   string = {'drop': ''}
   #
   # number of slaves
   #
   num_slaves = molecule['mpi_size'] - 1
   #
   # number of available slaves
   #
   slaves_avail = num_slaves
   #
   # define mpi message tags
   #
   tags = inc_corr_utils.enum('ready','done','exit','start')
   #
   # init job index
   #
   i = 0
   #
   # init stat counter
   #
   counter = 0
   #
   # wake up slaves
   #
   msg = {'task': 'energy_calc_mono_exp'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   while (slaves_avail >= 1):
      #
      # write string
      #
      if (i <= (n_tup[order-1]-1)): inc_corr_orb_rout.orb_string(molecule,l_limit,u_limit,tup[order-1][i][0],string)
      #
      # receive data dict
      #
      data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=molecule['mpi_stat'])
      #
      # probe for source
      #
      source = molecule['mpi_stat'].Get_source()
      #
      # probe for tag
      #
      tag = molecule['mpi_stat'].Get_tag()
      #
      if (tag == tags.ready):
         #
         if (i <= (n_tup[order-1]-1)):
            #
            # store job index
            #
            string['index'] = i
            #
            # send string dict
            #
            molecule['mpi_comm'].send(string, dest=source, tag=tags.start)
            #
            # increment job index
            #
            i += 1
         #
         else:
            #
            molecule['mpi_comm'].send(None, dest=source, tag=tags.exit)
      #
      elif (tag == tags.done):
         #
         # write tuple energy
         #
         tup[order-1][data['index']].append(data['e_tmp'])
         #
         # increment stat counter
         #
         counter += 1
         #
         # print status
         #
         inc_corr_utils.print_status(float(counter)/float(n_tup[order-1]),level)
         #
         # error check
         #
         if (data['error']):
            #
            print('problem with slave '+str(source)+' in energy_calc_mono_exp_par  ---  aborting...')
            #
            molecule['error'][0].append(True)
            #
            return molecule, tup
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   return molecule, tup

def energy_calc_mono_exp_est_ser(molecule,tup,n_tup,l_limit,u_limit,time,level):
   #
   string = {'drop': ''}
   #
   # init stat counter 
   #
   counter = 0
   #
   for k in range(1,molecule['max_est_order']+1):
      #
      # start time
      #
      start = timer()
      #
      for i in range(0,n_tup[k-1]):
         #
         # increment counter
         #
         counter += 1
         #
         # write string
         #
         inc_corr_orb_rout.orb_string(molecule,l_limit,u_limit,tup[k-1][i][0],string)
         #
         # run correlated calc
         #
         inc_corr_gen_rout.run_calc_corr(molecule,string['drop'],level)
         #
         # write tuple energy
         #
         tup[k-1][i].append(molecule['e_tmp'])
         #
         # print status
         #
         inc_corr_utils.print_status(float(counter)/float(sum(n_tup)),level)
         #
         # error check
         #
         if (molecule['error'][0][-1]):
            #
            return molecule
      #
      # collect time
      #
      time.append(timer()-start)
   #
   return molecule, tup

def energy_calc_mono_exp_est_par(molecule,tup,n_tup,l_limit,u_limit,time,level):
   #
   string = {'drop': ''}
   #
   # number of slaves
   #
   num_slaves = molecule['mpi_size'] - 1
   #
   # number of available slaves
   #
   slaves_avail = num_slaves
   #
   # define mpi message tags
   #
   tags = inc_corr_utils.enum('ready','done','exit','start')
   #
   # init job indices
   #
   k = molecule['max_est_order']
   i = 0
   #
   # init stat counter
   #
   counter = 0
   #
   # wake up slaves
   #
   msg = {'task': 'energy_calc_mono_exp_est'}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # start time
   #
   start = timer()
   #
   while (slaves_avail >= 1):
      #
      # write string
      #
      if (i <= (n_tup[k-1]-1)): inc_corr_orb_rout.orb_string(molecule,l_limit,u_limit,tup[k-1][i][0],string)
      #
      # receive data dict
      #
      data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=molecule['mpi_stat'])
      #
      # probe for source
      #
      source = molecule['mpi_stat'].Get_source()
      #
      # probe for tag
      #
      tag = molecule['mpi_stat'].Get_tag()
      #
      if (tag == tags.ready):
         #
         if (i <= (n_tup[k-1]-1)):
            #
            # store job indices
            #
            string['index-1'] = k
            #
            string['index-2'] = i
            #
            # send string dict
            #
            molecule['mpi_comm'].send(string, dest=source, tag=tags.start)
            #
            # increment job indices
            #
            if (i < (n_tup[k-1]-1)):
               #
               i += 1
            #
            else:
               #
               if (k > 1):
                  #
                  k -= 1
                  #
                  i = 0
               #
               else:
                  #
                  i += 1
         #
         else:
            #
            molecule['mpi_comm'].send(None, dest=source, tag=tags.exit)
      #
      elif (tag == tags.done):
         #
         # write tuple energy
         #
         tup[data['index-1']-1][data['index-2']].append(data['e_tmp'])
         #
         # increment stat counter
         #
         counter += 1
         #
         # print status
         #
         inc_corr_utils.print_status(float(counter)/float(sum(n_tup)),level)
         #
         # collect time and restart
         #
         if (data['index-2'] == (n_tup[data['index-1']-1]-1)):
            #
            time.append(timer()-start)
            #
            start = timer()
         #
         # error check
         #
         if (data['error']):
            #
            print('problem with slave '+str(source)+' in energy_calc_mono_exp_est_par  ---  aborting...')
            #
            molecule['error'][0].append(True)
            #
            return molecule, tup
      #
      elif (tag == tags.exit):
         #
         slaves_avail -= 1
   #
   # reverse the time list
   #
   time.reverse()
   #
   return molecule, tup

def set_max_est_order(molecule,n_tup,time):
   #
   diff_order = 0
   #
   for i in range(0,len(molecule['prim_n_tuples'][0])):
      #
      if ((molecule['prim_n_tuples'][0][i] < molecule['theo_work'][0][i]) and (molecule['prim_n_tuples'][0][i] > 0)):
         #
         diff_order = i
         #
         break
   #
   if (diff_order == 0):
      #
      molecule['max_est_order'] = 0
      #
      molecule['est_order'] = 0
      #
      for _ in range(0,len(molecule['e_tot'][0])):
         #
         n_tup.append(0)
         #
         molecule['e_est'][0].append(0.0)
         #
         time.append(0.0)
      #
      return molecule
   #
   elif ((diff_order + molecule['est_order']) > (len(molecule['prim_tuple'][0])-1)):
      #
      molecule['max_est_order'] = len(molecule['prim_tuple'][0])-1
      #
      molecule['est_order'] = (len(molecule['prim_tuple'][0])-1) - diff_order
   #
   else:
      #
      molecule['max_est_order'] = diff_order + molecule['est_order']
   #
   return molecule

def inc_corr_order(molecule,k,n_tup,tup,e_tot):
   #
   for j in range(0,n_tup[k-1]):
      #
      for i in range(k-1,0,-1):
         #
         for l in range(0,n_tup[i-1]):
            #
            if (set(tup[i-1][l][0]) < set(tup[k-1][j][0])):
               #
               tup[k-1][j][1] -= tup[i-1][l][1]
   #
   e_tmp = 0.0
   #
   for j in range(0,n_tup[k-1]):
      #
      e_tmp += tup[k-1][j][1]
   #
   if (k > 1):
      #
      e_tmp += e_tot[k-2]
   #
   e_tot.append(e_tmp)
   #
   return e_tot

def inc_corr_order_est(molecule,n_tup,tup,e_est):
   #
   for k in range(1,molecule['max_est_order']+1):
      #
      for j in range(0,n_tup[k-1]):
         #
         for i in range(k-1,0,-1):
            #
            for l in range(0,n_tup[i-1]):
               #
               if (set(tup[i-1][l][0]) < set(tup[k-1][j][0])):
                  #
                  tup[k-1][j][1] -= tup[i-1][l][1]
   #
   for k in range(1,molecule['max_est_order']+1):
      #
      e_tmp = 0.0
      #
      for j in range(0,n_tup[k-1]):
         #
         found = False
         #
         for l in range(0,molecule['prim_n_tuples'][0][k-1]):
            #
            if (set(tup[k-1][j][0]) == set(molecule['prim_tuple'][0][k-1][l][0])):
               #
               found = True
               #
               break
         #
         if (not found):
            #
            e_tmp += tup[k-1][j][1]
      #
      if (k > 1):
         #
         e_tmp += e_est[k-2]
      #
      e_est.append(e_tmp)
   #
   return e_est

