#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_restart.py: restart utilities for Bethe-Goldstone correlation calculation."""

import numpy as np
import os
import re

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def init_restart_dirs(molecule):
   #
   # base restart name
   #
   molecule['rst_name'] = 'rst'
   molecule['rst_dir'] = molecule['wrk_dir']+'/'+molecule['rst_name']
   #
   # sanity checks
   #
   if (molecule['rst'] and (not os.path.isdir(molecule['rst_dir']))):
      #
      print('restart requested but no rst directory present in work directory, aborting ...')
      #
      molecule['error'].append(True)
   #
   elif ((not molecule['rst']) and os.path.isdir(molecule['rst_dir'])):
      #
      print('no restart requested but rst directory present in work directory, aborting ...')
      #
      molecule['error'].append(True)  
   #
   # init main restart dir
   #
   os.mkdir(molecule['rst_dir'])
   #
   # init phase dir
   #
   molecule['rst_dir_init'] = molecule['rst_dir']+'/init'
   os.mkdir(molecule['rst_dir_init'])
   #
   # kernel phase dir
   #
   molecule['rst_dir_kernel'] = molecule['rst_dir']+'/kernel'
   os.mkdir(molecule['rst_dir_kernel'])
   #
   # final phase dir
   #
   molecule['rst_dir_final'] = molecule['rst_dir']+'/final'
   os.mkdir(molecule['rst_dir_final'])
   #
   return

def write_init_restart(molecule,tup,order,level):
   #
   # write tup[k-1]
   #
   if (not ((level == 'MACRO') and molecule['conv'][-1])): np.save(os.path.join(molecule['rst_dir_init'],'tup_'+str(level)+'_'+str(order)),tup[order-1])
   #
   # write mpi init timings
   #
   np.save(os.path.join(molecule['rst_dir_init'],'mpi_time_work_init_'+str(order-1)),np.asarray(molecule['mpi_time_work_init']))
   np.save(os.path.join(molecule['rst_dir_init'],'mpi_time_work_order_'+str(order-1)),np.asarray(molecule['mpi_time_work_kernel']))
   np.save(os.path.join(molecule['rst_dir_init'],'mpi_time_work_final_'+str(order-1)),np.asarray(molecule['mpi_time_work_final']))
   #
   return

def restart_main(molecule,level):
   #
   if (not molecule['rst']):
      #
      molecule['min_order'] = 1
   #
   else:
      #
      if (level == 'MACRO'):
         #
         read_init_restart_prim(molecule)
      #
      elif (level == 'CORRE'):
         #
         read_init_restart_corr(molecule)
   #
   return molecule

def read_init_restart_prim(molecule):
   #
   # list filenames in files list
   #
   files = [f for f in os.listdir(molecule['rst_dir_init']) if os.path.isfile(os.path.join(molecule['rst_dir_init'],f))]
   #
   print('files = '+str(files))
   #
   tup_order = 0
   time_order_work = 0; time_order_comm = 0; time_order_idle = 0
   #
   for i in range(0,len(files)):
      #
      # read tuples
      #
      if (('tup' in files[i]) and ('MACRO' in files[i])):
         #
         tup_order = max(int(re.search(r'\d+',files[i]).group()),tup_order)
         #
         molecule['prim_tuple'].append(np.load(os.path.join(molecule['rst_dir_init'],files[i])))
      #
      # read init timings
      #
      elif ('time' in files[i]):
         #
         if ('work' in files[i]):
            #
            time_order_work = max(int(re.search(r'\d+',files[i]).group()),time_order_work)
            #
            molecule['mpi_time_work_init'] = np.load(os.path.join(molecule['rst_dir_init'],files[i])).tolist()
         #
         elif ('comm' in files[i]):
            #
            time_order_comm = max(int(re.search(r'\d+',files[i]).group()),time_order_comm)
            #
            molecule['mpi_time_comm_init'] = np.load(os.path.join(molecule['rst_dir_init'],files[i])).tolist()
         #
         elif ('idle' in files[i]):
            #
            time_order_idle = max(int(re.search(r'\d+',files[i]).group()),time_order_idle)
            #
            molecule['mpi_time_idle_init'] = np.load(os.path.join(molecule['rst_dir_init'],files[i])).tolist()
   #
   # does the orders match up?
   #
   print('tup_order = '+str(tup_order))
   print('time_order_work = {0:} , time_order_comm = {1:} , time_order_idle = {2:}'.format(time_order_work,time_order_comm,time_order_idle))
   #
   fail = False
   #
   if ((time_order_work != time_order_comm) or (time_order_work != time_order_idle) or (time_order_comm != time_order_idle)): fail = True
   #
   if (not ((tup_order == time_order_work) or (tup_order == (time_order_work-1)))): fail = True
   #
   if (fail):
      #
      print('init restart failed - mismatch between tuple and timing restart files, aborting ...')
      #
      molecule['error'].append(True)
   #
   if (tup_order > 0): molecule['min_order'] = tup_order
   #
   return molecule

def read_init_restart_corr(molecule):
   #
   # list filenames in files list
   #
   files = [f for f in os.listdir(molecule['rst_dir_init']) if os.path.isfile(os.path.join(molecule['rst_dir_init'],f))]
   #
   tup_order = 0
   #
   # read tuples
   #
   for i in range(0,len(files)):
      #
      # read tuples
      #
      if (('tup' in files[i]) and ('CORRE' in files[i])):
         #
         tup_order = max(int(re.search(r'\d+',files[i]).group()),tup_order)
         #
         molecule['corr_tuple'].append(np.load(os.path.join(molecule['rst_dir_init'],files[i])))
   #
   if (tup_order > 0): molecule['min_corr_order'] = tup_order
   #
   return molecule

