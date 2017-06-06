#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_main.py: main MPI driver routine for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from os import getcwd, mkdir, chdir
from os.path import isfile
from shutil import copy, rmtree
from mpi4py import MPI

from bg_mpi_utils import print_mpi_table, prepare_calc
from bg_mpi_rst import rst_dist_slave
from bg_mpi_time import init_mpi_timings, collect_screen_mpi_time, collect_kernel_mpi_time, collect_summation_mpi_time
from bg_kernel_mpi import energy_kernel_slave
from bg_summation_mpi import energy_summation_par
from bg_entanglement_mpi import entanglement_abs_par
from bg_screening_mpi import tuple_generation_slave


### remove this function (move if-clause to bg_main) and set variables elsewhere
def init_mpi(molecule):
		""" init mpi """
		# set mpi logical
		molecule['mpi_parallel'] = MPI.COMM_WORLD.Get_size() > 1
		# slaves proceed to the main slave routine, master returns
		if (MPI.COMM_WORLD.Get_rank() != 0):
			main_slave(molecule)
		else:
			molecule['mpi_master'] = True
			molecule['mpi_rank'] = 0
		#
		return 


def main_slave(molecule):
		""" main slave routine """
		# set loop/waiting logical
		slave = True
		# start waiting
		while (slave):
			# receive task
			msg = MPI.COMM_WORLD.bcast(None,root=0)
			# branch depending on task id
			if (msg['task'] == 'bcast_mol_dict'):
				# receive molecule dict from master
				molecule = MPI.COMM_WORLD.bcast(None,root=0)
				# set current mpi proc to 'slave'
				molecule['mpi_master'] = False
				# init slave mpi timings
				init_mpi_timings(molecule)
				# overwrite wrk_dir in case this is different from the one on the master node
				molecule['wrk_dir'] = getcwd()
				# update with private mpi info
				molecule['mpi_comm'] = MPI.COMM_WORLD
				molecule['mpi_size'] = molecule['mpi_comm'].Get_size()
				molecule['mpi_rank'] = molecule['mpi_comm'].Get_rank()
				molecule['mpi_name'] = MPI.Get_processor_name()
				molecule['mpi_stat'] = MPI.Status()
			elif (msg['task'] == 'init_slave_env'):
				# private scr dir
				molecule['scr_dir'] = molecule['wrk_dir']+'/'+molecule['scr_name']+'-'+str(molecule['mpi_rank'])
				# init scr env
				mkdir(molecule['scr_dir'])
				chdir(molecule['scr_dir'])
				# init tuple list
				molecule['prim_tuple'] = []
				# init e_inc list
				molecule['prim_energy_inc'] = []
				# set starting order
				if (not molecule['rst']): molecule['min_order'] = 1
			elif (msg['task'] == 'print_mpi_table'):
				print_mpi_table(molecule)
			elif (msg['task'] == 'prepare_calc_par'):
				# set mol params
				molecule['nocc'] = msg['nocc']
				molecule['nvirt'] = msg['nvirt']
				molecule['ncore'] = msg['ncore']
				# prepare calc
				prepare_calc(molecule)
			elif (msg['task'] == 'rst_dist'):
				# distribute (receive) restart files
				rst_dist_slave(molecule) 
			elif (msg['task'] == 'entanglement_abs_par'):
				# orbital entanglement
				entanglement_abs_par(molecule,msg['l_limit'],msg['u_limit'],msg['order'],msg['calc_end'])
				collect_screen_mpi_time(molecule,msg['order'],msg['calc_end'])
			elif (msg['task'] == 'tuple_generation_par'):
				# generate tuples
				tuple_generation_slave(molecule,molecule['prim_tuple'],molecule['prim_energy_inc'],msg['thres'],msg['l_limit'],msg['u_limit'],msg['order'],'MACRO')
				collect_screen_mpi_time(molecule,msg['order'],True)
			elif (msg['task'] == 'energy_kernel_par'):
				# energy kernel
				energy_kernel_slave(molecule,molecule['prim_tuple'],molecule['prim_energy_inc'],msg['l_limit'],msg['u_limit'],msg['order'],'MACRO')
				collect_kernel_mpi_time(molecule,msg['order'])
			elif (msg['task'] == 'energy_summation_par'):
				# energy summation
				energy_summation_par(molecule,molecule['prim_tuple'],molecule['prim_energy_inc'],None,None,msg['order'],'MACRO')
				collect_summation_mpi_time(molecule,msg['order'])
			elif (msg['task'] == 'remove_slave_env'):
				# remove scr env
				chdir(molecule['wrk_dir'])
				rmtree(molecule['scr_dir'],ignore_errors=True)
			elif (msg['task'] == 'finalize_mpi'):
				slave = False
		#
		return


