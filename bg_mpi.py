#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi.py: MPI class for Bethe-Goldstone correlation calculations."""

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


class MPICls():
		""" mpi parameters """
		def __init__():
				""" init parameters """
				self.parallel = self.comm.Get_size() > 1
				if (self.parallel):
					self.comm = MPI.COMM_WORLD
					self.size = self.comm.Get_size()
					self.rank = self.comm.Get_rank()
					self.master = self.rank == 0
					self.name = MPI.Get_processor_name()
					self.stat = MPI.Status()
				#
				return self


		def bcast_hf_int(self, mol, calc):
				""" bcast hf and int info """
				if (self.master):
					# bcast to slaves
					self.comm.bcast(mol.hf, root=0)
					self.comm.bcast(mol.norb, root=0)
					self.comm.bcast(mol.nocc, root=0)
					self.comm.bcast(mol.nvirt, root=0)
					self.comm.bcast(calc.h1e, root=0)
					self.comm.bcast(calc.h2e, root=0)
				else:
					# receive from master
					mol.hf = self.comm.bcast(None, root=0)
					mol.norb = self.com.bcast(None, root=0)
					mol.nocc = self.comm.bcast(None, root=0)
					mol.nvirt = self.comm.bcast(None, root=0)
					calc.h1e = self.comm.bcast(None, root=0)
					calc.h2e = self.comm.bcast(None, root=0)
				#
				return


		def main_slave(self):
				""" main slave routine """
				# set loop/waiting logical
				slave = True
				# start waiting
				while (slave):
					# receive task
					msg = self.comm.bcast(None,root=0)
					# branch depending on task id
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


