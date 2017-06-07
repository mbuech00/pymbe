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


		def bcast_rst_master(self, exp, calc, time):
				""" master routine for distributing restart files """
				# wake up slaves 
				msg = {'task': 'bcast_rst'}
				# bcast
				self.comm.bcast(msg, root=0)
				# determine start index for energy kernel phase
				e_inc_end = np.argmax(exp.energy_inc[-1] == 0.0)
				if (e_inc_end == 0): e_inc_end = len(molecule['prim_energy_inc'][-1])
				# collect exp_info
				exp_info = {'len_tup': [len(exp.tuples[i]) for i in range(1,len(exp.tuples))],\
						'len_e_inc': [len(exp.energy_inc[i]) for i in range(0,len(exp.energy_inc))],\
						'min_order': calc.exp_min_order, 'e_inc_end': e_inc_end}
				# bcast info
				self.comm.bcast(exp_info, root=0)
				# bcast tuples
				for i in range(1,len(exp.tuples)):
					self.comm.Bcast([exp.tuples[i],MPI.INT], root=0)
				# bcast energy increments
				for i in range(len(exp.energy_inc)):
					if (i < (len(exp.energy_inc)-1)):
						self.comm.Bcast([exp.energy_inc[i],MPI.DOUBLE], root=0)
					else:
						self.comm.Bcast([exp.energy_inc[i][:e_inc_end],MPI.DOUBLE], root=0)
				# collect time_info
				for i in range(1,self.size):
					time_info = {'kernel': [time.mpi_time_work[1][i],
								time.mpi_time_comm[1][i],time.mpi_time_idle[1][i]],\
							'summation': [time.mpi_time_work[2][i],
								time.mpi_time_comm[2][i],time.mpi_time_idle[2][i]],\
							'screen': [time.mpi_time_work[0][i],
								time.mpi_time_comm[0][i],time.mpi_time_idle[0][i]]}
					self.comm.send(time_info, dest=i)
				#
				return
		
		
		def bcast_rst_slave(self, exp, calc, time):
				""" slave routine for distributing restart files """
				# receive exp_info
				info = self.comm.bcast(None, root=0)
				# set min_order
				calc.exp_min_order = info['min_order']
				# receive tuples
				for i in range(len(info['len_tup'])):
					buff = np.empty([info['len_tup'][i],i+2], dtype=np.int32)
					self.comm.Bcast([buff,MPI.INT], root=0)
					exp.tuples.append(buff)
				# receive e_inc
				for i in range(len(info['len_e_inc'])):
					buff = np.zeros(info['len_e_inc'][i], dtype=np.float64)
					if (i < (len(info['len_e_inc'])-1)):
						self.comm.Bcast([buff,MPI.DOUBLE], root=0)
					else:
						self.comm.Bcast([buff[:info['e_inc_end']],MPI.DOUBLE], root=0)
					exp.energy_inc.append(buff)
				# for e_inc[-1], make sure that this is distributed among the slaves
				for i in range(0,info['e_inc_end']):
					if ((i % (self.size-1)) != (self.rank-1)): exp.energy_inc[-1][i] = 0.0 
				# receive time_info
				time_info = self.comm.recv(source=0, status=self.stat)
				time.mpi_time_work_kernel = time_info['kernel'][0]
				time.mpi_time_comm_kernel = time_info['kernel'][1]
				time.mpi_time_idle_kernel = time_info['kernel'][2]
				time.mpi_time_work_summation = time_info['summation'][0]
				time.mpi_time_comm_summation = time_info['summation'][1]
				time.mpi_time_idle_summation = time_info['summation'][2]
				time.mpi_time_work_screen = time_info['screen'][0]
				time.mpi_time_comm_screen = time_info['screen'][1]
				time.mpi_time_idle_screen = time_info['screen'][2]
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
					elif (msg['task'] == 'finalize_mpi'):
						slave = False
				#
				return


