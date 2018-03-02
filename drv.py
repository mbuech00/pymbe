#!/usr/bin/env python
# -*- coding: utf-8 -*

""" drv.py: driver class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import numpy as np
from mpi4py import MPI

import mbe
from screen import ScrCls
from exp import ExpCls


class DrvCls():
		""" driver class """
		def __init__(self, mol, calc):
				""" init parameters and classes """
				# init required classes
				self.mbe = mbe.MBECls()
				self.screening = ScrCls(mol, calc.exp_type)
				#
				return


		def main(self, mpi, mol, calc, kernel, exp, prt, rst):
				""" main driver routine """
				# print and time logical
				do_print = mpi.global_master and (not ((calc.exp_type == 'combined') and (exp.level == 'micro')))
				# restart
				rst.rst_main(mpi, calc, exp)
				# exp class instantiation on slaves
				if (mpi.parallel):
					if (calc.exp_type in ['occupied','virtual']):
						msg = {'task': 'exp_cls', 'type': calc.exp_type, 'rst': rst.restart}
						# bcast msg
						mpi.local_comm.bcast(msg, root=0)
					else:
						if ((exp.level == 'macro') and (mpi.num_local_masters >= 1)):
							msg = {'task': 'exp_cls', 'rst': rst.restart, 'min_order': exp.min_order}
							# bcast msg
							mpi.master_comm.bcast(msg, root=0)
						else:
							msg = {'task': 'exp_cls', 'type': 'virtual', 'incl_idx': exp.incl_idx, 'min_order': exp.min_order}
							# bcast msg
							mpi.local_comm.bcast(msg, root=0)
							# compute and communicate distinct natural virtual orbitals
							if (calc.exp_virt == 'DNO'):
								kernel.trans_dno(mol, calc, exp) 
								mpi.bcast_mo_info(mol, calc, mpi.local_comm)
				# print expansion header
				if (do_print): prt.exp_header(calc, exp)
				# restart
				if (rst.restart):
					# bcast exp info
					if (mpi.parallel): mpi.bcast_exp(calc, exp)
					# if rst, print previous results
					if (do_print):
						for exp.order in range(exp.start_order, exp.min_order):
							prt.mbe_header(calc, exp)
							prt.mbe_microresults(calc, exp)
							prt.mbe_end(calc, exp)
							prt.mbe_results(mol, calc, exp, kernel)
							exp.thres = self.screening.update(calc, exp)
							prt.screen_header(calc, exp)
							prt.screen_end(calc, exp)
							rst.rst_freq = rst.update()
					# reset restart logical and init exp.order
					rst.restart = False
				# now do expansion
				for exp.order in range(exp.min_order, exp.max_order+1):
					#
					#** energy phase **#
					#
					if (do_print):
						# print mbe header
						prt.mbe_header(calc, exp)
					# init energies
					if (len(exp.energy['inc']) < (exp.order - (exp.start_order - 1))):
						inc = np.empty(len(exp.tuples[-1]), dtype=np.float64)
						inc.fill(np.nan)
						exp.energy['inc'].append(inc)
					# mbe calculations
					self.mbe.main(mpi, mol, calc, kernel, exp, prt, rst)
					if (do_print):
						# print micro results
						prt.mbe_microresults(calc, exp)
						# print mbe end
						prt.mbe_end(calc, exp)
						# write restart files
						rst.mbe_write(calc, exp, True)
						# print mbe results
						prt.mbe_results(mol, calc, exp, kernel)
					#
					#** screening phase **#
					#
					if (do_print):
						# print screen header
						prt.screen_header(calc, exp)
					# orbital screening
					if (exp.order < exp.max_order):
						# start time
						if (do_print): exp.time_screen.append(MPI.Wtime())
						# perform screening
						self.screening.main(mpi, mol, calc, exp, rst)
						if (do_print):
							# collect time
							exp.time_screen[-1] -= MPI.Wtime()
							exp.time_screen[-1] *= -1.0
							# write restart files
							if (not exp.conv_orb[-1]):
								rst.screen_write(exp)
							# print screen end
							prt.screen_end(calc, exp)
					else:
						if (do_print):
							# print screen end
							prt.screen_end(calc, exp)
							# collect time
							exp.time_screen.append(0.0)
					# update restart frequency
					if (do_print): rst.rst_freq = rst.update()
					#
					#** convergence check **#
					#
					if (exp.conv_orb[-1] or (exp.order == exp.max_order)):
						# recast as numpy array
						exp.energy['tot'] = np.array(exp.energy['tot'])
						# now break
						break
				#
				return


		def local_master(self, mpi, mol, calc, kernel, rst):
				""" local master routine """
				# set loop/waiting logical
				local_master = True
				# enter local master state
				while (local_master):
					# task id
					msg = mpi.master_comm.bcast(None, root=0)
					#
					#** exp class instantiation **#
					#
					if (msg['task'] == 'exp_cls'):
						exp = ExpCls(mol, calc, 'occupied')
						# mark expansion as macro
						exp.level = 'macro'
						# set min order
						exp.min_order = msg['min_order']
						# receive exp info
						rst.restart = msg['rst']
						if (rst.restart): mpi.bcast_exp(calc, exp)
						# reset restart logical
						rst.restart = False
					#
					#** energy phase **#
					#
					if (msg['task'] == 'mbe_local_master'):
						exp.order = msg['exp_order']
						self.mbe.slave(mpi, mol, calc, kernel, exp, rst)
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'screen_local_master'):
						exp.order = msg['exp_order']
						exp.thres = msg['thres']
						self.screening.slave(mpi, mol, calc, exp)
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_local_master'):
						local_master = False
				# finalize
				mpi.final(None)
	
	
		def slave(self, mpi, mol, calc, kernel):
				""" slave routine """
				# set loop/waiting logical
				slave = True
				# enter slave state
				while (slave):
					# task id
					msg = mpi.local_comm.bcast(None, root=0)
					#
					#** exp class instantiation **#
					#
					if (msg['task'] == 'exp_cls'):
						exp = ExpCls(mol, calc, msg['type'])
						# mark expansion as micro
						exp.level = 'micro'
						# distinguish between occ-virt expansions and combined expansions
						if (calc.exp_type == 'combined'):
							exp.incl_idx = msg['incl_idx']
							# receive distinct natural virtual orbitals
							if (calc.exp_virt == 'DNO'):
								mpi.bcast_mo_info(mol, calc, mpi.local_comm)
						else:
							# receive exp info
							if (msg['rst']): mpi.bcast_exp(calc, exp)
					#
					#** energy phase **#
					#
					if (msg['task'] == 'mbe_slave'):
						exp.order = msg['exp_order']
						self.mbe.slave(mpi, mol, calc, kernel, exp)
					#
					#** screening phase **#
					#
					elif (msg['task'] == 'screen_slave'):
						exp.order = msg['exp_order']
						exp.thres = msg['thres']
						self.screening.slave(mpi, mol, calc, exp)
					#
					#** exit **#
					#
					elif (msg['task'] == 'exit_slave'):
						slave = False
				# finalize
				mpi.final(None)
	
	
