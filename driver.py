#!/usr/bin/env python
# -*- coding: utf-8 -*

""" driver.py: driver module """

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

import rst
import mbe
import kernel
import output
import screen
import expansion


def main(mpi, mol, calc, exp):
		""" main driver routine """
		# print and time logical
		do_print = mpi.global_master and (not ((calc.exp_type == 'combined') and (exp.level == 'micro')))
		# exp class instantiation on slaves
		if (mpi.parallel):
			if (calc.exp_type in ['occupied','virtual']):
				msg = {'task': 'exp_cls', 'rst': calc.restart}
				# bcast msg
				mpi.local_comm.bcast(msg, root=0)
			else:
				if ((exp.level == 'macro') and (mpi.num_local_masters >= 1)):
					msg = {'task': 'exp_cls', 'rst': calc.restart, 'min_order': exp.min_order}
					# bcast msg
					mpi.master_comm.bcast(msg, root=0)
				else:
					msg = {'task': 'exp_cls', 'incl_idx': exp.incl_idx, 'min_order': exp.min_order}
					# bcast msg
					mpi.local_comm.bcast(msg, root=0)
					# compute and communicate distinct natural virtual orbitals
					if (calc.exp_virt == 'DNO'):
						kernel.trans_dno(mol, calc, exp) 
						mpi.bcast_mo_info(mol, calc, mpi.local_comm)
		# print expansion header
		if (do_print): output.exp_header(calc, exp)
		# restart
		if (calc.restart):
			# bcast exp info
			if (mpi.parallel): mpi.bcast_exp(calc, exp)
			# if rst, print previous results
			if (do_print):
				for exp.order in range(exp.start_order, exp.min_order):
					output.mbe_header(calc, exp)
					output.mbe_microresults(calc, exp)
					output.mbe_end(calc, exp)
					output.mbe_results(mol, calc, exp)
					exp.thres = screen.update(calc, exp)
					output.screen_header(calc, exp)
					output.screen_end(calc, exp)
					exp.rst_freq = int(max(exp.rst_freq / 2., 1.))
			# reset restart logical and init exp.order
			calc.restart = False
		# now do expansion
		for exp.order in range(exp.min_order, exp.max_order+1):
			#
			#** energy phase **#
			#
			if (do_print):
				# print mbe header
				output.mbe_header(calc, exp)
			# init energies
			if (len(exp.energy['inc']) < (exp.order - (exp.start_order - 1))):
				inc = np.empty(len(exp.tuples[-1]), dtype=np.float64)
				inc.fill(np.nan)
				exp.energy['inc'].append(inc)
			# mbe calculations
			mbe.main(mpi, mol, calc, exp)
			if (do_print):
				# print micro results
				output.mbe_microresults(calc, exp)
				# print mbe end
				output.mbe_end(calc, exp)
				# write restart files
				rst.mbe_write(calc, exp, True)
				# print mbe results
				output.mbe_results(mol, calc, exp)
			#
			#** screening phase **#
			#
			if (do_print):
				# print screen header
				output.screen_header(calc, exp)
			# orbital screening
			if (exp.order < exp.max_order):
				# start time
				if (do_print): exp.time_screen.append(MPI.Wtime())
				# perform screening
				screen.main(mpi, mol, calc, exp)
				if (do_print):
					# collect time
					exp.time_screen[-1] -= MPI.Wtime()
					exp.time_screen[-1] *= -1.0
					# write restart files
					if (not exp.conv_orb[-1]):
						rst.screen_write(exp)
					# print screen end
					output.screen_end(calc, exp)
			else:
				if (do_print):
					# print screen end
					output.screen_end(calc, exp)
					# collect time
					exp.time_screen.append(0.0)
			# update restart frequency
			if (do_print): exp.rst_freq = int(max(exp.rst_freq / 2., 1.))
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


def local_master(mpi, mol, calc):
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
				exp = expansion.ExpCls(mol, calc)
				# mark expansion as macro
				exp.level = 'macro'
				# set min order
				exp.min_order = msg['min_order']
				# receive exp info
				calc.restart = msg['rst']
				if (calc.restart): mpi.bcast_exp(calc, exp)
				# reset restart logical
				calc.restart = False
			#
			#** energy phase **#
			#
			if (msg['task'] == 'mbe_local_master'):
				exp.order = msg['exp_order']
				mbe.slave(mpi, mol, calc, exp)
			#
			#** screening phase **#
			#
			elif (msg['task'] == 'screen_local_master'):
				exp.order = msg['exp_order']
				exp.thres = msg['thres']
				screen.slave(mpi, mol, calc, exp)
			#
			#** exit **#
			#
			elif (msg['task'] == 'exit_local_master'):
				local_master = False
		# finalize
		mpi.final()


def slave(mpi, mol, calc):
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
				exp = expansion.ExpCls(mol, calc)
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
				mbe.slave(mpi, mol, calc, exp)
			#
			#** screening phase **#
			#
			elif (msg['task'] == 'screen_slave'):
				exp.order = msg['exp_order']
				exp.thres = msg['thres']
				screen.slave(mpi, mol, calc, exp)
			#
			#** exit **#
			#
			elif (msg['task'] == 'exit_slave'):
				slave = False
		# finalize
		mpi.final()
	
	
