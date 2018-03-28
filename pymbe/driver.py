#!/usr/bin/env python
# -*- coding: utf-8 -*

""" driver.py: driver module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import numpy as np
from mpi4py import MPI

import restart
import mbe
import kernel
import output
import screen
import expansion
import parallel


def main(mpi, mol, calc, exp):
		""" main driver routine """
		# print expansion headers
		if mpi.global_master:
			output.main_header()
			output.exp_header(calc, exp)
		# restart
		if mpi.global_master and calc.restart:
			exp.thres, exp.rst_freq, calc.restart = _rst_print(mol, calc, exp)
		# now do expansion
		for exp.order in range(exp.min_order, exp.max_order+1):
			#
			#** mbe phase **#
			#
			# init energies
			if len(exp.energy['inc']) < exp.order - (exp.start_order - 1):
				inc = np.empty(len(exp.tuples[-1]), dtype=np.float64)
				inc.fill(np.nan)
				exp.energy['inc'].append(inc)
			# mbe calculations
			mbe.main(mpi, mol, calc, exp)
			if mpi.global_master:
				# print mbe end
				output.mbe_end(exp)
				# write restart files
				restart.mbe_write(calc, exp, True)
				# print mbe results
				output.mbe_results(mol, calc, exp)
			#
			#** screening phase **#
			#
			# orbital screening
			if exp.order < exp.max_order:
				# start time
				if mpi.global_master: exp.time['screen'].append(MPI.Wtime())
				# perform screening
				screen.main(mpi, mol, calc, exp)
				if mpi.global_master:
					# collect time
					exp.time['screen'][-1] -= MPI.Wtime()
					exp.time['screen'][-1] *= -1.0
					# write restart files
					if not exp.conv_orb[-1]: restart.screen_write(exp)
					# print screen end
					output.screen_end(exp)
			else:
				if mpi.global_master:
					# print screen end
					output.screen_end(exp)
					# collect time
					exp.time['screen'].append(0.0)
			# update restart frequency
			if mpi.global_master: exp.rst_freq = int(max(exp.rst_freq / 2., 1.))
			# convergence check
			if exp.conv_orb[-1] or exp.order == exp.max_order:
				exp.energy['tot'] = np.array(exp.energy['tot'])
				break


def master(mpi, mol, calc, exp):
		""" local master routine """
		# set loop/waiting logical
		local_master = True
		# enter local master state
		while local_master:
			# task id
			msg = mpi.master_comm.bcast(None, root=0)
			#
			#** exp class instantiation **#
			#
			if msg['task'] == 'exp_cls':
				exp = expansion.ExpCls(mol, calc)
				# set min order
				exp.min_order = msg['min_order']
				# receive exp info
				calc.restart = msg['rst']
				if calc.restart: parallel.exp(calc, exp, mpi.master_comm)
				# reset restart logical
				calc.restart = False
			#
			#** energy phase **#
			#
			if msg['task'] == 'mbe_local_master':
				exp.order = msg['exp_order']
				mbe.slave(mpi, mol, calc, exp)
			#
			#** screening phase **#
			#
			elif msg['task'] == 'screen_local_master':
				exp.order = msg['exp_order']
				exp.thres = msg['thres']
				screen.slave(mpi, mol, calc, exp)
			#
			#** exit **#
			#
			elif msg['task'] == 'exit_local_master':
				local_master = False
		# finalize
		parallel.final(mpi)


def slave(mpi, mol, calc, exp):
		""" slave routine """
		# set loop/waiting logical
		slave = True
		# enter slave state
		while slave:
			# task id
			msg = mpi.local_comm.bcast(None, root=0)
			#
			#** mbe phase **#
			#
			if msg['task'] == 'mbe':
				exp.order = msg['order']
				mbe.main(mpi, mol, calc, exp)
			#
			#** screening phase **#
			#
			elif msg['task'] == 'screen':
				exp.order = msg['order']
				exp.thres = msg['thres']
				screen.main(mpi, mol, calc, exp)
			#
			#** exit **#
			#
			elif msg['task'] == 'exit':
				slave = False
		# finalize
		parallel.final(mpi)
	

def _rst_print(mol, calc, exp):
		""" print output in case of restart """
		# init rst_freq
		rst_freq = exp.rst_freq
		for exp.order in range(exp.start_order, exp.min_order):
			output.mbe_header(exp)
			output.mbe_end(exp)
			output.mbe_results(mol, calc, exp)
			thres = screen.update(calc, exp)
			output.screen_header(exp, thres)
			output.screen_end(exp)
			rst_freq = int(max(rst_freq / 2., 1.))
		return thres, rst_freq, False

	
