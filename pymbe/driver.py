#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
driver module containing main master and slave pymbe functions
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.6'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
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
import tools
import parallel


def master(mpi, mol, calc, exp):
		"""
		this function is the main master function

		:param mpi: pymbe mpi object
		:param mol: pymbe mol object
		:param calc: pymbe calc object
		:param exp: pymbe exp object
		"""
		# print expansion headers
		print(output.main_header())
		print(output.exp_header(calc.model['method']))

		# mbe expansion
		for exp.order in range(exp.start_order, exp.max_order+1):

			#################
			#** mbe phase **#
			#################

			# init time
			exp.time['mbe'].append(0.0)

			if len(exp.tuples) > len(exp.count):

				# print header
				print(output.mbe_header(exp.tuples[-1].shape[0], \
										calc.ref_space.size + exp.tuples[-1].shape[1], exp.order))

				# start time
				time = MPI.Wtime()

				# main mbe function
				mbe.main(mpi, mol, calc, exp)

				# collect time
				exp.time['mbe'][-1] = MPI.Wtime() - time

				# write restart files
				restart.mbe_write(calc, exp)

				# print mbe end
				print(output.mbe_end(exp.count[-1], calc.ref_space.size + exp.tuples[-1].shape[1], \
										exp.time['mbe'][-1], exp.order))

				# print mbe results
				print(output.mbe_results(mol, calc, exp))

			#######################
			#** screening phase **#
			#######################

			# init time
			exp.time['screen'].append(0.0)

			if exp.order < exp.max_order:

				# print header
				print(output.screen_header(exp.order))

				# start time
				time = MPI.Wtime()

				# main screening function
				hashes, tuples = screen.master(mpi, calc.exp_space, exp.min_order, exp.order, \
												exp.hashes[-1], exp.tuples[-1])

				# append tuples and hashes
				exp.tuples.append(tuples)
				exp.hashes.append(hashes)

				# collect time
				exp.time['screen'][-1] = MPI.Wtime() - time

				# write restart files
				if exp.tuples[-1].shape[0] > 0:
					restart.screen_write(exp)

				# print screen end
				print(output.screen_end(exp.tuples[-1].shape[0], \
										exp.time['screen'][-1], exp.order))

			#########################
			#** convergence check **#
			#########################

			if exp.tuples[-1].shape[0] == 0 or exp.order == exp.max_order:

				# timings
				exp.time['mbe'] = np.asarray(exp.time['mbe'])
				exp.time['screen'] = np.asarray(exp.time['screen'])
				exp.time['total'] = exp.time['mbe'] + exp.time['screen']

				# final results
				exp.prop[calc.target]['tot'] = np.asarray(exp.prop[calc.target]['tot'])

				break


def slave(mpi, mol, calc, exp):
		"""
		this function is the main slave function

		:param mpi: pymbe mpi object
		:param mol: pymbe mol object
		:param calc: pymbe calc object
		:param exp: pymbe exp object
		"""
		# set loop/waiting logical
		slave = True

		# enter slave state
		while slave:

			# task id
			msg = mpi.comm.bcast(None, root=0)

			#################
			#** mbe phase **#
			#################

			if msg['task'] == 'mbe':

				# receive order
				exp.order = msg['order']

				# main mbe function
				mbe.main(mpi, mol, calc, exp)

			#######################
			#** screening phase **#
			#######################

			elif msg['task'] == 'screen':

				# receive order
				exp.order = msg['order']

				# main screening function
				hashes = screen.slave(mpi, calc.occup, calc.prot['scheme'], calc.thres, \
										calc.ref_space, calc.exp_space, exp.min_order, exp.order, \
										exp.hashes[-1], exp.prop[calc.target]['inc'][-1])

				# append hashes
				exp.hashes.append(hashes)

			############
			#** exit **#
			############

			elif msg['task'] == 'exit':

				slave = False

		# finalize mpi
		parallel.finalize(mpi)
	

