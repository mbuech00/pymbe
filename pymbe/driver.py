#!/usr/bin/env python
# -*- coding: utf-8 -*

""" driver.py: driver module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
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
import tools
import parallel


def master(mpi, mol, calc, exp):
		""" master routine """
		# print expansion headers
		print(output.main_header())
		print(output.exp_header(calc.model['method']))
		# mpi assertion
		tools.assertion(mpi.size >= 2, 'PyMBE requires two or more MPI processes')
		# restart
		if calc.restart:
			_rst(mol, calc, exp)
		# now do expansion
		for exp.order in range(exp.start_order, exp.max_order+1):
			#** mbe phase **#
			# print header
			print(output.mbe_header(exp.tuples[-1].shape[0], \
									calc.ref_space.size + exp.tuples[-1].shape[1], exp.order))
			if len(exp.tuples) > len(exp.count):
				mbe.main(mpi, mol, calc, exp)
				# write restart files
				restart.mbe_write(calc, exp)
			# print mbe end
			print(output.mbe_end(exp.count[-1], \
									calc.ref_space.size + exp.tuples[-1].shape[1], \
									exp.time['mbe'][-1], exp.order))
			# print mbe results
			print(output.mbe_results(mol, calc, exp))
			#** screening phase **#
			if exp.order < exp.max_order:
				# perform screening
				screen.main(mpi, mol, calc, exp)
				# write restart files
				if exp.tuples[-1].shape[0] > 0: restart.screen_write(exp)
			else:
				# collect time
				exp.time['screen'].append(0.0)
			# convergence check
			if exp.tuples[-1].shape[0] == 0 or \
			(exp.count[-1] == 0 and exp.order > 1) or \
			exp.order == exp.max_order:
				# timings
				exp.time['mbe'] = np.asarray(exp.time['mbe'])
				exp.time['screen'] = np.asarray(exp.time['screen'])
				exp.time['total'] = exp.time['mbe'] + exp.time['screen']
				# final results
				exp.prop[calc.target]['tot'] = np.asarray(exp.prop[calc.target]['tot'])
				# print screen end
				if exp.order < exp.max_order:
					print(output.screen_end(exp.tuples[-1].shape[0], \
											exp.time['screen'][-1], exp.order, True))
				break
			else:
				# print screen end
				print(output.screen_end(exp.tuples[-1].shape[0], \
										exp.time['screen'][-1], exp.order, False))



def slave(mpi, mol, calc, exp):
		""" slave routine """
		# set loop/waiting logical
		slave = True
		# enter slave state
		while slave:
			# task id
			msg = mpi.comm.bcast(None, root=0)
			#** mbe phase **#
			if msg['task'] == 'mbe':
				exp.order = msg['order']
				mbe.main(mpi, mol, calc, exp)
			#** screening phase **#
			elif msg['task'] == 'screen':
				exp.order = msg['order']
				screen.main(mpi, mol, calc, exp)
			#** exit **#
			elif msg['task'] == 'exit':
				slave = False
		# finalize
		parallel.final(mpi)
	

def _rst(mol, calc, exp):
		""" print output in case of restart """
		print(output.main_header())
		for exp.order in range(1, exp.start_order):
			print(output.mbe_header(exp.tuples[exp.order-1].shape[0], calc.ref_space.size + exp.tuples[exp.order-1].shape[1], exp.order))
			print(output.mbe_end(exp.count[exp.order-1], calc.ref_space.size + exp.tuples[exp.order-1].shape[1], \
									exp.time['mbe'][exp.order-1], exp.order))
			print(output.mbe_results(mol, calc, exp))
			print(output.screen_header(exp.order))
			print(output.screen_end(exp.tuples[exp.order-1].shape[0], exp.time['screen'][exp.order-1], exp.order))

	
