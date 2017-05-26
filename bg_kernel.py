#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_kernel.py: energy kernel routines for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np

from bg_mpi_time import timer_mpi, collect_kernel_mpi_time
from bg_kernel_mpi import energy_kernel_master
from bg_utils import run_calc_corr, term_calc, orb_string
from bg_print import print_status
from bg_rst_write import rst_write_kernel


def energy_kernel(molecule, tup, e_inc, l_limit, u_limit, order, level):
		""" energy kernel phase """
		# mpi parallel version
		if (molecule['mpi_parallel']):
			energy_kernel_master(molecule,tup,e_inc,l_limit,u_limit,order,level)
			collect_kernel_mpi_time(molecule,order)
		else:
			# init string dict
			string = {'drop': ''}
			# determine start index
			if (molecule['rst'] and (order == molecule['min_order'])):
				start = np.argmax(e_inc[order-1] == 0.0)
			else:
				start = 0
			# loop over tuples
			for i in range(start,len(tup[order-1])):
				# start work time
				timer_mpi(molecule,'mpi_time_work_kernel',order)
				# write string
				orb_string(molecule,l_limit,u_limit,tup[order-1][i],string)
				# run correlated calc
				run_calc_corr(molecule,string['drop'],level)
				# write tuple energy
				e_inc[order-1][i] = molecule['e_tmp']
				# print status
				print_status(float(i+1)/float(len(tup[order-1])),level)
				# error handling
				if (molecule['error'][-1]):
					molecule['error_rank'] = 0
					molecule['error_drop'] = string['drop']
					term_calc(molecule)
				# collect work time
				timer_mpi(molecule,'mpi_time_work_kernel',order,True)
				# write restart files
				if (((i+1) % molecule['rst_freq']) == 0): rst_write_kernel(molecule,e_inc,order)
		#
		return


