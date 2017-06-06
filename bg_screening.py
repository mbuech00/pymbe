#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_screening.py: screening routines for Bethe-Goldstone correlation calculations. """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from itertools import combinations

from bg_mpi_time import timer_mpi, collect_screen_mpi_time
from bg_screening_mpi import tuple_generation_master
from bg_print import print_screening


def screening_main(molecule, tup, e_inc, thres, l_limit, u_limit, order, level):
		""" driver routine for input generation """
		tuple_generation(molecule, tup, e_inc, thres, l_limit, u_limit, order, level)
		# print screening results
		print_screen_results(molecule,tup,order,thres,level)
	    # collect timings
		if (molecule['mpi_parallel']): collect_screen_mpi_time(molecule,order,True)
		# update threshold and restart frequency
		update_thres_and_rst_freq(molecule,order)
        #	
		return


def tuple_generation(molecule, tup, e_inc, thres, l_limit, u_limit, order, level):
	""" input generation for subsequent order """
	if (molecule['mpi_parallel']):
		# mpi parallel version
		tuple_generation_master(molecule,tup,e_inc,thres,l_limit,u_limit,order,level)
	else:
		# start time
		timer_mpi(molecule,'mpi_time_work_screen',order)
		# determine which tuples have contributions below the threshold
		allow_tuple = tup[-1][np.where(np.abs(e_inc[-1]) >= thres)]
		# init bookkeeping variables
		molecule['screen_count'] = 0; tmp = []; combs = []
        # loop over parent tuples
		for i in range(0,len(tup[-1])):
			# generate list with all subsets of particular tuple
			combs = list(list(comb) for comb in combinations(tup[-1][i],order-1))
			# loop through possible orbitals to augment the combinations with
			for m in range(tup[-1][i][-1]+1,(l_limit+u_limit)+1):
				# init screening logical
				screen = False
				# loop over subset combinations
				for j in range(0,len(combs)):
					# check whether or not the particular tuple is actually allowed
					if (not np.equal(combs[j]+[m],tup[-1]).all(axis=1).any()):
						# screen away
						screen = True
						break
				if (not screen):
	                # loop over subset combinations
					for j in range(0,len(combs)):
						# check whether the particular tuple among negligible tuples
						if (not np.equal(combs[j]+[m],allow_tuple).all(axis=1).any()):
							# screen away
							screen = True
							break
				# if tuple is allowed, add to child tuple list, otherwise screen away
				if (not screen):
					tmp.append(tup[-1][i].tolist()+[m])
				else:
					molecule['screen_count'] += 1
		# when done, write to tup list or mark expansion as converged
		if (len(tmp) >= 1):
			tup.append(np.array(tmp,dtype=np.int32))
		else:
			molecule['conv_orb'].append(True)
		# delete local variables
		del combs; del tmp
		# end time
		timer_mpi(molecule,'mpi_time_work_screen',order,True)
		#
		return


def update_thres_and_rst_freq(molecule, order):
	""" update threshold and restart frequency """
    # update threshold with dampening
	molecule['prim_exp_thres'] = (molecule['prim_exp_scaling'])**(order) * molecule['prim_exp_thres_init']
	# update restart frequency by halving it
	molecule['rst_freq'] /= 2.
	#
	return


