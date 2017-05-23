#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_utils.py: general utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from itertools import combinations, chain
from scipy.misc import comb
from os import listdir, unlink, chdir
from os.path import join, isfile
from subprocess import call
from shutil import rmtree, copy, move
from glob import glob

from bg_mpi_wrapper import abort_rout
from bg_mpi_utils import remove_slave_env
from bg_print import print_ref_header, print_ref_end


def run_calc_hf(molecule):
		""" run HF calc """
		# write input file
		molecule['input_hf'](molecule)
		# run the calc
		with open('OUTPUT_'+str(molecule['mpi_rank'])+'.OUT','w') as output: \
					call(molecule['backend_prog_exe'],stdout=output,stderr=output)
		# recover occ and virt dimensions
		molecule['get_dim'](molecule)
		# remove files from calculation
		if (not molecule['error'][-1]): rm_dir_content(molecule)
		#
		return


def run_calc_corr(molecule, drop_string, level):
		""" run correlated calcs """
		# write input file
		molecule['input_corr'](molecule,drop_string,level)
		# run the calc
		with open('OUTPUT_'+str(molecule['mpi_rank'])+'.OUT','w') as output: \
					call(molecule['backend_prog_exe'],stdout=output,stderr=output)
		# recover correlation energy
		molecule['write_energy'](molecule,level)
		# remove files from calculation
		if (not molecule['error'][-1]): rm_dir_content(molecule)
		#
		return


def rm_dir_content(molecule):
		""" remove content of directory """
		# loop over files in dir
		for the_file in listdir(molecule['scr_dir']):
			# delete the file
			file_path = join(molecule['scr_dir'],the_file)
			try:
				if isfile(file_path): unlink(file_path)
			except Exception as e: print(e)
		#
		return


def term_calc(molecule, final=False):
		""" terminate BG calc """
        # cd to work directory
		chdir(molecule['wrk_dir'])
		# normal termination
		if (final):
			# rm scratch dir
			rmtree(molecule['scr_dir'],ignore_errors=True)
			# rm restart dir
			rmtree(molecule['rst_dir'],ignore_errors=True)
			# remove slave environments
			if (molecule['mpi_parallel']): remove_slave_env(molecule)
		# error handling
		if (molecule['error'][-1]): abort_rout(molecule)
		#
		return


def ref_calc(molecule):
		""" perform reference calc """
		# print header
		print_ref_header(molecule)
		# start time
		start_time = MPI.Wtime()
		# run calc
		run_calc_corr(molecule,'','REF')
		# collect time
		molecule['ref_time'] = MPI.Wtime()-start_time
		# print info
		print_ref_end(molecule)
		#
		return


def orb_string(molecule, l_limit, u_limit, tup, string):
		""" write DROP_MO string """
		# generate list with all occ/virt orbitals
		dim = range(l_limit+1,(l_limit+u_limit)+1)
		# generate list with all orbs the should be dropped (not part of the current tuple)
		drop = sorted(list(set(dim)-set(tup.tolist())))
		# for virt scheme, explicitly drop the core orbitals for frozen core
		if ((molecule['exp'] == 'virt') and molecule['frozen']):
			for i in range(molecule['ncore'],0,-1):
				drop.insert(0,i)
		# init local variables
		inc = 0; string['drop'] = ''
        # now write string
		for i in range(0,len(drop)):
			if (inc == 0):
				string['drop'] += 'DROP_MO='+str(drop[i])
			else:
				if (drop[i] == (drop[i-1]+1)):
					if (i < (len(drop)-1)):
						if (drop[i] != (drop[i+1]-1)):
							string['drop'] += '>'+str(drop[i])
					else:
						string['drop'] += '>'+str(drop[i])
				else:
					string['drop'] += '-'+str(drop[i])
			inc += 1
		# end string
		if (string['drop'] != ''): string['drop'] += '\n'
		#
		return

def comb_index(n, k):
		""" calculate combined index """
		count = comb(n,k,exact=True)
		index = np.fromiter(chain.from_iterable(combinations(range(n),k)),int,count=count*k)
		#
		return index.reshape(-1,k)


