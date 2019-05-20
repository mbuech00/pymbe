#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main pymbe program
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.6'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import os.path
import shutil
try:
	import numpy as np
except ImportError:
	sys.stderr.write('\nImportError : numpy module not found\n\n')
try:
	from mpi4py import MPI
except ImportError:
	sys.stderr.write('\nImportError : mpi4py module not found\n\n')
try:
	from pyscf import lib, scf
except ImportError:
	sys.stderr.write('\nImportError : pyscf module not found\n\n')

import setup
import driver
import tools
import results


def settings():
		"""
		this function sets and verifies some general settings
		"""
		# force OMP_NUM_THREADS = 1
		lib.num_threads(1)

		# mute scf checkpoint files
		scf.hf.MUTE_CHKFILE = True

		# PYTHONHASHSEED = 0
		pythonhashseed = os.environ.get('PYTHONHASHSEED', -1)
		tools.assertion(int(pythonhashseed) == 0, \
						'environment variable PYTHONHASHSEED must be set to zero')


def main():
		""" main program """
		# general settings
		settings()

		# init mpi, mol, calc, and exp objects
		mpi, mol, calc, exp = setup.main()

		if mpi.master:

			# rm out dir if present
			if os.path.isdir(tools.OUT):
				shutil.rmtree(tools.OUT, ignore_errors=True)

			# make out dir
			os.mkdir(tools.OUT)

			# init logger
			sys.stdout = tools.Logger(tools.OUT_FILE)

			# main master driver
			driver.master(mpi, mol, calc, exp)

			# re-init logger
			sys.stdout = tools.Logger(tools.RES_FILE, both=False)

			# print/plot results
			results.main(mpi, mol, calc, exp)

			# finalize mpi
			parallel.finalize(mpi)

		else:

			# main slave driver
			driver.slave(mpi, mol, calc, exp)


if __name__ == '__main__':
	main()


