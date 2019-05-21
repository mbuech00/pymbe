#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main pymbe program
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import os.path
import shutil

import setup
import driver
import tools
import output
import results
import parallel


def main():
		""" main program """
		# general settings
		setup.settings()

		# init mpi, mol, calc, and exp objects
		mpi, mol, calc, exp = setup.main()

		if mpi.master:

			# rm out dir if present
			if os.path.isdir(output.OUT):
				shutil.rmtree(output.OUT, ignore_errors=True)

			# make out dir
			os.mkdir(output.OUT)

			# init logger
			sys.stdout = tools.Logger(output.OUT_FILE)

			# main master driver
			driver.master(mpi, mol, calc, exp)

			# re-init logger
			sys.stdout = tools.Logger(results.RES_FILE, both=False)

			# print/plot results
			results.main(mpi, mol, calc, exp)

			# finalize mpi
			parallel.finalize(mpi)

		else:

			# main slave driver
			driver.slave(mpi, mol, calc, exp)


if __name__ == '__main__':
	main()


