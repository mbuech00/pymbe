#!/usr/bin/env python
# -*- coding: utf-8 -*

""" main.py: main program """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
try:
	from mpi4py import MPI
except ImportError:
	sys.stderr.write('\nImportError : mpi4py module not found\n\n')

from init import InitCls


def main():
		""" main program """
		# initialize the calculation
		pymbe = InitCls()
		# now branch
		if (not pymbe.mpi.global_master):
			if (pymbe.mpi.local_master):
				# proceed to local master driver
				pymbe.drv.local_master(pymbe.mpi, pymbe.mol, pymbe.calc, pymbe.rst)
			else:
				# proceed to slave driver
				pymbe.drv.slave(pymbe.mpi, pymbe.mol, pymbe.calc)
		else:
			# proceed to main driver
			pymbe.drv.main(pymbe.mpi, pymbe.mol, pymbe.calc, pymbe.exp, pymbe.prt, pymbe.rst)
			# print summary and plot results
			pymbe.res.main(pymbe.mpi, pymbe.mol, pymbe.calc, pymbe.exp)
			# finalize
			pymbe.mpi.final(pymbe.rst)


if __name__ == '__main__':
	main()


