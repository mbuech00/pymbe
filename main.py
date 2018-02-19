#!/usr/bin/env python
# -*- coding: utf-8 -*

""" main.py: main program """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
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
		bg = InitCls()
		# now branch
		if (not bg.mpi.global_master):
			if (bg.mpi.local_master):
				# proceed to local master driver
				bg.drv.local_master(bg.mpi, bg.mol, bg.calc, bg.kernel, bg.rst)
			else:
				# proceed to slave driver
				bg.drv.slave(bg.mpi, bg.mol, bg.calc, bg.kernel)
		else:
			# proceed to main driver
			bg.drv.main(bg.mpi, bg.mol, bg.calc, bg.kernel, bg.exp, bg.prt, bg.rst)
			# print summary and plot results
			bg.res.main(bg.mpi, bg.mol, bg.calc, bg.exp)
			# finalize
			bg.mpi.final(bg.rst)


if __name__ == '__main__':
	main()

