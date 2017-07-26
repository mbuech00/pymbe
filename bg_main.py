#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_main.py: python driver for Bethe-Goldstone correlation calculations. """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

try:
	from mpi4py import MPI
except ImportError:
	sys.stderr.write('\nImportError : mpi4py module not found\n\n')

from bg_init import InitCls


def main():
		""" main program """
		# initialize the calculation
		bg = InitCls()
		# now branch
		if (not bg.mpi.global_master):
			# proceed to main slave driver
			bg.driver.slave(bg.mpi, bg.mol, bg.calc, bg.pyscf, bg.time)
		else:
			# proceed to main driver
			bg.driver.main(bg.mpi, bg.mol, bg.calc, bg.pyscf, bg.exp, bg.time, bg.prt, bg.rst)
			# print summary and plot results
			bg.res.main(bg.mpi, bg.mol, bg.calc, bg.exp, bg.time)
			# finalize
			bg.mpi.final(bg.rst)


if __name__ == '__main__':
	main()


