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

from mpi4py import MPI

from bg_init import InitCls


def main():
		""" main bg program """
		# initialize the calculation
		bg = InitCls()
		# now branch
		if (not bg.mpi.master):
			# proceed to main slave driver
			bg.driver.slave(bg.mpi, bg.mol, bg.calc, bg.pyscf, bg.exp, bg.time, bg.err, bg.rst)
		else:
			# proceed to main master driver
			bg.driver.master(bg.mpi, bg.mol, bg.calc, bg.pyscf, bg.exp, bg.time, bg.err, bg.prt, bg.rst)
			# print summary and plot results
			bg.res.main(bg.mpi, bg.mol, bg.calc, bg.exp, bg.time)
			# finalize
			bg.mpi.final(bg.rst)


if __name__ == '__main__':
	main()


