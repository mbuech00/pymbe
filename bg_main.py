#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_main.py: python driver for Bethe-Goldstone correlation calculations. """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from mpi4py import MPI

from bg_init import InitCls
from bg_driver import driver
from bg_summary import summary
from bg_plotting import plot


def main():
		""" main bg program """
		# initialize the calculation
		bg = InitCls()
		# now branch
		if (not bg.mpi.master):
			# proceed to main slave routine
			bg.mpi.slave(bg.mol, bg.exp, bg.calc, bg.time)
		else:
			# print program header
			bg.prt.main_header()
			# call main driver
			driver(bg)
			# calculate timings
			bg.time.calc_time(bg.mpi, bg.exp)
			# print summary
			summary(bg)
			# plot results
			plot(bg)
		# finalize
		bg.mpi.comm.bcast({'task': 'exit_slave'}, root=0)
		bg.mpi.comm.Barrier()
		MPI.Finalize()


if __name__ == '__main__':
	main()


