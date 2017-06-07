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


def main():
		""" main bg program """
		# initialize the calculation
		bg = InitCls()
		# now branch
		if (not bg.mpi.master):
			# proceed to main slave routine
			bg.mpi.slave(bg.mol, bg.calc, bg.exp, bg.time)
		else:
			# proceed to main driver
			bg.drv.driver(bg.mpi, bg.mol, bg.calc, bg.exp,
							bg.prt, bg.time, bg.rst, bg.err)
			# print summary and plot results
			bg.res.main(bg.mpi, bg.mol, bg.calc, bg.exp, bg.time)
		# finalize
		bg.mpi.comm.bcast({'task': 'exit_slave'}, root=bg.mpi.rank)
		bg.mpi.comm.Barrier()
		MPI.Finalize()


if __name__ == '__main__':
	main()


