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

from bg_mpi_wrapper import finalize_mpi
from bg_mpi_main import init_mpi
from bg_mpi_time import calc_mpi_timings
from bg_setup import init_calc
from bg_utils import ref_calc, term_calc
from bg_print import print_main_header, print_main_end 
from bg_summary import summary_main
from bg_driver import main_drv
from bg_plotting import ic_plot

def main():
		""" main bg program """
		# initialize the calculation
		bg = InitCls()
		# now branch - slaves to main_slave, master to main_drv
		if (not bg.mpi.master):
			# proceed to main slave routine
			main_slave()
		else:
			# print program header
			print_main_header(molecule,'bg_output.out',True)
			# initialization done - start the calculation
			main_drv(molecule)
			# calculate timings
			calc_mpi_timings(molecule)
			# do (potential) reference calculation
			if (molecule['ref'] and (not molecule['error'][-1])): ref_calc(molecule)
			# print summary of the calculation
			summary_main(molecule)
			# plot results
			if (not molecule['error'][-1]): ic_plot(molecule)
			# terminate calculation and clean up
			term_calc(molecule,True)
			# print program end
			print_main_end(molecule)
		# finalize
		bg.mpi.comm.Barrier()
		MPI.Finalize()

if __name__ == '__main__':
	main()

