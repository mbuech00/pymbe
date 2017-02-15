#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_main.py: python driver for Bethe-Goldstone correlation calculations.

written by Janus J. Eriksen (jeriksen@uni-mainz.de), 2016-2017, Mainz, Germany."""

from mpi4py import MPI

from bg_mpi_utilities import init_mpi, finalize_mpi, abort_mpi
from bg_mpi_kernels import red_mpi_timings
from bg_setup import init_calc, term_calc
from bg_utilities import ref_calc
from bg_print import print_main_header, print_summary, print_main_end 
from bg_driver import main_drv
from bg_plotting import ic_plot

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def main():
   #
   #  ---  init molecule dict...  ---
   #
   molecule = {}
   #
   #  ---  init mpi...  ---
   #
   init_mpi(molecule)
   #
   #  ---  master only...  ---
   #
   if (MPI.COMM_WORLD.Get_rank() == 0):
      #
      #  ---  initialize the calculation...  ---
      #
      init_calc(molecule)
      #
      #
      if (molecule['error'][-1]): abort_mpi(molecule)
      #
      #  ---  print program header...  ---
      #
      print_main_header(molecule)
      #
      #  ---  initialization done - start the calculation...  ---
      #
      main_drv(molecule)
      #
      #  ---  collect mpi timings from slaves  ---
      #
      if (molecule['mpi_parallel']): red_mpi_timings(molecule)
      #
      #  ---  start (potential) reference calculation...  ---
      #
      if (molecule['ref'] and (not molecule['error'][-1])): ref_calc(molecule)
      #
      #  ---  print summary of the calculation  ---
      #
      print_summary(molecule)
      #
      #  ---  plot the results of the calculation  ---
      #
      if (not molecule['error'][-1]): ic_plot(molecule)
      #
      #  ---  terminate calculation and clean up...  ---
      #
      term_calc(molecule)
      #
      #  ---  print program end...  ---
      #
      print_main_end(molecule)
   #
   #  ---  finalize mpi...  ---
   #
   finalize_mpi(molecule)

if __name__ == '__main__':
   #
   main()

