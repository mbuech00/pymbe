#!/usr/bin/env python

#
# python driver for inc.-corr. calculations.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall/Winter 2016 + Winter/Spring 2017, Mainz, Germnay.
#

from mpi4py import MPI

from inc_corr_mpi import init_mpi, red_mpi_timings, finalize_mpi
from inc_corr_utils import init_calc, ref_calc, term_calc
from inc_corr_info import sanity_chk
from inc_corr_print import print_bg_header, print_summary, print_bg_end 
from inc_corr_driver import main_drv
from inc_corr_plot import ic_plot

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

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
      #  ---  print program header...  ---
      #
      print_bg_header(molecule)
      #
      #  ---  initialization done - start the calculation...  ---
      #
      main_drv(molecule)
      #
      #  ---  start (potential) reference calculation...  ---
      #
      if (molecule['ref'] and (not molecule['error'][0][-1])): ref_calc(molecule)
      #
      #  ---  collect mpi timings from slaves  ---
      #
      if (molecule['mpi_parallel']): red_mpi_timings(molecule)
      #
      #  ---  print summary of the calculation  ---
      #
      print_summary(molecule)
      #
      #  ---  plot the results of the calculation  ---
      #
      if (not molecule['error'][0][-1]): ic_plot(molecule)
      #
      #  ---  terminate calculation and clean up...  ---
      #
      term_calc(molecule)
      #
      #  ---  print program end...  ---
      #
      print_bg_end(molecule)
   #
   #  ---  finalize mpi...  ---
   #
   finalize_mpi(molecule)

if __name__ == '__main__':
   #
   main()

