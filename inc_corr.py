#!/usr/bin/env python

#
# python driver for inc.-corr. calculations using CFOUR as backend program.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#
# Requires the path of the cfour basis GENBAS file ($CFOURBASIS) and bin directory ($CFOURBIN)
#

import inc_corr_utils
import inc_corr_gen_rout
import inc_corr_e_rout
import inc_corr_plot

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def main():
   #
   #  ---  init molecule dictionary... ---
   #
   molecule = {}
   #
   #  ---  redirect stdout to output.out - if present in wrk dir (alongside plotting output), delete these files before proceeding...  ---
   #
   inc_corr_utils.redirect_stdout(molecule)
   #
   #  ---  initialize the calculation...  ---
   #
   inc_corr_utils.init_calc(molecule)
   #
   #  ---  setup of scratch directory...  ---
   #
   inc_corr_utils.setup_calc(molecule)
   #
   #  ---  run HF calc to determine problem size parameters...  ---
   #
   inc_corr_gen_rout.run_calc_hf(molecule)
   #
   #  ---  run a few sanity checks...  ---
   #
   inc_corr_utils.sanity_chk(molecule)
   #
   #  ---  initialize (potential) domains...  ---
   #
   inc_corr_utils.init_domains(molecule)
   #
   #  ---  initialization done - start the calculation...  ---
   #
   if (molecule['exp_ctrl']):
      #
      inc_corr_e_rout.inc_corr_tuple_thres(molecule)
   #
   else:
      #
      inc_corr_e_rout.inc_corr_tuple_order(molecule)
   #
   #  ---  start (potential) reference calculation...  ---
   #
   if (molecule['ref'] and (not molecule['error'][0][-1])):
      #
      inc_corr_gen_rout.ref_calc(molecule)
   #
   #  ---  print summary of the calculation  ---
   #
   inc_corr_utils.inc_corr_summary(molecule)
   #
   #  ---  plot the results of the calculation  ---
   #
   if (not molecule['error'][0][-1]):
      #
      inc_corr_plot.ic_plot(molecule)
   #
   #  ---  terminate calculation and clean up...  ---
   #
   inc_corr_utils.term_calc(molecule)
   #

if __name__ == '__main__':
   #
   main()

