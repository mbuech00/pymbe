#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_utilities.py: general utilities for Bethe-Goldstone correlation calculation."""

from os import listdir, unlink
from os.path import join, isfile
from subprocess import call
from timeit import default_timer

from bg_print import print_ref_header, print_ref_end

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.3'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def run_calc_hf(molecule):
   #
   molecule['input_hf'](molecule)
   #
   call(molecule['backend_prog_exe']+' &> OUTPUT.OUT',shell=True) 
   #
   molecule['get_dim'](molecule)
   #
   if (not molecule['error'][0][-1]): rm_dir_content(molecule)
   #
   return molecule

def run_calc_corr(molecule,drop_string,level):
   #
   molecule['input_corr'](molecule,drop_string,level)
   #
   call(molecule['backend_prog_exe']+' &> OUTPUT.OUT',shell=True)
   #
   molecule['write_energy'](molecule,level)
   #
   if (not molecule['error'][0][-1]): rm_dir_content(molecule)
   #
   return molecule

def rm_dir_content(molecule):
   #
   for the_file in listdir(molecule['scr']):
      #
      file_path = join(molecule['scr'],the_file)
      #
      try:
         #
         if isfile(file_path):
            #
            unlink(file_path)
      #
      except Exception as e:
         #
         print(e)
   #
   return

def ref_calc(molecule):
   #
   print_ref_header()
   #
   start = default_timer()
   #
   run_calc_corr(molecule,'','REF')
   #
   molecule['prim_time'][0].append(default_timer()-start)
   #
   print_ref_end(molecule)
   #
   return molecule


