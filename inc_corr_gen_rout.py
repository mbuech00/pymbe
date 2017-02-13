#!/usr/bin/env python

#
# generel, yet specific routines for inc-corr calcs.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import os
import shutil

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def mk_out_dir(directory):
   #
   command = 'mkdir '+directory
   os.system(command)
   #
   return

def mk_scr_dir(directory):
   #
   os.mkdir(directory)
   #
   return

def rm_scr_dir(directory):
   #
   os.rmdir
   #
   return

def cd_dir(directory):
   #
   os.chdir(directory)
   #
   return

def save_err_out(directory):
   #
   shutil.copy(molecule['scr']+'/OUTPUT.OUT',molecule['wrk']+'/OUTPUT.OUT')
   #
   return

def setup_calc(directory):
   #
   mk_scr_dir(directory)
   #
   cd_dir(directory)
   #
   return

