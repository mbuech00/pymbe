#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_utilities.py: MPI utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def enum(*sequential,**named):
   #
   # hardcoded enums
   #
   enums = dict(zip(sequential,range(len(sequential))),**named)
   #
   return type('Enum',(), enums)

def add_tup(dict_1,dict_2,data_type):
   #
   # MPI.SUM for dictionaries of tuples
   #
   for item in dict_2:
      #
      if (item in dict_1):
         #
         dict_1[item] = [[a[0],a[1]+b[1]] for a,b in zip(dict_1[item],dict_2[item])]
      #
      else:
         #
         dict_1[item] = dict_2[item]
   #
   return dict_1

def add_time(dict_1,dict_2,data_type):
   #
   # MPI.SUM for dictionaries of timings
   #
   for item in dict_2:
      #
      if (item in dict_1):
         #
         dict_1[item] += dict_2[item]
      #
      else:
         #
         dict_1[item] = dict_2[item]
   #
   return dict_1

