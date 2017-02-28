#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mpi_orbitals.py: MPI orbital-related routines for Bethe-Goldstone correlation calculations."""

import numpy as np
from mpi4py import MPI

from bg_mpi_time import timer_mpi

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def bcast_dom_master(molecule,dom,l_limit,u_limit,k,level):
   #
   #  ---  master routine
   #
   # wake up slaves
   #
   timer_mpi(molecule,'mpi_time_comm_init',k)
   #
   msg = {'task': 'orb_generator_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': k, 'level': level}
   #
   molecule['mpi_comm'].bcast(msg,root=0)
   #
   # bcast orbital domains and lower/upper limits
   #
   dom_info = {'dom': dom}
   #
   molecule['mpi_comm'].bcast(dom_info,root=0)
   #
   timer_mpi(molecule,'mpi_time_comm_init',k,True)
   #
   dom_info.clear()
   #
   return

def bcast_dom_slave(molecule,k):
   #
   #  ---  slave routine
   #
   timer_mpi(molecule,'mpi_time_comm_init',k)
   #
   # receive domains
   #
   dom_info = MPI.COMM_WORLD.bcast(None,root=0)
   #
   molecule['dom'] = dom_info['dom']
   #
   timer_mpi(molecule,'mpi_time_comm_init',k,True)
   #
   dom_info.clear()
   #
   return molecule

