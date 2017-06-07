#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_error.py: error utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from mpi4py import MPI
import sys
import traceback
from contextlib import redirect_stdout


class ErrCls():
		""" error handling """
		def __init__(out_dir):
				""" init parameters """
				self.error_msg = ''
				self.error_tup = ''
				self.error_rank = -1
				self.error_out = out_dir+'/bg_output.out'
				# set custom exception hook
				self.set_exc_hook()
				#
				return self


		def set_exc_hook(self):
				""" set an exception hook for aborting mpi """
				# save sys.excepthook
				sys_excepthook = sys.excepthook
				# define mpi exception hook
				def mpi_excepthook(t, v, tb):
					sys_excepthook(t, v, tb)
					traceback.print_last(file=self.error_out)
					self.abort()
				# overwrite sys.excepthook
				sys.excepthook = mpi_excepthook
				#
				return


		def abort():
				""" abort bg calculation in case of error """
				# write error log to bg_output.out
				with open(self.error_out,'a') as f:
					with redirect_stdout(f):
						print('')
						print('!!!!!!!!!!!!!')
						print('ERROR\n')
						if (self.error_tup == ''):
							print(' - master quits with error:\n')
						else:
							print(' - mpi proc. # {0:} quits with correlated calc. error:\n'.format(self.error_rank))
							print(self.error_msg)
							print('\nprint of the string of dropped MOs:\n')
							print(self.error_tup)
						print('\nERROR')
						print('!!!!!!!!!!!!!')
						print('')
				# abort
				MPI.COMM_WORLD.Abort()
				#
				return


