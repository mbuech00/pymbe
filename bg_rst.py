#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst.py: restart utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from os import mkdir, listdir
from os.path import join, isfile, isdir
from shutil import rmtree
from re import search
from copy import deepcopy


class RstCls():
		""" restart class """
		def __init__(self, _out, _mpi):
				""" init restart env and parameters """
				self.rst_dir = _out.wrk_dir+'/rst'
				self.rst_freq = 50000.0
				if (not isdir(self.rst_dir)):
					self.restart = False
					mkdir(self.rst_dir)
				else:
					self.restart = True
				#
				return


		def rm_rst(self):
				""" remove rst directory in case of successful calc """
				rmtree(self.rst_dir, ignore_errors=True)
				#
				return


		def rst_main(self, _mpi, _calc, _exp):
				""" main restart driver """
				if (not self.restart):
					# set start order for expansion
					_exp.min_order = 1
				else:
					# read in restart files
					self.read_main(_mpi, _calc, _exp)
					# update restart frequency
					for _ in range(1, _exp.min_order): self.rst_freq = self.update()
					# bcast rst data
					if (_mpi.parallel): _mpi.bcast_rst(_calc, _exp)
					# reset restart logical
					self.restart = False
				#
				return
		
		
		def update(self):
				""" update restart freq according to start order """
				#
				return self.rst_freq / 2.


		def write_kernel(self, _mpi, _calc, _exp, _final):
				""" write energy kernel restart files """
				if (not ((_calc.exp_type == 'combined') and (_exp.level == 'micro'))):
					# write e_inc
					np.save(join(self.rst_dir, 'e_inc_' + str(_exp.order)),
							_exp.energy_inc[_exp.order - 1])
					if (_final):
						np.save(join(self.rst_dir, 'e_tot_' + str(_exp.order)),
								np.asarray(_exp.energy_tot[_exp.order - 1]))
				#
				return
		
		
		def write_screen(self, _mpi, _calc, _exp):
				""" write screening restart files """
				if (not ((_calc.exp_type == 'combined') and (_exp.level == 'micro'))):
					# write tuples
					np.save(join(self.rst_dir, 'tup_' + str(_exp.order + 1)),
							_exp.tuples[_exp.order])
				#
				return


		def read_main(self, _mpi, _calc, _exp):
				""" driver for reading of restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort()
				# loop over files
				for i in range(len(files)):
					# read tuples
					if ('tup' in files[i]):
						_exp.tuples.append(np.load(join(self.rst_dir,
											files[i])))
					# read e_inc
					elif ('e_inc' in files[i]):
						_exp.energy_inc.append(np.load(join(self.rst_dir,
														files[i])))
					# read e_tot
					elif ('e_tot' in files[i]):
						_exp.energy_tot.append(np.load(join(self.rst_dir,
														files[i])).tolist())
				# set start order for expansion
				_exp.min_order = len(_exp.tuples)
				#
				return


