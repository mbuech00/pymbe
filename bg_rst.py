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
				if (_mpi.global_master):
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
					# read in _exp restart files
					self.read_exp(_exp)
					# update restart frequency
					for _ in range(1, _exp.min_order): self.rst_freq = self.update()
				# init _exp.order
				_exp.order = _exp.min_order
				#
				return
		
		
		def update(self):
				""" update restart freq according to start order """
				#
				return self.rst_freq / 2.


		def write_hf_trans(self, _mol):
				""" write hf_dens and trans_mat restart files """
				# write hf_dens
				np.save(join(self.rst_dir, 'hf_dens'), _mol.hf_dens)
				# write trans_mat
				np.save(join(self.rst_dir, 'trans_mat'), _mol.trans_mat)
				# write e_zero
				np.save(join(self.rst_dir, 'e_zero'), _mol.e_zero)
				#
				return


		def write_kernel(self, _calc, _exp, _final):
				""" write energy kernel restart files """
				# write e_inc
				np.save(join(self.rst_dir, 'e_inc_'+str(_exp.order)), _exp.energy_inc[-1])
				# write micro_conv
				if (_calc.exp_type == 'combined'):
					np.save(join(self.rst_dir, 'micro_conv_'+str(_exp.order)), np.asarray(_exp.micro_conv[-1]))
				# write time
				np.save(join(self.rst_dir, 'time_kernel_'+str(_exp.order)), np.asarray(_exp.time_kernel[-1]))
				# write e_tot
				if (_final):
					np.save(join(self.rst_dir, 'e_tot_'+str(_exp.order)), np.asarray(_exp.energy_tot[-1]))
				#
				return
		
		
		def write_screen(self, _exp):
				""" write screening restart files """
				# write tuples
				np.save(join(self.rst_dir, 'tup_'+str(_exp.order+1)), _exp.tuples[-1])
				# write time
				np.save(join(self.rst_dir, 'time_screen_'+str(_exp.order)), np.asarray(_exp.time_screen[-1]))
				#
				return


		def read_hf_trans(self, _mol):
				""" driver for reading _mol restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort()
				# loop over files
				for i in range(len(files)):
					# read hf_dens
					if ('hf_dens' in files[i]):
						_mol.hf_dens = np.load(join(self.rst_dir, files[i]))
					# read trans_mat
					elif ('trans_mat' in files[i]):
						_mol.trans_mat = np.load(join(self.rst_dir, files[i]))
					# read e_zero
					elif ('e_zero' in files[i]):
						_mol.e_zero = np.load(join(self.rst_dir, files[i]))
				#
				return


		def read_exp(self, _exp):
				""" driver for reading _exp restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort()
				# loop over files
				for i in range(len(files)):
					# read tuples
					if ('tup' in files[i]):
						_exp.tuples.append(np.load(join(self.rst_dir, files[i])))
					# read e_inc
					elif ('e_inc' in files[i]):
						_exp.energy_inc.append(np.load(join(self.rst_dir, files[i])))
					# read e_tot
					elif ('e_tot' in files[i]):
						_exp.energy_tot.append(np.load(join(self.rst_dir, files[i])).tolist())
					# read micro_conv
					elif ('micro_conv' in files[i]):
						_exp.micro_conv.append(np.load(join(self.rst_dir, files[i])).tolist())
					# read timings
					elif ('time_kernel' in files[i]):
						_exp.time_kernel.append(np.load(join(self.rst_dir, files[i])).tolist())
					elif ('time_screen' in files[i]):
						_exp.time_screen.append(np.load(join(self.rst_dir, files[i])).tolist())
				# set start order for expansion
				_exp.min_order = len(_exp.tuples)
				#
				return


