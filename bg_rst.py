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
					_exp.min_order = len(_exp.tuples[0][0])
				else:
					# read in _exp restart files
					self.read_exp(_exp)
					# set min_order
					_exp.min_order = len(_exp.tuples[-1][0])
				#
				return
		
		
		def update(self):
				""" update restart freq according to start order """
				#
				return self.rst_freq / 2.


		def write_hf_trans(self, _calc):
				""" write hf_mo_coeff and trans_mat restart files """
				# write hf_mo_coeff
				np.save(join(self.rst_dir, 'hf_mo_coeff'), _calc.hf_mo_coeff)
				# write hf_mo_occ
				np.save(join(self.rst_dir, 'hf_mo_occ'), np.array(_calc.hf_mo_occ))
				# write trans_mat
				np.save(join(self.rst_dir, 'trans_mat'), _calc.trans_mat)
				# write e_zero
				np.save(join(self.rst_dir, 'e_zero'), _calc.e_zero)
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
				# write screen_count
				np.save(join(self.rst_dir, 'screen_count_'+str(_exp.order)), _exp.screen_count[-1])
				# write time
				np.save(join(self.rst_dir, 'time_screen_'+str(_exp.order)), np.asarray(_exp.time_screen[-1]))
				#
				return


		def read_hf_trans(self, _calc):
				""" driver for reading _mol restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort()
				# loop over files
				for i in range(len(files)):
					# read hf_mo_coeff
					if ('hf_mo_coeff' in files[i]):
						_calc.hf_mo_coeff = np.load(join(self.rst_dir, files[i]))
					# read hf_mo_occ
					if ('hf_mo_occ' in files[i]):
						_calc.hf_mo_occ = np.load(join(self.rst_dir, files[i])).tolist()
					# read trans_mat
					elif ('trans_mat' in files[i]):
						_calc.trans_mat = np.load(join(self.rst_dir, files[i]))
					# read e_zero
					elif ('e_zero' in files[i]):
						_calc.e_zero = np.load(join(self.rst_dir, files[i]))
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
					# read screen_count
					elif ('screen_count' in files[i]):
						_exp.screen_count.append(np.load(join(self.rst_dir, files[i])).tolist())
					# read micro_conv
					elif ('micro_conv' in files[i]):
						_exp.micro_conv.append(np.load(join(self.rst_dir, files[i])).tolist())
					# read timings
					elif ('time_kernel' in files[i]):
						_exp.time_kernel.append(np.load(join(self.rst_dir, files[i])).tolist())
					elif ('time_screen' in files[i]):
						_exp.time_screen.append(np.load(join(self.rst_dir, files[i])).tolist())
				#
				return


