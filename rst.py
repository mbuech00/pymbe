#!/usr/bin/env python
# -*- coding: utf-8 -*

""" rst.py: restart class """

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
from re import search, split
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
				if (self.restart):
					# read in _exp restart files
					self.read_exp(_exp)
				# set min_order
				_exp.min_order = _exp.tuples[-1].shape[1]
				#
				return
		
		
		def update(self):
				""" update restart freq according to start order """
				#
				return self.rst_freq / 2.


		def write_mbe(self, _calc, _exp, _final):
				""" write energy mbe restart files """
				# write e_inc
				np.save(join(self.rst_dir, 'e_inc_'+str(_exp.order)), _exp.energy['inc'][-1])
				# write micro_conv
				if (_calc.exp_type == 'combined'):
					np.save(join(self.rst_dir, 'micro_conv_'+str(_exp.order)), np.asarray(_exp.micro_conv[-1]))
				# write time
				np.save(join(self.rst_dir, 'time_mbe_'+str(_exp.order)), np.asarray(_exp.time_mbe[-1]))
				# write e_tot
				if (_final):
					np.save(join(self.rst_dir, 'e_tot_'+str(_exp.order)), np.asarray(_exp.energy['tot'][-1]))
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


		def read_exp(self, _exp):
				""" driver for reading _exp restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort(key=natural_keys)
				# loop over files
				for i in range(len(files)):
					# read tuples
					if ('tup' in files[i]):
						_exp.tuples.append(np.load(join(self.rst_dir, files[i])))
					# read e_inc
					elif ('e_inc' in files[i]):
						_exp.energy['inc'].append(np.load(join(self.rst_dir, files[i])))
					# read e_tot
					elif ('e_tot' in files[i]):
						_exp.energy['tot'].append(np.load(join(self.rst_dir, files[i])).tolist())
					# read micro_conv
					elif ('micro_conv' in files[i]):
						_exp.micro_conv.append(np.load(join(self.rst_dir, files[i])).tolist())
					# read timings
					elif ('time_mbe' in files[i]):
						_exp.time_mbe.append(np.load(join(self.rst_dir, files[i])).tolist())
					elif ('time_screen' in files[i]):
						_exp.time_screen.append(np.load(join(self.rst_dir, files[i])).tolist())
				#
				return


def convert(txt):
		""" convert strings with numbers in them """
		return int(txt) if txt.isdigit() else txt


def natural_keys(txt):
		"""
		alist.sort(key=natural_keys) sorts in human order
		http://nedbatchelder.com/blog/200712/human_sorting.html
		cf. https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
		"""
		return [ convert(c) for c in split('(\d+)', txt) ]


