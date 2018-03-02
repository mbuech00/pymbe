#!/usr/bin/env python
# -*- coding: utf-8 -*

""" rst.py: restart class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
import json
from os import mkdir, listdir
from os.path import join, isfile, isdir
from shutil import rmtree
from re import search, split
from copy import deepcopy


class RstCls():
		""" restart class """
		def __init__(self, out, mpi):
				""" init restart env and parameters """
				if (mpi.global_master):
					self.rst_dir = out.wrk_dir+'/rst'
					self.rst_freq = 50000
					if (not isdir(self.rst_dir)):
						self.restart = False
						mkdir(self.rst_dir)
					else:
						self.restart = True
				#
				return


		def rm(self):
				""" remove rst directory in case of successful calc """
				rmtree(self.rst_dir, ignore_errors=True)
				#
				return


		def rst_main(self, mpi, calc, exp):
				""" main restart driver """
				if (self.restart):
					# read in exp restart files
					self.readexp(exp)
				# set min_order
				exp.min_order = exp.tuples[-1].shape[1]
				#
				return
		
		
		def update(self):
				""" update restart freq """
				#
				return int(max(self.rst_freq / 2., 1.))


		def write_fund(self, mol, calc):
				""" write fundamental info restart files """
				# write dimensions
				dims = {'nocc': mol.nocc, 'nvirt': mol.nvirt, 'no_act': calc.no_act}
				with open(join(self.rst_dir, 'dims.rst'), 'w') as f:
					json.dump(dims, f)
				# write hf and base energies
				e_hf_base = {'hf': calc.energy['hf'], 'base': calc.energy['base']}
				with open(join(self.rst_dir, 'e_hf_base.rst'), 'w') as f:
					json.dump(e_hf_base, f)
				# write expansion spaces
				np.save(join(self.rst_dir, 'ref_space'), calc.ref_space)
				np.save(join(self.rst_dir, 'exp_space'), calc.exp_space)
				# occupation
				np.save(join(self.rst_dir, 'occup'), calc.occup)
				# write orbitals
				np.save(join(self.rst_dir, 'mo'), calc.mo)
				#
				return


		def read_fund(self, mol, calc):
				""" read fundamental info restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort(key=natural_keys)
				# loop over files
				for i in range(len(files)):
					# read dimensions
					if ('dims' in files[i]):
						with open(join(self.rst_dir, files[i]), 'r') as f:
							dims = json.load(f)
						mol.nocc = dims['nocc']; mol.nvirt = dims['nvirt']; calc.no_act = dims['no_act']
					# read hf and base energies
					elif ('e_hf_base' in files[i]):
						with open(join(self.rst_dir, files[i]), 'r') as f:
							e_hf_base = json.load(f)
						calc.energy['hf'] = e_hf_base['hf']; calc.energy['base'] = e_hf_base['base'] 
					# read expansion spaces
					elif ('ref_space' in files[i]):
						calc.ref_space = np.load(join(self.rst_dir, files[i]))
					elif ('exp_space' in files[i]):
						calc.exp_space = np.load(join(self.rst_dir, files[i]))
					# read occupation
					elif ('occup' in files[i]):
						calc.occup = np.load(join(self.rst_dir, files[i]))
					# read orbitals
					elif ('mo' in files[i]):
						calc.mo = np.load(join(self.rst_dir, files[i]))
				# norb
				mol.norb = mol.nocc + mol.nvirt
				#
				return


		def mbe_write(self, calc, exp, final):
				""" write energy mbe restart files """
				# write e_inc
				np.save(join(self.rst_dir, 'e_inc_'+str(exp.order)), exp.energy['inc'][-1])
				# write micro_conv
				if (calc.exp_type == 'combined'):
					np.save(join(self.rst_dir, 'micro_conv_'+str(exp.order)), np.asarray(exp.micro_conv[-1]))
				# write time
				np.save(join(self.rst_dir, 'time_mbe_'+str(exp.order)), np.asarray(exp.time_mbe[-1]))
				# write e_tot
				if (final):
					np.save(join(self.rst_dir, 'e_tot_'+str(exp.order)), np.asarray(exp.energy['tot'][-1]))
				#
				return
		
		
		def screen_write(self, exp):
				""" write screening restart files """
				# write tuples
				np.save(join(self.rst_dir, 'tup_'+str(exp.order+1)), exp.tuples[-1])
				# write time
				np.save(join(self.rst_dir, 'time_screen_'+str(exp.order)), np.asarray(exp.time_screen[-1]))
				#
				return


		def readexp(self, exp):
				""" read expansion restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort(key=natural_keys)
				# loop over files
				for i in range(len(files)):
					# read tuples
					if ('tup' in files[i]):
						exp.tuples.append(np.load(join(self.rst_dir, files[i])))
					# read e_inc
					elif ('e_inc' in files[i]):
						exp.energy['inc'].append(np.load(join(self.rst_dir, files[i])))
					# read e_tot
					elif ('e_tot' in files[i]):
						exp.energy['tot'].append(np.load(join(self.rst_dir, files[i])).tolist())
					# read micro_conv
					elif ('micro_conv' in files[i]):
						exp.micro_conv.append(np.load(join(self.rst_dir, files[i])).tolist())
					# read timings
					elif ('time_mbe' in files[i]):
						exp.time_mbe.append(np.load(join(self.rst_dir, files[i])).tolist())
					elif ('time_screen' in files[i]):
						exp.time_screen.append(np.load(join(self.rst_dir, files[i])).tolist())
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
		return [convert(c) for c in split('(\d+)', txt)]


