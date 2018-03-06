#!/usr/bin/env python
# -*- coding: utf-8 -*

""" restart.py: restart module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
import json
import os
import os.path
import shutil
import re


# rst parameters
_rst = os.getcwd()+'/rst'


def restart():
		""" restart logical """
		if not os.path.isdir(_rst):
			os.mkdir(_rst)
			return False
		else:
			return True


def rm():
		""" remove rst directory in case of successful calc """
		shutil.rmtree(_rst, ignore_errors=True)


def main(calc, exp):
		""" main restart driver """
		if not calc.restart:
			return exp.start_order
		else:
			# list filenames in files list
			files = [f for f in os.listdir(_rst) if os.path.isfile(os.path.join(_rst, f))]
			# sort the list of files
			files.sort(key=_natural_keys)
			# loop over files
			for i in range(len(files)):
				# read tuples
				if 'tup' in files[i]:
					exp.tuples.append(np.load(os.path.join(_rst, files[i])))
				# read e_inc
				elif 'e_inc' in files[i]:
					exp.energy['inc'].append(np.load(os.path.join(_rst, files[i])))
				# read e_tot
				elif 'e_tot' in files[i]:
					exp.energy['tot'].append(np.load(os.path.join(_rst, files[i])).tolist())
				# read timings
				elif 'time_mbe' in files[i]:
					exp.time['mbe'].append(np.load(os.path.join(_rst, files[i])).tolist())
				elif 'time_screen' in files[i]:
					exp.time['screen'].append(np.load(os.path.join(_rst, files[i])).tolist())
			return exp.tuples[-1].shape[1]


def write_fund(mol, calc):
		""" write fundamental info restart files """
		# write dimensions
		dims = {'nocc': mol.nocc, 'nvirt': mol.nvirt, 'no_act': calc.no_act}
		with open(os.path.join(_rst, 'dims.rst'), 'w') as f:
			json.dump(dims, f)
		# write hf and base energies
		energies = {'hf': calc.energy['hf'], 'base': calc.energy['base'], \
					'ref': calc.energy['ref'], 'ref_base': calc.energy['ref_base']}
		with open(os.path.join(_rst, 'energies.rst'), 'w') as f:
			json.dump(energies, f)
		# write expansion spaces
		np.save(os.path.join(_rst, 'ref_space'), calc.ref_space)
		np.save(os.path.join(_rst, 'exp_space'), calc.exp_space)
		# occupation
		np.save(os.path.join(_rst, 'occup'), calc.occup)
		# write orbitals
		np.save(os.path.join(_rst, 'mo'), calc.mo)


def read_fund(mol, calc):
		""" read fundamental info restart files """
		# list filenames in files list
		files = [f for f in os.listdir(_rst) if os.path.isfile(os.path.join(_rst, f))]
		# sort the list of files
		files.sort(key=_natural_keys)
		# loop over files
		for i in range(len(files)):
			# read dimensions
			if 'dims' in files[i]:
				with open(os.path.join(_rst, files[i]), 'r') as f:
					dims = json.load(f)
				mol.nocc = dims['nocc']; mol.nvirt = dims['nvirt']; calc.no_act = dims['no_act']
			# read hf and base energies
			elif 'energies' in files[i]:
				with open(os.path.join(_rst, files[i]), 'r') as f:
					energies = json.load(f)
				calc.energy['hf'] = energies['hf']; calc.energy['base'] = energies['base'] 
				calc.energy['ref'] = energies['ref']; calc.energy['ref_base'] = energies['ref_base']
			# read expansion spaces
			elif 'ref_space' in files[i]:
				calc.ref_space = np.load(os.path.join(_rst, files[i]))
			elif 'exp_space' in files[i]:
				calc.exp_space = np.load(os.path.join(_rst, files[i]))
			# read occupation
			elif 'occup' in files[i]:
				calc.occup = np.load(os.path.join(_rst, files[i]))
			# read orbitals
			elif 'mo' in files[i]:
				calc.mo = np.load(os.path.join(_rst, files[i]))
		# norb
		mol.norb = mol.nocc + mol.nvirt


def mbe_write(calc, exp, final):
		""" write energy mbe restart files """
		# write e_inc
		np.save(os.path.join(_rst, 'e_inc_'+str(exp.order)), exp.energy['inc'][-1])
		# write time
		np.save(os.path.join(_rst, 'time_mbe_'+str(exp.order)), np.asarray(exp.time['mbe'][-1]))
		# write e_tot
		if final:
			np.save(os.path.join(_rst, 'e_tot_'+str(exp.order)), np.asarray(exp.energy['tot'][-1]))


def screen_write(exp):
		""" write screening restart files """
		# write tuples
		np.save(os.path.join(_rst, 'tup_'+str(exp.order+1)), exp.tuples[-1])
		# write time
		np.save(os.path.join(_rst, 'time_screen_'+str(exp.order)), np.asarray(exp.time['screen'][-1]))


def _natural_keys(txt):
		"""
		alist.sort(key=natural_keys) sorts in human order
		http://nedbatchelder.com/blog/200712/human_sorting.html
		cf. https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
		"""
		return [_convert(c) for c in re.split('(\d+)', txt)]


def _convert(txt):
		""" convert strings with numbers in them """
		return int(txt) if txt.isdigit() else txt


