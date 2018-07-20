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
RST = os.getcwd()+'/rst'


def restart():
		""" restart logical """
		if not os.path.isdir(RST):
			os.mkdir(RST)
			return False
		else:
			return True


def rm():
		""" remove rst directory in case of successful calc """
		shutil.rmtree(RST, ignore_errors=True)


def main(calc, exp):
		""" main restart driver """
		if not calc.restart:
			return exp.start_order
		else:
			# list filenames in files list
			files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]
			# sort the list of files
			files.sort(key=_natural_keys)
			# loop over files
			for i in range(len(files)):
				# read tuples
				if 'tup' in files[i]:
					exp.tuples.append(np.load(os.path.join(RST, files[i])))
				# read hashes
				elif 'hash' in files[i]:
					exp.hashes.append(np.load(os.path.join(RST, files[i])))
				# read e_inc
				elif 'e_inc' in files[i]:
					exp.prop['energy'][int(re.findall('\d+', files[i])[-1])]['inc'].append(np.load(os.path.join(RST, files[i])))
				# read dip_inc
				elif 'dip_inc' in files[i]:
					exp.prop['dipole'][int(re.findall('\d+', files[i])[-1])]['inc'].append(np.load(os.path.join(RST, files[i])))
				# read trans_inc
				elif 'trans_inc' in files[i]:
					exp.prop['trans'][int(re.findall('\d+', files[i])[-1])-1]['inc'].append(np.load(os.path.join(RST, files[i])))
				# read e_tot
				elif 'e_tot' in files[i]:
					exp.prop['energy'][int(re.findall('\d+', files[i])[-1])]['tot'].append(np.load(os.path.join(RST, files[i])).tolist())
				# read dip_tot
				elif 'dip_tot' in files[i]:
					exp.prop['dipole'][int(re.findall('\d+', files[i])[-1])]['tot'].append(np.load(os.path.join(RST, files[i])).tolist())
				# read trans_tot
				elif 'trans_tot' in files[i]:
					exp.prop['trans'][int(re.findall('\d+', files[i])[-1])-1]['tot'].append(np.load(os.path.join(RST, files[i])).tolist())
				# read timings
				elif 'time_mbe' in files[i]:
					exp.time['mbe'].append(np.load(os.path.join(RST, files[i])).tolist())
				elif 'time_screen' in files[i]:
					exp.time['screen'].append(np.load(os.path.join(RST, files[i])).tolist())
			return exp.tuples[-1].shape[1]


def write_fund(mol, calc):
		""" write fundamental info restart files """
		# write dimensions
		dims = {'nocc': mol.nocc, 'nvirt': mol.nvirt, 'no_exp': calc.no_exp, \
				'ne_act': calc.ne_act, 'no_act': calc.no_act}
		with open(os.path.join(RST, 'dims.rst'), 'w') as f:
			json.dump(dims, f)
		# write hf, reference, and base properties
		energies = {'hf': calc.prop['hf']['energy'], 'base': calc.base['energy'], \
					'ref': calc.prop['ref']['energy'], 'ref_base': calc.base['ref']}
		with open(os.path.join(RST, 'energies.rst'), 'w') as f:
			json.dump(energies, f)
		if calc.target['dipole']:
			dipoles = {'hf': calc.prop['hf']['dipole'].tolist(), \
						'ref': [calc.prop['ref']['dipole'][i].tolist() for i in range(len(calc.prop['ref']['dipole']))]}
			with open(os.path.join(RST, 'dipoles.rst'), 'w') as f:
				json.dump(dipoles, f)
		if calc.target['trans']:
			transitions = {'ref': [calc.prop['ref']['trans'][i].tolist() for i in range(len(calc.prop['ref']['trans']))]}
			with open(os.path.join(RST, 'transitions.rst'), 'w') as f:
				json.dump(transitions, f)
		# write expansion spaces
		np.save(os.path.join(RST, 'ref_space'), calc.ref_space)
		np.save(os.path.join(RST, 'exp_space'), calc.exp_space)
		# occupation
		np.save(os.path.join(RST, 'occup'), calc.occup)
		# write orbs
		np.save(os.path.join(RST, 'mo'), calc.mo)


def read_fund(mol, calc):
		""" read fundamental info restart files """
		# list filenames in files list
		files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]
		# sort the list of files
		files.sort(key=_natural_keys)
		# loop over files
		for i in range(len(files)):
			# read dimensions
			if 'dims' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					dims = json.load(f)
				mol.nocc = dims['nocc']; mol.nvirt = dims['nvirt']; calc.no_exp = dims['no_exp']
				calc.ne_act = dims['ne_act']; calc.no_act = dims['no_act']
			# read hf and base properties
			elif 'energies' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					energies = json.load(f)
				calc.prop['hf']['energy'] = energies['hf']; calc.base['energy'] = energies['base'] 
				calc.prop['ref']['energy'] = energies['ref']; calc.base['ref'] = energies['ref_base']
			elif 'dipoles' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					dipoles = json.load(f)
				calc.prop['hf']['dipole'] = np.asarray(dipoles['hf'])
				calc.prop['ref']['dipole'] = [np.asarray(dipoles['ref'][i]) for i in range(len(dipoles['ref']))]
			elif 'transitions' in files[i]:
				with open(os.path.join(RST, files[i]), 'r') as f:
					transitions = json.load(f)
				calc.prop['ref']['trans'] = [np.asarray(transitions['ref'][i]) for i in range(len(transitions['ref']))]
			# read expansion spaces
			elif 'ref_space' in files[i]:
				calc.ref_space = np.load(os.path.join(RST, files[i]))
			elif 'exp_space' in files[i]:
				calc.exp_space = np.load(os.path.join(RST, files[i]))
			# read occupation
			elif 'occup' in files[i]:
				calc.occup = np.load(os.path.join(RST, files[i]))
			# read orbs
			elif 'mo' in files[i]:
				calc.mo = np.load(os.path.join(RST, files[i]))
		# norb
		mol.norb = mol.nocc + mol.nvirt


def mbe_write(calc, exp):
		""" write mbe restart files """
		# write incremental and total quantities
		for i in range(calc.nroots):
			# *_inc
			np.save(os.path.join(RST, 'e_inc_{:}_state_{:}'.format(exp.order, i)), exp.prop['energy'][i]['inc'][-1])
			if calc.target['dipole']:
				np.save(os.path.join(RST, 'dip_inc_{:}_state_{:}'.format(exp.order, i)), exp.prop['dipole'][i]['inc'][-1])
			if calc.target['trans'] and i >= 1:
				np.save(os.path.join(RST, 'trans_inc_{:}_states_{:}_{:}'.format(exp.order, 0, i)), exp.prop['trans'][i-1]['inc'][-1])
			# *_tot
			np.save(os.path.join(RST, 'e_tot_{:}_state_{:}'.format(exp.order, i)), exp.prop['energy'][i]['tot'][-1])
			if calc.target['dipole']:
				np.save(os.path.join(RST, 'dip_tot_{:}_state_{:}'.format(exp.order, i)), exp.prop['dipole'][i]['tot'][-1])
			if calc.target['trans'] and i >= 1:
				np.save(os.path.join(RST, 'trans_tot_{:}_states_{:}_{:}'.format(exp.order, 0, i)), exp.prop['trans'][i-1]['tot'][-1])
		# write time
		np.save(os.path.join(RST, 'time_mbe_'+str(exp.order)), np.asarray(exp.time['mbe'][-1]))


def screen_write(exp):
		""" write screening restart files """
		# write tuples
		np.save(os.path.join(RST, 'tup_'+str(exp.order+1)), exp.tuples[-1])
		# write hashes
		np.save(os.path.join(RST, 'hash_'+str(exp.order+1)), exp.hashes[-1])
		# write time
		np.save(os.path.join(RST, 'time_screen_'+str(exp.order)), np.asarray(exp.time['screen'][-1]))


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


