#!/usr/bin/env python
# -*- coding: utf-8 -*

""" tools.py: various pure functions """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import os
import sys
import subprocess
import numpy as np
import math

# array of degenerate (dooh) orbsym IDs
# E1gx (2) , E1gy (3)
# E1uy (6) , E1ux (7)
DEG_ID = np.array([2, 6]) 


def enum(*sequential, **named):
		""" hardcoded enums
		see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
		"""
		enums = dict(zip(sequential, range(len(sequential))), **named)
		return type('Enum', (), enums)


def fsum(a):
		""" use math.fsum to safely sum 1d array or 2d array (column-wise) """
		if a.ndim == 1:
			return math.fsum(a)
		elif a.ndim == 2:
			return np.fromiter(map(math.fsum, a.T), dtype=a.dtype, count=a.shape[1])
		else:
			NotImplementedError('tools.py: _fsum()')


def hash_2d(a):
		""" convert a 2d numpy array to a 1d array of hashes """
		return np.fromiter(map(hash_1d, a), dtype=np.int64, count=a.shape[0])


def hash_1d(a):
		""" convert a 1d numpy array to a hash """
		return hash(a.tobytes())


def hash_compare(a, b):
		""" find occurences of b in a from binary searches """
		left = a.searchsorted(b, side='left')
		right = a.searchsorted(b, side='right')
		return ((right - left) > 0).nonzero()[0], left, right


def dict_conv(old_dict):
		""" convert dict keys """
		new_dict = {}
		for key, value in old_dict.items():
			if key.lower() in ['method', 'active', 'type', 'occ', 'virt', 'scheme']:
				new_dict[key.lower()] = value.lower()
			else:
				new_dict[key.lower()] = value
		return new_dict


def git_version():
		""" return the git revision as a string
		see: https://github.com/numpy/numpy/blob/master/setup.py#L70-L92
		"""
		def _minimal_ext_cmd(cmd):
			env = {}
			for k in ['SYSTEMROOT', 'PATH', 'HOME']:
				v = os.environ.get(k)
				if v is not None:
					env[k] = v
			# LANGUAGE is used on win32
			env['LANGUAGE'] = 'C'
			env['LANG'] = 'C'
			env['LC_ALL'] = 'C'
			out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, cwd=get_pymbe_path()).communicate()[0]
			return out
		try:
			out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
			GIT_REVISION = out.strip().decode('ascii')
		except OSError:
			GIT_REVISION = "Unknown"
		return GIT_REVISION


def get_pymbe_path():
		""" return path to pymbe """
		return os.path.dirname(os.path.realpath(sys.argv[0]))


def mbe_tasks(n_tuples, num_slaves, task_size):
		""" task list for mbe phase """
		if num_slaves * task_size < n_tuples:
			n_tasks = n_tuples // task_size
		else:
			n_tasks = num_slaves
		return np.array_split(np.arange(n_tuples), n_tasks)


def screen_tasks(n_tuples, num_slaves):
		""" task list for screen phase """
		return np.array_split(np.arange(n_tuples), num_slaves)


def core_cas(mol, exp, tup):
		""" define core and cas spaces """
		cas_idx = np.asarray(sorted(exp.incl_idx + sorted(tup.tolist())))
		core_idx = np.asarray(sorted(list(set(range(mol.nocc)) - set(cas_idx))))
		return core_idx, cas_idx


def lz_mbe(orbsym, tup):
		""" lz symmetry check for mbe phase """
		# loop over IDs
		for sym in DEG_ID:
			# given set of x and y pi orbs
			pi_orbs = np.where((orbsym[tup] == sym) | (orbsym[tup] == (sym+1)))[0]
			if pi_orbs.size % 2 > 0:
				# uneven number of pi orbs
				return False
		return True


def lz_prune(orbsym, tup):
		""" lz pruning for screening phase """
		# loop over IDs
		for sym in DEG_ID:
			# given set of x and y pi orbs
			pi_orbs = np.where((orbsym[tup] == sym) | (orbsym[tup] == (sym+1)))[0]
			if set([14, 15, 22, 23]) <= set(tup):
				print('\ntup-lz = {:} , orbsym = {:} , pi_orbs = {:}\n'.format(tup, orbsym[tup], pi_orbs))
			if pi_orbs.size > 0:
				if pi_orbs.size % 2 > 0:
					# uneven number of pi orbs
					if orbsym[tup[-1]] not in [sym, sym+1]:
						# last orbital is not a pi orbital
						return False
					else:
						if orbsym[tup[-1]-1] in [sym, sym+1]:
							# this is the second in the set of degenerated pi orbs (in a list ranked by mo energies))
							return False
				else:
					# even number of pi orbs
					if np.count_nonzero(np.ediff1d(tup[pi_orbs]) == 1) < pi_orbs.size // 2:
						# the pi orbs are not degenerated (i.e., not placed as successive orbs in a list ranked by mo energies)
						return False
		return True


def filter(c, f, cas_idx):
		""" return result of filter condition """
		cond = False
		if f == 'c2_sg+':
			# C2 Sigma^+_g state
			if np.abs(np.abs(c[1, 1]) - np.abs(c[2, 2])) < 1.0e-05 and np.sign(c[1, 1]) == np.sign(c[2, 2]):
				cond = True
		elif f == 'c2_dg':
			# C2 Delta_g state
			if np.abs(np.abs(c[1, 1]) - np.abs(c[2, 2])) < 1.0e-05 and np.sign(c[1, 1]) != np.sign(c[2, 2]):
				cond = True
		if not cond:
			print('filter mismatch ({:}): cas = {:} , c[0,0] = {:.6f} , c[1,1] = {:.6f} , c[2,2] = {:.6f}'.\
					format(f, [i for i in cas_idx], c[0,0], c[1,1], c[2,2]))
		return cond


