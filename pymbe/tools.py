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
# E2gx (10) , E2gy (11)
# E3gx (12) , E3gy (13)
# E2uy (14) , E2ux (15)
# E3uy (16) , E3ux (17)
# E4gx (20) , E4gy (21)
# E5gx (22) , E5gy (23)
# E4uy (24) , E4ux (25)
# E5uy (26) , E5ux (27)
DEG_ID = np.array([2, 6, 10, 12, 14, 16, 20, 22, 24, 26]) 


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


def lz_check(mol, calc, exp, idx):
		""" lz symmetry check """
		# get dooh orbsym IDs for expansion space
		exp.cas_idx = core_cas(mol, exp, exp.tuples[-1][idx])[-1]
		dooh_orbs = calc.orbsym_dooh[exp.cas_idx[(calc.ref_space.size+calc.no_exp):]]
		# loop over IDs
		for sym in DEG_ID:
			# given set of x- and y-orbs
			pi_orbs = np.where((dooh_orbs == sym) | (dooh_orbs == (sym+1)))[0]
			if pi_orbs.size > 0:
				# uneven number of orbs
				if pi_orbs.size % 2 > 0:
					return False
				# are the (d2h) pi orbs not degenerated (i.e., not placed as successive orbs in a list rank by mo energies)
				d2h_orbs = np.array([x for x in exp.cas_idx[(calc.ref_space.size+calc.no_exp):][pi_orbs]])
				if np.where(np.ediff1d(d2h_orbs) == 1)[0].size < d2h_orbs.size // 2:
					return False
		return True


def filter(c, f):
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
		return cond


