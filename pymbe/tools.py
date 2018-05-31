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


def enum(*sequential, **named):
		""" hardcoded enums
		see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
		"""
		enums = dict(zip(sequential, range(len(sequential))), **named)
		return type('Enum', (), enums)


def tasks(n_tasks, procs):
		""" determine mpi batch sizes """
		base = int(n_tasks * 0.75 // procs) # make one large batch per proc corresponding to approx. 75 % of the tasks
		tasks = []
		for i in range(n_tasks-base*procs):
			tasks += [i+2 for p in range(procs-1)] # extra slaves tasks
			if np.sum(tasks) > float(n_tasks-base*procs):
				tasks = tasks[:-(procs-1)]
				tasks += [base for p in range(procs-1) if base > 0] # add large slave batches
				tasks = tasks[::-1]
				tasks += [1 for j in range(base)] # add master tasks
				tasks += [1 for j in range(n_tasks - int(np.sum(tasks)))] # add extra single tasks
				return tasks


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


