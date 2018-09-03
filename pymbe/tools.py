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


UNDEF = 0
LZMAP = (
 1    ,  # 0  A1g
 UNDEF,  # 1  A2g
 5    ,  # 2  E1gx
-5    ,  # 3  E1gy
 UNDEF,  # 4  A2u
 2    ,  # 5  A1u
-6    ,  # 6  E1uy
 6    ,  # 7  E1ux
 UNDEF,  # 8
 UNDEF,  # 9
 7    ,  # 10 E2gx
-7    ,  # 11 E2gy
 9    ,  # 12 E3gx
-9    ,  # 13 E3gy
-8    ,  # 14 E2uy
 8    ,  # 15 E2ux
-10   ,  # 16 E3uy
 10   ,  # 17 E3ux
 UNDEF,  # 18
 UNDEF,  # 19
 11   ,  # 20 E4gx
-11   ,  # 21 E4gy
 13   ,  # 22 E5gx
-13   ,  # 23 E5gy
-12   ,  # 24 E4uy
 12   ,  # 25 E4ux
-14   ,  # 26 E5uy
 14   ,  # 27 E5ux
)


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


def filter(c, f):
		""" return result of filter condition """
		cond = False
		if f == 'c2_short':
			# C2 X^1 Sigma^+_g state before point of avoided crossing
			if np.abs(c[1, 1] - c[2, 2]) < 1.0e-05:
				if np.sign(c[1, 1]) * np.sign(c[2, 2]) > 0:
					if np.abs(c[0, 0]) > np.abs(c[1, 1]):
						cond = True
		elif f == 'c2_long':
			# C2 X^1 Sigma^+_g state after point of avoided crossing
			if np.abs(c[1, 1] - c[2, 2]) < 1.0e-05:
				if np.sign(c[1, 1]) * np.sign(c[2, 2]) > 0:
					if np.abs(c[0, 0]) < np.abs(c[1, 1]):
						cond = True
		return cond


