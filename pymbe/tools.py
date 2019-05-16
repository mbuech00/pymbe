#!/usr/bin/env python
# -*- coding: utf-8 -*

""" tools.py: various pure functions """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import os
import sys
import traceback
import subprocess
import numpy as np
import scipy.special
import functools
import itertools
import math
from pyscf import lo

import parallel

# output folder and files
OUT = os.getcwd()+'/output'
OUT_FILE = OUT+'/output.out'
RES_FILE = OUT+'/results.out'
# array of degenerate (dooh) orbsym IDs
# E1gx (2) , E1gy (3)
# E1uy (6) , E1ux (7)
DEG_ID = np.array([2, 3, 6, 7]) 


class Logger(object):
		""" write to both stdout and output_file """
		def __init__(self, output_file, both=True):
			self.terminal = sys.stdout
			self.log = open(output_file, 'a')
			self.both = both
		def write(self, message):
			self.log.write(message)
			if self.both:
				self.terminal.write(message)
		def flush(self):
			pass


def enum(*sequential, **named):
		""" hardcoded enums
		see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
		"""
		enums = dict(zip(sequential, range(len(sequential))), **named)
		return type('Enum', (), enums)


def time_str(time):
		""" write time as HH:MM:SS string """
		# hours, minutes, and seconds
		hours = int(time // 3600)
		minutes = int((time - (time // 3600) * 3600.)//60)
		seconds = time - hours * 3600. - minutes * 60.
		# init time string
		string = ''
		form = ()
		# write time string
		if hours > 0:
			string += '{:}h '
			form += (hours,)
		if minutes > 0:
			string += '{:}m '
			form += (minutes,)
		string += '{:.2f}s'
		form += (seconds,)
		return string.format(*form)


def fsum(a):
		""" use math.fsum to safely sum 1d array or 2d array (column-wise) """
		if a.ndim == 1:
			return math.fsum(a)
		elif a.ndim == 2:
			return np.fromiter(map(math.fsum, a.T), dtype=a.dtype, count=a.shape[1])
		else:
			raise NotImplementedError('tools.py: _fsum()')


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
		if ((right - left) > 0).all():
			return left
		else:
			return None


def dict_conv(old_dict):
		""" convert dict keys """
		new_dict = {}
		for key, value in old_dict.items():
			if key.lower() in ['method', 'active']:
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


def tasks(n_tuples, n_slaves, task_size):
		""" task list """
		if n_slaves * task_size < n_tuples:
			n_tasks = n_tuples // task_size
		else:
			n_tasks = n_slaves
		return np.array_split(np.arange(n_tuples), n_tasks)


def cas(ref_space, tup):
		""" define cas space """
		return np.sort(np.append(ref_space, tup))


def cas_allow(occup, ref_space, tup):
		""" check for presence of both occupied and virtual orbitals in cas space """
		return np.any(occup[cas(ref_space, tup)] > 0.0) and np.any(occup[cas(ref_space, tup)] == 0.0)


def core_cas(mol, ref_space, tup):
		""" define core and cas spaces """
		cas_idx = cas(ref_space, tup)
		core_idx = np.setdiff1d(np.arange(mol.nocc), cas_idx)
		return core_idx, cas_idx


def _cas_idx_cart(cas_idx):
		""" generate a cartesian product of (cas_idx, cas_idx) """
		return np.array(np.meshgrid(cas_idx, cas_idx)).T.reshape(-1, 2)


def _coor_to_idx(ij):
		""" compute lower triangular index corresponding to (i, j) (ij[0], ij[1]) """
		i = ij[0]; j = ij[1]
		if i >= j:
			return i * (i + 1) // 2 + j
		else:
			return j * (j + 1) // 2 + i


def cas_idx_tril(cas_idx):
		""" compute lower triangular cas_idx """
		cas_idx_cart = _cas_idx_cart(cas_idx)
		return np.unique(np.fromiter(map(_coor_to_idx, cas_idx_cart), \
										dtype=cas_idx_cart.dtype, count=cas_idx_cart.shape[0]))


def mat_idx(site, nx, ny):
		""" get x and y indices of a matrix """
		x = site % nx
		y = int(math.floor(float(site) / ny))
		return x, y


def near_nbrs(site_xy, nx, ny):
		""" get list of nearest neighbour indices """
		left = ((site_xy[0] - 1) % nx, site_xy[1])
		right = ((site_xy[0] + 1) % nx, site_xy[1])
		down = (site_xy[0], (site_xy[1] + 1) % ny)
		up = (site_xy[0], (site_xy[1] - 1) % ny)
		return [left, right, down, up]


def nelec(occup, tup):
		""" number of electrons in tuple of orbitals """
		occup_tup = occup[tup]
		return (np.count_nonzero(occup_tup > 0.), np.count_nonzero(occup_tup > 1.))


def ndets(occup, cas_idx, n_elec=None):
		""" estimated number of determinants in given CASCI calculation (ignoring point group symmetry) """
		if n_elec is None:
			n_elec = nelec(occup, cas_idx)
		n_orbs = cas_idx.size
		return scipy.special.binom(n_orbs, n_elec[0]) * scipy.special.binom(n_orbs, n_elec[1])


def assertion(condition, reason):
		""" assertion of condition """
		if not condition:
			# get stack
			stack = ''.join(traceback.format_stack()[:-1])
			# print stack
			print('\n\n'+stack)
			print('\n\n*** PyMBE assertion error: '+reason+' ***\n\n')
			# abort calculation
			parallel.abort()


