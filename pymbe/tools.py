#!/usr/bin/env python
# -*- coding: utf-8 -*

""" tools.py: various functions """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np

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


def hash_conv(a):
		""" convert a 1d numpy array to a hash """
		return hash(a.tobytes())


def hash_compare(a, b):
		""" find occurences of b in a """
		return np.where(np.in1d(a, b))[0]


def upper(old_dict):
		""" capitalize dict keys """
		new_dict = {}
		for key, value in old_dict.items():
			if key.upper() in ['METHOD', 'ACTIVE', 'TYPE', 'OCC', 'VIRT', 'SCHEME']:
				new_dict[key.upper()] = value.upper()
			else:
				new_dict[key.upper()] = value
		return new_dict


