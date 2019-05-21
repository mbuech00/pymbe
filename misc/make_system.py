#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import numpy as np


if len(sys.argv) != 5:
    sys.exit('\n missing or too many arguments: python make_system.py shape atom_type n_atom b\n')

shape = sys.argv[1]
atom_type = sys.argv[2]
n_atoms = int(sys.argv[3])
b = float(sys.argv[4])


def main(shape, atom_type, n_atoms, b):
		""" monoatomic system of given shape with bond lengths b """
		string = '\n'
		form = ()
		if shape == 'chain':
			# chain
			for z in np.linspace(-n_atoms * b / 2. + b / 2., n_atoms * b / 2. - b / 2., num=n_atoms):
				string += '{:<3s} {:>10.5f} {:>10.5f} {:>10.5f}\n'
				form += (atom_type.upper(), 0.0, 0.0, z,)
		elif shape == 'ring':
			# ring (adapted from pyscf/tools/ring.py)
			r = b / 2. / np.sin(np.pi / n_atoms)
			for atom in range(n_atoms):
				theta = atom * (2. * np.pi / n_atoms)
				string += '{:<3s} {:>10.5f} {:>10.5f} {:>10.5f}\n'
				form += (atom_type.upper(), r * np.cos(theta), r * np.sin(theta), 0.0,)
		else:
			sys.exit('\n unknown shape. currently implemented: chain and ring')
		return string.format(*form)


print(main(shape, atom_type, n_atoms, b))


