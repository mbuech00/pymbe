#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_mol.py: molecule class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from os.path import isfile
from pyscf import gto


class MolCls(gto.Mole):
		""" molecule class (inherited from pyscf gto.Mole class) """
		def __init__(self, _err):
				""" init parameters """
				# gto.Mole instance
				gto.Mole.__init__(self)
				# set geometric parameters
				self.atom = self.set_geo(_err)
				# set molecular parameters
				self.charge, self.spin, self.symmetry, self.basis, \
					self.unit, self.frozen, self.verbose  = \
							self.set_mol(_err)
				# build mol
				self.build()
				# set number of core orbs
				self.ncore = self.set_ncore()
				#
				return


		def set_geo(self, _err):
				""" set geometry from bg-geo.inp file """
				# error handling
				if (not isfile('bg-geo.inp')):
					_err.error_msg = 'bg-geo.inp not found'
					_err.abort()
				# read input file
				with open('bg-geo.inp') as f:
					content = f.readlines()
					atom = ''
					for i in range(len(content)):
						atom += content[i]
				#
				return atom


		def set_mol(self, _err):
				""" set molecular parameters from bg-mol.inp file """
				# error handling
				if (not isfile('bg-mol.inp')):
					_err.error_msg = 'bg-mol.inp not found'
					_err.abort()
				# read input file
				with open('bg-mol.inp') as f:
					content = f.readlines()
					for i in range(len(content)):
						if (content[i].split()[0] == 'charge'):
							charge = int(content[i].split()[2])
						elif (content[i].split()[0] == 'spin'):
							spin = int(content[i].split()[2])
						elif (content[i].split()[0] == 'symmetry'):
							symmetry = content[i].split()[2].upper() == 'TRUE'
						elif (content[i].split()[0] == 'basis'):
							basis = content[i].split()[2]
						elif (content[i].split()[0] == 'unit'):
							unit = content[i].split()[2]
						elif (content[i].split()[0] == 'frozen'):
							frozen = content[i].split()[2].upper() == 'TRUE'
						# error handling
						else:
							_err.error_msg = content[i].split()[2] + \
											' keyword in bg-mol.inp not recognized'
							_err.abort()
				# silence pyscf output
				verbose = 0
				#
				return charge, spin, symmetry, basis, unit, frozen, verbose


		def set_ncore(self):
				""" set ncore """
				ncore = 0
				if (self.frozen):
					for i in range(self.natm):
						if (self.atom_charge(i) > 2): ncore += 1
						if (self.atom_charge(i) > 12): ncore += 4
						if (self.atom_charge(i) > 20): ncore += 4
						if (self.atom_charge(i) > 30): ncore += 6
				#
				return ncore


