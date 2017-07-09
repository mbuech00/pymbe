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
import sys


class MolCls(gto.Mole):
		""" molecule class (inherited from pyscf gto.Mole class) """
		def __init__(self, _mpi, _rst):
				""" init parameters """
				# gto.Mole instance
				gto.Mole.__init__(self)
				# set geometric and molecular parameters
				if (_mpi.master):
					# set default value for FC
					self.frozen = False
					# set geometry
					self.atom = self.set_geo(_rst)
					# set Mole
					self.charge, self.spin, self.symmetry, self.basis, \
						self.unit, self.frozen, self.verbose  = \
								self.set_mol(_rst)
					# build mol (master)
					try:
						self.build()
					except RuntimeWarning as err:
						try:
							raise RuntimeError
						except RuntimeError:
							_rst.rm_rst()
							sys.stderr.write('\nValueError: non-sensible input in bg-mol.inp\n'
												'PySCF error : {0:}\n\n'.format(err))
							raise
					if (_mpi.parallel): _mpi.bcast_mol_info(self)
				else:
					_mpi.bcast_mol_info(self)
					self.build()
				# set number of core orbs
				self.ncore = self.set_ncore()
				#
				return


		def set_geo(self, _rst):
				""" set geometry from bg-geo.inp file """
				# read input file
				try:
					with open('bg-geo.inp') as f:
						content = f.readlines()
						atom = ''
						for i in range(len(content)):
							atom += content[i]
				except IOError:
					_rst.rm_rst()
					sys.stderr.write('\nIOError: bg-geo.inp not found\n\n')
				#
				return atom


		def set_mol(self, _rst):
				""" set molecular parameters from bg-mol.inp file """
				# read input file
				try:
					with open('bg-mol.inp') as f:
						content = f.readlines()
						for i in range(len(content)):
							if (content[i].split()[0] == 'charge'):
								self.charge = int(content[i].split()[2])
							elif (content[i].split()[0] == 'spin'):
								self.spin = int(content[i].split()[2])
							elif (content[i].split()[0] == 'symmetry'):
								self.symmetry = content[i].split()[2]
							elif (content[i].split()[0] == 'basis'):
								self.basis = content[i].split()[2]
							elif (content[i].split()[0] == 'unit'):
								self.unit = content[i].split()[2]
							elif (content[i].split()[0] == 'frozen'):
								self.frozen = content[i].split()[2].upper() == 'TRUE'
							# error handling
							else:
								try:
									raise RuntimeError('\''+content[i].split()[0]+'\'' + \
													' keyword in bg-mol.inp not recognized')
								except Exception as err:
									_rst.rm_rst()
									sys.stderr.write('\nInputError : {0:}\n\n'.format(err))
				except IOError:
					_rst.rm_rst()
					sys.stderr.write('\nIOError: bg-mol.inp not found\n\n')
				# silence pyscf output
				self.verbose = 0
				#
				return self.charge, self.spin, self.symmetry, self.basis, \
						self.unit, self.frozen, self.verbose


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


