#!/usr/bin/env python
# -*- coding: utf-8 -*

""" mol.py: molecule class """

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
import re
import sys


class MolCls(gto.Mole):
		""" molecule class (inherited from pyscf gto.Mole class) """
		def __init__(self, _mpi, _rst):
				""" init parameters """
				# gto.Mole instantiation
				gto.Mole.__init__(self)
				# silence pyscf output
				self.verbose = 1
				# set geometric and molecular parameters
				if (_mpi.global_master):
					# default C1 symmetry
					self.symmetry = 'C1'
					# init occupation
					self.irrep_nelec = {}
					# set default value for FC
					self.frozen = False
					# init max_memory
					self.max_memory = None
					# verbose
					self.verbose_prt = False
					# set geometry
					self.atom = self.set_geo(_rst)
					# set Mole
					self.charge, self.spin, self.symmetry, self.irrep_nelec, \
						self.basis, self.unit, self.frozen, self.verbose_prt = self.set_mol(_rst)
					# store symmetry
					self.comp_symmetry = self.symmetry
				#
				return


		def make(self, _mpi, _rst):
				""" build Mole object """
				if (_mpi.global_master):
					try:
						self.build(dump_input=False, parse_arg=False)
					except RuntimeWarning as err:
						try:
							raise RuntimeError
						except RuntimeError:
							_rst.rm_rst()
							sys.stderr.write('\nValueError: non-sensible input in mol.inp\n'
												'PySCF error : {0:}\n\n'.format(err))
							raise
				else:
					self.build(dump_input=False, parse_arg=False)
				#
				return


		def set_geo(self, _rst):
				""" set geometry from geo.inp file """
				# read input file
				try:
					with open('geo.inp') as f:
						content = f.readlines()
						atom = ''
						for i in range(len(content)):
							if (content[i].split()[0][0] == '#'):
								continue
							else:
								atom += content[i]
				except IOError:
					_rst.rm_rst()
					sys.stderr.write('\nIOError: geo.inp not found\n\n')
					raise
				#
				return atom


		def set_mol(self, _rst):
				""" set molecular parameters from mol.inp file """
				# read input file
				try:
					with open('mol.inp') as f:
						content = f.readlines()
						for i in range(len(content)):
							if (content[i].split()[0][0] == '#'):
								continue
							elif (re.split('=',content[i])[0].strip() == 'charge'):
								self.charge = int(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'spin'):
								self.spin = int(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'sym'):
								self.symmetry = re.split('=',content[i])[1].strip()
							elif (re.split('=',content[i])[0].strip() == 'basis'):
								self.basis = re.split('=',content[i])[1].strip()
							elif (re.split('=',content[i])[0].strip() == 'unit'):
								self.unit = re.split('=',content[i])[1].strip()
							elif (re.split('=',content[i])[0].strip() == 'frozen'):
								self.frozen = re.split('=',content[i])[1].strip().upper() == 'TRUE'
							elif (re.split('=',content[i])[0].strip() == 'occ'):
								self.irrep_nelec = eval(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'verbose'):
								self.verbose_prt = re.split('=',content[i])[1].strip().upper() == 'TRUE'
							# error handling
							else:
								try:
									raise RuntimeError('\''+content[i].split()[0].strip()+'\'' + \
													' keyword in mol.inp not recognized')
								except Exception as err:
									_rst.rm_rst()
									sys.stderr.write('\nInputError : {0:}\n\n'.format(err))
									raise
				except IOError:
					_rst.rm_rst()
					sys.stderr.write('\nIOError: mol.inp not found\n\n')
					raise
				#
				return self.charge, self.spin, self.symmetry, self.irrep_nelec, \
						self.basis, self.unit, self.frozen, self.verbose_prt


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


