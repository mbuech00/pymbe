#!/usr/bin/env python
# -*- coding: utf-8 -*

""" molecule.py: molecule class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from pyscf import gto
import re
import sys

import restart


class MolCls(gto.Mole):
		""" molecule class (inherited from pyscf gto.Mole class) """
		def __init__(self, mpi):
				""" init parameters """
				# gto.Mole instantiation
				gto.Mole.__init__(self)
				# silence pyscf output
				self.verbose = 1
				# set geometric and molecular parameters
				if mpi.global_master:
					# default C1 symmetry
					self.symmetry = 'C1'
					# init occupation
					self.irrep_nelec = {}
					# set default value for FC
					self.frozen = False
					# init max_memory
					self.max_memory = None
					# verbose
					self.verbose = False
					# set geometry
					self.atom = self.set_geo()
					# set Mole
					self.charge, self.spin, self.symmetry, self.irrep_nelec, \
						self.basis, self.unit, self.frozen, self.verbose = self.set_mol()


		def make(self, mpi):
				""" build Mole object """
				try:
					self.build(dump_input=False, parse_arg=False)
				except RuntimeWarning as err:
					try:
						raise RuntimeError
					except RuntimeError:
						if mpi.global_master:
							restart.rm()
							sys.stderr.write('\nValueError: non-sensible input in mol.inp\n'
												'PySCF error : {0:}\n\n'.format(err))
							raise
				# set core region
				self.ncore = self._set_ncore()


		def set_geo(self):
				""" set geometry from geo.inp file """
				# read input file
				try:
					with open('geo.inp') as f:
						content = f.readlines()
						atom = ''
						for i in range(len(content)):
							if content[i].split()[0][0] == '#':
								continue
							else:
								atom += content[i]
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError: geo.inp not found\n\n')
					raise
				return atom


		def set_mol(self):
				""" set molecular parameters from mol.inp file """
				# read input file
				try:
					with open('mol.inp') as f:
						content = f.readlines()
						for i in range(len(content)):
							if content[i].split()[0][0] == '#':
								continue
							elif re.split('=',content[i])[0].strip() == 'charge':
								self.charge = int(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'spin':
								self.spin = int(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'sym':
								self.symmetry = re.split('=',content[i])[1].strip()
							elif re.split('=',content[i])[0].strip() == 'basis':
								try:
									self.basis = eval(re.split('=',content[i])[1].strip())
								except Exception:	
									self.basis = re.split('=',content[i])[1].strip()
							elif re.split('=',content[i])[0].strip() == 'unit':
								self.unit = re.split('=',content[i])[1].strip()
							elif re.split('=',content[i])[0].strip() == 'frozen':
								self.frozen = re.split('=',content[i])[1].strip().upper() == 'TRUE'
							elif re.split('=',content[i])[0].strip() == 'occup':
								self.irrep_nelec = eval(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'verbose':
								self.verbose = re.split('=',content[i])[1].strip().upper() == 'TRUE'
							# error handling
							else:
								try:
									raise RuntimeError('\''+content[i].split()[0].strip()+'\'' + \
													' keyword in mol.inp not recognized')
								except Exception as err:
									restart.rm()
									sys.stderr.write('\nInputError : {0:}\n\n'.format(err))
									raise
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError: mol.inp not found\n\n')
					raise
				return self.charge, self.spin, self.symmetry, self.irrep_nelec, \
						self.basis, self.unit, self.frozen, self.verbose


		def _set_ncore(self):
				""" set ncore """
				ncore = 0
				if self.frozen:
					for i in range(self.natm):
						if self.atom_charge(i) > 2: ncore += 1
						if self.atom_charge(i) > 12: ncore += 4
						if self.atom_charge(i) > 20: ncore += 4
						if self.atom_charge(i) > 30: ncore += 6
				return ncore


