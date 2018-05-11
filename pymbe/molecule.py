#!/usr/bin/env python
# -*- coding: utf-8 -*

""" molecule.py: molecule class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import re
import sys
import os
import ast
from pyscf import gto, symm

import tools
import restart


class MolCls(gto.Mole):
		""" molecule class (inherited from pyscf gto.Mole class) """
		def __init__(self, mpi):
				""" init parameters """
				# gto.Mole instantiation
				gto.Mole.__init__(self)
				# set defaults
				self.atom = ''
				self.mol = {'CHARGE': 0, 'SPIN': 0, 'SYM': 'c1', 'BASIS': 'sto-3g', 'UNIT': 'ang', \
							'FROZEN': False, 'OCCUP': {}, 'VERBOSE': 1, 'DEBUG': False}
				# set geometric and molecular parameters
				if mpi.global_master:
					# read atom and molecule settings
					self.atom, self.mol = self.set_mol()
					# sanity check
					self.sanity_chk()
					# translate to Mole input
					self.charge = self.mol['CHARGE']
					self.spin = self.mol['SPIN']
					self.symmetry = symm.addons.std_symb(self.mol['SYM'])
					self.basis = self.mol['BASIS']
					self.irrep_nelec = self.mol['OCCUP']
					self.verbose = self.mol['VERBOSE']
					self.unit = self.mol['UNIT']
					# add pymbe parameters
					self.frozen = self.mol['FROZEN']
					self.debug = self.mol['DEBUG']
					self.e_core = None


		def set_mol(self):
				""" set molecular parameters from input file """
				# read input file
				try:
					with open(os.getcwd()+'/input') as f:
						content = f.readlines()
						for i in range(len(content)):
							if content[i].strip():
								if content[i].split()[0][0] == '#':
									continue
								# atom
								elif re.split('=',content[i])[0].strip() == 'atom':
									for j in range(i+1, len(content)):
										if content[j][:3] == "'''" or content[j][:3] == '"""':
											break
										else:
											self.atom += content[j]
								# mol 
								elif re.split('=',content[i])[0].strip() == 'mol':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in molecule dict (mol) must be strings, dicts, ints, and bools')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.mol[key] = val
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError : input file not found\n\n')
					raise
				#
				return self.atom, self.mol


		def sanity_chk(self):
				""" sanity check for molecular parameters """
				try:
					# atom
					if not isinstance(self.atom, str):
						raise ValueError('wrong input -- atom input in geo.xyz must be a str')
					# charge
					if not isinstance(self.mol['CHARGE'], int):
						raise ValueError('wrong input -- charge input in mol dict (charge) must be an int')
					# spin
					if not isinstance(self.mol['SPIN'], int):
						raise ValueError('wrong input -- spin input (2S) in mol dict (spin) must be an int >= 0')
					if self.mol['SPIN'] < 0:
						raise ValueError('wrong input -- spin input (2S) in mol dict (spin) must be an int >= 0')
					# sym
					if not isinstance(self.mol['SYM'], str):
						raise ValueError('wrong input -- symmetry input in mol dict (sym) must be a str')
					if self.mol['SPIN'] < 0:
						raise ValueError('wrong input -- spin input (2S) in mol dict (spin) must be int >= 0')
					# spin
					if not isinstance(self.mol['SYM'], str):
						raise ValueError('wrong input -- symmetry input in mol dict (sym) must be a str')
					if symm.addons.std_symb(self.mol['SYM']) not in symm.param.POINTGROUP:
						raise ValueError('wrong input -- spin input (2S) in mol dict (spin) must be int >= 0')
					# basis
					if not isinstance(self.mol['BASIS'], (str, dict)):
						raise ValueError('wrong input -- basis set input in mol dict (basis) must be a str or a dict')
					# occup
					if not isinstance(self.mol['OCCUP'], dict):
						raise ValueError('wrong input -- occupation input in mol dict (occup) must be a dict')
					# verbose
					if not isinstance(self.mol['VERBOSE'], int):
						raise ValueError('wrong input -- verbosity input in mol dict (verbose) must be an int >= 0')
					if self.mol['VERBOSE'] < 0:
						raise ValueError('wrong input -- verbosity input in mol dict (verbose) must be an int >= 0')
					# unit
					if not isinstance(self.mol['UNIT'], str):
						raise ValueError('wrong input -- unit input in mol dict (unit) must be a str')
					# frozen
					if not isinstance(self.mol['FROZEN'], bool):
						raise ValueError('wrong input -- frozen core input in mol dict (frozen) must be a bool')
					# debug
					if not isinstance(self.mol['DEBUG'], bool):
						raise ValueError('wrong input -- debug input in mol dict (debug) must be a bool')
				except Exception as err:
					restart.rm()
					sys.stderr.write('\nValueError : {0:}\n\n'.format(err))
					raise


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
							sys.stderr.write('\nValueError: non-sensible molecule input\n'
												'PySCF error : {0:}\n\n'.format(err))
							raise
				# set core region
				self.ncore = self._set_ncore()


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


