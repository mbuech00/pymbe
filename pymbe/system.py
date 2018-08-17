#!/usr/bin/env python
# -*- coding: utf-8 -*

""" system.py: ab initio / model hamiltonian class """

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
from pyscf import gto, symm, ao2mo

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
				self.system = {'charge': 0, 'spin': 0, 'sym': 'c1', 'basis': 'sto-3g', 'unit': 'ang', \
							'frozen': False, 'occup': {}, 'verbose': 1, 'debug': False, \
							't': 1.0, 'u': 1.0, 'dim': 1, 'nsites': 6, 'pbc': True, 'nelec': 0}
				# set geometric and molecular parameters
				if mpi.global_master:
					# read atom and molecule settings
					self.atom, self.system = self.set_system()
					# sanity check
					self.sanity_chk()
					# translate to Mole input
					self.incore_anyway = True
					self.verbose = self.system['verbose']
					self.irrep_nelec = self.system['occup']
					self.charge = self.system['charge']
					self.spin = self.system['spin']
					self.symmetry = symm.addons.std_symb(self.system['sym'])
					self.basis = self.system['basis']
					self.unit = self.system['unit']
					# hubbard hamiltonian
					if not self.atom:
						self.atom = []
						self.symmetry = False
						self.ncore = 0
						self.nelectron = self.system['nelec']
						self.t = self.system['t']
						self.u = self.system['u']
						self.dim = self.system['dim']
						self.nsites = self.system['nsites']
						self.nelectron = self.system['nelec']
						self.pbc = self.system['pbc']
					# add pymbe parameters
					self.frozen = self.system['frozen']
					self.debug = self.system['debug']
					self.e_core = None


		def set_system(self):
				""" set system parameters from input file """
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
								# system 
								elif re.split('=',content[i])[0].strip() == 'system':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in system dict (system) must be strings, dicts, ints, and bools')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.system[key] = val
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError : input file not found\n\n')
					raise
				return self.atom, self.system


		def sanity_chk(self):
				""" sanity check for system parameters """
				try:
					# charge
					if not isinstance(self.system['charge'], int):
						raise ValueError('wrong input -- charge input in system dict (charge) must be an int')
					# spin
					if not isinstance(self.system['spin'], int):
						raise ValueError('wrong input -- spin input (2S) in system dict (spin) must be an int >= 0')
					if self.system['spin'] < 0:
						raise ValueError('wrong input -- spin input (2S) in system dict (spin) must be an int >= 0')
					# sym
					if not isinstance(self.system['sym'], str):
						raise ValueError('wrong input -- symmetry input in system dict (sym) must be a str')
					if symm.addons.std_symb(self.system['sym']) not in symm.param.POINTGROUP + ('Dooh', 'Coov',):
						raise ValueError('wrong input -- illegal symmetry input in system dict (sym)')
					# basis
					if not isinstance(self.system['basis'], (str, dict)):
						raise ValueError('wrong input -- basis set input in system dict (basis) must be a str or a dict')
					# occup
					if not isinstance(self.system['occup'], dict):
						raise ValueError('wrong input -- occupation input in system dict (occup) must be a dict')
					# verbose
					if not isinstance(self.system['verbose'], int):
						raise ValueError('wrong input -- verbosity input in system dict (verbose) must be an int >= 0')
					if self.system['verbose'] < 0:
						raise ValueError('wrong input -- verbosity input in system dict (verbose) must be an int >= 0')
					# unit
					if not isinstance(self.system['unit'], str):
						raise ValueError('wrong input -- unit input in system dict (unit) must be a str')
					# frozen
					if not isinstance(self.system['frozen'], bool):
						raise ValueError('wrong input -- frozen core input in system dict (frozen) must be a bool')
					# debug
					if not isinstance(self.system['debug'], bool):
						raise ValueError('wrong input -- debug input in system dict (debug) must be a bool')
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
							sys.stderr.write('\nValueError: non-sensible system input\n'
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


