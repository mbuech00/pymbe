#!/usr/bin/env python
# -*- coding: utf-8 -*

""" system.py: ab initio / model hamiltonian class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import re
import sys
import os
import ast
import math
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
				self.system = {'charge': 0, 'spin': 0, 'sym': 'c1', 'hf_sym': None, \
							'hf_init_guess': 'minao', 'basis': 'sto-3g', 'cart': False, \
							'unit': 'ang', 'frozen': False, 'occup': {}, 'debug': 0, \
							'u': 1.0, 'n': 1.0, 'matrix': (1, 6), 'pbc': True}
				# set geometric and molecular parameters
				if mpi.master:
					# read atom and molecule settings
					self.atom, self.system = self.set_system()
					# sanity check
					self.sanity_chk()
					# translate to Mole input
					self.incore_anyway = True
					self.irrep_nelec = self.system['occup']
					self.charge = self.system['charge']
					self.spin = self.system['spin']
					self.symmetry = symm.addons.std_symb(self.system['sym'])
					self.hf_sym = symm.addons.std_symb(self.system['hf_sym'])
					self.hf_init_guess = self.system['hf_init_guess']
					self.basis = self.system['basis']
					self.cart = self.system['cart']
					self.unit = self.system['unit']
					# hubbard hamiltonian
					if not self.atom:
						self.atom = []
						self.symmetry = self.hf_sym = False
						self.ncore = 0
						self.u = self.system['u']
						self.n = self.system['n']
						self.matrix = self.system['matrix']
						self.nsites = self.matrix[0] * self.matrix[1]
						self.nelectron = math.floor(self.nsites * self.system['n'])
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
										raise ValueError('wrong input -- error in reading in system dictionary')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.system[key] = val
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError : input file not found\n\n')
					raise
				# hf symmetry
				if self.system['hf_sym'] is None:
					self.system['hf_sym'] = self.system['sym']
				return self.atom, self.system


		def sanity_chk(self):
				""" sanity check for system parameters """
				# charge
				tools.assertion(isinstance(self.system['charge'], int), \
								'charge input in system dict (charge) must be an int')
				# spin
				tools.assertion(isinstance(self.system['spin'], int) and self.system['spin'] >= 0, \
								'spin input (2S) in system dict (spin) must be an int >= 0')
				# sym
				tools.assertion(isinstance(self.system['sym'], str), \
								'symmetry input in system dict (sym) must be a str')
				tools.assertion(symm.addons.std_symb(self.system['sym']) in symm.param.POINTGROUP + ('Dooh', 'Coov',), \
								'illegal symmetry input in system dict (sym)')
				# hf_sym
				tools.assertion(isinstance(self.system['hf_sym'], str), \
								'HF symmetry input in system dict (hf_sym) must be a str')
				tools.assertion(symm.addons.std_symb(self.system['hf_sym']) in symm.param.POINTGROUP + ('Dooh', 'Coov',), \
								'illegal HF symmetry input in system dict (hf_sym)')
				# hf_init_guess
				tools.assertion(isinstance(self.system['hf_init_guess'], str), \
								'HF initial guess in system dict (hf_init_guess) must be a str')
				tools.assertion(self.system['hf_init_guess'] in ['minao', 'atom', '1e'], \
								'valid HF initial guesses in system dict (hf_init_guess) are: minao, atom, and 1e')
				# basis
				tools.assertion(isinstance(self.system['basis'], (str, dict)), \
								'basis set input in system dict (basis) must be a str or a dict')
				# cart
				tools.assertion(isinstance(self.system['cart'], bool), \
								'cartesian gto basis input in system dict (cart) must be a bool')
				# occup
				tools.assertion(isinstance(self.system['occup'], dict), \
								'occupation input in system dict (occup) must be a dict')
				# unit
				tools.assertion(isinstance(self.system['unit'], str), \
								'unit input in system dict (unit) must be a str')
				# frozen
				tools.assertion(isinstance(self.system['frozen'], bool), \
								'frozen core input in system dict (frozen) must be a bool')
				# debug
				tools.assertion(type(self.system['debug']) is int, \
								'debug input in system dict (debug) must be an int')
				tools.assertion(self.system['debug'] >= 0, \
								'debug input in system dict (debug) must be an int >= 0')
				if not self.atom:
					# matrix
					tools.assertion(isinstance(self.system['matrix'], tuple), \
									'hubbard matrix input in system dict (matrix) must be a tuple')
					tools.assertion(len(self.system['matrix']) == 2, \
									'hubbard matrix input in system dict (matrix) must have a dimension of 2')
					tools.assertion(isinstance(self.system['matrix'][0], int) and isinstance(self.system['matrix'][1], int), \
									'hubbard matrix input in system dict (matrix) must be a tuple of ints')
					# u parameter
					tools.assertion(isinstance(self.system['u'], float), \
									'hubbard on-site repulsion parameter (u) must be a float')
					tools.assertion(self.system['u'] > 0.0, \
									'only repulsive hubbard models are implemented (u > 0.0)')
					# n parameter
					tools.assertion(isinstance(self.system['n'], float), \
									'hubbard model filling parameter (n) must be a float')
					tools.assertion(self.system['n'] > 0.0 and self.system['n'] < 2.0, \
									'hubbard model filling parameter (n) must be a float between 0.0 < n < 2.0')
					# periodic boundary conditions
					tools.assertion(isinstance(self.system['pbc'], bool), \
									'hubbard model pbc parameter (pbc) must be a bool')


		def make(self, mpi):
				""" build Mole object """
				try:
					self.build(dump_input=False, parse_arg=False, verbose=0)
				except RuntimeWarning as err:
					try:
						raise RuntimeError
					except RuntimeError:
						if mpi.master:
							restart.rm()
							sys.stderr.write('\nValueError: non-sensible system input\n'
												'PySCF error : {:}\n\n'.format(err))
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


