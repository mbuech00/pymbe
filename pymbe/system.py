#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
system module containing all ab initio / model hamiltonian attributes
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.6'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
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
		"""
		this class contains all molecule attributes (inherited from the pyscf gto.Mole class)
		"""
		def __init__(self):
				"""
				init molecule attributes
				"""
				# gto.Mole instantiation
				gto.Mole.__init__(self)

				# set defaults
				self.atom = ''
				self.system = {'charge': 0, 'spin': 0, 'symmetry': 'c1', 'hf_symmetry': None, \
							'hf_init_guess': 'minao', 'basis': 'sto-3g', 'cart': False, \
							'unit': 'ang', 'frozen': False, 'ncore': 0, 'irrep_nelec': {}, 'debug': 0, \
							'u': 1.0, 'n': 1.0, 'matrix': (1, 6), 'pbc': True}
				self.max_memory = 1e10
				self.incore_anyway = True


		def make(self, mpi):
				"""
				this function builds the pyscf Mole object

				:param mpi: pymbe mpi object
				"""
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
				if self.frozen:
					self.ncore = self._set_ncore()


		def _set_ncore(self):
				"""
				this function sets ncore

				:return: number of core MOs. integer
				"""
				# init ncore
				ncore = 0

				# loop over atoms
				for i in range(self.natm):

					if self.atom_charge(i) > 2: ncore += 1
					if self.atom_charge(i) > 12: ncore += 4
					if self.atom_charge(i) > 20: ncore += 4
					if self.atom_charge(i) > 30: ncore += 6

				return ncore


def set_system(mol):
		"""
		this function sets system attributes from input file

		:param mol: pymbe mol object
		:return: updated mol object
		"""
		# read input file
		try:

			with open(os.getcwd()+'/input') as f:

				content = f.readlines()

				for i in range(len(content)):

					if content[i].strip():
						if content[i].split()[0][0] == '#':
							continue

						elif re.split('=',content[i])[0].strip() == 'atom':

							for j in range(i+1, len(content)):

								if content[j][:3] == "'''" or content[j][:3] == '"""':
									break
								else:
									mol.atom += content[j]

						elif re.split('=',content[i])[0].strip() == 'system':

							try:
								inp = ast.literal_eval(re.split('=',content[i])[1].strip())
							except ValueError:
								raise ValueError('wrong input -- error in reading in system dictionary')

							# update system
							mol.system = {**mol.system, **inp}

		except IOError:

			restart.rm()
			sys.stderr.write('\nIOError : input file not found\n\n')
			raise

		return mol


def translate_system(mol):
		"""
		this function translates system input to mol attributes

		:param mol: pymbe mol object
		:return: updated mol object
		"""
		# copy all attributes
		for key, val in mol.system.items():
			setattr(mol, key, val)

		# hf symmetry
		if mol.hf_symmetry is None:
			mol.hf_symmetry = mol.symmetry

		# recast symmetries as standard symbols
		mol.symmetry = symm.addons.std_symb(mol.symmetry)
		mol.hf_symmetry = symm.addons.std_symb(mol.hf_symmetry)

		# hubbard hamiltonian
		if not mol.atom:

			mol.atom = []
			mol.symmetry = mol.hf_symmetry = False
			mol.nsites = mol.matrix[0] * mol.matrix[1]
			mol.nelectron = math.floor(mol.nsites * mol.n)

		return mol


def sanity_chk(mol):
		"""
		this function performs sanity checks of mol attributes

		:param mol: pymbe mol object
		"""
		# charge
		tools.assertion(isinstance(mol.charge, int), \
						'charge input in system dict (charge) must be an int')

		# spin
		tools.assertion(isinstance(mol.spin, int) and mol.spin >= 0, \
						'spin input (2S) in system dict (spin) must be an int >= 0')

		# symmetry
		tools.assertion(isinstance(mol.symmetry, str), \
						'symmetry input in system dict (symmetry) must be a str')
		tools.assertion(symm.addons.std_symb(mol.symmetry) in symm.param.POINTGROUP + ('Dooh', 'Coov',), \
						'illegal symmetry input in system dict (symmetry)')

		# hf_symmetry
		tools.assertion(isinstance(mol.hf_symmetry, str), \
						'HF symmetry input in system dict (hf_symmetry) must be a str')
		tools.assertion(symm.addons.std_symb(mol.hf_symmetry) in symm.param.POINTGROUP + ('Dooh', 'Coov',), \
						'illegal HF symmetry input in system dict (hf_symmetry)')

		# hf_init_guess
		tools.assertion(isinstance(mol.hf_init_guess, str), \
						'HF initial guess in system dict (hf_init_guess) must be a str')
		tools.assertion(mol.hf_init_guess in ['minao', 'atom', '1e'], \
						'valid HF initial guesses in system dict (hf_init_guess) are: minao, atom, and 1e')

		# basis
		tools.assertion(isinstance(mol.basis, (str, dict)), \
						'basis set input in system dict (basis) must be a str or a dict')

		# cart
		tools.assertion(isinstance(mol.cart, bool), \
						'cartesian gto basis input in system dict (cart) must be a bool')

		# irrep_nelec
		tools.assertion(isinstance(mol.irrep_nelec, dict), \
						'occupation input in system dict (irrep_nelec) must be a dict')

		# unit
		tools.assertion(isinstance(mol.unit, str), \
						'unit input in system dict (unit) must be a str')

		# frozen
		tools.assertion(isinstance(mol.frozen, bool), \
						'frozen core input in system dict (frozen) must be a bool')

		# debug
		tools.assertion(type(mol.debug) is int, \
						'debug input in system dict (debug) must be an int')
		tools.assertion(mol.debug >= 0, \
						'debug input in system dict (debug) must be an int >= 0')

		# hubbard
		if not mol.atom:

			# matrix
			tools.assertion(isinstance(mol.matrix, tuple), \
							'hubbard matrix input in system dict (matrix) must be a tuple')
			tools.assertion(len(mol.matrix) == 2, \
							'hubbard matrix input in system dict (matrix) must have a dimension of 2')
			tools.assertion(isinstance(mol.matrix[0], int) and isinstance(mol.matrix[1], int), \
							'hubbard matrix input in system dict (matrix) must be a tuple of ints')

			# u parameter
			tools.assertion(isinstance(mol.u, float), \
							'hubbard on-site repulsion parameter (u) must be a float')
			tools.assertion(mol.u > 0.0, \
							'only repulsive hubbard models are implemented (u > 0.0)')

			# n parameter
			tools.assertion(isinstance(mol.n, float), \
							'hubbard model filling parameter (n) must be a float')
			tools.assertion(mol.n > 0.0 and mol.n < 2.0, \
							'hubbard model filling parameter (n) must be a float between 0.0 < n < 2.0')

			# periodic boundary conditions
			tools.assertion(isinstance(mol.pbc, bool), \
							'hubbard model pbc parameter (pbc) must be a bool')


