#!/usr/bin/env python
# -*- coding: utf-8 -*

""" calculation.py: calculation class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import re
import sys
import ast
import copy
import numpy as np
from pyscf import symm

import restart


class CalcCls():
		""" calculation class """
		def __init__(self, mpi, mol):
				""" init parameters """
				# set defaults
				self.model = {'METHOD': 'FCI', 'TYPE': 'VIRT'}
				self.prop = {'ENERGY': True, 'DIPOLE': False, 'EXCITATION': False}
				self.protocol = copy.deepcopy(self.prop)
				self.ref = {'METHOD': 'HF'}
				self.base = {'METHOD': None}
				self.state = {'WFNSYM': symm.addons.irrep_id2name(mol.symmetry, 0), 'ROOT': 0}
				self.thres = 1.0e-10
				self.relax = 1.0
				self.max_order = 1000000
				self.orbitals = {'OCC': 'CAN', 'VIRT': 'CAN'}
				self.async = False
				# init mo
				self.mo = None
				# set calculation parameters
				if mpi.global_master:
					self.model, self.prop, self.protocol, self.ref, \
						self.base, self.thres, self.relax, \
						self.state, self.max_order, \
						self.orbitals, self.async, \
						mol.max_memory, mpi.num_local_masters = self.set_calc(mpi, mol)
					# sanity check
					self.sanity_chk(mpi, mol)
					# restart logical
					self.restart = restart.restart()
				# init property dict
				self.property = {}
				self.property['energy'] = {}
				self.property['dipole'] = {}
				self.property['excitation'] = {}


		def set_calc(self, mpi, mol):
				""" set calculation and mpi parameters from calc.inp file """
				# read input file
				try:
					with open('calc.inp') as f:
						content = f.readlines()
						for i in range(len(content)):
							if content[i].split()[0][0] == '#':
								continue
							elif re.split('=',content[i])[0].strip() == 'model':
								try:
									tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
								except ValueError:
									raise ValueError('wrong input -- values in model dict (model) must be strings')
								tmp = self._upper(tmp)
								for key, val in tmp.items():
									self.model[key] = val
							elif re.split('=',content[i])[0].strip() == 'prop':
								try:
									tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
								except ValueError:
									raise ValueError('wrong input -- values in property dict (prop) must be bools (True, False)')
								tmp = self._upper(tmp)
								for key, val in tmp.items():
									self.prop[key] = val
							elif re.split('=',content[i])[0].strip() == 'prot':
								try:
									tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
								except ValueError:
									raise ValueError('wrong input -- values in protocol dict (prot) must be bools (True, False)')
								tmp = self._upper(tmp)
								for key, val in tmp.items():
									self.protocol[key] = val
							elif re.split('=',content[i])[0].strip() == 'ref':
								self.ref = ast.literal_eval(re.split('=',content[i])[1].strip())
								self.ref = self._upper(self.ref)
							elif re.split('=',content[i])[0].strip() == 'base':
								self.base = ast.literal_eval(re.split('=',content[i])[1].strip())
								self.base = self._upper(self.base)
							elif re.split('=',content[i])[0].strip() == 'thres':
								self.thres = float(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'relax':
								self.relax = float(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'state':
								try:
									tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
								except ValueError:
									raise ValueError('wrong input -- values in state dict (state) must be strings and ints')
								tmp = self._upper(tmp)
								for key, val in tmp.items():
									if key == 'WFNSYM':
										self.state[key] = symm.addons.std_symb(val)
									else:
										self.state[key] = val
							elif re.split('=',content[i])[0].strip() == 'order':
								self.max_order = int(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'orbitals':
								self.orbitals = ast.literal_eval(re.split('=',content[i])[1].strip())
								self.orbitals = self._upper(self.orbitals)
							elif re.split('=',content[i])[0].strip() == 'async':
								self.async = re.split('=',content[i])[1].strip().upper() == 'TRUE'
							elif re.split('=',content[i])[0].strip() == 'mem':
								mol.max_memory = int(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'num_local_masters':
								mpi.num_local_masters = int(re.split('=',content[i])[1].strip())
							# error handling
							else:
								try:
									raise RuntimeError('\''+content[i].split()[0].strip()+'\'' + \
														' keyword in calc.inp not recognized')
								except Exception as err:
									restart.rm()
									sys.stderr.write('\nInputError : {0:}\n\n'.format(err))
									raise
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError : calc.inp not found\n\n')
					raise
				#
				return self.model, self.prop, self.protocol, self.ref, self.base, \
							self.thres, self.relax, self.state, \
							self.max_order, self.orbitals, self.async, \
							mol.max_memory, mpi.num_local_masters


		def sanity_chk(self, mpi, mol):
				""" sanity check for calculation and mpi parameters """
				try:
					# expansion model
					if not all(isinstance(i, str) for i in self.model.values()):
						raise ValueError('wrong input -- values in model input (model) must be strings')
					if not set(list(self.model.keys())) <= set(['METHOD', 'TYPE']):
						raise ValueError('wrong input -- valid input in model dict is: method and type')
					if self.model['METHOD'] not in ['CISD', 'CCSD', 'CCSD(T)', 'SCI', 'FCI']:
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: CISD, CCSD, CCSD(T), SCI, and FCI')
					if self.model['TYPE'] not in ['OCC', 'VIRT', 'COMB']:
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are: occ, virt, and comb')
					# reference model
					if self.ref['METHOD'] not in ['HF', 'CASCI', 'CASSCF']:
						raise ValueError('wrong input -- valid reference models are currently: HF, CASCI, and CASSCF')
					if self.ref['METHOD'] in ['CASCI', 'CASSCF']:
						if self.ref['METHOD'] == 'CASSCF' and self.model['METHOD'] not in ['SCI', 'FCI']:
							raise ValueError('wrong input -- a CASSCF reference is only meaningful for SCI or FCI expansion models')
						if 'ACTIVE' not in self.ref:
							raise ValueError('wrong input -- an active space (active) choice is required for CASCI/CASSCF references')
					if 'ACTIVE' in self.ref:
						if self.ref['METHOD'] == 'HF':
							raise ValueError('wrong input -- an active space is only meaningful for CASCI/CASSCF references')
						if self.ref['ACTIVE'] == 'MANUAL':
							if 'SELECT' not in self.ref:
								raise ValueError('wrong input -- a selection (select) of HF orbitals is required for manual active space')
							if not isinstance(self.ref['SELECT'], list): 
								raise ValueError('wrong input -- select key (select) for active space must be a list')
							if 'NELEC' in self.ref:
								if not isinstance(self.ref['NELEC'], tuple):
									raise ValueError('wrong input -- number of electrons (nelec) in active space must be a tuple (alpha,beta)')
							else:
								raise ValueError('wrong input -- number of electrons (nelec) in active space must be specified')
						elif self.ref['ACTIVE'] == 'AVAS':
							if 'AO_LABELS' not in self.ref:
								raise ValueError('wrong input -- AO labels (AO_lABELS) is required for avas active space')
							if not isinstance(self.ref['AO_LABELS'], list): 
								raise ValueError('wrong input -- AO labels key (AO_LABELS) for active space must be a list')
						else:
							raise ValueError('wrong input -- active space choices are currently: MANUAL and AVAS')
					# base model
					if self.base['METHOD'] not in [None, 'CISD', 'CCSD', 'CCSD(T)', 'SCI']:
						raise ValueError('wrong input -- valid base models ' + \
										'are currently: CISD, CCSD, CCSD(T), SCI, and FCI')
					# state
					try:
						self.state['WFNSYM'] = symm.addons.irrep_name2id(mol.symmetry, self.state['WFNSYM'])
					except Exception as err_2:
						raise ValueError('wrong input -- illegal choice of wfnsym -- PySCF error: {0:}'.format(err_2))
					if self.state['WFNSYM'] != 0:
						if self.model['METHOD'] not in ['SCI','FCI']:
							raise ValueError('wrong input -- illegal choice of wfnsym for chosen expansion model')
					if self.state['ROOT'] < 0:
						raise ValueError('wrong input -- choice of target state (root) must be integer: 0 <= root')
					if self.state['ROOT'] > 0:
						if self.model['METHOD'] not in ['SCI','FCI']:
							raise ValueError('wrong input -- excited states only implemented for SCI and FCI expansion models')
						if not self.prop['EXCITATION']:
							raise ValueError('wrong input -- excited states necessitate the calculation of excitation energy in prop dict')
					# properties
					if not all(isinstance(i, bool) for i in self.prop.values()):
						raise ValueError('wrong input -- values in property input (prop) must be bools (True, False)')
					if not set(list(self.prop.keys())) <= set(['ENERGY', 'DIPOLE', 'EXCITATION']):
						raise ValueError('wrong input -- valid choices for properties are: energy and dipole')
					if not self.prop['ENERGY']:
						raise ValueError('wrong input -- calculation of ground state energy (energy) is mandatory')
					if self.prop['DIPOLE'] and self.base['METHOD'] is not None:
						raise ValueError('wrong input -- calculation of dipole moment (dipole) is only allowed in the absence of a base model')
					if self.prop['EXCITATION'] and self.state['ROOT'] == 0:
						raise ValueError('wrong input -- calculation of excitation energy (excit) requires a state root different from 0')
					# screening protocol
					if not all(isinstance(i, bool) for i in self.protocol.values()):
						raise ValueError('wrong input -- values in protocol input (prot) must be bools (True, False)')
					if not set(list(self.protocol.keys())) <= set(['ENERGY', 'DIPOLE', 'EXCITATION']):
						raise ValueError('wrong input -- valid choices for properties are: energy, dipole, and excitation')
					if not self.protocol['ENERGY'] and sum(self.prop.values()) == 1:
						raise ValueError('wrong input -- non-energy screening requires other properties to be requested in prop input')
					for key in self.protocol.keys():
						if self.protocol[key] and not self.prop[key]:
							raise ValueError('wrong input -- screening wrt a given property requires that this is also requested in prop input')
					# max order
					if self.max_order < 0:
						raise ValueError('wrong input -- maximum expansion order (order) must be integer >= 1')
					# expansion thresholds
					if self.thres < 0.0:
						raise ValueError('wrong input -- expansion threshold parameter (thres) must be float: 0.0 <= thres')
					if self.relax < 1.0:
						raise ValueError('wrong input -- threshold relaxation parameter (relax) must be float: 1.0 <= relax')
					# orbital representation
					if self.orbitals['OCC'] not in ['CAN', 'PM', 'FB', 'IBO-1', 'IBO-2', 'CISD', 'CCSD', 'SCI']:
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations (occ) are currently: canonical (CAN), local (PM or FB), ' + \
										'intrinsic bond orbitals (IBO-1 or IBO-2), or natural orbitals (CISD, CCSD, or SCI)')
					if self.orbitals['VIRT'] not in ['CAN', 'PM', 'FB', 'CISD', 'CCSD', 'SCI']:
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations (virt) are currently: canonical (CAN), local (PM or FB), ' + \
										'or natural orbitals (CISD, CCSD, or SCI)')
					if self.orbitals['OCC'] in ['PM', 'FB', 'IBO-1', 'IBO-2'] or self.orbitals['VIRT'] in ['PM', 'FB']:
						if mol.symmetry != 'C1':
							raise ValueError('wrong input -- the combination of local orbitals and point group symmetry ' + \
											'different from C1 is not allowed')
					# mpi groups
					if mpi.parallel:
						if mpi.num_local_masters < 0:
							raise ValueError('wrong input -- number of local mpi masters (num_local_masters) ' + \
											'must be a positive number >= 1')
						elif mpi.num_local_masters == 0:
							if self.model['TYPE'] == 'COMB':
								raise ValueError('wrong input -- combined expansions are only valid in ' + \
												'combination with at least one local mpi master (num_local_masters >= 1)')
						else:
							if self.model['TYPE'] != 'COMB':
								raise ValueError('wrong input -- the use of local mpi masters ' + \
												'is currently not implemented for occ and virt expansions')
						if mpi.global_size <= 2 * mpi.num_local_masters:
							raise ValueError('wrong input -- total number of mpi processes ' + \
											'must be larger than twice the number of local mpi masters (num_local_masters)')
					# memory
					if mol.max_memory is None:
						raise ValueError('wrong input -- the memory keyword (mem) is missing')
				except Exception as err:
					restart.rm()
					sys.stderr.write('\nValueError : {0:}\n\n'.format(err))
					raise


		def _upper(self, old_dict):
				""" capitalize keys """
				new_dict = {}
				for key, value in old_dict.items():
					if key.upper() in ['METHOD', 'ACTIVE', 'TYPE', 'OCC', 'VIRT']:
						new_dict[key.upper()] = value.upper()
					else:
						new_dict[key.upper()] = value
				return new_dict


