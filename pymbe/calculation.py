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
import os
import ast
import numpy as np
from pyscf import symm

import tools
import restart


class CalcCls():
		""" calculation class """
		def __init__(self, mpi, mol):
				""" init parameters """
				# set defaults
				self.model = {'METHOD': 'FCI', 'TYPE': 'VIRT'}
				self.prop = {'ENERGY': True, 'DIPOLE': False}
				self.prot = {'SCHEME': 'NEW', 'GS_ENERGY_ONLY': False}
				self.ref = {'METHOD': 'HF'}
				self.base = {'METHOD': None}
				self.state = {'WFNSYM': symm.addons.irrep_id2name(mol.symmetry, 0), 'ROOT': 0}
				self.thres = {'INIT': 1.0e-10, 'RELAX': 1.0}
				self.misc = {'MEM': 2000, 'ORDER': None, 'ASYNC': False}
				self.orbs = {'OCC': 'CAN', 'VIRT': 'CAN'}
				self.mpi = {'MASTERS': 1}
				# init mo
				self.mo = None
				# set calculation parameters
				if mpi.global_master:
					# read parameters
					self.model, self.prop, self.prot, self.ref, \
						self.base, self.thres, self.state, \
						self.misc, self.orbs, self.mpi = self.set_calc()
					# sanity check
					self.sanity_chk(mpi, mol)
					# restart logical
					self.restart = restart.restart()
				# init property dict
				self.property = {}
				self.property['hf'] = {}
				self.property['ref'] = {}


		def set_calc(self):
				""" set calculation and mpi parameters from input file """
				# read input file
				try:
					with open(os.getcwd()+'/input') as f:
						content = f.readlines()
						for i in range(len(content)):
							if content[i].strip():
								if content[i].split()[0][0] == '#':
									continue
								# model
								elif re.split('=',content[i])[0].strip() == 'model':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in model dict (model) must be strings')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.model[key] = val
								# prop
								elif re.split('=',content[i])[0].strip() == 'prop':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in property dict (prop) must be bools')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.prop[key] = val
								# prot
								elif re.split('=',content[i])[0].strip() == 'prot':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in prot dict (prot) must be strings and bools')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.prot[key] = val
								# ref
								elif re.split('=',content[i])[0].strip() == 'ref':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in reference dict (ref) must be strings, lists, and tuples')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.ref[key] = val
								# base
								elif re.split('=',content[i])[0].strip() == 'base':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in base dict (base) must be strings')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.base[key] = val
								# thres
								elif re.split('=',content[i])[0].strip() == 'thres':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in threshold dict (thres) must be floats')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.thres[key] = val
								# state
								elif re.split('=',content[i])[0].strip() == 'state':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in state dict (state) must be strings and ints')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										if key == 'WFNSYM':
											self.state[key] = symm.addons.std_symb(val)
										else:
											self.state[key] = val
								# misc
								elif re.split('=',content[i])[0].strip() == 'misc':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in misc dict (misc) must be ints and bools')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.misc[key] = val
								# orbs
								elif re.split('=',content[i])[0].strip() == 'orbs':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in orbital dict (orbs) must be strings')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.orbs[key] = val
								# mpi
								elif re.split('=',content[i])[0].strip() == 'mpi':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in mpi dict (mpi) must be ints')
									tmp = tools.upper(tmp)
									for key, val in tmp.items():
										self.mpi[key] = val
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError : input file not found\n\n')
					raise
				#
				return self.model, self.prop, self.prot, self.ref, self.base, \
							self.thres, self.state, self.misc, self.orbs, self.mpi


		def sanity_chk(self, mpi, mol):
				""" sanity check for calculation and mpi parameters """
				try:
					# expansion model
					if not all(isinstance(i, str) for i in self.model.values()):
						raise ValueError('wrong input -- values in model input (model) must be strings')
					if not set(list(self.model.keys())) <= set(['METHOD', 'TYPE']):
						raise ValueError('wrong input -- valid input in model dict is: method and type')
					if self.model['METHOD'] not in ['CISD', 'CCSD', 'CCSD(T)', 'FCI']:
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: CISD, CCSD, CCSD(T), and FCI')
					if self.model['TYPE'] not in ['OCC', 'VIRT', 'COMB']:
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are: occ, virt, and comb')
					# reference model
					if self.ref['METHOD'] not in ['HF', 'CASCI', 'CASSCF']:
						raise ValueError('wrong input -- valid reference models are currently: HF, CASCI, and CASSCF')
					if self.ref['METHOD'] in ['CASCI', 'CASSCF']:
						if self.ref['METHOD'] == 'CASSCF' and self.model['METHOD'] != 'FCI':
							raise ValueError('wrong input -- a CASSCF reference is only meaningful for an FCI expansion model')
						if 'ACTIVE' not in self.ref:
							raise ValueError('wrong input -- an active space (active) choice is required for CASCI/CASSCF references')
					if 'ACTIVE' in self.ref:
						if self.ref['METHOD'] == 'HF':
							raise ValueError('wrong input -- an active space is only meaningful for CASCI/CASSCF references')
						if self.ref['ACTIVE'] == 'MANUAL':
							if 'SELECT' not in self.ref:
								raise ValueError('wrong input -- a selection (select) of HF orbs is required for manual active space')
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
					if self.base['METHOD'] not in [None, 'CISD', 'CCSD', 'CCSD(T)']:
						raise ValueError('wrong input -- valid base models are currently: CISD, CCSD, and CCSD(T)')
					# state
					try:
						self.state['WFNSYM'] = symm.addons.irrep_name2id(mol.symmetry, self.state['WFNSYM'])
					except Exception as err_2:
						raise ValueError('wrong input -- illegal choice of wfnsym -- PySCF error: {0:}'.format(err_2))
					if self.state['WFNSYM'] != 0 and self.model['METHOD'] != 'FCI':
						raise ValueError('wrong input -- illegal choice of wfnsym for chosen expansion model')
					if self.state['ROOT'] < 0:
						raise ValueError('wrong input -- choice of target state (root) must be an int >= 0')
					if self.state['ROOT'] > 0 and self.model['METHOD'] != 'FCI':
						raise ValueError('wrong input -- excited states only implemented for an FCI expansion model')
					# properties
					if not all(isinstance(i, bool) for i in self.prop.values()):
						raise ValueError('wrong input -- values in property input (prop) must be bools (True, False)')
					if not set(list(self.prop.keys())) <= set(['ENERGY', 'DIPOLE']):
						raise ValueError('wrong input -- valid choices for properties are: energy and dipole')
					if not self.prop['ENERGY']:
						raise ValueError('wrong input -- calculation of ground state energy (energy) is mandatory')
					if self.prop['DIPOLE'] and self.base['METHOD'] is not None:
						raise ValueError('wrong input -- calculation of dipole moment (dipole) is only allowed in the absence of a base model')
					# screening protocol
					if not all(isinstance(i, (str, bool)) for i in self.prot.values()):
						raise ValueError('wrong input -- values in prot input (prot) must be string and bools (True, False)')
					if self.prot['SCHEME'] not in ['NEW', 'OLD']:
						raise ValueError('wrong input -- valid protocol schemes are: new and old')
					# expansion thresholds
					if not all(isinstance(i, float) for i in self.thres.values()):
						raise ValueError('wrong input -- values in threshold input (thres) must be floats')
					if not set(list(self.thres.keys())) <= set(['INIT', 'RELAX']):
						raise ValueError('wrong input -- valid input in thres dict is: init and relax')
					if self.thres['INIT'] < 0.0:
						raise ValueError('wrong input -- initial threshold (init) must be a float >= 0.0')
					if self.thres['RELAX'] < 1.0:
						raise ValueError('wrong input -- threshold relaxation (relax) must be a float >= 1.0')
					# orbital representation
					if self.orbs['OCC'] not in ['CAN', 'PM', 'FB', 'IBO-1', 'IBO-2', 'CISD', 'CCSD']:
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations (occ) are currently: canonical (CAN), local (PM or FB), ' + \
										'intrinsic bond orbs (IBO-1 or IBO-2), or natural orbs (CISD or CCSD)')
					if self.orbs['VIRT'] not in ['CAN', 'PM', 'FB', 'CISD', 'CCSD']:
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations (virt) are currently: canonical (CAN), local (PM or FB), ' + \
										'or natural orbs (CISD or CCSD)')
					if self.orbs['OCC'] in ['PM', 'FB', 'IBO-1', 'IBO-2'] or self.orbs['VIRT'] in ['PM', 'FB']:
						if mol.symmetry != 'C1':
							raise ValueError('wrong input -- the combination of local orbs and point group symmetry ' + \
											'different from C1 is not allowed')
					# misc
					if not isinstance(self.misc['MEM'], int):
						raise ValueError('wrong input -- maximum memory (mem) in units of MB must be an int >= 1')
					if self.misc['MEM'] < 0:
						raise ValueError('wrong input -- maximum memory (mem) in units of MB must be an int >= 1')
					if not isinstance(self.misc['ORDER'], (int, type(None))):
						raise ValueError('wrong input -- maximum expansion order (order) must be an int >= 1')
					if self.misc['ORDER'] is not None:
						if self.misc['ORDER'] < 0:
							raise ValueError('wrong input -- maximum expansion order (order) must be an int >= 1')
					if not isinstance(self.misc['ASYNC'], bool):
						raise ValueError('wrong input -- asynchronous key (async) must be a bool (True, False)')
					# mpi
					if not isinstance(self.mpi['MASTERS'], int):
						raise ValueError('wrong input -- number of mpi masters (masters) must be an int >= 1')
					if mpi.parallel:
						if self.mpi['MASTERS'] < 1:
							raise ValueError('wrong input -- number of mpi masters (masters) must be an int >= 1')
						elif self.mpi['MASTERS'] == 1:
							if self.model['TYPE'] == 'COMB':
								raise ValueError('wrong input -- combined expansions are only valid in ' + \
												'combination with at least one local mpi master (i.e., masters > 1)')
						else:
							if self.model['TYPE'] != 'COMB':
								raise ValueError('wrong input -- the use of local mpi masters (i.e., masters > 1) ' + \
												'is currently not implemented for occ and virt expansions')
						if mpi.global_size <= 2 * self.mpi['MASTERS']:
							raise ValueError('wrong input -- total number of mpi processes ' + \
											'must be larger than twice the number of local mpi masters (masters+1)')
					else:
						if self.mpi['MASTERS'] > 1:
							raise ValueError('wrong input -- local masters requested in mpi dict (mpi), but non-mpi run requested')
				except Exception as err:
					restart.rm()
					sys.stderr.write('\nValueError : {0:}\n\n'.format(err))
					raise




