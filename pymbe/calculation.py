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
				self.model = {'method': 'fci', 'type': 'virt'}
				self.target = {'energy': True, 'dipole': False, 'trans': False}
				self.prot = {'scheme': 'new', 'specific': False}
				self.ref = {'method': 'hf'}
				self.base = {'method': None}
				self.state = {'wfnsym': symm.addons.irrep_id2name(mol.symmetry, 0), 'root': 0}
				self.thres = {'init': 1.0e-10, 'relax': 1.0}
				self.misc = {'mem': 2000, 'order': None, 'async': False}
				self.orbs = {'occ': 'can', 'virt': 'can'}
				self.mpi = {'masters': 1}
				# init mo
				self.mo = None
				# set calculation parameters
				if mpi.global_master:
					# read parameters
					self.model, self.target, self.prot, self.ref, \
						self.base, self.thres, self.state, \
						self.misc, self.orbs, self.mpi = self.set_calc()
					# sanity check
					self.sanity_chk(mpi, mol)
					# restart logical
					self.restart = restart.restart()
				# set nroots
				if self.state['root'] == 0:
					self.nroots = 1
				else:
					if self.prot['specific']:
						self.nroots = 2
					else:
						self.nroots = self.state['root'] + 1
				# init prop dict
				self.prop = {'hf': {}, 'ref': {}}


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
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.model[key] = val
								# target
								elif re.split('=',content[i])[0].strip() == 'target':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in target dict (target) must be bools')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.target[key] = val
								# prot
								elif re.split('=',content[i])[0].strip() == 'prot':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in prot dict (prot) must be strings and bools')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.prot[key] = val
								# ref
								elif re.split('=',content[i])[0].strip() == 'ref':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in reference dict (ref) must be strings, lists, and tuples')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.ref[key] = val
								# base
								elif re.split('=',content[i])[0].strip() == 'base':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in base dict (base) must be strings')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.base[key] = val
								# thres
								elif re.split('=',content[i])[0].strip() == 'thres':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in threshold dict (thres) must be floats')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.thres[key] = val
								# state
								elif re.split('=',content[i])[0].strip() == 'state':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in state dict (state) must be strings and ints')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										if key == 'wfnsym':
											self.state[key] = symm.addons.std_symb(val)
										else:
											self.state[key] = val
								# misc
								elif re.split('=',content[i])[0].strip() == 'misc':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in misc dict (misc) must be ints and bools')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.misc[key] = val
								# orbs
								elif re.split('=',content[i])[0].strip() == 'orbs':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in orbital dict (orbs) must be strings')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.orbs[key] = val
								# mpi
								elif re.split('=',content[i])[0].strip() == 'mpi':
									try:
										tmp = ast.literal_eval(re.split('=',content[i])[1].strip())
									except ValueError:
										raise ValueError('wrong input -- values in mpi dict (mpi) must be ints')
									tmp = tools.dict_conv(tmp)
									for key, val in tmp.items():
										self.mpi[key] = val
				except IOError:
					restart.rm()
					sys.stderr.write('\nIOError : input file not found\n\n')
					raise
				#
				return self.model, self.target, self.prot, self.ref, self.base, \
							self.thres, self.state, self.misc, self.orbs, self.mpi


		def sanity_chk(self, mpi, mol):
				""" sanity check for calculation and mpi parameters """
				try:
					# expansion model
					if not all(isinstance(i, str) for i in self.model.values()):
						raise ValueError('wrong input -- values in model input (model) must be strings')
					if not set(list(self.model.keys())) <= set(['method', 'type']):
						raise ValueError('wrong input -- valid input in model dict is: method and type')
					if self.model['method'] not in ['cisd', 'ccsd', 'ccsd(t)', 'fci']:
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: cisd, ccsd, ccsd(t), and fci')
					if self.model['type'] not in ['occ', 'virt', 'comb']:
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are: occ, virt, and comb')
					# reference model
					if self.ref['method'] not in ['hf', 'casci', 'casscf']:
						raise ValueError('wrong input -- valid reference models are currently: hf, casci, and casscf')
					if self.ref['method'] in ['casci', 'casscf']:
						if self.ref['method'] == 'casscf' and self.model['method'] != 'fci':
							raise ValueError('wrong input -- a casscf reference is only meaningful for an fci expansion model')
						if 'active' not in self.ref:
							raise ValueError('wrong input -- an active space (active) choice is required for casci/casscf references')
					if 'active' in self.ref:
						if self.ref['method'] == 'hf':
							raise ValueError('wrong input -- an active space is only meaningful for casci/casscf references')
						if self.ref['active'] == 'manual':
							if 'select' not in self.ref:
								raise ValueError('wrong input -- a selection (select) of hf orbs is required for manual active space')
							if not isinstance(self.ref['select'], list): 
								raise ValueError('wrong input -- select key (select) for active space must be a list')
							if 'nelec' in self.ref:
								if not isinstance(self.ref['nelec'], tuple):
									raise ValueError('wrong input -- number of electrons (nelec) in active space must be a tuple (alpha,beta)')
							else:
								raise ValueError('wrong input -- number of electrons (nelec) in active space must be specified')
						elif self.ref['active'] == 'avas':
							if 'ao_labels' not in self.ref:
								raise ValueError('wrong input -- AO labels (ao_labels) is required for avas active space')
							if not isinstance(self.ref['ao_labels'], list): 
								raise ValueError('wrong input -- AO labels key (ao_labels) for active space must be a list')
						else:
							raise ValueError('wrong input -- active space choices are currently: manual and avas')
					# base model
					if self.base['method'] not in [None, 'cisd', 'ccsd', 'ccsd(t)']:
						raise ValueError('wrong input -- valid base models are currently: cisd, ccsd, and ccsd(t)')
					# state
					try:
						self.state['wfnsym'] = symm.addons.irrep_name2id(mol.symmetry, self.state['wfnsym'])
					except Exception as err_2:
						raise ValueError('wrong input -- illegal choice of wfnsym -- PySCF error: {0:}'.format(err_2))
					if self.state['wfnsym'] != 0 and self.model['method'] != 'fci':
						raise ValueError('wrong input -- illegal choice of wfnsym for chosen expansion model')
					if self.state['root'] < 0:
						raise ValueError('wrong input -- choice of target state (root) must be an int >= 0')
					if self.state['root'] > 0 and self.model['method'] != 'fci':
						raise ValueError('wrong input -- excited states only implemented for an fci expansion model')
					# targets
					if not all(isinstance(i, bool) for i in self.target.values()):
						raise ValueError('wrong input -- values in target input (target) must be bools')
					if not set(list(self.target.keys())) <= set(['energy', 'dipole', 'trans']):
						raise ValueError('wrong input -- valid choices for target properties are: energy, dipole, and transition dipole (trans)')
					if not self.target['energy']:
						raise ValueError('wrong input -- calculation of ground state energy (energy) is mandatory')
					if self.target['dipole'] and self.base['method'] is not None:
						raise ValueError('wrong input -- calculation of dipole moment (dipole) is only allowed in the absence of a base model')
					if self.target['trans'] and self.base['method'] is not None:
						raise ValueError('wrong input -- calculation of transition dipole moment (trans) is only allowed in the absence of a base model')
					if self.target['trans'] and self.state['root'] == 0:
						raise ValueError('wrong input -- calculation of transition dipole moment (trans) requires target state root >= 1')
					# screening protocol
					if not all(isinstance(i, (str, bool)) for i in self.prot.values()):
						raise ValueError('wrong input -- values in prot input (prot) must be string and bools')
					if self.prot['scheme'] not in ['new', 'old']:
						raise ValueError('wrong input -- valid protocol schemes are: new and old')
					# expansion thresholds
					if not all(isinstance(i, float) for i in self.thres.values()):
						raise ValueError('wrong input -- values in threshold input (thres) must be floats')
					if not set(list(self.thres.keys())) <= set(['init', 'relax']):
						raise ValueError('wrong input -- valid input in thres dict is: init and relax')
					if self.thres['init'] < 0.0:
						raise ValueError('wrong input -- initial threshold (init) must be a float >= 0.0')
					if self.thres['relax'] < 1.0:
						raise ValueError('wrong input -- threshold relaxation (relax) must be a float >= 1.0')
					# orbital representation
					if self.orbs['occ'] not in ['can', 'pm', 'fb', 'ibo-1', 'ibo-2', 'cisd', 'ccsd']:
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations (occ) are currently: canonical (can), local (pm or fb), ' + \
										'intrinsic bond orbs (ibo-1 or ibo-2), or natural orbs (cisd or ccsd)')
					if self.orbs['virt'] not in ['can', 'pm', 'fb', 'cisd', 'ccsd']:
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations (virt) are currently: canonical (can), local (pm or fb), ' + \
										'or natural orbs (cisd or ccsd)')
					if self.orbs['occ'] in ['pm', 'fb', 'ibo-1', 'ibo-2'] or self.orbs['virt'] in ['pm', 'fb']:
						if mol.symmetry != 'c1':
							raise ValueError('wrong input -- the combination of local orbs and point group symmetry ' + \
											'different from c1 is not allowed')
					# misc
					if not isinstance(self.misc['mem'], int):
						raise ValueError('wrong input -- maximum memory (mem) in units of MB must be an int >= 1')
					if self.misc['mem'] < 0:
						raise ValueError('wrong input -- maximum memory (mem) in units of MB must be an int >= 1')
					if not isinstance(self.misc['order'], (int, type(None))):
						raise ValueError('wrong input -- maximum expansion order (order) must be an int >= 1')
					if self.misc['order'] is not None:
						if self.misc['order'] < 0:
							raise ValueError('wrong input -- maximum expansion order (order) must be an int >= 1')
					if not isinstance(self.misc['async'], bool):
						raise ValueError('wrong input -- asynchronous key (async) must be a bool')
					# mpi
					if not isinstance(self.mpi['masters'], int):
						raise ValueError('wrong input -- number of mpi masters (masters) must be an int >= 1')
					if mpi.parallel:
						if self.mpi['masters'] < 1:
							raise ValueError('wrong input -- number of mpi masters (masters) must be an int >= 1')
						elif self.mpi['masters'] == 1:
							if self.model['type'] == 'comb':
								raise ValueError('wrong input -- combined expansions are only valid in ' + \
												'combination with at least one local mpi master (i.e., masters > 1)')
						else:
							if self.model['type'] != 'comb':
								raise ValueError('wrong input -- the use of local mpi masters (i.e., masters > 1) ' + \
												'is currently not implemented for occ and virt expansions')
						if mpi.global_size <= 2 * self.mpi['masters']:
							raise ValueError('wrong input -- total number of mpi processes ' + \
											'must be larger than twice the number of local mpi masters (masters+1)')
					else:
						if self.mpi['masters'] > 1:
							raise ValueError('wrong input -- local masters requested in mpi dict (mpi), but non-mpi run requested')
				except Exception as err:
					restart.rm()
					sys.stderr.write('\nValueError : {0:}\n\n'.format(err))
					raise




