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
from pyscf import symm

import restart


class CalcCls():
		""" calculation class """
		def __init__(self, mpi, mol):
				""" init parameters """
				# set defaults
				self.model = {'METHOD': 'FCI'}
				self.typ = 'occupied'
				self.ref = {'METHOD': 'HF'}
				self.base = {'METHOD': None}
				self.thres = 1.0e-10
				self.relax = 1.0
				self.wfnsym = symm.addons.irrep_id2name(mol.symmetry, 0)
				self.target = 0
				self.max_order = 1000000
				self.occ = 'REF'
				self.virt = 'REF'
				# init energy dict and mo
				self.energy = {}
				self.mo = None
				# set calculation parameters
				if mpi.global_master:
					self.model, self.typ, self.ref, self.base, \
						self.thres, self.relax, \
						self.wfnsym, self.target, self.max_order, \
						self.occ, self.virt, \
						mol.max_memory, mpi.num_local_masters = self.set_calc(mpi, mol)
					# sanity check
					self.sanity_chk(mpi, mol)
					# restart logical
					self.restart = restart.restart()


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
								self.model = eval(re.split('=',content[i])[1].strip())
								self.model = self._upper(self.model)
							elif re.split('=',content[i])[0].strip() == 'type':
								self.typ = re.split('=',content[i])[1].strip()
							elif re.split('=',content[i])[0].strip() == 'ref':
								self.ref = eval(re.split('=',content[i])[1].strip())
								self.ref = self._upper(self.ref)
							elif re.split('=',content[i])[0].strip() == 'base':
								self.base = eval(re.split('=',content[i])[1].strip())
								self.base = self._upper(self.base)
							elif re.split('=',content[i])[0].strip() == 'thres':
								self.thres = float(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'relax':
								self.relax = float(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'wfnsym':
								self.wfnsym = symm.addons.std_symb(eval(re.split('=',content[i])[1].strip()))
							elif re.split('=',content[i])[0].strip() == 'target':
								self.target = int(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'order':
								self.max_order = int(re.split('=',content[i])[1].strip())
							elif re.split('=',content[i])[0].strip() == 'occ':
								self.occ = re.split('=',content[i])[1].strip().upper()
							elif re.split('=',content[i])[0].strip() == 'virt':
								self.virt = re.split('=',content[i])[1].strip().upper()
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
				return self.model, self.typ, self.ref, self.base, \
							self.thres, self.relax, self.wfnsym, self.target, \
							self.max_order, self.occ, self.virt, \
							mol.max_memory, mpi.num_local_masters


		def sanity_chk(self, mpi, mol):
				""" sanity check for calculation and mpi parameters """
				try:
					# expansion model
					if self.model['METHOD'] not in ['CISD','CCSD','CCSD(T)','SCI','FCI']:
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: CISD, CCSD, CCSD(T), SCI, and FCI')
					# type of expansion
					if self.typ not in ['occupied','virtual','combined']:
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are occupied, virtual, and combined')
					# reference model
					if self.ref['METHOD'] not in ['HF','CASCI','CASSCF']:
						raise ValueError('wrong input -- valid reference models are currently: HF, CASCI, and CASSCF')
					if self.ref['METHOD'] in ['CASCI','CASSCF']:
						if self.ref['METHOD'] == 'CASSCF' and self.model['METHOD'] not in ['SCI','FCI']:
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
					if self.ref['METHOD'] == 'CASSCF' and self.base['METHOD'] not in [None,'SCI']:
						raise ValueError('wrong input -- invalid base model for CASSCF reference model')
					if self.base['METHOD'] not in [None,'CISD','CCSD','CCSD(T)','SCI']:
						raise ValueError('wrong input -- valid base models ' + \
										'are currently: CISD, CCSD, CCSD(T), SCI, and FCI')
					# max order
					if self.max_order < 0:
						raise ValueError('wrong input -- maximum expansion order (order) must be integer >= 1')
					# wfnsym
					try:
						self.wfnsym = symm.addons.irrep_name2id(mol.symmetry, self.wfnsym)
					except Exception as err_2:
						raise ValueError('wrong input -- illegal choice of wfnsym -- PySCF error: {0:}'.format(err_2))
					if self.wfnsym != 0:
						if self.model['METHOD'] not in ['SCI','FCI']:
							raise ValueError('wrong input -- illegal choice of wfnsym for chosen expansion model')
					# target state
					if self.target > 0 and self.model['METHOD'] not in ['SCI','FCI']:
						raise ValueError('wrong input -- maximum expansion order (order) must be integer >= 1')
					# expansion and convergence thresholds
					if self.thres < 0.0:
						raise ValueError('wrong input -- expansion threshold parameter (thres) must be float: 0.0 <= thres')
					if self.relax < 1.0:
						raise ValueError('wrong input -- threshold relaxation parameter (relax) must be float: 1.0 <= relax')
					# orbital representation
					if self.occ not in ['REF','PM','FB','IBO-1','IBO-2','NO']:
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations are currently: REF, local (PM or FB), ' + \
										'intrinsic bond orbitals (IBO-1 or IBO-2), or base model natural orbitals (NO)')
					if self.virt not in ['REF','PM','FB','NO','DNO']:
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations are currently: REF, local (PM or FB), ' + \
										'or base model (distinctive) natural orbitals (NO or DNO)')
					if self.occ in ['PM','FB','IBO-1','IBO-2'] or self.virt in ['PM','FB']:
						if mol.symmetry != 'C1':
							raise ValueError('wrong input -- the combination of local orbitals and point group symmetry ' + \
											'different from C1 is not allowed')
					if self.occ == 'NO' or self.virt in ['NO','DNO']:
						if self.base['METHOD'] is None:
							raise ValueError('wrong input -- the use of (distinctive) natural orbitals (NOs/DNOs) ' + \
											'requires the use of a CC or SCI base model for the expansion')
					if self.virt == 'DNO':
						if self.typ != 'combined':
							raise ValueError('wrong input -- the use of distinctive virtual natural orbitals (DNOs) ' + \
											'is only valid in combination with combined (dual) expansions')
					if self.occ == 'NO' and self.virt == 'DNO':
						raise ValueError('wrong input -- the use of distinctive virtual natural orbitals (DNOs) ' + \
										'excludes the use of occupied natural orbitals')
					# mpi groups
					if mpi.parallel:
						if mpi.num_local_masters < 0:
							raise ValueError('wrong input -- number of local mpi masters (num_local_masters) ' + \
											'must be a positive number >= 1')
						elif mpi.num_local_masters == 0:
							if self.typ == 'combined':
								raise ValueError('wrong input -- combined expansions are only valid in ' + \
												'combination with at least one local mpi master (num_local_masters >= 1)')
						else:
							if self.typ != 'combined':
								raise ValueError('wrong input -- the use of local mpi masters ' + \
												'is currently not implemented for occupied and virtual expansions')
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
					if key.upper() in ['METHOD', 'ACTIVE']:
						new_dict[key.upper()] = value.upper()
					else:
						new_dict[key.upper()] = value
				return new_dict


