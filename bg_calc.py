#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_calc.py: calculation class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from os.path import isfile
import re
import sys


class CalcCls():
		""" calculation class """
		def __init__(self, _mpi, _rst, _mol):
				""" init parameters """
				# set default parameters
				self.exp_model = {'METHOD': 'FCI'}
				self.exp_type = 'occupied'
				self.exp_ref = {'METHOD': 'HF'}
				self.exp_base = {'METHOD': None}
				self.exp_thres = 1.0e-10
				self.exp_relax = 1.0
				self.exp_max_order = 1000000
				self.exp_occ = 'REF'
				self.exp_virt = 'REF'
				self.tolerance = 0.0
				# init ref_mo_coeff, hf_mo_occ, and transformation matrix
				self.ref_mo_coeff = None; self.hf_mo_occ = None; self.trans_mat = None
				# set calculation parameters
				if (_mpi.global_master):
					self.exp_model, self.exp_type, self.exp_ref, self.exp_base, \
						self.exp_thres, self.exp_max_order, self.exp_occ, self.exp_virt, \
						self.exp_relax, self.tolerance, \
						_mol.max_memory, _mpi.num_local_masters = self.set_calc(_mpi, _rst, _mol)
					# sanity check
					self.sanity_chk(_mpi, _rst, _mol)
				#
				return


		def set_calc(self, _mpi, _rst, _mol):
				""" set calculation and mpi parameters from bg-calc.inp file """
				# read input file
				try:
					with open('bg-calc.inp') as f:
						content = f.readlines()
						for i in range(len(content)):
							if (content[i].split()[0][0] == '#'):
								continue
							elif (re.split('=',content[i])[0].strip() == 'model'):
								self.exp_model = eval(re.split('=',content[i])[1].strip())
								self.exp_model = self.upper(self.exp_model)
							elif (re.split('=',content[i])[0].strip() == 'type'):
								self.exp_type = re.split('=',content[i])[1].strip()
							elif (re.split('=',content[i])[0].strip() == 'ref'):
								self.exp_ref = eval(re.split('=',content[i])[1].strip())
								self.exp_ref = self.upper(self.exp_ref)
							elif (re.split('=',content[i])[0].strip() == 'base'):
								self.exp_base = eval(re.split('=',content[i])[1].strip())
								self.exp_base = self.upper(self.exp_base)
							elif (re.split('=',content[i])[0].strip() == 'thres'):
								self.exp_thres = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'max_order'):
								self.exp_max_order = int(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'occ'):
								self.exp_occ = re.split('=',content[i])[1].strip().upper()
							elif (re.split('=',content[i])[0].strip() == 'virt'):
								self.exp_virt = re.split('=',content[i])[1].strip().upper()
							elif (re.split('=',content[i])[0].strip() == 'relax'):
								self.exp_relax = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'tolerance'):
								self.tolerance = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'mem'):
								_mol.max_memory = int(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'num_local_masters'):
								_mpi.num_local_masters = int(re.split('=',content[i])[1].strip())
							# error handling
							else:
								try:
									raise RuntimeError('\''+content[i].split()[0].strip()+'\'' + \
														' keyword in bg-calc.inp not recognized')
								except Exception as err:
									_rst.rm_rst()
									sys.stderr.write('\nInputError : {0:}\n\n'.format(err))
									raise
				except IOError:
					_rst.rm_rst()
					sys.stderr.write('\nIOError : bg-calc.inp not found\n\n')
					raise
				#
				return self.exp_model, self.exp_type, self.exp_ref, self.exp_base, self.exp_thres, \
							self.exp_max_order, self.exp_occ, self.exp_virt, self.exp_relax, \
							self.tolerance, _mol.max_memory, _mpi.num_local_masters


		def sanity_chk(self, _mpi, _rst, _mol):
				""" sanity check for calculation and mpi parameters """
				try:
					# expansion model
					if (not (self.exp_model['METHOD'] in ['CISD','CCSD','CCSD(T)','SCI','FCI'])):
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: CISD, CCSD, CCSD(T), SCI, and FCI')
					# type of expansion
					if (not (self.exp_type in ['occupied','virtual','combined'])):
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are occupied, virtual, and combined')
					# reference model
					if (not (self.exp_ref['METHOD'] in ['HF','CASCI','CASSCF'])):
						raise ValueError('wrong input -- valid reference models are currently: HF, CASCI, and CASSCF')
					if (self.exp_ref['METHOD'] in ['CASCI','CASSCF']):
						if (not (self.exp_model['METHOD'] in ['SCI','FCI'])):
							raise ValueError('wrong input -- a CASCI/CASSCF reference is only meaningful for SCI or FCI expansion models')
						if (_mol.spin != 0):
							raise NotImplementedError('not implemented -- a CASCI/CASSCF reference is only implemented for closed-shell cases')
						if (not ('ACTIVE' in self.exp_ref)):
							raise ValueError('wrong input -- an active space (active) choice is required for CASCI/CASSCF references')
					if ('ACTIVE' in self.exp_ref):
						if (self.exp_ref['METHOD'] == 'HF'):
							raise ValueError('wrong input -- an active space is only meaningful for CASCI/CASSCF references')
						if (self.exp_ref['ACTIVE'] == 'MANUAL'):
							if (not ('SELECT' in self.exp_ref)):
								raise ValueError('wrong input -- a selection (select) of HF orbitals is required for manual active space')
							if (not isinstance(self.exp_ref['SELECT'], list)): 
								raise ValueError('wrong input -- select key (select) for active space must be a list')
							if (not ('NELEC' in self.exp_ref)):
								raise ValueError('wrong input -- number of electrons (nelec) in active space must be specified')
							if (('NELEC' in self.exp_ref) and (not isinstance(self.exp_ref['NELEC'], tuple))):
								raise ValueError('wrong input -- number of electrons (nelec) in active space must be a tuple (alpha,beta)')
						elif (self.exp_ref['ACTIVE'] == 'AVAS'):
							if (not ('AO_LABELS' in self.exp_ref)):
								raise ValueError('wrong input -- AO labels (AO_lABELS) is required for avas active space')
							if (not isinstance(self.exp_ref['AO_LABELS'], list)): 
								raise ValueError('wrong input -- AO labels key (AO_LABELS) for active space must be a list')
						else:
							raise ValueError('wrong input -- active space choices are currently: MANUAL and AVAS')
					# base model
					if ((self.exp_ref['METHOD'] != 'HF') and (not (self.exp_base['METHOD'] in [None,'SCI']))):
						raise ValueError('wrong input -- invalid base model for choice of reference model')
					if (not (self.exp_base['METHOD'] in [None,'CISD','CCSD','CCSD(T)','SCI'])):
						raise ValueError('wrong input -- valid base models ' + \
										'are currently: CISD, CCSD, CCSD(T), SCI, and FCI')
					# max order
					if (self.exp_max_order < 0):
						raise ValueError('wrong input -- wrong maximum ' + \
										'expansion order (must be integer >= 1)')
					# expansion and convergence thresholds
					if (self.exp_thres < 0.0):
						raise ValueError('wrong input -- expansion threshold parameter ' + \
										'(thres) must be float: 0.0 <= thres')
					if (self.exp_relax < 1.0):
						raise ValueError('wrong input -- threshold relaxation parameter ' + \
										'(relax) must be float: 1.0 <= relax')
					# orbital representation
					if (not (self.exp_occ in ['REF','PM','FB','IBO-1','IBO-2','NO'])):
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations are currently: REF, local (PM or FB), ' + \
										'intrinsic bond orbitals (IBO-1 or IBO-2), or base model natural orbitals (NO)')
					if (not (self.exp_virt in ['REF','PM','FB','NO','DNO'])):
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations are currently: REF, local (PM or FB), ' + \
										'or base model (distinctive) natural orbitals (NO or DNO)')
					if (((self.exp_occ == 'NO') or (self.exp_virt in ['NO','DNO'])) and (self.exp_base['METHOD'] is None)):
						raise ValueError('wrong input -- the use of (distinctive) natural orbitals (NOs/DNOs) ' + \
										'requires the use of a CC or SCI base model for the expansion')
					if ((self.exp_type != 'combined') and (self.exp_virt == 'DNO')):
						raise ValueError('wrong input -- the use of distinctive virtual natural orbitals (DNOs) ' + \
										'is only valid in combination with combined (dual) expansions')
					if ((self.exp_occ == 'NO') and (self.exp_virt == 'DNO')):
						raise ValueError('wrong input -- the use of distinctive virtual natural orbitals (DNOs) ' + \
										'excludes the use of occupied natural orbitals')
					# mpi groups
					if (_mpi.parallel):
						if (_mpi.num_local_masters < 0):
							raise ValueError('wrong input -- number of local mpi masters (num_local_masters) ' + \
											'must be a positive number >= 1')
						if ((self.exp_type == 'combined') and (_mpi.num_local_masters == 0)):
							raise ValueError('wrong input -- combined expansions are only valid in ' + \
											'combination with at least one local mpi master (num_local_masters >= 1)')
						if ((self.exp_type != 'combined') and (_mpi.num_local_masters >= 1)):
							raise ValueError('wrong input -- the use of local mpi masters ' + \
											'is currently not implemented for occupied and virtual expansions')
						if (_mpi.global_size <= 2 * _mpi.num_local_masters):
							raise ValueError('wrong input -- total number of mpi processes ' + \
											'must be larger than twice the number of local mpi masters (num_local_masters)')
					# memory
					if (_mol.max_memory is None):
						raise ValueError('wrong input -- the memory keyword (mem) appears to be missing')
				except Exception as err:
					_rst.rm_rst()
					sys.stderr.write('\nValueError : {0:}\n\n'.format(err))
					raise
				#
				return


		def upper(self, old_dict):
				""" capitalize keys """
				new_dict = {}
				for key, value in old_dict.items():
					if (key.upper() in ['METHOD','ACTIVE']):
						new_dict[key.upper()] = value.upper()
					else:
						new_dict[key.upper()] = value
				#
				return new_dict


