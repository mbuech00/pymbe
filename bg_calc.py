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
				self.exp_base = None
				self.exp_thres = 0.0
				self.exp_max_order = 0
				self.exp_occ = 'CAN'
				self.exp_virt = 'CAN'
				self.energy_thres = 0.0
				self.tolerance = 0.0
				# init hf_mo_coeff, hf_mo_occ, and transformation matrix
				self.hf_mo_coeff = None; self.hf_mo_occ = None; self.trans_mat = None
				# init h1e and h2e
				self.h1e = None; self.h2e = None
				# set calculation parameters
				if (_mpi.global_master):
					self.exp_model, self.exp_type, self.exp_ref, self.exp_base, \
						self.exp_thres, self.exp_max_order, self.exp_occ, self.exp_virt, \
						self.energy_thres, self.tolerance, _mol.verbose, _mol.max_memory, \
						_mpi.num_local_masters = self.set_calc(_mpi, _rst, _mol)
					# if not given, set exp_base equal to exp_ref
					if (self.exp_base is None): self.exp_base = {'METHOD': self.exp_ref['METHOD']}
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
							elif (re.split('=',content[i])[0].strip() == 'exp_model'):
								self.exp_model = eval(re.split('=',content[i])[1].strip())
								self.exp_model = self.upper(self.exp_model)
							elif (re.split('=',content[i])[0].strip() == 'exp_type'):
								self.exp_type = re.split('=',content[i])[1].strip()
							elif (re.split('=',content[i])[0].strip() == 'exp_ref'):
								self.exp_ref = eval(re.split('=',content[i])[1].strip())
								self.exp_ref = self.upper(self.exp_ref)
							elif (re.split('=',content[i])[0].strip() == 'exp_base'):
								self.exp_base = eval(re.split('=',content[i])[1].strip())
								self.exp_base = self.upper(self.exp_base)
							elif (re.split('=',content[i])[0].strip() == 'exp_thres'):
								self.exp_thres = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'exp_max_order'):
								self.exp_max_order = int(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'exp_occ'):
								self.exp_occ = re.split('=',content[i])[1].strip().upper()
							elif (re.split('=',content[i])[0].strip() == 'exp_virt'):
								self.exp_virt = re.split('=',content[i])[1].strip().upper()
							elif (re.split('=',content[i])[0].strip() == 'energy_thres'):
								self.energy_thres = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'tolerance'):
								self.tolerance = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'verbose'):
								_mol.verbose = int(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'memory'):
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
							self.exp_max_order, self.exp_occ, self.exp_virt, self.energy_thres, \
							self.tolerance, _mol.verbose, _mol.max_memory, _mpi.num_local_masters


		def sanity_chk(self, _mpi, _rst, _mol):
				""" sanity check for calculation and mpi parameters """
				try:
					# expansion model
					if (not ('METHOD' in self.exp_model)):
						raise ValueError('wrong input -- exp_model dictionary must contain "method" key ' + \
										'with method value given as a string')
					if (not (self.exp_model['METHOD'] in ['CISD','CCSD','CCSD(T)','FCI'])):
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: CISD, CCSD, CCSD(T), and FCI')
					# type of expansion
					if (not (self.exp_type in ['occupied','virtual','combined'])):
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are occupied, virtual, and combined')
					# reference model
					if (not ('METHOD' in self.exp_ref)):
						raise ValueError('wrong input -- exp_ref dictionary must contain "method" key ' + \
										'with method value ("HF" or "CASCI") given as a string')
					if (not (self.exp_ref['METHOD'] in ['HF','CASCI'])):
						raise ValueError('wrong input -- valid reference models are currently: HF and CASCI')
					if (_mol.spin > 0):
						if ((self.exp_ref['METHOD'] == 'HF') and (self.exp_type == 'occupied')):
							raise ValueError('wrong input -- open-shell CASCI reference is required for occupied expansions, ' + \
											'since open-shell "spin" value (2S = n_a - n_b > 0) given in bg_mol.inp')
						if ((self.exp_ref['METHOD'] == 'CASCI') and (self.exp_type == 'virtual')):
							raise ValueError('wrong input -- open-shell CASCI reference is equal to HF reference for virtual expansions, ' + \
											'please rerun with exp_ref "method": "HF"')
					if ((self.exp_ref['METHOD'] == 'CASCI') and (_mol.spin == 0)):
						raise ValueError('wrong input -- open-shell CASCI reference requested, but ' + \
										'closed-shell "spin" value (2S = n_a - n_b = 0) given in bg_mol.inp')
					# base model
					if (not ('METHOD' in self.exp_base)):
						raise ValueError('wrong input -- exp_base dictionary must contain "method" key ' + \
										'with method value given as a string')
					if (not (self.exp_base['METHOD'] in [self.exp_ref['METHOD'],'CISD','CCSD','CCSD(T)'])):
						raise ValueError('wrong input -- invalid base model')
					if (((self.exp_base['METHOD'] == 'CISD') and (self.exp_model['METHOD'] in ['CISD'])) or \
						((self.exp_base['METHOD'] == 'CCSD') and (self.exp_model['METHOD'] in ['CISD','CCSD'])) or \
						((self.exp_base['METHOD'] == 'CCSD(T)') and (self.exp_model['METHOD'] in ['CISD','CCSD','CCSD(T)']))):
						raise ValueError('wrong input -- invalid base model for choice ' + \
										'of expansion model')
					# max order
					if (self.exp_max_order < 0):
						raise ValueError('wrong input -- wrong maximum ' + \
										'expansion order (must be integer >= 1)')
					# expansion and energy thresholds
					if (self.exp_thres < 0.0):
						raise ValueError('wrong input -- expansion threshold parameter ' + \
										'(exp_thres) must be float >= 0.0')
					if (self.energy_thres < 0.0):
						raise ValueError('wrong input -- energy threshold parameter ' + \
										'(energy_thres) must be float >= 0.0')
					# orbital representation
					if (not (self.exp_occ in ['CAN','PM','FB','IBO-1','IBO-2','NO'])):
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations are currently: CAN, local (PM or FB), ' + \
										'intrinsic bond orbitals (IBO-1 or IBO-2), or base model natural orbitals (NO)')
					if (not (self.exp_virt in ['CAN','PM','FB','NO','DNO'])):
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations are currently: CAN, local (PM or FB), ' + \
										'or base model (distinctive) natural orbitals (NO or DNO)')
					if (((self.exp_occ == 'NO') or (self.exp_virt in ['NO','DNO'])) and (self.exp_ref['METHOD'] == self.exp_base['METHOD'])):
						raise ValueError('wrong input -- the use of (distinctive) natural orbitals (NOs/DNOs) ' + \
										'requires the use of a correlated base model for the expansion')
					if (((self.exp_occ == 'NO') or (self.exp_virt in ['NO','DNO'])) and (_mol.spin > 0)):
						raise ValueError('wrong input -- the use of (distinctive) natural orbitals (NOs/DNOs) ' + \
										'is currently only implemented for closed-shell states (2S = n_a - n_b = 0)')
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
						raise ValueError('wrong input -- the memory keyword (memory) appears to be missing')
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
					if (key.upper() in ['METHOD', 'TYPE']):
						new_dict[key.upper()] = value.upper()
					else:
						new_dict[key.upper()] = value
				#
				return new_dict


