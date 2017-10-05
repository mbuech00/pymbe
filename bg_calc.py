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
				self.exp_model = 'CCSD'
				self.exp_type = 'occupied'
				self.exp_ref = {'METHOD': 'HF'}
				self.exp_base = 'REF'
				self.exp_thres = 0.0
				self.exp_max_order = 0
				self.exp_occ = 'REF'
				self.exp_virt = 'REF'
				self.energy_thres = 0.0
				self.tolerance = 0.0
				# init hf_dens, ref_dens, and transformation matrix
				self.hf_dens = None; self.ref_dens = None; self.trans_mat = None
				# init h1e and h2e
				self.h1e = None; self.h2e = None
				# set calculation parameters
				if (_mpi.global_master):
					self.exp_model, self.exp_type, self.exp_ref, self.exp_base, \
						self.exp_thres, self.exp_max_order, self.exp_occ, self.exp_virt, \
						self.energy_thres, self.tolerance, _mol.verbose, _mol.max_memory, \
						_mpi.num_local_masters = self.set_calc(_mpi, _rst, _mol)
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
								self.exp_model = re.split('=',content[i])[1].strip().upper()
							elif (re.split('=',content[i])[0].strip() == 'exp_type'):
								self.exp_type = re.split('=',content[i])[1].strip()
							elif (re.split('=',content[i])[0].strip() == 'exp_ref'):
								self.exp_ref = eval(re.split('=',content[i])[1].strip())
								self.exp_ref = self.upper(self.exp_ref)
							elif (re.split('=',content[i])[0].strip() == 'exp_base'):
								self.exp_base = re.split('=',content[i])[1].strip().upper()
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
					if (not (self.exp_model in ['CISD','CCSD','CCSD(T)','FCI'])):
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: CISD, CCSD, CCSD(T), and FCI')
					# type of expansion
					if (not (self.exp_type in ['occupied','virtual','combined'])):
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are occupied, virtual, and combined')
					# reference model
					if (not ('METHOD' in self.exp_ref)):
						raise ValueError('wrong input -- exp_ref dictionary must contain "method" key ' + \
										'with method value given as a string')
					if (not (self.exp_ref['METHOD'] in ['HF','DFT','CASSCF'])):
						raise ValueError('wrong input -- invalid reference model')
					if ((self.exp_ref['METHOD'] != 'HF') and _mol.frozen):
						raise ValueError('wrong input -- non-HF reference is not allowed for frozen-core calculations')
					if ((self.exp_ref['METHOD'] == 'DFT') and (not ('XC' in self.exp_ref))):
						raise ValueError('wrong input -- missing "xc" key in exp_ref dictionary for ' + \
										'DFT reference model (with choice of xc given as a string)')
					if ((self.exp_ref['METHOD'] == 'CASSCF') and (not ('AO_LABELS' in self.exp_ref))):
						raise ValueError('wrong input -- missing "ao_labels" key in exp_ref dictionary for ' + \
										'CASSCF reference model (with ao_labels given as a list of strings)')
					# base model
					if (not (self.exp_base in ['REF','CISD','CCSD','CCSD(T)'])):
						raise ValueError('wrong input -- invalid base model')
					if (((self.exp_base == 'CISD') and (self.exp_model in ['CISD'])) or \
						((self.exp_base == 'CCSD') and (self.exp_model in ['CISD','CCSD'])) or \
						((self.exp_base == 'CCSD(T)') and (self.exp_model in ['CISD','CCSD','CCSD(T)']))):
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
					if (not (self.exp_occ in ['REF','PM','FB','IBO-1','IBO-2','NO'])):
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations are currently: REF, local (PM or FB), ' + \
										'intrinsic bond orbitals (IBO-1 or IBO-2), or base model natural orbitals (NO)')
					if (not (self.exp_virt in ['REF','PM','FB','NO','DNO'])):
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations are currently: REF, local (PM or FB), ' + \
										'or base model (distinctive) natural orbitals (NO or DNO)')
					if (((self.exp_occ == 'NO') or (self.exp_virt in ['NO','DNO'])) and (self.exp_base == 'REF')):
						raise ValueError('wrong input -- the use of (distinctive) natural orbitals (NOs/DNOs) ' + \
										'requires the use of a correlated base model for the expansion')
					if ((_mol.symmetry.upper() != 'C1') and ((self.exp_occ in ['PM','FB','IBO-1','IBO-2']) or \
							(self.exp_virt in ['PM','FB']))):
						raise ValueError('wrong input -- the use of local orbitals (PM or FB) ' + \
										'or intrinsic bond orbitals (IBO-1 or IBO-2) excludes the use of symmetry (must be C1)')
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
					if (key.upper() in ['METHOD', 'XC']):
						new_dict[key.upper()] = value.upper()
					else:
						new_dict[key.upper()] = value
				#
				return new_dict


