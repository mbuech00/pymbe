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
				self.exp_base = 'HF'
				self.exp_thres_init = None
				self.exp_relax = 10.0
				self.exp_max_order = 0
				self.exp_occ = 'HF'
				self.exp_virt = 'HF'
				self.protocol = 1
				self.energy_thres = 0.0
				self.tolerance = 0.0
				# init hf_dens and transformation matrix
				self.hf_dens = None; self.trans_mat = None
				# init h1e and h2e
				self.h1e = None; self.h2e = None
				# set calculation parameters
				if (_mpi.global_master):
					self.exp_model, self.exp_type, self.exp_base, self.exp_thres_init, self.exp_realx, \
						self.exp_max_order, self.exp_occ, self.exp_virt, self.protocol, \
						self.energy_thres, self.tolerance, _mol.verbose, _mol.max_memory, \
						_mpi.num_local_masters = self.set_calc(_mpi, _rst, _mol)
					# set default exp_thres_init
					if (self.exp_thres_init is None):
						if (self.protocol == 1):
							self.exp_thres_init = 1.0e-10
						elif (self.protocol == 2):
							self.exp_thres_init = 100.0
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
							elif (re.split('=',content[i])[0].strip() == 'exp_base'):
								self.exp_base = re.split('=',content[i])[1].strip().upper()
							elif (re.split('=',content[i])[0].strip() == 'exp_thres_init'):
								self.exp_thres_init = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'exp_relax'):
								self.exp_relax = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'exp_max_order'):
								self.exp_max_order = int(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'exp_occ'):
								self.exp_occ = re.split('=',content[i])[1].strip().upper()
							elif (re.split('=',content[i])[0].strip() == 'exp_virt'):
								self.exp_virt = re.split('=',content[i])[1].strip().upper()
							elif (re.split('=',content[i])[0].strip() == 'energy_thres'):
								self.energy_thres = float(re.split('=',content[i])[1].strip())
							elif (re.split('=',content[i])[0].strip() == 'protocol'):
								self.protocol = int(re.split('=',content[i])[1].strip())
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
				return self.exp_model, self.exp_type, self.exp_base, self.exp_thres_init, self.exp_relax, \
							self.exp_max_order, self.exp_occ, self.exp_virt, self.protocol, self.energy_thres, \
							self.tolerance, _mol.verbose, _mol.max_memory, _mpi.num_local_masters


		def sanity_chk(self, _mpi, _rst, _mol):
				""" sanity check for calculation and mpi parameters """
				try:
					# expansion model
					if (not (self.exp_model in ['MP2','CISD','CCSD','FCI'])):
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: MP2, CISD, CCSD, and FCI')
					# type of expansion
					if (not (self.exp_type in ['occupied','virtual','combined'])):
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are occupied, virtual, and combined')
					# base model
					if (not (self.exp_base in ['HF','MP2','CISD','CCSD'])):
						raise ValueError('wrong input -- valid base models ' + \
										'are currently: HF, MP2, and CCSD')
					if (((self.exp_base == 'MP2') and (self.exp_model == 'MP2')) or \
						((self.exp_base == 'CISD') and (self.exp_model in ['MP2','CISD'])) or \
						((self.exp_base == 'CCSD') and (self.exp_model in ['MP2','CISD','CCSD']))):
							raise ValueError('wrong input -- invalid base model for choice ' + \
											'of expansion model')
					# max order
					if (self.exp_max_order < 0):
						raise ValueError('wrong input -- wrong maximum ' + \
										'expansion order (must be integer >= 1)')
					# expansion and energy thresholds
					if (self.exp_thres_init < 0.0):
						raise ValueError('wrong input -- initial expansion threshold ' + \
										'(exp_thres_init) must be float >= 0.0')
					if (self.exp_relax < 0.0):
						raise ValueError('wrong input -- expansion relaxation ' + \
										'(exp_relax) must be float >= 0.0')
					if (self.energy_thres < 0.0):
						raise ValueError('wrong input -- energy threshold ' + \
										'(energy_thres) must be float >= 0.0')
					# orbital representation
					if (not (self.exp_occ in ['HF','PM','ER','BOYS','NO'])):
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations are currently: HF, local (PM, ER, or Boys), ' + \
										'or base model natural orbitals (NO)')
					if (not (self.exp_virt in ['HF','PM','ER','BOYS','NO','DNO'])):
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations are currently: HF local (PM, ER, or Boys), ' + \
										'or base model (distinctive) natural orbitals (NO or DNO)')
					if (((self.exp_occ == 'NO') or (self.exp_virt in ['NO','DNO'])) and (self.exp_base == 'HF')):
						raise ValueError('wrong input -- the use of (distinctive) natural orbitals (NOs/DNOs) ' + \
										'requires the use of a correlated base model for the expansion')
					if ((_mol.symmetry.upper() != 'C1') and ((self.exp_occ in ['PM','ER','BOYS']) or (self.exp_virt in ['PM','ER','BOYS']))):
						raise ValueError('wrong input -- the use of local orbitals (PM, ER, or Boys) ' + \
										'excludes the use of symmetry (must be C1)')
					if ((self.exp_type != 'combined') and (self.exp_virt == 'DNO')):
						raise ValueError('wrong input -- the use of distinctive virtual natural orbitals (DNOs) ' + \
										'is only valid in combination with combined (dual) expansions')
					if ((self.exp_occ == 'NO') and (self.exp_virt == 'DNO')):
						raise ValueError('wrong input -- the use of distinctive virtual natural orbitals (DNOs) ' + \
										'excludes the use of occupied natural orbitals')
					# screening protocol
					if ((self.protocol < 1) or (self.protocol > 2)):
						raise ValueError('wrong input -- valid values for the screening protocol (protocol) ' + \
										'are 1 and 2')
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


