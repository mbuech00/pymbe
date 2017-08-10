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
import sys


class CalcCls():
		""" calculation class """
		def __init__(self, _mpi, _rst):
				""" init parameters """
				# set default parameters
				self.exp_model = 'CCSD'
				self.exp_type = 'occupied'
				self.exp_base = 'HF'
				self.exp_thres = 10.0
				self.exp_max_order = 0
				self.exp_occ = 'HF'
				self.exp_virt = 'HF'
				self.energy_thres = 3.8e-05
				self.tolerance = 0.0
				# set calculation parameters
				if (_mpi.global_master):
					self.exp_model, self.exp_type, self.exp_base, self.exp_thres, \
						self.exp_max_order, self.exp_occ, self.exp_virt, \
						self.energy_thres, self.tolerance, \
						_mpi.num_local_masters = self.set_calc(_mpi, _rst)
					# sanity check
					self.sanity_chk(_mpi, _rst)
				if (_mpi.parallel):
					# bcast calc and mpi info
					_mpi.bcast_calc_info(self)
					_mpi.bcast_mpi_info()
				# set local groups
				_mpi.set_local_groups()
				#
				return


		def set_calc(self, _mpi, _rst):
				""" set calculation and mpi parameters from bg-calc.inp file """
				# read input file
				try:
					with open('bg-calc.inp') as f:
						content = f.readlines()
						for i in range(len(content)):
							if (content[i].split()[0][0] == '#'):
								continue
							elif (content[i].split()[0] == 'exp_model'):
								self.exp_model = content[i].split()[2].upper()
							elif (content[i].split()[0] == 'exp_type'):
								self.exp_type = content[i].split()[2]
							elif (content[i].split()[0] == 'exp_base'):
								self.exp_base = content[i].split()[2].upper()
							elif (content[i].split()[0] == 'exp_thres'):
								self.exp_thres = float(content[i].split()[2])
							elif (content[i].split()[0] == 'exp_max_order'):
								self.exp_max_order = int(content[i].split()[2])
							elif (content[i].split()[0] == 'exp_occ'):
								self.exp_occ = content[i].split()[2].upper()
							elif (content[i].split()[0] == 'exp_virt'):
								self.exp_virt = content[i].split()[2].upper()
							elif (content[i].split()[0] == 'energy_thres'):
								self.energy_thres = float(content[i].split()[2])
							elif (content[i].split()[0] == 'tolerance'):
								self.tolerance = float(content[i].split()[2])
							elif (content[i].split()[0] == 'num_local_masters'):
								_mpi.num_local_masters = int(content[i].split()[2])
							# error handling
							else:
								try:
									raise RuntimeError('\''+content[i].split()[0]+'\'' + \
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
				return self.exp_model, self.exp_type, self.exp_base, self.exp_thres, \
							self.exp_max_order, self.exp_occ, self.exp_virt, self.energy_thres, self.tolerance, \
							_mpi.num_local_masters


		def sanity_chk(self, _mpi, _rst):
				""" sanity check for calculation and mpi parameters """
				try:
					# expansion model
					if (not (self.exp_model in ['MP2','CCSD','FCI'])):
						raise ValueError('wrong input -- valid expansion models ' + \
										'are currently: MP2, CCSD, and FCI')
					# type of expansion
					if (not (self.exp_type in ['occupied','virtual','combined'])):
						raise ValueError('wrong input -- valid choices for ' + \
										'expansion scheme are occupied, virtual, and combined')
					# base model
					if (not (self.exp_base in ['HF','MP2','CCSD'])):
						raise ValueError('wrong input -- valid base models ' + \
										'are currently: HF, MP2, and CCSD')
					if (((self.exp_base == 'MP2') and (self.exp_model == 'MP2')) or \
						((self.exp_base == 'CCSD') and (self.exp_model in ['MP2','CCSD']))):
							raise ValueError('wrong input -- invalid base model for choice ' + \
											'of expansion model')
					# max order
					if (self.exp_max_order < 0):
						raise ValueError('wrong input -- wrong maximum ' + \
										'expansion order (must be integer >= 1)')
					# expansion and energy thresholds
					if (self.exp_thres < 0.0):
						raise ValueError('wrong input -- expansion threshold ' + \
										'(exp_thres) must be float >= 0.0')
					if (self.energy_thres < 0.0):
						raise ValueError('wrong input -- energy threshold ' + \
										'(energy_thres) must be float >= 0.0')
					# orbital representation
					if (not (self.exp_occ in ['HF','PM','ER','BOYS','NO'])):
						raise ValueError('wrong input -- valid occupied orbital ' + \
										'representations are currently: HF, local (PM, ER, or Boys), or base model natural orbitals')
					if (not (self.exp_virt in ['HF','NO','DNO'])):
						raise ValueError('wrong input -- valid virtual orbital ' + \
										'representations are currently: HF or base model (distinctive) natural orbitals (NOs or DNOs)')
					if (((self.exp_occ == 'NO') or (self.exp_virt in ['NO','DNO'])) and (self.exp_base == 'HF')):
						raise ValueError('wrong input -- the use of (distinctive) natural orbitals (NOs/DNOs) ' + \
										'requires the use of a correlated base model for the expansion')
					if ((self.exp_type != 'combined') and (self.exp_virt == 'DNO')):
						raise ValueError('wrong input -- the use of distinctive virtual natural orbitals (DNOs) ' + \
										'is only valid in combination with combined (dual) expansions')
					if ((self.exp_occ == 'NO') and (self.exp_virt == 'DNO')):
						raise ValueError('wrong input -- the use of distinctive virtual natural orbitals (DNOs) ' + \
										'excludes the use of occupied natural orbitals')
					# mpi groups
					if (_mpi.parallel):
						if (_mpi.num_local_masters < 0):
							raise ValueError('wrong input -- number of local mpi masters ' + \
											'must be a positive number >= 1')
						if ((self.exp_type == 'combined') and (_mpi.num_local_masters == 0)):
							raise ValueError('wrong input -- combined expansions are only valid in ' + \
											'combination with at least one local mpi master (num_local_masters >= 1)')
						if ((self.exp_type != 'combined') and (_mpi.num_local_masters >= 1)):
							raise ValueError('wrong input -- the use of local mpi masters ' + \
											'is currently not implemented for occupied and virtual expansions')
						if (_mpi.global_size <= 2 * _mpi.num_local_masters):
							raise ValueError('wrong input -- total number of mpi processes ' + \
											'must be larger than twice the number of local mpi masters')
				except Exception as err:
					_rst.rm_rst()
					sys.stderr.write('\nValueError : {0:}\n\n'.format(err))
					raise
				#
				return


