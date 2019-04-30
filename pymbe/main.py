#!/usr/bin/env python
# -*- coding: utf-8 -*

""" main.py: main program """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import os
import os.path
import shutil
import numpy as np
try:
	from mpi4py import MPI
except ImportError:
	sys.stderr.write('\nImportError : mpi4py module not found\n\n')
try:
	from pyscf import lib, scf
except ImportError:
	sys.stderr.write('\nImportError : pyscf module not found\n\n')

import parallel
import system
import calculation
import expansion
import kernel
import driver
import restart
import results
import tools


def main():
		""" main program """
		# general settings and sanity checks
		_setup()
		# mpi, mol, calc, and exp objects
		mpi, mol, calc, exp = _init()
		# branch
		if not mpi.master:
			# proceed to slave driver
			driver.slave(mpi, mol, calc, exp)
		else:
			# rm out if present
			if os.path.isdir(tools.OUT):
				shutil.rmtree(tools.OUT, ignore_errors=True)
			# mkdir out
			os.mkdir(tools.OUT)
			# init logger
			sys.stdout = tools.Logger(tools.OUT_FILE)
			# proceed to main driver
			driver.master(mpi, mol, calc, exp)
			# re-init logger
			sys.stdout = tools.Logger(tools.RES_FILE, both=False)
			# print/plot results
			results.main(mpi, mol, calc, exp)
			# finalize
			parallel.final(mpi)


def _init():
		""" init mpi, mol, calc, and exp objects """
		# mpi, mol, and calc objects
		mpi = parallel.MPICls()
		mol = _mol(mpi)
		calc = _calc(mpi, mol)
		# exp object
		exp = _exp(mpi, mol, calc)
		# bcast restart info
		if calc.restart: parallel.exp(mpi, calc, exp)
		return mpi, mol, calc, exp


def _mol(mpi):
		""" init mol object """
		# mol object
		mol = system.MolCls(mpi)
		parallel.mol(mpi, mol)
		mol.make(mpi)
		return mol


def _calc(mpi, mol):
		""" init calc object """
		# calc object
		calc = calculation.CalcCls(mpi, mol)
		parallel.calc(mpi, calc)
		return calc


def _exp(mpi, mol, calc):
		""" init exp object """
		if mpi.master:
			# restart
			if calc.restart:
				# get dipole integrals
				calc.dipole = kernel.dipole_ints(mol) if calc.target in ['dipole', 'trans'] else None
				# read fundamental info
				restart.read_fund(mol, calc)
				# get ao integrals
				mol.hcore, mol.eri = kernel.ao_ints(mol)
				# get mo integrals
				mol.hcore, mol.vhf, mol.eri = kernel.mo_ints(mol, calc.mo_coeff)
				# exp object
				exp = expansion.ExpCls(mol, calc)
			# no restart
			else:
				# get ao integrals
				mol.hcore, mol.eri = kernel.ao_ints(mol)
				# hf calculation
				mol.nocc, mol.nvirt, mol.norb, calc.hf, mol.e_nuc, \
					calc.prop['hf']['energy'], calc.prop['hf']['dipole'], \
					calc.occup, calc.orbsym, \
					calc.mo_energy, calc.mo_coeff = kernel.hf(mol, calc)
				# reference and expansion spaces and mo coefficients
				calc.mo_energy, calc.mo_coeff, calc.nelec, calc.ref_space, calc.exp_space = kernel.ref_mo(mol, calc)
				# get mo integrals
				mol.hcore, mol.vhf, mol.eri = kernel.mo_ints(mol, calc.mo_coeff)
				# base energy
				calc.prop['base']['energy'] = kernel.base(mol, calc)
				# exp object
				exp = expansion.ExpCls(mol, calc)
				# reference space properties
				calc.prop['ref'][calc.target] = kernel.ref_prop(mol, calc, exp)
				# write fundamental info
				restart.write_fund(mol, calc)
		else:
			# get dipole integrals
			calc.dipole = kernel.dipole_ints(mol) if calc.target in ['dipole', 'trans'] else None
		# bcast fundamental info
		parallel.fund(mpi, mol, calc)
		# exp object on slaves
		if not mpi.master:
			# exp object
			exp = expansion.ExpCls(mol, calc)
		# pi-orbitals in case of pi-pruning, treat non-degenerate and degenerate orbitals separately
		if calc.extra['pi_pruning']:
			calc.pi_orbs = tools.pi_orbs(calc.mo_energy, calc.orbsym, calc.exp_space)
			calc.pi_hashes = tools.hash_2d(calc.pi_orbs)
			calc.pi_hashes.sort()
			# check for even number of pi-orbitals if pi-pruning is requested 
			if calc.extra['pi_pruning']:
				tools.assertion(tools.pi_pruning(calc.orbsym, calc.pi_hashes, calc.ref_space), \
								'uneven number of degenerate pi-orbitals in reference space')
			calc.exp_space = tools.sigma_orbs(calc.orbsym, calc.exp_space)
		# init tuples and hashes
		if mpi.master:
			exp.tuples, exp.hashes = expansion.init_tup(mol, calc)
		else:
			exp.hashes = expansion.init_tup(mol, calc, hashes_only=True)
		# restart
		if mpi.master:
			exp.start_order = restart.main(calc, exp)
		return exp


def _setup():
		""" set general settings and perform sanity checks"""
		# force OMP_NUM_THREADS = 1
		lib.num_threads(1)
		# mute scf checkpoint files
		scf.hf.MUTE_CHKFILE = True
		# PYTHONHASHSEED
		pythonhashseed = os.environ.get('PYTHONHASHSEED', -1)
		tools.assertion(int(pythonhashseed) == 0, \
						'environment variable PYTHONHASHSEED must be set to zero')


if __name__ == '__main__':
	main()


