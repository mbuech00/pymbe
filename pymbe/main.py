#!/usr/bin/env python
# -*- coding: utf-8 -*

""" main.py: main program """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
try:
	from mpi4py import MPI
except ImportError:
	sys.stderr.write('\nImportError : mpi4py module not found\n\n')
try:
	from pyscf import lib, scf
except ImportError:
	sys.stderr.write('\nImportError : pyscf module not found\n\n')

import parallel
import molecule
import calculation
import expansion
import kernel
import driver
import restart
import results


def main():
		""" main program """
		# general pyscf settings
		_pyscf_set()
		# mpi, mol, calc, and exp objects
		mpi, mol, calc, exp = _init()
		# branch
		if not mpi.global_master:
			if mpi.local_master:
				# proceed to local master driver
				driver.master(mpi, mol, calc, exp)
			else:
				# proceed to slave driver
				driver.slave(mpi, mol, calc, exp)
		else:
			# proceed to main driver
			driver.main(mpi, mol, calc, exp)
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
		# configure mpi
		parallel.set_mpi(mpi)
		# exp object
		exp = _exp(mpi, mol, calc)
		# bcast restart info
		if mpi.parallel and calc.restart: parallel.exp(mpi, calc, exp, mpi.global_comm)
		return mpi, mol, calc, exp


def _mol(mpi):
		""" init mol object """
		# mol object
		mol = molecule.MolCls(mpi)
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
		if mpi.global_master:
			# restart
			if calc.restart:
				# get ao integrals
				mol.hcore, mol.eri, mol.dipole = kernel.ao_ints(mol, calc)
				# read fundamental info
				restart.read_fund(mol, calc)
				# exp object
				if calc.typ != 'combined':
					exp = expansion.ExpCls(mol, calc, calc.typ)
				else:
					# exp.typ = 'occupied' for occ-virt and exp.typ = 'virtual' for virt-occ combined expansions
					raise NotImplementedError('combined expansion not implemented')
			# no restart
			else:
				# hf calculation
				calc.hf, calc.property['energy']['hf'], calc.property['dipole']['hf'], \
					calc.occup, calc.orbsym, calc.mo = kernel.hf(mol, calc)
				# get ao integrals
				mol.hcore, mol.eri, mol.dipole = kernel.ao_ints(mol, calc)
				# reference and expansion spaces
				calc.ref_space, calc.exp_space, \
					calc.no_exp, calc.no_act, calc.ne_act = kernel.active(mol, calc)
				# exp object
				if calc.typ != 'combined':
					exp = expansion.ExpCls(mol, calc, calc.typ)
				else:
					# exp.typ = 'occupied' for occ-virt and exp.typ = 'virtual' for virt-occ combined expansions
					raise NotImplementedError('combined expansion not implemented')
				# reference calculation
				calc.property['energy']['ref'], calc.property['energy']['ref_base'], \
					calc.property['dipole']['ref'], calc.mo = kernel.ref(mol, calc, exp)
				# base energy and transformation matrix
				calc.property['energy']['base'] = kernel.base(mol, calc, exp)
				# write fundamental info
				restart.write_fund(mol, calc)
		else:
			# get ao integrals
			mol.hcore, mol.eri, mol.dipole = kernel.ao_ints(mol, calc)
		# bcast fundamental info
		if mpi.parallel: parallel.fund(mpi, mol, calc)
		# restart and exp object on slaves
		if mpi.global_master:
			exp.min_order = restart.main(calc, exp)
		else:
			# exp object
			if calc.typ != 'combined':
				exp = expansion.ExpCls(mol, calc, calc.typ)
			else:
				# exp.typ = 'virtual' for occ-virt and exp.typ = 'occupied' for virt-occ combined expansions
				raise NotImplementedError('combined expansion not implemented')
		return exp


def _pyscf_set():
		""" set general pyscf settings """
		# force OMP_NUM_THREADS = 1
		lib.num_threads(1)
		# mute scf checkpoint files
		scf.hf.MUTE_CHKFILE = True


if __name__ == '__main__':
	main()


