#!/usr/bin/env python
# -*- coding: utf-8 -*

""" main.py: main program """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
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

import parallel
import molecule
import calculation
import expansion
import kernel
import driver
import restart
import output
import results


def main():
		""" main program """
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
		# mpi, mol, and calc instantiations
		mpi = parallel.MPICls()
		mol = _mol(mpi)
		calc = _calc(mpi, mol)
		# configure mpi
		parallel.set_mpi(mpi)
		# exp instantiation
		exp = _exp(mpi, mol, calc)
		# bcast restart info
		if mpi.parallel and calc.restart: parallel.exp(calc, exp, mpi.global_comm)
		return mpi, mol, calc, exp


def _mol(mpi):
		""" init mol object """
		mol = molecule.MolCls(mpi)
		parallel.mol(mpi, mol)
		mol.make(mpi)
		return mol


def _calc(mpi, mol):
		""" init calc object """
		calc = calculation.CalcCls(mpi, mol)
		parallel.calc(mpi, calc)
		return calc


def _exp(mpi, mol, calc):
		""" init exp object """
		if mpi.global_master:
			# restart
			if calc.restart:
				# get hcore and eri
				mol.hcore, mol.eri = kernel.hcore_eri(mol)
				# read fundamental info
				restart.read_fund(mol, calc)
				# expansion instantiation
				exp = expansion.ExpCls(mol, calc)
			# no restart
			else:
				# hf calculation
				calc.hf, calc.energy['hf'], calc.occup, calc.orbsym, calc.mo = kernel.hf(mol, calc)
				# get hcore and eri
				mol.hcore, mol.eri = kernel.hcore_eri(mol)
				# reference and expansion spaces
				calc.ref_space, calc.exp_space, calc.no_act = kernel.active(mol, calc)
				# expansion instantiation
				exp = expansion.ExpCls(mol, calc)
				# reference calculation
				calc.energy['ref'], calc.energy['ref_base'], calc.mo = kernel.ref(mol, calc, exp)
				# base energy and transformation matrix
				calc.energy['base'] = kernel.base(mol, calc, exp)
				# write fundamental info
				restart.write_fund(mol, calc)
		else:
			# get hcore and eri
			mol.hcore, mol.eri = kernel.hcore_eri(mol)
		# bcast fundamental info
		if mpi.parallel: parallel.fund(mpi, mol, calc)
		# restart and expansion instantiation on slaves
		if mpi.global_master:
			exp.min_order = restart.main(calc, exp)
		else:
			# expansion instantiation
			exp = expansion.ExpCls(mol, calc)
		return exp


if __name__ == '__main__':
	main()


