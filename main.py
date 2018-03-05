#!/usr/bin/env python
# -*- coding: utf-8 -*

""" main.py: main program """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
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
		# mpi instantiation
		mpi = parallel.MPICls()
		# molecule instantiation
		mol = molecule.MolCls(mpi)
		parallel.mol(mpi, mol)
		mol.make(mpi)
		# calculation instantiation
		calc = calculation.CalcCls(mpi, mol)
		parallel.calc(mpi, calc)
		# configure mpi
		parallel.set_mpi(mpi)
		# hf and ref calculations
		if mpi.global_master:
			# restart
			if calc.restart:
				# get hcore and eri
				mol.hcore, mol.eri = kernel.hcore_eri(mol)
				# read fundamental info
				restart.read_fund(mol, calc)
				# expansion instantiation
				exp = expansion.ExpCls(mol, calc)
#				elif (calc.typ == 'combined'):
#					exp = expansion.ExpCls(mol, calc)
#					# mark expansion as macro
#					exp.level = 'macro'
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
#				elif (calc.typ == 'combined'):
#					exp = expansion.ExpCls(mol, calc, 'occupied')
#					# mark expansion as macro
#					exp.level = 'macro'
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
		# now branch
		if not mpi.global_master:
			if mpi.local_master:
				# proceed to local master driver
				driver.local_master(mpi, mol, calc)
			else:
				# proceed to slave driver
				driver.slave(mpi, mol, calc)
		else:
			# print main header
			output.main_header()
			# restart
			exp.min_order = restart.main(calc, exp)
			# proceed to main driver
			driver.main(mpi, mol, calc, exp)
			# print summary and plot results
			results.main(mpi, mol, calc, exp)
			# finalize
			parallel.final(mpi)


if __name__ == '__main__':
	main()


