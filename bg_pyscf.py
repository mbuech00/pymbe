#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_pyscf.py: pyscf-related routines for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
import scipy as sp
from functools import reduce
from pyscf import gto, scf, ao2mo, mp, ci, cc, fci


class PySCFCls():
		""" pyscf class """
		def hf_calc(self, _mol):
				""" underlying hf calculation """
				# perform hf calc
				hf = scf.RHF(_mol)
				hf.conv_tol = 1.0e-10
				hf.kernel()
				# determine dimensions
				norb = hf.mo_coeff.shape[1]
				nocc = int(hf.mo_occ.sum()) // 2
				nvirt = norb - nocc
				# duplicate total hf energy
				e_hf = hf.e_tot
				#
				return hf, e_hf, norb, nocc, nvirt


		def int_trans(self, _mol, _calc):
				""" integral transformation """
				# e_ref
				if (_calc.exp_base == 'HF'):
					e_ref = 0.0
				elif (_calc.exp_base == 'MP2'):
					# calculate mp2 energy
					mp2 = mp.MP2(_mol.hf)
					e_ref = mp2.kernel()[0]
				elif (_calc.exp_base == 'CCSD'):
					# calculate ccsd energy
					ccsd = cc.CCSD(_mol.hf)
					e_ref = ccsd.kernel()[0]
				# integrals
				if (_calc.exp_virt == 'HF'):
					trans_mat = _mol.hf.mo_coeff
				elif (_calc.exp_virt == 'MP2'):
					if (_calc.exp_base != 'MP2'):
						mp2 = mp.MP2(_mol.hf)
						mp2.kernel()
					dm = mp2.make_rdm1()
					occup, no = sp.linalg.eigh(dm[_mol.nocc:,_mol.nocc:])
					mo_coeff_virt = np.dot(_mol.hf.mo_coeff[:,_mol.nocc:], no[:,::-1])
					trans_mat = _mol.hf.mo_coeff
					trans_mat[:,_mol.nocc:] = mo_coeff_virt
				elif (_calc.exp_virt == 'CCSD'):
					if (_calc.exp_base != 'CCSD'):
						ccsd = cc.CCSD(_mol.hf)
						ccsd.kernel()
					dm = ccsd.make_rdm1()
					occup, no = sp.linalg.eigh(dm[_mol.nocc:,_mol.nocc:])
					mo_coeff_virt = np.dot(_mol.hf.mo_coeff[:,_mol.nocc:], no[:,::-1])
					trans_mat = _mol.hf.mo_coeff
					trans_mat[:,_mol.nocc:] = mo_coeff_virt
				# transform 1- and 2-electron integrals
				h1e = reduce(np.dot, (np.transpose(trans_mat), _mol.hf.get_hcore(), trans_mat))
				h2e = ao2mo.kernel(_mol, trans_mat) # with four-fold permutation symmetry
				h2e = ao2mo.restore(1, h2e, _mol.norb) # remove symmetry
				#
				return e_ref, h1e, h2e


		def corr_input(self, _mol, _calc, _exp, _tup):
				""" generate input for correlated calculation """
				# generate orbital lists
				cas_idx = sorted(_exp.incl_idx + _tup.tolist())
				core_idx = sorted(list(set(range(_mol.nocc)) - set(cas_idx)))
				# extract core and cas integrals and calculate core energy
				if (len(core_idx) > 0):
					vhf_core = np.einsum('iipq->pq', _mol.h2e[core_idx][:,core_idx]) * 2
					vhf_core -= np.einsum('piiq->pq', _mol.h2e[:,core_idx][:,:,core_idx])
					h1e_cas = (_mol.h1e + vhf_core)[cas_idx][:,cas_idx]
				else:
					h1e_cas = _mol.h1e[cas_idx][:,cas_idx]
				h2e_cas = _mol.h2e[cas_idx][:,cas_idx][:,:,cas_idx][:,:,:,cas_idx]
				# set core energy
				if (len(core_idx) > 0):
					e_core = _mol.h1e[core_idx][:,core_idx].trace() * 2 + \
								vhf_core[core_idx][:,core_idx].trace() + \
								_mol.energy_nuc()
				else:
					e_core = _mol.energy_nuc()
				#
				return cas_idx, core_idx, h1e_cas, h2e_cas, e_core


		def corr_calc(self, _mol, _calc, _exp):
				""" correlated cas calculation """
				# init solver
				if (_calc.exp_model != 'FCI'):
					solver_cas = ModelSolver(_calc.exp_model)
				else:
					if (_mol.spin == 0):
						solver_cas = fci.direct_spin0.FCI()
					else:
						solver_cas = fci.direct_spin1.FCI()
				# settings
				solver_cas.conv_tol = 1.0e-08
				solver_cas.max_memory = _mol.max_memory
				# cas calculation
				e_cas = solver_cas.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx),
											_mol.nelectron - 2 * len(_exp.core_idx))[0]
				# base calculation
				if (_calc.exp_base == 'HF'):
					e_corr = (e_cas + _exp.e_core) - _mol.e_hf
				else:
					solver_base = ModelSolver(_calc.exp_base)
					solver_base.conv_tol = 1.0e-08
					solver_base.max_memory = _mol.max_memory
					# base calculation
					e_base = solver_base.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx),
												_mol.nelectron - 2 * len(_exp.core_idx))[0]
					e_corr = e_cas - e_base
				print('cas_idx = {0:} , e_corr = {1:.6f}'.format(_exp.cas_idx,e_corr))
				#
				return e_corr


class ModelSolver():
		""" MP2 or CCSD as active space solver, 
		adapted from cc test: 42-as_casci_fcisolver.py of the pyscf test suite
		"""
		def __init__(self, model):
				""" init model object """
				self.model_type = model
				self.model = None
				self.eris = None
				#
				return


		def kernel(self, _h1e, _h2e, _norb, _nelec):
				""" model kernel """
				cas_mol = gto.M(verbose=0)
				cas_mol.nelectron = _nelec
				cas_hf = scf.RHF(cas_mol)
				cas_hf._eri = ao2mo.restore(8, _h2e, _norb)
				cas_hf.get_hcore = lambda *args: _h1e
				cas_hf.get_ovlp = lambda *args: np.eye(_norb)
				cas_hf.kernel()
				if (self.model_type == 'MP2'):
					self.model = mp.MP2(cas_hf)
					e_corr = self.model.kernel()[0]
				elif (self.model_type == 'CCSD'):
					self.model = cc.CCSD(cas_hf)
					self.eris = self.model.ao2mo()
					e_corr = self.model.kernel(eris=self.eris)[0]
				e_tot = cas_hf.e_tot + e_corr
				#
				return e_tot, None


