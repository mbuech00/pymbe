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

import sys
import numpy as np
import scipy as sp
from functools import reduce
try:
	from pyscf import gto, scf, ao2mo, mp, ci, cc, fci
except ImportError:
	sys.stderr.write('\nImportError : pyscf module not found\n\n')


class PySCFCls():
		""" pyscf class """
		def start(self, _mol, _calc):
				""" underlying calculation """
				# perform hf calc
				hf = scf.RHF(_mol)
				hf.conv_tol = 1.0e-12
				hf.kernel()
				# determine dimensions
				norb = hf.mo_coeff.shape[1]
				nocc = int(hf.mo_occ.sum()) // 2
				nvirt = norb - nocc
				# zeroth-order energy
				if (_calc.exp_base == 'HF'):
					e_zero = 0.0
				elif (_calc.exp_base == 'MP2'):
					# calculate mp2 energy
					mp2 = mp.MP2(hf)
					mp2.frozen = _mol.ncore
					e_zero = mp2.kernel()[0]
				elif (_calc.exp_base == 'CCSD'):
					# calculate ccsd energy
					ccsd = cc.CCSD(hf)
					ccsd.conv_tol = 1.0e-10
					ccsd.frozen = _mol.ncore
					e_zero = ccsd.kernel()[0]
				#
				return hf.mo_coeff, hf.get_hcore(), hf.e_tot, norb, nocc, nvirt, e_zero


		def int_trans(self, _mol, _calc, _exp):
				""" integral transformation """
				# define frozen list
				if (_calc.exp_type in ['occupied','virtual']):
					frozen = list(range(_mol.ncore)) 
				else:
					frozen = sorted(list(set(range(_mol.nocc)) - set(_exp.incl_idx))) 
				# perform hf calc
				mol = gto.M(verbose=0)
				mol.nelectron = _mol.nelectron
				hf = scf.RHF(mol)
				hf.conv_tol = 1.0e-12
				h2e = ao2mo.kernel(_mol, _mol.mo_coeff)
				hf._eri = ao2mo.restore(8, h2e, _mol.norb)
				h1e = reduce(np.dot, (np.transpose(_mol.mo_coeff), _mol.hcore, _mol.mo_coeff))
				hf.get_hcore = lambda *args: h1e
				hf.get_ovlp = lambda *args: np.eye(_mol.norb)
				hf.kernel()
				# e_ref
				if (_calc.exp_base == 'MP2'):
					# calculate mp2 energy
					mp2 = mp.MP2(hf)
					mp2.frozen = frozen
					e_ref = mp2.kernel()[0]
				elif (_calc.exp_base == 'CCSD'):
					# calculate ccsd energy
					ccsd = cc.CCSD(hf)
					ccsd.conv_tol = 1.0e-10
					ccsd.frozen = frozen
					e_ref = ccsd.kernel()[0]
				# integrals
				if (_calc.exp_virt == 'HF'):
					h2e = ao2mo.restore(1, h2e, _mol.norb)
				else:
					if (_calc.exp_virt == 'MP2'):
						if (_calc.exp_base != 'MP2'):
							mp2 = mp.MP2(hf)
							mp2.frozen = frozen
							mp2.kernel()
						dm = mp2.make_rdm1()
						occup, no = sp.linalg.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):])
						mo_coeff_virt = np.dot(_mol.mo_coeff[:, _mol.nocc:], no[:, ::-1])
						trans_mat = _mol.mo_coeff
						trans_mat[:,_mol.nocc:] = mo_coeff_virt
					elif (_calc.exp_virt == 'CCSD'):
						if (_calc.exp_base != 'CCSD'):
							ccsd = cc.CCSD(hf)
							ccsd.frozen = frozen
							ccsd.kernel()
						dm = ccsd.make_rdm1()
						occup, no = sp.linalg.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):])
						mo_coeff_virt = np.dot(_mol.mo_coeff[:, _mol.nocc:], no[:, ::-1])
						trans_mat = _mol.mo_coeff
						trans_mat[:,_mol.nocc:] = mo_coeff_virt
					h1e = reduce(np.dot, (np.transpose(trans_mat), _mol.hcore, trans_mat))
					h2e = ao2mo.kernel(_mol, trans_mat)
					h2e = ao2mo.restore(1, h2e, _mol.norb)
				#
				return h1e, h2e


		def corr_input(self, _mol, _calc, _exp, _tup):
				""" generate input for correlated calculation """
				# generate orbital lists
				cas_idx = sorted(_exp.incl_idx + _tup.tolist())
				core_idx = sorted(list(set(range(_mol.nocc)) - set(cas_idx)))
				# extract core and cas integrals and calculate core energy
				if (len(core_idx) > 0):
					vhf_core = np.einsum('iipq->pq', _exp.h2e[core_idx][:,core_idx]) * 2
					vhf_core -= np.einsum('piiq->pq', _exp.h2e[:,core_idx][:,:,core_idx])
					h1e_cas = (_exp.h1e + vhf_core)[cas_idx][:,cas_idx]
				else:
					h1e_cas = _exp.h1e[cas_idx][:,cas_idx]
				h2e_cas = _exp.h2e[cas_idx][:,cas_idx][:,:,cas_idx][:,:,:,cas_idx]
				# set core energy
				if (len(core_idx) > 0):
					e_core = _exp.h1e[core_idx][:,core_idx].trace() * 2 + \
								vhf_core[core_idx][:,core_idx].trace() + \
								_mol.energy_nuc()
				else:
					e_core = _mol.energy_nuc()
				#
				return core_idx, cas_idx, h1e_cas, h2e_cas, e_core


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
				solver_cas.conv_tol = 1.0e-10
				solver_cas.max_memory = _mol.max_memory
				solver_cas.max_cycle = 100
				# cas calculation
				e_cas = solver_cas.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx),
											_mol.nelectron - 2 * len(_exp.core_idx))[0]
				# base calculation
				if (_calc.exp_base == 'HF'):
					e_corr = (e_cas + _exp.e_core) - _mol.e_hf
				else:
					solver_base = ModelSolver(_calc.exp_base)
					# base calculation
					e_base = solver_base.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx),
												_mol.nelectron - 2 * len(_exp.core_idx))[0]
					e_corr = e_cas - e_base
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
				cas_hf.conv_tol = 1.0e-12
				cas_hf._eri = ao2mo.restore(8, _h2e, _norb)
				cas_hf.get_hcore = lambda *args: _h1e
				cas_hf.get_ovlp = lambda *args: np.eye(_norb)
				cas_hf.kernel()
				if (self.model_type == 'MP2'):
					self.model = mp.MP2(cas_hf)
					e_corr = self.model.kernel()[0]
				elif (self.model_type == 'CCSD'):
					self.model = cc.CCSD(cas_hf)
					self.model.conv_tol = 1.0e-10
					self.model.diis_space = 12
					self.model.max_cycle = 100
					self.eris = self.model.ao2mo()
					try:
						e_corr = self.model.kernel(eris=self.eris)[0]
					except Exception as err:
						try:
							self.model.diis_start_cycle = 2
							e_corr = self.model.kernel(eris=self.eris)[0]
						except Exception as err:
							try:
								raise RuntimeError
							except RuntimeError:
								sys.stderr.write('\nCASCCSD Error\n\n')
								raise
				e_tot = cas_hf.e_tot + e_corr
				#
				return e_tot, None


