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
				# compute interals
				h1e = reduce(np.dot, (np.transpose(hf.mo_coeff), hf.get_hcore(), hf.mo_coeff))
				h2e = ao2mo.kernel(_mol, hf.mo_coeff)
				h2e = ao2mo.restore(8, h2e, norb)
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
				# integrals
				if (_calc.exp_base == 'HF'):
					trans_mat = hf.mo_coeff
				else:
					# compute NOs
					if (_calc.exp_base == 'MP2'):
						dm = mp2.make_rdm1()
					elif (_calc.exp_base == 'CCSD'):
						dm = ccsd.make_rdm1()
					# init trans_mat
					trans_mat = hf.mo_coeff
					# occ-occ block
					if (_calc.exp_occ != 'HF'):
						occup, no = sp.linalg.eigh(dm[:(nocc-_mol.ncore), :(nocc-_mol.ncore)])
						mo_coeff_no = np.dot(hf.mo_coeff[:, _mol.ncore:nocc], no[:, ::-1])
						trans_mat[:, _mol.ncore:nocc] = mo_coeff_no
					# virt-virt block
					if (_calc.exp_virt != 'HF'):
						occup, no = sp.linalg.eigh(dm[(nocc-_mol.ncore):, (nocc-_mol.ncore):])
						mo_coeff_no = np.dot(hf.mo_coeff[:, nocc:], no[:, ::-1])
						trans_mat[:, nocc:] = mo_coeff_no
				# transform integrals
				h1e = reduce(np.dot, (np.transpose(trans_mat), hf.get_hcore(), trans_mat))
				h2e = ao2mo.kernel(_mol, trans_mat)
				h2e = ao2mo.restore(1, h2e, norb)
				return h1e, h2e, hf.e_tot, norb, nocc, nvirt, e_zero


		def int_trans(self, _mol, _calc, _exp):
				""" integral transformation """
				# define frozen list
				core_idx, active_idx, h1e_active, h2e_active, e_core = \
						self.prepare(_mol, _calc, _exp, list(range(_mol.nocc, _mol.norb)))
				solver_base = ModelSolver(_calc.exp_base)
				mol_active, hf_active = solver_base.fake_hf(h1e_active, h2e_active, len(active_idx), \
												_mol.nelectron - 2 * len(core_idx))
				e_active, dm_active = solver_base.kernel(hf_active, _dens=True)
				e_ref = (e_active + e_core) - _mol.e_hf
				# compute NOs
				occup, no = sp.linalg.eigh(dm[(_mol.nocc-len(core_idx)):, (_mol.nocc-len(core_idx)):])
				mo_coeff_virt = np.dot(_mol.mo_coeff[active_idx, _mol.nocc:], no[:, ::-1])
				trans_mat = _mol.mo_coeff[active_idx, active_idx]
				trans_mat[(_mol.nocc-len(core_idx)):, (_mol.nocc-len(core_idx)):] = mo_coeff_virt
				# transform integrals
				h1e_active = reduce(np.dot, (np.transpose(trans_mat), _mol.hcore[active_idx, active_idx], trans_mat))
				h2e_active = ao2mo.kernel(mol_active, trans_mat)
				h2e_active = ao2mo.restore(1, h2e_active, len(active_idx))
				#
				return h1e, h2e, e_ref


		def prepare(self, _mol, _calc, _exp, _tup):
				""" generate input for correlated calculation """
				# generate orbital lists
				cas_idx = sorted(_exp.incl_idx + _tup.tolist())
				core_idx = sorted(list(set(range(_mol.nocc)) - set(cas_idx)))
				# extract core and cas integrals and calculate core energy
				if (len(core_idx) > 0):
					vhf_core = np.einsum('iipq->pq', _calc.h2e[core_idx][:,core_idx]) * 2
					vhf_core -= np.einsum('piiq->pq', _calc.h2e[:,core_idx][:,:,core_idx])
					h1e_cas = (_calc.h1e + vhf_core)[cas_idx][:,cas_idx]
				else:
					h1e_cas = _calc.h1e[cas_idx][:,cas_idx]
				h2e_cas = _calc.h2e[cas_idx][:,cas_idx][:,:,cas_idx][:,:,:,cas_idx]
				# set core energy
				if (len(core_idx) > 0):
					e_core = _calc.h1e[core_idx][:,core_idx].trace() * 2 + \
								vhf_core[core_idx][:,core_idx].trace() + \
								_mol.energy_nuc()
				else:
					e_core = _mol.energy_nuc()
				#
				return core_idx, cas_idx, h1e_cas, h2e_cas, e_core


		def calc(self, _mol, _calc, _exp):
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
				if (_calc.exp_model != 'FCI'):
					hf_cas = solver_cas.fake_hf(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
												_mol.nelectron - 2 * len(_exp.core_idx))[1]
					e_cas = solver_cas.kernel(hf_cas)[0]
				else:
					e_cas = solver_cas.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
												_mol.nelectron - 2 * len(_exp.core_idx))[0]
				# base calculation
				if (_calc.exp_base == 'HF'):
					e_corr = (e_cas + _exp.e_core) - _mol.e_hf
				else:
					solver_base = ModelSolver(_calc.exp_base)
					# base calculation
					hf_base = solver_base.fake_hf(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
												_mol.nelectron - 2 * len(_exp.core_idx))[1]
					e_base = solver_base.kernel(hf_base)[0]
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
				#
				return


		def fake_hf(self, _h1e, _h2e, _norb, _nelec):
				""" form active space hf """
				cas_mol = gto.M(verbose=0)
				cas_mol.nelectron = _nelec
				cas_hf = scf.RHF(cas_mol)
				cas_hf.conv_tol = 1.0e-12
				cas_hf._eri = ao2mo.restore(8, _h2e, _norb)
				cas_hf.get_hcore = lambda *args: _h1e
				cas_hf.get_ovlp = lambda *args: np.eye(_norb)
				cas_hf.kernel()
				#
				return cas_mol, cas_hf


		def kernel(self, _cas_hf, _dens=False):
				""" model kernel """
				if (self.model_type == 'MP2'):
					self.model = mp.MP2(_cas_hf)
					e_corr = self.model.kernel()[0]
				elif (self.model_type == 'CCSD'):
					self.model = cc.CCSD(_cas_hf)
					self.model.conv_tol = 1.0e-10
					self.model.diis_space = 12
					self.model.max_cycle = 100
					try:
						e_corr = self.model.kernel()[0]
					except Exception as err:
						for i in [2,4,6,8]:
							try:
								self.model.diis_start_cycle = i
								e_corr = self.model.kernel()[0]
							except Exception as err:
								try:
									raise RuntimeError
								except RuntimeError:
									sys.stderr.write('\nCASCCSD Error\n\n')
									raise
							else:
								break
					if (_dens):
						dm = self.model.make_rdm1()
					else:
						dm = None
				#
				return _cas_hf.e_tot + e_corr, dm


