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
	from pyscf import gto, scf, ao2mo, lo, mp, ci, cc, fci
except ImportError:
	sys.stderr.write('\nImportError : pyscf module not found\n\n')


class PySCFCls():
		""" pyscf class """
		def hf(self, _mol, _calc):
				""" determine dimensions """
				# perform hf calc
				hf = scf.RHF(_mol)
				hf.conv_tol = 1.0e-12
				hf.kernel()
				# determine dimensions
				norb = hf.mo_coeff.shape[1]
				nocc = int(hf.mo_occ.sum()) // 2
				nvirt = norb - nocc
				#
				return hf, norb, nocc, nvirt


		def int_trans(self, _mol, _calc, _exp):
				""" determine dimensions """
				# set frozen list
				if ((_calc.exp_type in ['occupied','virtual']) or (_calc.exp_virt == 'NO')):
					frozen = list(range(_mol.ncore))
				else:
					frozen = sorted(list(set(range(_mol.nocc)) - set(_exp.incl_idx)))
				# proceed or return
				if ((_calc.exp_type in ['occupied','virtual']) or \
					((_calc.exp_virt != 'DNO') and (_mol.trans_mat is None)) or \
					(_calc.exp_virt == 'DNO')):
					# zeroth-order energy
					if (_calc.exp_base == 'HF'):
						_mol.e_zero = 0.0
					elif (_calc.exp_base == 'MP2'):
						# calculate mp2 energy
						mp2 = mp.MP2(_mol.hf)
						mp2.frozen = frozen
						_mol.e_zero = mp2.kernel()[0]
						if ((_calc.exp_occ == 'NO') or (_calc.exp_virt in ['NO','DNO'])):
							dm = mp2.make_rdm1()
					elif (_calc.exp_base == 'CCSD'):
						# calculate ccsd energy
						ccsd = cc.CCSD(_mol.hf)
						ccsd.conv_tol = 1.0e-10
						ccsd.frozen = frozen
						_mol.e_zero = ccsd.kernel()[0]
						if ((_calc.exp_occ == 'NO') or (_calc.exp_virt in ['NO','DNO'])):
							dm = ccsd.make_rdm1()
					# init transformation matrix
					if (_mol.trans_mat is None): _mol.trans_mat = _mol.hf.mo_coeff
					# occ-occ block
					if ((_calc.exp_occ != 'HF') and (_exp.order == _exp.min_order)):
						if (_calc.exp_occ == 'NO'):
							occup, no = sp.linalg.eigh(dm[:(_mol.nocc-len(frozen)), :(_mol.nocc-len(frozen))])
							mo_coeff_occ = np.dot(_mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc], no[:, ::-1])
						elif (_calc.exp_occ == 'PM'):
							mo_coeff_occ = lo.PM(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
						elif (_calc.exp_occ == 'ER'):
							mo_coeff_occ = lo.ER(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
						elif (_calc.exp_occ == 'BOYS'):
							mo_coeff_occ = lo.Boys(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
						_mol.trans_mat[:, _mol.ncore:_mol.nocc] = mo_coeff_occ
					# virt-virt block
					if (_calc.exp_virt != 'HF'):
						occup, no = sp.linalg.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):])
						mo_coeff_virt = np.dot(_mol.hf.mo_coeff[:, _mol.nocc:], no[:, ::-1])
						_mol.trans_mat[:, _mol.nocc:] = mo_coeff_virt
					_mol.h1e = reduce(np.dot, (np.transpose(_mol.trans_mat), _mol.hf.get_hcore(), _mol.trans_mat))
					_mol.h2e = ao2mo.kernel(_mol, _mol.trans_mat)
					_mol.h2e = ao2mo.restore(1, _mol.h2e, _mol.norb)
				#
				return


		def prepare(self, _mol, _calc, _exp, _tup):
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
					e_corr = (e_cas + _exp.e_core) - _mol.hf.e_tot
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


