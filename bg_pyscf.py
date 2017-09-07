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
	from pyscf import gto, symm, scf, ao2mo, lo, mp, ci, cc, fci
except ImportError:
	sys.stderr.write('\nImportError : pyscf module not found\n\n')


class PySCFCls():
		""" pyscf class """
		def hf(self, _mol, _calc):
				""" determine dimensions """
				# perform hf calc
				hf = scf.RHF(_mol)
				hf.conv_tol = 1.0e-12
				hf.max_cycle = 100
				hf.irrep_nelec = _mol.irrep_nelec
				for i in list(range(0, 12, 2)):
					hf.diis_start_cycle = i
					try:
						hf.kernel()
					except sp.linalg.LinAlgError: pass
					if (hf.converged): break
				if (not hf.converged):
					try:
						raise RuntimeError('\nHF Error : no convergence\n\n')
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				# determine dimensions
				norb = hf.mo_coeff.shape[1]
				nocc = int(hf.mo_occ.sum()) // 2
				nvirt = norb - nocc
				# orbsym
				orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, hf.mo_coeff)
				#
				return hf, norb, nocc, nvirt, orbsym


		def int_trans(self, _mol, _calc, _exp):
				""" determine dimensions """
				# set frozen list
				if ((_calc.exp_type in ['occupied','virtual']) or (_calc.exp_virt == 'NO')):
					frozen = list(range(_mol.ncore))
				else:
					frozen = sorted(list(set(range(_mol.nocc)) - set(_exp.incl_idx)))
				# proceed or return
				if ((_calc.exp_type in ['occupied','virtual']) or \
					((_calc.exp_virt != 'DNO') and ((_mol.trans_mat_occ is None) and (_mol.trans_mat_virt is None))) or \
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
					elif (_calc.exp_base == 'CISD'):
						# calculate ccsd energy
						cisd = ci.CISD(_mol.hf)
						cisd.conv_tol = 1.0e-10
						cisd.max_cycle = 100
						cisd.max_space = 30
						cisd.frozen = frozen
						_mol.e_zero = cisd.kernel()[0]
						if ((_calc.exp_occ == 'NO') or (_calc.exp_virt in ['NO','DNO'])):
							dm = cisd.make_rdm1()
					elif (_calc.exp_base == 'CCSD'):
						# calculate ccsd energy
						ccsd = cc.CCSD(_mol.hf)
						ccsd.conv_tol = 1.0e-10
						ccsd.max_cycle = 100
						ccsd.diis_space = 10
						ccsd.frozen = frozen
						for i in list(range(0, 12, 2)):
							ccsd.diis_start_cycle = i
							try:
								_mol.e_zero = ccsd.kernel()[0]
							except sp.linalg.LinAlgError: pass
							if (ccsd.converged): break
						if (not ccsd.converged):
							try:
								raise RuntimeError('\nCCSD (int-trans) Error : no convergence\n\n')
							except Exception as err:
								sys.stderr.write(str(err))
								raise
						if ((_calc.exp_occ == 'NO') or (_calc.exp_virt in ['NO','DNO'])):
							dm = ccsd.make_rdm1()
					# sum up total zeroth-order energy
					_mol.e_zero_tot = _mol.hf.e_tot + _mol.e_zero
					# set transformation matrix
					if (_mol.trans_mat_occ is None):
						# init transformation matrix
						_mol.trans_mat_occ = _mol.hf.mo_coeff[:, :_mol.nocc]
						# occ-occ block (local, intrinsic AOs, or symmetry-adapted AOs)
						if (_calc.exp_occ != 'HF'):
							if (_calc.exp_occ == 'NO'):
								occup, no = symm.eigh(dm[:(_mol.nocc-len(frozen)), :(_mol.nocc-len(frozen))], \
														_mol.orbsym[sorted(list(set(range(_mol.nocc)) - set(frozen)))])
								mo_coeff_occ = np.dot(_mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc], no[:, ::-1])
							elif (_calc.exp_occ == 'PM'):
								mo_coeff_occ = lo.PM(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
							elif (_calc.exp_occ == 'ER'):
								mo_coeff_occ = lo.ER(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
							elif (_calc.exp_occ == 'BOYS'):
								mo_coeff_occ = lo.Boys(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
							_mol.trans_mat_occ[:, _mol.ncore:] = mo_coeff_occ
					if ((_mol.trans_mat_virt is None) or (_calc.exp_virt == 'DNO')):
						# init transformation matrix
						_mol.trans_mat_virt = _mol.hf.mo_coeff[:, _mol.nocc:]
						# virt-virt block (symmetry-adapted NOs)
						if (_calc.exp_virt != 'HF'):
							if (_calc.exp_virt in ['NO','DNO']):
								occup, no = symm.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):], \
														_mol.orbsym[_mol.nocc:])
								mo_coeff_virt = np.dot(_mol.hf.mo_coeff[:, _mol.nocc:], no[:, ::-1])
							elif (_calc.exp_virt == 'PM'):
								mo_coeff_virt = lo.PM(_mol, _mol.hf.mo_coeff[:, _mol.nocc:]).kernel()
							elif (_calc.exp_virt == 'ER'):
								mo_coeff_virt = lo.ER(_mol, _mol.hf.mo_coeff[:, _mol.nocc:]).kernel()
							elif (_calc.exp_virt == 'BOYS'):
								mo_coeff_virt = lo.Boys(_mol, _mol.hf.mo_coeff[:, _mol.nocc:]).kernel()
							_mol.trans_mat_virt = mo_coeff_virt
					# concatenate transformation matrices
					_mol.trans_mat = np.concatenate((_mol.trans_mat_occ, _mol.trans_mat_virt), axis=1)
				# perform integral transformation
				_mol.h1e = reduce(np.dot, (np.transpose(_mol.trans_mat), _mol.hf.get_hcore(), _mol.trans_mat))
				_mol.h2e = ao2mo.kernel(_mol, _mol.trans_mat)
				_mol.h2e = ao2mo.restore(1, _mol.h2e, _mol.norb)
				# overwrite orbsym
				if ((_calc.exp_occ == 'NO') or (_calc.exp_virt in ['NO','DNO'])):
					_mol.orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, _mol.trans_mat)
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
				if ((_calc.exp_base in ['CISD','CCSD']) and (_exp.order == 1)):
					e_corr = 0.0
				else:
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
					solver_cas.max_cycle = 100
					solver_cas.max_space = 30
					# initial guess
					na = fci.cistring.num_strings(len(_exp.cas_idx), (_mol.nelectron - 2 * len(_exp.core_idx)) // 2)
					hf_as_civec = np.zeros((na, na))
					hf_as_civec[0, 0] = 1
					# cas calculation
					if (_calc.exp_model != 'FCI'):
						hf_cas = solver_cas.fake_hf(_mol, _exp.h1e_cas, _exp.h2e_cas, _exp.core_idx, _exp.cas_idx)[1]
						e_cas = solver_cas.kernel(hf_cas, _exp.core_idx, _exp.cas_idx)
					else:
						e_cas = solver_cas.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
													_mol.nelectron - 2 * len(_exp.core_idx), ci0=hf_as_civec, \
													orbsym=_mol.orbsym)[0]
					# base calculation
					if (_calc.exp_base == 'HF'):
						e_corr = (e_cas + _exp.e_core) - _mol.hf.e_tot
					else:
						# base calculation
						solver_base = ModelSolver(_calc.exp_base)
						hf_base = solver_base.fake_hf(_mol, _exp.h1e_cas, _exp.h2e_cas, _exp.core_idx, _exp.cas_idx)[1]
						e_base = solver_base.kernel(hf_base, _exp.core_idx, _exp.cas_idx, _e_cas=e_cas)
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


		def fake_hf(self, _mol, _h1e, _h2e, _core_idx, _cas_idx):
				""" form active space hf """
				cas_mol = gto.M(verbose=0)
				cas_mol.nelectron = _mol.nelectron - 2 * len(_core_idx)
				cas_hf = scf.RHF(cas_mol)
				cas_hf.conv_tol = 1.0e-12
				cas_hf.max_cycle = 100
				cas_hf._eri = ao2mo.restore(8, _h2e, len(_cas_idx))
				cas_hf.get_hcore = lambda *args: _h1e
				cas_hf.get_ovlp = lambda *args: np.eye(len(_cas_idx))
				dm0 = np.diag(_mol.hf.mo_occ[_cas_idx])
				for i in list(range(0, 12, 2)):
					cas_hf.diis_start_cycle = i
					try:
						cas_hf.kernel(dm0)
					except sp.linalg.LinAlgError: pass
					if (cas_hf.converged): break
				if (not cas_hf.converged):
					try:
						raise RuntimeError(('\nCAS-HF Error : no convergence\n'
											'core_idx = {0:} , cas_idx = {1:}\n\n').\
											format(_core_idx,_cas_idx))
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				#
				return cas_mol, cas_hf


		def kernel(self, _cas_hf, _core_idx, _cas_idx, _e_cas=None):
				""" model kernel """
				if (self.model_type == 'MP2'):
					self.model = mp.MP2(_cas_hf)
					e_corr = self.model.kernel()[0]
				elif (self.model_type == 'CISD'):
					self.model = ci.CISD(_cas_hf)
					self.model.conv_tol = 1.0e-10
					self.model.max_cycle = 100
					self.model.max_space = 30
					for i in range(5,-1,-1):
						self.model.level_shift = 1.0 / 10.0 ** (i)
						try:
							e_corr = self.model.kernel()[0]
						except sp.linalg.LinAlgError: pass
						if (self.model.converged): break
					if (not self.model.converged):
						try:
							raise RuntimeError(('\nCAS-CISD Error : no convergence\n'
												'core_idx = {0:} , cas_idx = {1:}\n\n').\
												format(_core_idx,_cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					if (_e_cas is not None):
						if ((_cas_hf.e_tot + e_corr) < _e_cas):
							try:
								raise RuntimeError(('\nCAS-CISD Error : wrong convergence\n'
													'core_idx = {0:} , cas_idx = {1:}\n\n').\
													format(_core_idx,_cas_idx))
							except Exception as err:
								sys.stderr.write(str(err))
								raise
				elif (self.model_type == 'CCSD'):
					self.model = cc.CCSD(_cas_hf)
					self.model.conv_tol = 1.0e-10
					self.model.diis_space = 10
					self.model.max_cycle = 100
					for i in list(range(0, 12, 2)):
						self.model.diis_start_cycle = i
						try:
							e_corr = self.model.kernel()[0]
						except sp.linalg.LinAlgError: pass
						if (self.model.converged): break
					if (not self.model.converged):
						try:
							raise RuntimeError(('\nCAS-CCSD Error : no convergence\n'
												'core_idx = {0:} , cas_idx = {1:}\n\n').\
												format(_core_idx,_cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
				#
				return _cas_hf.e_tot + e_corr


