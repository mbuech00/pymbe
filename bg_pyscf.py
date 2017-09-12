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
				if (_mol.hf_dens is None):
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
					_mol.hf_dens = hf.make_rdm1()
				else:
					hf.kernel(_mol.hf_dens)
				# determine dimensions
				_mol.norb = hf.mo_coeff.shape[1]
				_mol.nocc = int(hf.mo_occ.sum()) // 2
				_mol.nvirt = _mol.norb - _mol.nocc
				# mo_occ
				_mol.mo_occ = hf.mo_occ
				# orbsym
				_mol.orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, hf.mo_coeff)
				#
				return hf


		def trans_main(self, _mol, _calc):
				""" determine main transformation matrices """
				# set frozen list
				frozen = list(range(_mol.ncore))
				# zeroth-order energy
				if (_calc.exp_base == 'HF'):
					_mol.e_zero = 0.0
				elif (_calc.exp_base == 'MP2'):
					# calculate mp2 energy
					mp2 = mp.MP2(_mol.hf)
					mp2.frozen = frozen
					_mol.e_zero = mp2.kernel()[0]
					if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')): dm = mp2.make_rdm1()
				elif (_calc.exp_base == 'CISD'):
					# calculate ccsd energy
					cisd = ci.CISD(_mol.hf)
					cisd.conv_tol = 1.0e-10
					cisd.max_cycle = 100
					cisd.max_space = 30
					cisd.frozen = frozen
					for i in range(5,-1,-1):
						cisd.level_shift = 1.0 / 10.0 ** (i)
						try:
							_mol.e_zero = cisd.kernel()[0]
						except sp.linalg.LinAlgError: pass
						if (cisd.converged): break
					if (not cisd.converged):
						try:
							raise RuntimeError('\nCISD (main int-trans) Error : no convergence\n\n')
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')): dm = cisd.make_rdm1()
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
							raise RuntimeError('\nCCSD (main int-trans) Error : no convergence\n\n')
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')): dm = ccsd.make_rdm1()
				# init transformation matrix
				_mol.trans_mat = np.copy(_mol.hf.mo_coeff)
				# occ-occ block (local or symmetry-adapted NOs)
				if (_calc.exp_occ != 'HF'):
					if (_calc.exp_occ == 'NO'):
						occup, no = symm.eigh(dm[:(_mol.nocc-_mol.ncore), :(_mol.nocc-_mol.ncore)], \
												_mol.orbsym[_mol.ncore:_mol.nocc])
						_mol.trans_mat[:, _mol.ncore:_mol.nocc] = np.dot(_mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc], no[:, ::-1])
					elif (_calc.exp_occ == 'PM'):
						_mol.trans_mat[:, _mol.ncore:_mol.nocc] = lo.PM(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
					elif (_calc.exp_occ == 'ER'):
						_mol.trans_mat[:, _mol.ncore:_mol.nocc] = lo.ER(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
					elif (_calc.exp_occ == 'BOYS'):
						_mol.trans_mat[:, _mol.ncore:_mol.nocc] = lo.Boys(_mol, _mol.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
				# virt-virt block (local or symmetry-adapted NOs)
				if (_calc.exp_virt != 'HF'):
					if (_calc.exp_virt == 'NO'):
						occup, no = symm.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):], \
												_mol.orbsym[_mol.nocc:])
						_mol.trans_mat[:, _mol.nocc:] = np.dot(_mol.hf.mo_coeff[:, _mol.nocc:], no[:, ::-1])
					elif (_calc.exp_virt == 'PM'):
						_mol.trans_mat[:, _mol.nocc:] = lo.PM(_mol, _mol.hf.mo_coeff[:, _mol.nocc:]).kernel()
					elif (_calc.exp_virt == 'ER'):
						_mol.trans_mat[:, _mol.nocc:] = lo.ER(_mol, _mol.hf.mo_coeff[:, _mol.nocc:]).kernel()
					elif (_calc.exp_virt == 'BOYS'):
						_mol.trans_mat[:, _mol.nocc:] = lo.Boys(_mol, _mol.hf.mo_coeff[:, _mol.nocc:]).kernel()
				#
				return


		def trans_dno(self, _mol, _calc, _exp):
				""" determine dno transformation matrices """
				# set frozen list
				frozen = sorted(list(set(range(_mol.nocc)) - set(_exp.incl_idx))) 
				# zeroth-order energy
				if (_calc.exp_base == 'MP2'):
					# calculate mp2 energy
					mp2 = mp.MP2(_mol.hf)
					mp2.frozen = frozen
					mp2.kernel()
					dm = mp2.make_rdm1()
				elif (_calc.exp_base == 'CISD'):
					# calculate ccsd energy
					cisd = ci.CISD(_mol.hf)
					cisd.conv_tol = 1.0e-10
					cisd.max_cycle = 100
					cisd.max_space = 30
					cisd.frozen = frozen
					for i in range(5,-1,-1):
						cisd.level_shift = 1.0 / 10.0 ** (i)
						try:
							cisd.kernel()
						except sp.linalg.LinAlgError: pass
						if (cisd.converged): break
					if (not cisd.converged):
						try:
							raise RuntimeError('\nCISD (dno int-trans) Error : no convergence\n\n')
						except Exception as err:
							sys.stderr.write(str(err))
							raise
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
							ccsd.kernel()
						except sp.linalg.LinAlgError: pass
						if (ccsd.converged): break
					if (not ccsd.converged):
						try:
							raise RuntimeError('\nCCSD (dno int-trans) Error : no convergence\n\n')
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					dm = ccsd.make_rdm1()
				# generate dnos
				occup, no = symm.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):], \
										_mol.orbsym[_mol.nocc:])
				_mol.trans_mat[:, _mol.nocc:] = np.dot(_mol.hf.mo_coeff[:, _mol.nocc:], no[:, ::-1])
				#
				return


		def int_trans(self, _mol, _calc):
				""" integral transformation """
				# perform integral transformation
				_mol.h1e = reduce(np.dot, (np.transpose(_mol.trans_mat), _mol.hf.get_hcore(), _mol.trans_mat))
				_mol.h2e = ao2mo.kernel(_mol, _mol.trans_mat)
				_mol.h2e = ao2mo.restore(1, _mol.h2e, _mol.norb)
				# overwrite orbsym
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
					hf_cas = solver_cas.hf(_mol, _exp.h1e_cas, _exp.h2e_cas, _exp.core_idx, _exp.cas_idx)
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
					hf_base = solver_base.hf(_mol, _exp.h1e_cas, _exp.h2e_cas, _exp.core_idx, _exp.cas_idx)
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


		def hf(self, _mol, _h1e, _h2e, _core_idx, _cas_idx):
				""" form active space hf """
				cas_mol = gto.M(verbose=0)
				cas_mol.nelectron = _mol.nelectron - 2 * len(_core_idx)
				cas_hf = scf.RHF(cas_mol)
				cas_hf.conv_tol = 1.0e-12
				cas_hf.max_cycle = 100
				cas_hf._eri = ao2mo.restore(8, _h2e, len(_cas_idx))
				cas_hf.get_hcore = lambda *args: _h1e
				cas_hf.get_ovlp = lambda *args: np.eye(len(_cas_idx))
				dm0 = np.diag(_mol.mo_occ[_cas_idx])
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
				return cas_hf


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


