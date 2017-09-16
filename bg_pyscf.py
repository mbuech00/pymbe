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
				# restart calc?
				if (_calc.hf_dens is None):
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
					# calculate converged hf dens
					_calc.hf_dens = hf.make_rdm1()
					# determine dimensions
					_mol.norb = hf.mo_coeff.shape[1]
					_mol.nocc = int(hf.mo_occ.sum()) // 2
					_mol.nvirt = _mol.norb - _mol.nocc
				else:
					# restart from converged density
					hf.kernel(_calc.hf_dens)
					# determine dimensions
					_mol.norb = hf.mo_coeff.shape[1]
					_mol.nocc = int(hf.mo_occ.sum()) // 2
					_mol.nvirt = _mol.norb - _mol.nocc
					# overwrite occupied MOs
					if (_calc.exp_occ != 'HF'):
						hf.mo_coeff[:, _mol.ncore:_mol.nocc] = _calc.trans_mat[:, _mol.ncore:_mol.nocc]
				# save mo_occ
				_calc.mo_occ = hf.mo_occ
				# save orbsym
				_calc.orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, hf.mo_coeff)
				#
				return hf


		def trans_main(self, _mol, _calc):
				""" determine main transformation matrices """
				# set frozen list
				frozen = list(range(_mol.ncore))
				# zeroth-order energy
				if (_calc.exp_base == 'HF'):
					_calc.e_zero = 0.0
				elif (_calc.exp_base == 'MP2'):
					# calculate mp2 energy
					mp2 = mp.MP2(_calc.hf)
					mp2.frozen = frozen
					_calc.e_zero = mp2.kernel()[0]
					if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')): dm = mp2.make_rdm1()
				elif (_calc.exp_base == 'CISD'):
					# calculate ccsd energy
					cisd = ci.CISD(_calc.hf)
					cisd.conv_tol = 1.0e-10
					cisd.max_cycle = 100
					cisd.max_space = 30
					cisd.frozen = frozen
					for i in range(5,-1,-1):
						cisd.level_shift = 1.0 / 10.0 ** (i)
						try:
							_calc.e_zero = cisd.kernel()[0]
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
					ccsd = cc.CCSD(_calc.hf)
					ccsd.conv_tol = 1.0e-10
					ccsd.max_cycle = 100
					ccsd.diis_space = 10
					ccsd.frozen = frozen
					for i in list(range(0, 12, 2)):
						ccsd.diis_start_cycle = i
						try:
							_calc.e_zero = ccsd.kernel()[0]
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
				_calc.trans_mat = np.copy(_calc.hf.mo_coeff)
				# occ-occ block (local or symmetry-adapted NOs)
				if (_calc.exp_occ != 'HF'):
					if (_calc.exp_occ == 'NO'):
						occup, no = symm.eigh(dm[:(_mol.nocc-_mol.ncore), :(_mol.nocc-_mol.ncore)], \
												_calc.orbsym[_mol.ncore:_mol.nocc])
						_calc.trans_mat[:, _mol.ncore:_mol.nocc] = np.dot(_calc.hf.mo_coeff[:, _mol.ncore:_mol.nocc], no[:, ::-1])
					elif (_calc.exp_occ == 'PM'):
						_calc.trans_mat[:, _mol.ncore:_mol.nocc] = lo.PM(_mol, _calc.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
					elif (_calc.exp_occ == 'ER'):
						_calc.trans_mat[:, _mol.ncore:_mol.nocc] = lo.ER(_mol, _calc.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
					elif (_calc.exp_occ == 'BOYS'):
						_calc.trans_mat[:, _mol.ncore:_mol.nocc] = lo.Boys(_mol, _calc.hf.mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
				# virt-virt block (local or symmetry-adapted NOs)
				if (_calc.exp_virt != 'HF'):
					if (_calc.exp_virt == 'NO'):
						occup, no = symm.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):], \
												_calc.orbsym[_mol.nocc:])
						_calc.trans_mat[:, _mol.nocc:] = np.dot(_calc.hf.mo_coeff[:, _mol.nocc:], no[:, ::-1])
					elif (_calc.exp_virt == 'PM'):
						_calc.trans_mat[:, _mol.nocc:] = lo.PM(_mol, _calc.hf.mo_coeff[:, _mol.nocc:]).kernel()
					elif (_calc.exp_virt == 'ER'):
						_calc.trans_mat[:, _mol.nocc:] = lo.ER(_mol, _calc.hf.mo_coeff[:, _mol.nocc:]).kernel()
					elif (_calc.exp_virt == 'BOYS'):
						_calc.trans_mat[:, _mol.nocc:] = lo.Boys(_mol, _calc.hf.mo_coeff[:, _mol.nocc:]).kernel()
				#
				return


		def trans_dno(self, _mol, _calc, _exp):
				""" determine dno transformation matrices """
				# set frozen list
				frozen = sorted(list(set(range(_mol.nocc)) - set(_exp.incl_idx))) 
				# zeroth-order energy
				if (_calc.exp_base == 'MP2'):
					# calculate mp2 energy
					mp2 = mp.MP2(_calc.hf)
					mp2.frozen = frozen
					mp2.kernel()
					dm = mp2.make_rdm1()
				elif (_calc.exp_base == 'CISD'):
					# calculate ccsd energy
					cisd = ci.CISD(_calc.hf)
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
					ccsd = cc.CCSD(_calc.hf)
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
										_calc.orbsym[_mol.nocc:])
				_calc.trans_mat[:, _mol.nocc:] = np.dot(_calc.hf.mo_coeff[:, _mol.nocc:], no[:, ::-1])
				#
				return


		def int_trans(self, _mol, _calc):
				""" integral transformation """
				# perform integral transformation
				_calc.h1e = reduce(np.dot, (np.transpose(_calc.trans_mat), _calc.hf.get_hcore(), _calc.trans_mat))
				_calc.h2e = ao2mo.kernel(_mol, _calc.trans_mat)
				_calc.h2e = ao2mo.restore(1, _calc.h2e, _mol.norb)
				# overwrite orbsym
				_calc.orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, _calc.trans_mat)
				#
				return


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
				solver_cas.max_cycle = 100
				solver_cas.max_space = 30
				# initial guess
				na = fci.cistring.num_strings(len(_exp.cas_idx), (_mol.nelectron - 2 * len(_exp.core_idx)) // 2)
				hf_as_civec = np.zeros((na, na))
				hf_as_civec[0, 0] = 1
				# cas calculation
				if (_calc.exp_model != 'FCI'):
					hf_cas = solver_cas.hf(_mol, _calc, _exp.h1e_cas, _exp.h2e_cas, _exp.core_idx, _exp.cas_idx)
					e_cas = solver_cas.kernel(hf_cas, _exp.core_idx, _exp.cas_idx)
				else:
					try:
						e_cas, c_cas = solver_cas.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
													_mol.nelectron - 2 * len(_exp.core_idx), ci0=hf_as_civec, \
													orbsym=_calc.orbsym)
					except Exception as err:
						try:
							raise RuntimeError(('\nCAS-CI Error :\n'
												'core_idx = {0:} , cas_idx = {1:}\n'
												'PySCF Error: {2:}\n\n').\
												format(_exp.core_idx, _exp.cas_idx, err))
						except Exception as err_2:
							sys.stderr.write(str(err_2))
							raise
					# calculate spin
					cas_s, cas_mult = fci.spin_op.spin_square(c_cas, len(_exp.cas_idx), _mol.nelectron - 2 * len(_exp.core_idx))
					# check for correct spin
					if (int(round(cas_s)) != _mol.spin):
						try:
							raise RuntimeError(('\nCAS-CI Error : wrong spin\n'
												'2*S + 1 = {0:.3f}\n'
												'core_idx = {1:} , cas_idx = {2:}\n\n').\
												format(cas_mult, _exp.core_idx, _exp.cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
				# base calculation
				if (_calc.exp_base == 'HF'):
					e_corr = (e_cas + _exp.e_core) - _calc.hf.e_tot
				else:
					# base calculation
					solver_base = ModelSolver(_calc.exp_base)
					hf_base = solver_base.hf(_mol, _calc, _exp.h1e_cas, _exp.h2e_cas, _exp.core_idx, _exp.cas_idx)
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


		def hf(self, _mol, _calc, _h1e, _h2e, _core_idx, _cas_idx):
				""" form active space hf """
				cas_mol = gto.M(verbose=0)
				cas_mol.nelectron = _mol.nelectron - 2 * len(_core_idx)
				cas_hf = scf.RHF(cas_mol)
				cas_hf.spin = _mol.spin
				cas_hf.nelectron = cas_mol.nelectron
				cas_hf.conv_tol = 1.0e-12
				cas_hf.max_cycle = 100
				cas_hf._eri = ao2mo.restore(8, _h2e, len(_cas_idx))
				cas_hf.get_hcore = lambda *args: _h1e
				cas_hf.get_ovlp = lambda *args: np.eye(len(_cas_idx))
				dm0 = np.diag(_calc.mo_occ[_cas_idx])
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
											format(_core_idx, _cas_idx))
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
							e_corr, c_cascisd = self.model.kernel()
						except sp.linalg.LinAlgError: pass
						if (self.model.converged): break
					# check for convergence
					if (not self.model.converged):
						try:
							raise RuntimeError(('\nCAS-CISD Error : no convergence\n'
												'core_idx = {0:} , cas_idx = {1:}\n\n').\
												format(_core_idx, _cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					# check for energy
					if (_e_cas is not None):
						if ((_cas_hf.e_tot + e_corr) < _e_cas):
							if (np.abs(_e_cas - (_cas_hf.e_tot + e_corr)) > 1.0e-10):
								try:
									raise RuntimeError(('\nCAS-CISD Error : wrong convergence\n'
														'CAS-CISD = {0:.6f} , CAS-CI = {1:.6f}\n'
														'core_idx = {2:} , cas_idx = {3:}\n\n').\
														format(_cas_hf.e_tot + e_corr, _e_cas, _core_idx, _cas_idx))
								except Exception as err:
									sys.stderr.write(str(err))
									raise
					# check for spin
					c_cascisd_fci = ci.cisd.to_fci(c_cascisd, len(_cas_idx), _cas_hf.nelectron)
					cascisd_s, cascisd_mult = fci.spin_op.spin_square(c_cascisd_fci, len(_cas_idx), _cas_hf.nelectron)
					if (int(round(cascisd_s)) != _cas_hf.spin):
						try:
							raise RuntimeError(('\nCAS-CISD Error : wrong spin\n'
												'2*S + 1 = {0:.3f}\n'
												'core_idx = {1:} , cas_idx = {2:}\n\n').\
												format(cascisd_mult, _core_idx, _cas_idx))
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
												format(_core_idx, _cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
				#
				return _cas_hf.e_tot + e_corr


