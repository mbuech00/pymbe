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
	from pyscf import gto, scf, ao2mo, lo, ci, cc, mcscf, fci
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
				if (_calc.hf_mo_coeff is None):
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
					_mol.norb = hf.mo_coeff.shape[1]
					_mol.nocc = int(hf.mo_occ.sum()) // 2
					_mol.nvirt = _mol.norb - _mol.nocc
				else:
					# construct density
					hf_dens = scf.hf.make_rdm1(_calc.hf_mo_coeff, _calc.hf_mo_occ)
					# restart from converged density
					hf.kernel(hf_dens)
					# determine dimensions
					_mol.norb = hf.mo_coeff.shape[1]
					_mol.nocc = int(hf.mo_occ.sum()) // 2
					_mol.nvirt = _mol.norb - _mol.nocc
					# overwrite occupied MOs
					if (_calc.exp_occ != 'CAN'):
						hf.mo_coeff[:, _mol.ncore:_mol.nocc] = _calc.trans_mat[:, _mol.ncore:_mol.nocc]
				# store mo_coeff, mo_occ, and e_tot
				_calc.hf_mo_coeff = hf.mo_coeff
				_calc.hf_mo_occ = hf.mo_occ
				_calc.hf_e_tot = hf.e_tot
				#
				return hf


		def ref(self, _mol, _calc):
				""" determine dimensions """
				# hf reference model
				if (_calc.exp_ref['METHOD'] == 'HF'):
					# store energy
					ref_e_tot = _calc.hf_e_tot
				# casci reference model
				elif (_calc.exp_ref['METHOD'] == 'CASCI'):
					# import avas
					from pyscf.mcscf import avas
					# select active space
					_calc.no_act, _calc.ne_act, mo = avas.avas(_calc.hf, _calc.exp_ref['AO_LABELS'], canonicalize=True)
					# store information
					_calc.no_o_act = sorted([i-1 for i in range(_mol.nocc, _mol.nocc - (_calc.ne_act // 2), -1)])
					_calc.no_v_act = [i for i in range(_mol.nocc, _mol.nocc + (_calc.no_act - len(_calc.no_o_act)))]
					# perform reference calc
					casci = mcscf.CASCI(_calc.hf, _calc.no_act, _calc.ne_act)
					casci.conv_tol = 1.0e-12
					casci.max_cycle_macro = 100
					ref_e_tot = casci.kernel(mo)[0]
				#
				return ref_e_tot


		def trans_main(self, _mol, _calc):
				""" determine main transformation matrices """
				# set frozen list
				frozen = list(range(_mol.ncore))
				# zeroth-order energy
				if (_calc.exp_base['METHOD'] == _calc.exp_ref['METHOD']):
					_calc.e_zero = 0.0
				elif (_calc.exp_base['METHOD'] == 'CISD'):
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
				elif (_calc.exp_base['METHOD'] in ['CCSD','CCSD(T)']):
					# calculate ccsd energy
					ccsd = cc.CCSD(_calc.hf)
					ccsd.conv_tol = 1.0e-10
					ccsd.max_cycle = 100
					ccsd.diis_space = 10
					ccsd.frozen = frozen
					eris = ccsd.ao2mo()
					for i in list(range(0, 12, 2)):
						ccsd.diis_start_cycle = i
						try:
							ccsd.kernel(eris=eris)
						except sp.linalg.LinAlgError: pass
						if (ccsd.converged):
							_calc.e_zero = ccsd.e_corr
							break
					if (not ccsd.converged):
						try:
							raise RuntimeError('\nCCSD (main int-trans) Error : no convergence\n\n')
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')): dm = ccsd.make_rdm1()
				# init transformation matrix
				_calc.trans_mat = np.copy(_calc.hf_mo_coeff)
				# occ-occ block (local or NOs)
				if (_calc.exp_occ != 'CAN'):
					if (_calc.exp_occ == 'NO'):
						occup, no = sp.linalg.eigh(dm[:(_mol.nocc-_mol.ncore), :(_mol.nocc-_mol.ncore)])
						_calc.trans_mat[:, _mol.ncore:_mol.nocc] = np.dot(_calc.hf_mo_coeff[:, _mol.ncore:_mol.nocc], no[:, ::-1])
					elif (_calc.exp_occ == 'PM'):
						_calc.trans_mat[:, _mol.ncore:_mol.nocc] = lo.PM(_mol, _calc.hf_mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
					elif (_calc.exp_occ == 'FB'):
						_calc.trans_mat[:, _mol.ncore:_mol.nocc] = lo.Boys(_mol, _calc.hf_mo_coeff[:, _mol.ncore:_mol.nocc]).kernel()
					elif (_calc.exp_occ in ['IBO-1','IBO-2']):
						iao = lo.iao.iao(_mol, _calc.hf_mo_coeff[:, _mol.ncore:_mol.nocc])
						if (_calc.exp_occ == 'IBO-1'):
							iao = lo.vec_lowdin(iao, _calc.hf.get_ovlp())
							_calc.trans_mat[:, _mol.ncore:_mol.nocc] = lo.ibo.ibo(_mol, _calc.hf_mo_coeff[:, _mol.ncore:_mol.nocc], iao)
						elif (_calc.exp_occ == 'IBO-2'):
							_calc.trans_mat[:, _mol.ncore:_mol.nocc] = lo.ibo.PM(_mol, _calc.hf_mo_coeff[:, _mol.ncore:_mol.nocc], iao).kernel()
				# virt-virt block (local or NOs)
				if (_calc.exp_virt != 'CAN'):
					if (_calc.exp_virt == 'NO'):
						occup, no = sp.linalg.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):])
						_calc.trans_mat[:, _mol.nocc:] = np.dot(_calc.hf_mo_coeff[:, _mol.nocc:], no[:, ::-1])
					elif (_calc.exp_virt == 'PM'):
						_calc.trans_mat[:, _mol.nocc:] = lo.PM(_mol, _calc.hf_mo_coeff[:, _mol.nocc:]).kernel()
					elif (_calc.exp_virt == 'FB'):
						_calc.trans_mat[:, _mol.nocc:] = lo.Boys(_mol, _calc.hf_mo_coeff[:, _mol.nocc:]).kernel()
				# add (t) correction
				if (_calc.exp_base['METHOD'] == 'CCSD(T)'):
					if ((_calc.exp_occ == 'CAN') and (_calc.exp_virt == 'CAN')):
						_calc.e_zero += ccsd.ccsd_t(eris=eris, t1=ccsd.t1, t2=ccsd.t2)
					else:
						h1e = reduce(np.dot, (np.transpose(_calc.trans_mat), _calc.hf.get_hcore(), _calc.trans_mat))
						h2e = ao2mo.kernel(_mol, _calc.trans_mat)
						mol = gto.M(verbose=0)
						hf = scf.RHF(mol)
						hf._eri = h2e
						hf.get_hcore = lambda *args: h1e
						ccsd_2 = cc.ccsd.CCSD(hf, mo_coeff=np.eye(_mol.norb), mo_occ=_calc.hf_mo_occ)
						ccsd_2.conv_tol = 1.0e-10
						ccsd_2.max_cycle = 100
						ccsd_2.diis_space = 10
						ccsd_2.frozen = frozen
						ccsd_2.diis_start_cycle = ccsd.diis_start_cycle
						eris = ccsd_2.ao2mo()
						ccsd_2.kernel(eris=eris)
						_calc.e_zero += ccsd_2.ccsd_t(eris=eris, t1=ccsd_2.t1, t2=ccsd_2.t2)
				#
				return


		def trans_dno(self, _mol, _calc, _exp):
				""" determine dno transformation matrices """
				# set frozen list
				frozen = sorted(list(set(range(_mol.nocc)) - set(_exp.incl_idx))) 
				# zeroth-order energy
				if (_calc.exp_base['METHOD'] == 'CISD'):
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
				elif (_calc.exp_base['METHOD'] == 'CCSD'):
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
				occup, no = sp.linalg.eigh(dm[(_mol.nocc-len(frozen)):, (_mol.nocc-len(frozen)):])
				_calc.trans_mat[:, _mol.nocc:] = np.dot(_calc.hf_mo_coeff[:, _mol.nocc:], no[:, ::-1])
				#
				return


		def int_trans(self, _mol, _calc):
				""" integral transformation """
				# perform integral transformation
				_calc.h1e = reduce(np.dot, (np.transpose(_calc.trans_mat), _calc.hf.get_hcore(), _calc.trans_mat))
				_calc.h2e = ao2mo.kernel(_mol, _calc.trans_mat)
				_calc.h2e = ao2mo.restore(1, _calc.h2e, _mol.norb)
				#
				return


		def prepare(self, _mol, _calc, _exp, _tup):
				""" generate input for correlated calculation """
				# generate orbital lists
				cas_idx = sorted(_exp.incl_idx + _tup.tolist())
				core_idx = sorted(list(set(range(_mol.nocc)) - set(cas_idx)))
				# extract core and one-electron cas integrals and calculate core energy
				if (len(core_idx) > 0):
					if ((_calc.exp_type == 'occupied') or (_exp.e_core is None)):
						_calc.vhf_core = np.einsum('iipq->pq', _calc.h2e[core_idx][:,core_idx]) * 2
						_calc.vhf_core -= np.einsum('piiq->pq', _calc.h2e[:,core_idx][:,:,core_idx])
						_exp.e_core = _calc.h1e[core_idx][:,core_idx].trace() * 2 + \
										_calc.vhf_core[core_idx][:,core_idx].trace() + \
										_mol.energy_nuc()
					h1e_cas = (_calc.h1e + _calc.vhf_core)[cas_idx][:,cas_idx]
				else:
					h1e_cas = _calc.h1e[cas_idx][:,cas_idx]
					_exp.e_core = _mol.energy_nuc()
				# extract two-electron cas integrals
				h2e_cas = _calc.h2e[cas_idx][:,cas_idx][:,:,cas_idx][:,:,:,cas_idx]
				#
				return core_idx, cas_idx, h1e_cas, h2e_cas


		def calc(self, _mol, _calc, _exp):
				""" correlated cas calculation """
				# init solver
				if (_calc.exp_model['METHOD'] != 'FCI'):
					solver_cas = ModelSolver(_calc.exp_model['METHOD'])
				else:
					if (_mol.spin == 0):
						solver_cas = fci.direct_spin0.FCI()
					else:
						solver_cas = fci.direct_spin1.FCI()
				# settings
				solver_cas.conv_tol = 1.0e-10
				solver_cas.max_cycle = 100
				solver_cas.max_space = 10
				solver_cas.davidson_only = True
				# initial guess
				na = fci.cistring.num_strings(len(_exp.cas_idx), (_mol.nelectron - 2 * len(_exp.core_idx)) // 2)
				hf_as_civec = np.zeros((na, na))
				hf_as_civec[0, 0] = 1
				# cas calculation
				if (_calc.exp_model['METHOD'] != 'FCI'):
					hf_cas = solver_cas.hf(_mol, _calc, _exp.h1e_cas, _exp.h2e_cas, _exp.core_idx, _exp.cas_idx)
					e_cas = solver_cas.kernel(hf_cas, _exp.core_idx, _exp.cas_idx)
				else:
					try:
						e_cas, c_cas = solver_cas.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
													_mol.nelectron - 2 * len(_exp.core_idx), ci0=hf_as_civec)
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
				if (_calc.exp_ref['METHOD'] == _calc.exp_base['METHOD']):
					e_corr = (e_cas + _exp.e_core) - _calc.hf_e_tot
				else:
					# base calculation
					solver_base = ModelSolver(_calc.exp_base)
					hf_base = solver_base.hf(_mol, _calc, _exp.h1e_cas, _exp.h2e_cas, _exp.core_idx, _exp.cas_idx)
					e_base = solver_base.kernel(hf_base, _exp.core_idx, _exp.cas_idx, _e_cas=e_cas)
					e_corr = e_cas - e_base
				#
				return e_corr


class ModelSolver():
		""" CISD, CCSD, or CCSD(T) as active space solver, 
		adapted from cc test: 42-as_casci_fcisolver.py of the pyscf test suite
		"""
		def __init__(self, model):
				""" init model object """
				self.model_type = model['METHOD']
				self.model = None
				#
				return


		def hf(self, _mol, _calc, _h1e, _h2e, _core_idx, _cas_idx):
				""" form active space hf """
				cas_mol = gto.M(verbose=0)
				cas_mol.nelectron = _mol.nelectron - 2 * len(_core_idx)
				cas_hf = scf.RHF(cas_mol)
				cas_hf.spin = _mol.spin
				cas_hf._eri = ao2mo.restore(8, _h2e, len(_cas_idx))
				cas_hf.get_hcore = lambda *args: _h1e
				cas_hf.nelectron = cas_mol.nelectron
				cas_hf.mo_occ = np.zeros(len(_cas_idx)); cas_hf.mo_occ[:(_mol.nocc - len(_core_idx))] = 2
				cas_hf.e_tot = scf.hf.energy_elec(cas_hf, dm=np.diag(cas_hf.mo_occ))[0]
				#
				return cas_hf


		def kernel(self, _cas_hf, _core_idx, _cas_idx, _e_cas=None):
				""" model kernel """
				if (self.model_type == 'CISD'):
					self.model = ci.cisd.CISD(_cas_hf, mo_coeff=np.eye(len(_cas_idx)), mo_occ=_cas_hf.mo_occ)
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
				elif (self.model_type in ['CCSD','CCSD(T)']):
					self.model = cc.ccsd.CCSD(_cas_hf, mo_coeff=np.eye(len(_cas_idx)), mo_occ=_cas_hf.mo_occ)
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
					if (self.model_type == 'CCSD(T)'): e_corr += self.model.ccsd_t()
				#
				return e_corr + _cas_hf.e_tot


