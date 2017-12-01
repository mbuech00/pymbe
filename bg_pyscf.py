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
	from pyscf import gto, symm, scf, ao2mo, lo, ci, cc, mcscf, fci
except ImportError:
	sys.stderr.write('\nImportError : pyscf module not found\n\n')


class PySCFCls():
		""" pyscf class """
		def hcore_eri(self, _mol):
				""" get core hamiltonian and AO eris """
				hcore = _mol.intor_symmetric('int1e_kin') + _mol.intor_symmetric('int1e_nuc')
				eri = _mol.intor('int2e_sph', aosym=4)
				#
				return hcore, eri


		def hf(self, _mol, _calc):
				""" hartree-fock calculation """
				# perform hf calc
				hf = scf.RHF(_mol)
				hf.conv_tol = 1.0e-10
				hf.max_cycle = 500
				hf.irrep_nelec = _mol.irrep_nelec
				# perform hf calc
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
				_mol.norb, _mol.occ, _mol.nocc, _mol.virt, _mol.nvirt = self.dim(hf, _mol.ncore, _calc.exp_type)
				# store energy, mo_occ, and orbsym
				_calc.hf_e_tot = hf.e_tot
				_calc.hf_mo_occ = hf.mo_occ
				_calc.hf_orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, hf.mo_coeff)
				#
				return hf


		def dim(self, _hf, _ncore, _type):
				""" determine dimensions """
				# occupied and virtual lists
				if (_type == 'occupied'):
					occ = np.where(_hf.mo_occ == 2.)[0]
					virt = np.where(_hf.mo_occ < 2.)[0]
				elif (_type == 'virtual'):
					occ = np.where(_hf.mo_occ > 0.)[0]
					virt = np.where(_hf.mo_occ == 0.)[0]
				# nocc, nvirt, and norb
				nocc = len(occ)
				nvirt = len(virt)
				norb = nocc + nvirt
				# update occ orbitals according to _ncore
				occ = occ[_ncore:]
				#
				return norb, occ, nocc, virt, nvirt


		def active(self, _mol, _calc):
				""" set active space """
				# hf reference model
				if (_calc.exp_ref['METHOD'] == 'HF'):
					# no cas space and set of active orbitals
					no_act = _mol.nocc
					ne_act = (len(np.where(_calc.hf_mo_occ > 0.)[0]), len(np.where(_calc.hf_mo_occ == 2.)[0]))
					cas_space = np.array([])
					act_orbs = np.array([])
				# casci/casscf reference model
				elif (_calc.exp_ref['METHOD'] in ['CASCI','CASSCF']):
					# number of orbitals
					no_act = len(_calc.exp_ref['ACTIVE'])
					# set cas space
					cas_space = np.array(_calc.exp_ref['ACTIVE'])
					assert(np.count_nonzero(cas_space < _mol.ncore) == 0)
					# set of active orbitals
					num_occ = np.count_nonzero(cas_space < _mol.nocc)
					num_virt = np.count_nonzero(cas_space >= _mol.nocc)
					if (_calc.exp_type == 'occupied'):
						act_orbs = _mol.occ[:num_occ]
					elif (_calc.exp_type == 'virtual'):
						act_orbs = _mol.virt[:num_virt]
					# number of electrons
					ne_act = _calc.exp_ref['NELEC']
					assert((ne_act[0]+ne_act[1]) <= num_occ * 2)
				# debug print
				if (_mol.verbose_prt):
					print('\n cas  = {0:} , act_orbs = {1:} , no_act = {2:} , ne_act = {3:}'.\
							format(cas_space, act_orbs, no_act, ne_act))
				#
				return no_act, ne_act, cas_space, act_orbs


		def ref(self, _mol, _calc):
				""" reference calc """
				if (_calc.exp_ref['METHOD'] == 'HF'):
					# hf MOs
					ref_mo_coeff = np.asarray(_calc.hf.mo_coeff, order='C')
				elif (_calc.exp_ref['METHOD'] in ['CASCI','CASSCF']):
					# casci (no) or casscf results
					if (_calc.exp_ref['METHOD'] == 'CASCI'):
						cas = mcscf.CASCI(_calc.hf, _calc.no_act, _calc.ne_act)
					elif (_calc.exp_ref['METHOD'] == 'CASSCF'):
						cas = mcscf.CASSCF(_calc.hf, _calc.no_act, _calc.ne_act)
					if (_mol.spin == 0):
						cas.fcisolver = fci.direct_spin0_symm.FCI(_mol)
					else:
						cas.fcisolver = fci.direct_spin1_symm.FCI(_mol)
					cas.fcisolver.conv_tol = 1.0e-10
					if (_mol.verbose_prt): cas.verbose = 4
					if (_calc.exp_ref['METHOD'] == 'CASSCF'):
						cas.conv_tol = 1.0e-10
						cas.max_stepsize = .01
						cas.max_cycle_micro = 1
						cas.frozen = _mol.ncore
					# initial guess
					na = fci.cistring.num_strings(_calc.no_act, _calc.ne_act[0])
					nb = fci.cistring.num_strings(_calc.no_act, _calc.ne_act[1])
					hf_as_civec = np.zeros((na, nb))
					hf_as_civec[0, 0] = 1
					# fix spin if non-singlet
					if (_mol.spin > 0):
						sz = abs(_mol.ne_act[0]-_mol.ne_act[1]) * .5
						cas.fix_spin_(ss=sz * (sz + 1.))
					# sort mo
					mo = cas.sort_mo(_calc.cas_space, base=0)
					try:
						cas.kernel(mo, ci0=hf_as_civec)
					except Exception as err:
						try:
							raise RuntimeError(('\n{0:} Error :\n'
												'no_act = {1:} , ne_act = {2:}\n'
												'PySCF Error: {3:}\n\n').\
												format(_calc.exp_ref['METHOD'], _calc.no_act, _calc.ne_act, err))
						except Exception as err_2:
							sys.stderr.write(str(err_2))
							raise
					if (_calc.exp_ref['METHOD'] == 'CASSCF'):
						if (not cas.converged):
							try:
								raise RuntimeError('\nCASSCF Error : no convergence\n\n')
							except Exception as err:
								sys.stderr.write(str(err))
								raise
					# calculate spin
					s, mult = fci.spin_op.spin_square(cas.ci, _calc.no_act, _calc.ne_act)
					# check for correct spin
					if (float(_mol.spin) - s > 1.0e-05):
						try:
							raise RuntimeError(('\n{0:} Error : spin contamination\n'
												'2*S + 1 = {1:.3f}\n\n').\
												format(_calc.exp_ref['METHOD'], mult))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					# save MOs
					ref_mo_coeff = np.asarray(cas.mo_coeff, order='C')
				# calculate ref_e_tot
				ref_e_tot = self.e_mf(_mol, ref_mo_coeff)
				#
				return ref_e_tot, ref_mo_coeff


		def e_mf(self, _mol, _mo):
				""" calculate mean-field energy """
				dm = np.dot(_mo[:, :_mol.nocc], np.transpose(_mo[:, :_mol.nocc])) * 2
				vj, vk = scf.hf.get_jk(_mol, dm)
				vhf = vj - vk * .5
				e_mf = _mol.energy_nuc()
				e_mf += np.einsum('ij,ji', dm, _mol.hcore)
				e_mf += np.einsum('ij,ji', dm, vhf) * .5
				#
				return e_mf


		def trans_main(self, _mol, _calc, _exp):
				""" determine main transformation matrices """
				# zeroth-order energy
				if (_calc.exp_base['METHOD'] is None):
					_calc.e_zero = 0.0
				# cisd base
				elif (_calc.exp_base['METHOD'] == 'CISD'):
					_calc.e_zero, dm = self.ci_base(_calc)
				# ccsd base
				elif (_calc.exp_base['METHOD'] in ['CCSD','CCSD(T)']):
					_calc.e_zero, dm = self.cc_base(_mol, _calc, _exp, _calc.ref_mo_coeff, \
												(_calc.exp_occ == 'REF') and (_calc.exp_virt == 'REF'))
				# sci base
				elif (_calc.exp_base['METHOD'] == 'SCI'):
					_calc.e_zero, dm = self.sci_base(_mol, _calc, _exp, _calc.ref_mo_coeff)
				# init transformation matrix
				_calc.trans_mat = np.copy(_calc.ref_mo_coeff)
				# occ-occ block (local or NOs)
				if (_calc.exp_occ != 'REF'):
					if (_calc.exp_occ == 'NO'):
						if (_mol.spin > 0): dm = dm[0] + dm[1]
						occup, no = symm.eigh(dm[:len(_mol.occ), :len(_mol.occ)], _calc.hf_orbsym[_mol.occ])
						_calc.trans_mat[:, _mol.occ] = np.dot(_calc.ref_mo_coeff[:, _mol.occ], no[:, ::-1])
					elif (_calc.exp_occ == 'PM'):
						_calc.trans_mat[:, _mol.occ] = lo.PM(_mol, _calc.ref_mo_coeff[:, _mol.occ]).kernel()
					elif (_calc.exp_occ == 'FB'):
						_calc.trans_mat[:, _mol.occ] = lo.Boys(_mol, _calc.ref_mo_coeff[:, _mol.occ]).kernel()
					elif (_calc.exp_occ in ['IBO-1','IBO-2']):
						iao = lo.iao.iao(_mol, _calc.ref_mo_coeff[:, _mol.occ])
						if (_calc.exp_occ == 'IBO-1'):
							iao = lo.vec_lowdin(iao, _calc.hf.get_ovlp())
							_calc.trans_mat[:, _mol.occ] = lo.ibo.ibo(_mol, _calc.ref_mo_coeff[:, _mol.occ], iao)
						elif (_calc.exp_occ == 'IBO-2'):
							_calc.trans_mat[:, _mol.occ] = lo.ibo.PM(_mol, _calc.ref_mo_coeff[:, _mol.occ], iao).kernel()
				# virt-virt block (local or NOs)
				if (_calc.exp_virt != 'REF'):
					if (_calc.exp_virt == 'NO'):
						if ((_mol.spin > 0) and (_calc.exp_occ != 'NO')): dm = dm[0] + dm[1]
						occup, no = symm.eigh(dm[-len(_mol.virt):, -len(_mol.virt):], _calc.hf_orbsym[_mol.virt])
						_calc.trans_mat[:, _mol.virt] = np.dot(_calc.ref_mo_coeff[:, _mol.virt], no[:, ::-1])
					elif (_calc.exp_virt == 'PM'):
						_calc.trans_mat[:, _mol.virt] = lo.PM(_mol, _calc.ref_mo_coeff[:, _mol.virt]).kernel()
					elif (_calc.exp_virt == 'FB'):
						_calc.trans_mat[:, _mol.virt] = lo.Boys(_mol, _calc.ref_mo_coeff[:, _mol.virt]).kernel()
				# (t) correction for NOs
				if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')):
					if (_calc.exp_base['METHOD'] == 'CCSD(T)'):
						_calc.e_zero, dm = self.cc_base(_mol, _calc, _exp, _calc.trans_mat, True)
					elif (_calc.exp_base['METHOD'] == 'SCI'):
						_calc.e_zero, dm = self.sci_base(_mol, _calc, _exp, _calc.trans_mat)
				#
				return


		def ci_base(self, _calc):
				""" cisd base calc """
				# set core and cas spaces
				if (_calc.exp_type == 'occupied'):
					_exp.core_idx, _exp.cas_idx = self.core_cas(_mol, _exp, np.array(range(_mol.ncore, _mol.nocc)))
				if (_calc.exp_type == 'virtual'):
					_exp.core_idx, _exp.cas_idx = self.core_cas(_mol, _exp, np.array(range(_mol.nocc, _mol.norb)))
				# get integrals
				h1e, h2e = self.prepare(_mol, _calc, _exp, _mo)
				mol = gto.M(verbose=1)
				if (_mol.spin == 0):
					hf = scf.RHF(mol)
				else:
					hf = scf.UHF(mol)
				hf.get_hcore = lambda *args: h1e
				hf._eri = h2e 
				# init ccsd
				if (_mol.spin == 0):
					cisd = ci.cisd.CISD(hf, mo_coeff=np.eye(len(_exp.cas_idx)), mo_occ=_calc.hf_mo_occ[_exp.cas_idx])
				else:
					cisd = ci.ucisd.UCISD(hf, mo_coeff=np.array((np.eye(len(_exp.cas_idx)), np.eye(len(_exp.cas_idx)))), \
											mo_occ=np.array((_calc.hf_mo_occ[_exp.cas_idx]>0, _calc.hf_mo_occ[_exp.cas_idx]==2), dtype=np.double))
				cisd.max_cycle = 500
				cisd.diis_space = 10
				cisd.mol.incore_anyway = True
				eris = cisd.ao2mo()
				# calculate cisd energy
				for i in range(5,-1,-1):
					cisd.level_shift = 1.0 / 10.0 ** (i)
					try:
						cisd.kernel(eris=eris)
					except sp.linalg.LinAlgError: pass
					if (cisd.converged): break
				if (not cisd.converged):
					try:
						raise RuntimeError('\nCISD base Error : no convergence\n\n')
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				# e_zero
				e_zero = cisd.e_corr
				# cisd dm
				if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')):
					dm = cisd.make_rdm1()
				else:
					dm = None
				#
				return e_zero, dm


		def cc_base(self, _mol, _calc, _exp, _mo, _pt_corr=False):
				""" ccsd / ccsd(t) base calc """
				# set core and cas spaces
				if (_calc.exp_type == 'occupied'):
					_exp.core_idx, _exp.cas_idx = self.core_cas(_mol, _exp, np.array(range(_mol.ncore, _mol.nocc)))
				if (_calc.exp_type == 'virtual'):
					_exp.core_idx, _exp.cas_idx = self.core_cas(_mol, _exp, np.array(range(_mol.nocc, _mol.norb)))
				# get integrals
				h1e, h2e = self.prepare(_mol, _calc, _exp, _mo)
				mol = gto.M(verbose=1)
				if (_mol.spin == 0):
					hf = scf.RHF(mol)
				else:
					hf = scf.UHF(mol)
				hf.get_hcore = lambda *args: h1e
				hf._eri = h2e 
				# init ccsd
				if (_mol.spin == 0):
					ccsd = cc.ccsd.CCSD(hf, mo_coeff=np.eye(len(_exp.cas_idx)), mo_occ=_calc.hf_mo_occ[_exp.cas_idx])
				else:
					ccsd = cc.uccsd.UCCSD(hf, mo_coeff=np.array((np.eye(len(_exp.cas_idx)), np.eye(len(_exp.cas_idx)))), \
											mo_occ=np.array((_calc.hf_mo_occ[_exp.cas_idx]>0, _calc.hf_mo_occ[_exp.cas_idx]==2), dtype=np.double))
				ccsd.max_cycle = 500
				ccsd.diis_space = 10
				ccsd.mol.incore_anyway = True
				eris = ccsd.ao2mo()
				# calculate ccsd energy
				for i in list(range(0, 12, 2)):
					ccsd.diis_start_cycle = i
					try:
						ccsd.kernel(eris=eris)
					except sp.linalg.LinAlgError: pass
					if (ccsd.converged): break
				if (not ccsd.converged):
					try:
						raise RuntimeError('\nCCSD base Error : no convergence\n\n')
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				# e_zero
				e_zero = ccsd.e_corr
				# ccsd dm
				if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')):
					dm = ccsd.make_rdm1()
				else:
					dm = None
				# calculate (t) correction
				if (_pt_corr): e_zero += ccsd.ccsd_t(eris=eris)
				#
				return e_zero, dm


		def sci_base(self, _mol, _calc, _exp, _mo):
				""" sci base calc """
				# init sci
				if (_mol.spin == 0):
					sci_solver = fci.select_ci_spin0_symm.SCI(_mol)
				else:
					sci_solver = fci.select_ci_symm.SCI(_mol)
				sci_solver.conv_tol = 1.0e-10
				sci_solver.max_cycle = 500
				sci_solver.max_space = 10
				sci_solver.max_memory = _mol.max_memory
				sci_solver.davidson_only = True
				# set core and cas spaces
				if (_calc.exp_type == 'occupied'):
					_exp.core_idx, _exp.cas_idx = self.core_cas(_mol, _exp, np.array(range(_mol.ncore, _mol.nocc)))
				if (_calc.exp_type == 'virtual'):
					_exp.core_idx, _exp.cas_idx = self.core_cas(_mol, _exp, np.array(range(_mol.nocc, _mol.norb)))
				# get integrals
				h1e, h2e = self.prepare(_mol, _calc, _exp, _mo)
				# cas electrons
				nelec_cas = (_mol.nelec[0] - len(_exp.core_idx), _mol.nelec[1] - len(_exp.core_idx))
				# initial guess
				ci_strs = (np.asarray([int('1'*nelec_cas[0], 2)]), np.asarray([int('1'*nelec_cas[1], 2)]))
				hf_as_scivec = fci.select_ci._as_SCIvector(np.ones((1,1)), ci_strs)
				hf_as_scivec = sci_solver.enlarge_space(hf_as_scivec, h2e, len(_exp.cas_idx), nelec_cas)
				# orbital symmetry
				orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, _mo[:, _exp.cas_idx])
				# fix spin if non-singlet
				if (_mol.spin > 0):
					sz = abs(nelec_cas[0]-nelec_cas[1]) * .5
					fci.addons.fix_spin(sci_solver, ss=sz * (sz + 1.))
				# calculate sci energy
				try:
					e_sci, c_sci = sci_solver.kernel(h1e, h2e, len(_exp.cas_idx), nelec_cas, \
														ecore=_exp.e_core, orbsym=orbsym, ci0=hf_as_scivec)
				except Exception as err:
					try:
						raise RuntimeError(('\nSCI base Error :\n'
											'PySCF Error: {0:}\n\n').\
											format(err))
					except Exception as err_2:
						sys.stderr.write(str(err_2))
						raise
				# calculate spin
				s_sci, mult_sci = sci_solver.spin_square(c_sci, len(_exp.cas_idx), nelec_cas)
				# check for correct spin
				if (float(_mol.spin) - s_sci > 1.0e-05):
					try:
						raise RuntimeError(('\nSCI base Error : spin contamination\n\n'
											'2*S + 1 = {0:.3f}\n\n').\
											format(mult_sci))
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				# e_zero
				e_zero = e_sci - _calc.ref_e_tot
				# sci dm
				if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')):
					dm = sci_solver.make_rdm1(c_sci, len(_exp.cas_idx), nelec_cas)
				else:
					dm = None
				#
				return e_zero, dm


		def trans_dno(self, _mol, _calc, _exp):
				""" determine dno transformation matrices """
				# set frozen list
				frozen = sorted(list(set(range(_mol.nocc)) - set(_exp.incl_idx))) 
				# zeroth-order energy
				if (_calc.exp_base['METHOD'] == 'CISD'):
					# calculate ccsd energy
					cisd = ci.CISD(_calc.hf)
					cisd.conv_tol = 1.0e-10
					cisd.max_cycle = 500
					cisd.max_space = 10
					cisd.mol.incore_anyway = True
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
					ccsd.conv_tol_normt = 1.0e-10
					ccsd.max_cycle = 500
					ccsd.diis_space = 10
					ccsd.mol.incore_anyway = True
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
				_calc.trans_mat[:, _mol.virt] = np.dot(_calc.ref_mo_coeff[:, _mol.virt], no[:, ::-1])
				#
				return


		def core_cas(self, _mol, _exp, _tup):
				""" define core and cas spaces """
				cas_idx = sorted(_exp.incl_idx + sorted(_tup.tolist()))
				core_idx = sorted(list(set(range(_mol.nocc)) - set(cas_idx)))
				#
				return core_idx, cas_idx


		def prepare(self, _mol, _calc, _exp, _orbs):
				""" generate input for correlated calculation """
				# extract cas integrals and calculate core energy
				if (len(_exp.core_idx) > 0):
					if ((_calc.exp_type == 'occupied') or (_exp.e_core is None)):
						core_dm = np.dot(_orbs[:, _exp.core_idx], np.transpose(_orbs[:, _exp.core_idx])) * 2
						vj, vk = scf.hf.get_jk(_mol, core_dm)
						_exp.core_vhf = vj - vk * .5
						_exp.e_core = _mol.energy_nuc() + np.einsum('ij,ji', core_dm, _mol.hcore)
						_exp.e_core += np.einsum('ij,ji', core_dm, _exp.core_vhf) * .5
				else:
					_exp.e_core = _mol.energy_nuc()
					_exp.core_vhf = 0
				h1e_cas = reduce(np.dot, (np.transpose(_orbs[:, _exp.cas_idx]), \
										_mol.hcore + _exp.core_vhf, _orbs[:, _exp.cas_idx]))
				h2e_cas = ao2mo.incore.full(_mol.eri, _orbs[:, _exp.cas_idx])
				#
				return h1e_cas, h2e_cas


		def calc(self, _mol, _calc, _exp):
				""" correlated cas calculation """
				# init solver
				if (_calc.exp_model['METHOD'] != 'FCI'):
					solver_cas = ModelSolver(_calc.exp_model)
				else:
					if (_mol.spin == 0):
						solver_cas = fci.direct_spin0_symm.FCI()
					else:
						solver_cas = fci.direct_spin1_symm.FCI()
					# fci settings
					solver_cas.conv_tol = max(_exp.thres, 1.0e-10)
					solver_cas.max_cycle = 500
					solver_cas.max_space = 10
					solver_cas.max_memory = _mol.max_memory
					solver_cas.davidson_only = True
				# cas calculation
				if (_calc.exp_model['METHOD'] != 'FCI'):
					hf_cas = solver_cas.hf(_mol, _calc, _exp.h1e_cas, _exp.h2e_cas, _exp.cas_idx, _exp.e_core, _exp.thres)
					e_cas = solver_cas.kernel(hf_cas, _exp.core_idx, _exp.cas_idx)
				else:
					# cas electrons
					nelec_cas = (_mol.nelec[0] - len(_exp.core_idx), _mol.nelec[1] - len(_exp.core_idx))
					# initial guess
					na = fci.cistring.num_strings(len(_exp.cas_idx), nelec_cas[0])
					nb = fci.cistring.num_strings(len(_exp.cas_idx), nelec_cas[1])
					hf_as_civec = np.zeros((na, nb))
					hf_as_civec[0, 0] = 1
					# orbital symmetry
					orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, _calc.trans_mat[:, _exp.cas_idx])
					# fix spin if non-singlet
					if (_mol.spin > 0):
						sz = abs(nelec_cas[0]-nelec_cas[1]) * .5
						fci.addons.fix_spin(solver_cas, ss=sz * (sz + 1.))
					# perform calc
					try:
						e_cas, c_cas = solver_cas.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
															nelec_cas, orbsym=orbsym, ci0=hf_as_civec)
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
					cas_s, cas_mult = fci.spin_op.spin_square(c_cas, len(_exp.cas_idx), nelec_cas)
					# check for correct spin
					if (float(_mol.spin) - cas_s > 1.0e-05):
						try:
							raise RuntimeError(('\nCAS-CI Error : spin contamination\n\n'
												'2*S + 1 = {0:.3f}\n'
												'core_idx = {1:} , cas_idx = {2:}\n\n').\
												format(cas_mult, _exp.core_idx, _exp.cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
				# base calculation
				if (_calc.exp_base['METHOD'] is None):
					e_corr = (e_cas + _exp.e_core) - _calc.ref_e_tot
#					if (_exp.order < _exp.max_order): e_corr += (e_cas + _exp.e_core) - _calc.ref_e_tot + 0.001 * np.random.random_sample()
				elif (_calc.exp_base['METHOD'] == 'SCI'):
					# init solver
					if (_mol.spin == 0):
						solver_base = fci.select_ci_spin0_symm.SCI()
					else:
						solver_base = fci.select_ci_symm.SCI()
					# sci settings
					solver_base.conv_tol = max(_exp.thres, 1.0e-10)
					solver_base.max_cycle = 500
					solver_base.max_space = 10
					solver_base.max_memory = _mol.max_memory
					solver_base.davidson_only = True
					# initial guess
					ci_strs = (np.asarray([int('1'*nelec_cas[0], 2)]), np.asarray([int('1'*nelec_cas[1], 2)]))
					hf_as_scivec = fci.select_ci._as_SCIvector(np.ones((1,1)), ci_strs)
					hf_as_scivec = solver_base.enlarge_space(hf_as_scivec, _exp.h2e_cas, len(_exp.cas_idx), nelec_cas)
					# fix spin if non-singlet
					if (_mol.spin > 0):
						sz = abs(nelec_cas[0]-nelec_cas[1]) * .5
						fci.addons.fix_spin(solver_base, ss=sz * (sz + 1.))
					# perform calc
					try:
						e_base, c_base = solver_base.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
																nelec_cas, orbsym=orbsym, ci0=hf_as_scivec)
					except Exception as err:
						try:
							raise RuntimeError(('\nCAS-SCI Error :\n'
												'core_idx = {0:} , cas_idx = {1:}\n'
												'PySCF Error: {2:}\n\n').\
												format(_exp.core_idx, _exp.cas_idx, err))
						except Exception as err_2:
							sys.stderr.write(str(err_2))
							raise
					# calculate spin
					base_s, base_mult = solver_base.spin_square(c_base, len(_exp.cas_idx), nelec_cas)
					# check for correct spin
					if (float(_mol.spin) - base_s > 1.0e-05):
						try:
							raise RuntimeError(('\nCAS-SCI Error : spin contamination\n\n'
												'2*S + 1 = {0:.3f}\n'
												'core_idx = {1:} , cas_idx = {2:}\n\n').\
												format(base_mult, _exp.core_idx, _exp.cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					e_corr = e_cas - e_base
#					if (_exp.order < _exp.max_order): e_corr += e_cas - e_base + 0.001 * np.random.random_sample()
				else:
					# base calculation
					solver_base = ModelSolver(_calc.exp_base)
					hf_base = solver_base.hf(_mol, _calc, _exp.h1e_cas, _exp.h2e_cas, _exp.cas_idx, _exp.e_core, _exp.thres)
					e_base = solver_base.kernel(hf_base, _exp.core_idx, _exp.cas_idx)
					e_corr = e_cas - e_base
#					if (_exp.order < _exp.max_order): e_corr += e_cas - e_base + 0.001 * np.random.random_sample()
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


		def hf(self, _mol, _calc, _h1e, _h2e, _cas_idx, _e_core, _thres):
				""" form active space hf """
				cas_mol = gto.M(verbose=1)
				cas_mol.max_memory = _mol.max_memory
				cas_mol.spin = _mol.spin
				if (_mol.spin == 0):
					cas_hf = scf.RHF(cas_mol)
				else:
					cas_hf = scf.UHF(cas_mol)
				cas_hf._eri = _h2e
				cas_hf.get_hcore = lambda *args: _h1e
				# store quantities needed in kernel()
				cas_hf.mo_occ = _calc.hf_mo_occ[_cas_idx]
				cas_hf.e_tot = _calc.ref_e_tot - _e_core
				cas_hf.conv_tol = max(_thres, 1.0e-10)
				#
				return cas_hf


		def kernel(self, _cas_hf, _core_idx, _cas_idx):
				""" model kernel """
				if (self.model_type == 'CISD'):
					self.model = ci.CISD(_cas_hf)
					if (_cas_hf.mol.spin == 0):
						self.model = ci.cisd.CISD(_cas_hf, mo_coeff=np.eye(len(_cas_idx)), mo_occ=_cas_hf.mo_occ)
					else:
						self.model = ci.ucisd.UCISD(_cas_hf, mo_coeff=np.array((np.eye(len(_cas_idx)), np.eye(len(_cas_idx)))), \
												mo_occ=np.array((_cas_hf.mo_occ>0, _cas_hf.mo_occ==2), dtype=np.double))
					self.model.conv_tol = _cas_hf.conv_tol
					self.model.max_cycle = 500
					self.model.max_space = 10
					self.model.max_memory = _cas_hf.mol.max_memory
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
				elif (self.model_type in ['CCSD','CCSD(T)']):
					if (_cas_hf.mol.spin == 0):
						self.model = cc.ccsd.CCSD(_cas_hf, mo_coeff=np.eye(len(_cas_idx)), mo_occ=_cas_hf.mo_occ)
					else:
						self.model = cc.uccsd.UCCSD(_cas_hf, mo_coeff=np.array((np.eye(len(_cas_idx)), np.eye(len(_cas_idx)))), \
												mo_occ=np.array((_cas_hf.mo_occ>0, _cas_hf.mo_occ==2), dtype=np.double))
					self.model.conv_tol = _cas_hf.conv_tol
					self.model.max_cycle = 500
					self.model.diis_space = 10
					self.model.max_memory = _cas_hf.mol.max_memory
					eris = self.model.ao2mo()
					for i in list(range(0, 12, 2)):
						self.model.diis_start_cycle = i
						try:
							e_corr = self.model.kernel(eris=eris)[0]
						except sp.linalg.LinAlgError: pass
						if (self.model.converged): break
					# check for convergence
					if (not self.model.converged):
						try:
							raise RuntimeError(('\nCAS-CCSD Error : no convergence\n'
												'core_idx = {0:} , cas_idx = {1:}\n\n').\
												format(_core_idx, _cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					# add (t) correction
					if (self.model_type == 'CCSD(T)'): e_corr += self.model.ccsd_t(eris=eris)
				#
				return e_corr + _cas_hf.e_tot


