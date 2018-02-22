#!/usr/bin/env python
# -*- coding: utf-8 -*

""" kernel.py: kernel routines """

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


class KernCls():
		""" kernel class """
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
				# fixed occupation
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
				_mol.norb, _mol.nocc, _mol.nvirt = self.dim(hf, _mol.ncore, _calc.exp_type)
				# store energy, occupation, and orbsym
				_calc.energy['hf'] = hf.e_tot
				_calc.occup = hf.mo_occ
				_calc.orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, hf.mo_coeff)
				#
				return hf, np.asarray(hf.mo_coeff, order='C')


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
				#
				return norb, nocc, nvirt


		def active(self, _mol, _calc):
				""" set active space """
				# hf reference model
				if (_calc.exp_ref['METHOD'] == 'HF'):
					# no active space
					no_act = ne_act = None
				# casci/casscf reference model
				elif (_calc.exp_ref['METHOD'] in ['CASCI','CASSCF']):
					# active electrons
					ne_act = _calc.exp_ref['NELEC']
					assert((ne_act[0]+ne_act[1]) <= _mol.nelectron - (2*_mol.ncore))
					# active orbitals
					assert(np.count_nonzero(np.array(_calc.exp_ref['ACTIVE']) < _mol.ncore) == 0)
					no_act = len(_calc.exp_ref['ACTIVE'])
					if (_calc.exp_type == 'occupied'):
						assert(np.count_nonzero(np.array(_calc.exp_ref['ACTIVE']) >= _mol.nocc) == _mol.nvirt)
					elif (_calc.exp_type == 'virtual'):
						assert(np.count_nonzero(np.array(_calc.exp_ref['ACTIVE']) < _mol.nocc) == (_mol.nocc-_mol.ncore))
				# reference and expansion spaces
				if (_calc.exp_type == 'occupied'):
					ref_space = np.array(range(_mol.nocc, _mol.norb))
					exp_space = np.array(range(_mol.ncore, _mol.nocc))
				elif (_calc.exp_type == 'virtual'):
					ref_space = np.array(range(_mol.ncore, _mol.nocc))
					exp_space = np.array(range(_mol.nocc, _mol.norb))
				# debug print
				if (_mol.verbose_prt): print('\n ref_space = {0:} , exp_space = {1:}'.format(ref_space, exp_space))
				#
				return ref_space, exp_space, no_act, ne_act


		def e_mf(self, _mol, _calc, _mo):
				""" calculate mean-field energy """
				mo_a = _mo[:, np.where(_calc.occup > 0.)[0]]
				mo_b = _mo[:, np.where(_calc.occup == 2.)[0]]
				dm_a = np.dot(mo_a, np.transpose(mo_a))
				dm_b = np.dot(mo_b, np.transpose(mo_b))
				dm = np.array((dm_a, dm_b))
				vj, vk = scf.hf.get_jk(_mol, dm)
				vhf = vj[0] + vj[1] - vk
				e_mf = _mol.energy_nuc()
				e_mf += np.einsum('ij,ij', _mol.hcore.conj(), dm[0] + dm[1])
				e_mf += (np.einsum('ij,ji', vhf[0], dm[0]) + np.einsum('ij,ji', vhf[1], dm[1])) * .5
				#
				return e_mf


		def main_calc(self, _mol, _calc, _exp, _method):
				""" calculate correlation energy """
				# fci calc
				if (_method == 'FCI'):
					e_corr = self.fci(_mol, _calc, _exp, _calc.mo, False)
				# sci base
				elif (_method == 'SCI'):
					e_corr, _ = self.sci(_mol, _calc, _exp, _calc.mo, False)
				# cisd calc
				elif (_method == 'CISD'):
					e_corr, _ = self.ci(_mol, _calc, _exp, _calc.mo, False)
				# ccsd / ccsd(t) calc
				elif (_method in ['CCSD','CCSD(T)']):
					e_corr, _ = self.cc(_mol, _calc, _exp, _calc.mo, False, (_method == 'CCSD(T)'))
				#
				return e_corr


		def main_mo(self, _mol, _calc, _exp):
				""" calculate base energy and mo coefficients """
				# set core and cas spaces
				_exp.core_idx, _exp.cas_idx = self.core_cas(_mol, _exp, _calc.exp_space)
				# sort mo coefficients
				if (_calc.exp_ref['METHOD'] in ['CASCI','CASSCF']):
					idx = np.asarray([i for i in range(_mol.norb) if i not in _calc.exp_ref['ACTIVE']])
					mo = np.hstack((_calc.mo[:, idx[:_mol.ncore]], _calc.mo[:, _calc.exp_ref['ACTIVE']], _calc.mo[:, idx[_mol.ncore:]]))
					_calc.mo = np.asarray(mo, order='C')
				# casscf ref calc
				if (_calc.exp_ref['METHOD'] == 'CASSCF'):
					# exp model
					_calc.energy['cas_model'], _calc.mo = self.casscf(_mol, _calc, _exp, _calc.exp_model['METHOD'])
					# base model
					if (_calc.exp_base['METHOD'] is not None):
						_calc.energy['cas_base'], _ = self.casscf(_mol, _calc, _exp, _calc.exp_base['METHOD'])
				# zeroth-order energy
				if (_calc.exp_base['METHOD'] is None):
					e_base = np.float64(0.0)
				# cisd base
				elif (_calc.exp_base['METHOD'] == 'CISD'):
					e_base, dm = self.ci(_mol, _calc, _exp, _calc.mo, True)
					if ((_mol.spin > 0) and (dm is not None)): dm = dm[0] + dm[1]
				# ccsd / ccsd(t) base
				elif (_calc.exp_base['METHOD'] in ['CCSD','CCSD(T)']):
					e_base, dm = self.cc(_mol, _calc, _exp, _calc.mo, True, \
												(_calc.exp_base['METHOD'] == 'CCSD(T)') and \
												((_calc.exp_occ == 'REF') and (_calc.exp_virt == 'REF')))
					if ((_mol.spin > 0) and (dm is not None)): dm = dm[0] + dm[1]
				# sci base
				elif (_calc.exp_base['METHOD'] == 'SCI'):
					e_base, dm = self.sci(_mol, _calc, _exp, _calc.mo, True)
				# copy mo
				mo = np.copy(_calc.mo)
				# occ-occ block (local or NOs)
				if (_calc.exp_occ != 'REF'):
					if (_calc.exp_occ == 'NO'):
						occup, no = symm.eigh(dm[:(_mol.nocc-_mol.ncore), :(_mol.nocc-_mol.ncore)], _calc.orbsym[_mol.ncore:_mol.nocc])
						mo[:, _mol.ncore:_mol.nocc] = np.dot(_calc.mo[:, _mol.ncore:_mol.nocc], no[:, ::-1])
					elif (_calc.exp_occ == 'PM'):
						mo[:, _mol.ncore:_mol.nocc] = lo.PM(_mol, _calc.mo[:, _mol.ncore:_mol.nocc]).kernel()
					elif (_calc.exp_occ == 'FB'):
						mo[:, _mol.ncore:_mol.nocc] = lo.Boys(_mol, _calc.mo[:, _mol.ncore:_mol.nocc]).kernel()
					elif (_calc.exp_occ in ['IBO-1','IBO-2']):
						iao = lo.iao.iao(_mol, _calc.mo[:, _mol.core:_mol.nocc])
						if (_calc.exp_occ == 'IBO-1'):
							iao = lo.vec_lowdin(iao, _calc.hf.get_ovlp())
							mo[:, _mol.ncore:_mol.nocc] = lo.ibo.ibo(_mol, _calc.mo[:, _mol.ncore:_mol.nocc], iao)
						elif (_calc.exp_occ == 'IBO-2'):
							mo[:, _mol.ncore:_mol.nocc] = lo.ibo.PM(_mol, _calc.mo[:, _mol.ncore:_mol.nocc], iao).kernel()
				# virt-virt block (local or NOs)
				if (_calc.exp_virt != 'REF'):
					if (_calc.exp_virt == 'NO'):
						occup, no = symm.eigh(dm[-_mol.nvirt:, -_mol.nvirt:], _calc.orbsym[_mol.nocc:])
						mo[:, _mol.nocc:] = np.dot(_calc.mo[:, _mol.nocc:], no[:, ::-1])
					elif (_calc.exp_virt == 'PM'):
						mo[:, _mol.nocc:] = lo.PM(_mol, _calc.mo[:, _mol.nocc:]).kernel()
					elif (_calc.exp_virt == 'FB'):
						mo[:, _mol.nocc:] = lo.Boys(_mol, _calc.mo[:, _mol.nocc:]).kernel()
				# (t) correction for NOs
				if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')):
					if (_calc.exp_base['METHOD'] == 'CCSD(T)'):
						e_base, dm = self.cc(_mol, _calc, _exp, mo, False, True)
					elif (_calc.exp_base['METHOD'] == 'SCI'):
						e_base, dm = self.sci(_mol, _calc, _exp, mo, False)
				#
				return e_base, mo


		def casscf(self, _mol, _calc, _exp, _method):
				""" casscf calc """
				# casscf ref
				cas = mcscf.CASSCF(_calc.hf, _calc.no_act, _calc.ne_act)
				if (abs(_calc.ne_act[0]-_calc.ne_act[1]) == 0):
					if (_method == 'FCI'):
						cas.fcisolver = fci.direct_spin0_symm.FCI(_mol)
					elif (_method == 'SCI'):
						cas.fcisolver = fci.select_ci_spin0_symm.SCI(_mol)
				else:
					if (_method == 'FCI'):
						cas.fcisolver = fci.direct_spin1_symm.FCI(_mol)
					elif (_method == 'SCI'):
						cas.fcisolver = fci.select_ci_symm.SCI(_mol)
				cas.fcisolver.conv_tol = 1.0e-10
				cas.conv_tol = 1.0e-10
				cas.max_stepsize = .01
				cas.max_cycle_micro = 1
				# ncore
				cas.frozen = _mol.ncore
				# wfnsym
				cas.fcisolver.wfnsym = _calc.wfnsym
				# verbose print
				if (_mol.verbose_prt): cas.verbose = 4
				# fix spin if non-singlet
				if (_mol.spin > 0):
					sz = abs(_mol.ne_act[0]-_mol.ne_act[1]) * .5
					cas.fix_spin_(ss=sz * (sz + 1.))
				# run casscf calc
				try:
					cas.kernel(_calc.mo)
				except Exception as err:
					try:
						raise RuntimeError(('\nCASSCF Error :\n'
											'PySCF Error: {0:}\n\n').\
											format(err))
					except Exception as err_2:
						sys.stderr.write(str(err_2))
						raise
				if (not cas.converged):
					try:
						raise RuntimeError('\nCASSCF Error : no convergence\n\n')
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				# calculate spin
				s, mult = cas.fcisolver.spin_square(cas.ci, _calc.no_act, _calc.ne_act)
				# check for correct spin
				if (float(_mol.spin) - s > 1.0e-05):
					try:
						raise RuntimeError(('\nCASSCF Error : spin contamination\n'
											'2*S + 1 = {0:.3f}\n\n').\
											format(mult))
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				# save MOs and energy
				e_corr = cas.e_tot - _calc.energy['hf']
				mo = np.asarray(cas.mo_coeff, order='C')
				#
				return e_corr, mo


		def fci(self, _mol, _calc, _exp, _mo, _base):
				""" fci calc """
				# init fci solver
				if (_mol.spin == 0):
					solver = fci.direct_spin0_symm.FCI(_mol)
				else:
					solver = fci.direct_spin1_symm.FCI(_mol)
				# settings
				solver.conv_tol = 1.0e-10
				solver.max_cycle = 500
				solver.max_space = 10
				solver.davidson_only = True
				# wfnsym
				solver.wfnsym = _calc.wfnsym
				# get integrals and core energy
				h1e, h2e, e_core = self.prepare(_mol, _calc, _exp, _mo)
				# electrons
				nelec = (_mol.nelec[0] - len(_exp.core_idx), _mol.nelec[1] - len(_exp.core_idx))
				# orbital symmetry
				solver.orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, _mo[:, _exp.cas_idx])
				# fix spin if non-singlet
				if (_mol.spin > 0):
					sz = abs(nelec[0]-nelec[1]) * .5
					fci.addons.fix_spin(solver, ss=sz * (sz + 1.))
				# init guess (does it exist?)
				try:
					ci0 = fci.addons.symm_initguess(len(_exp.cas_idx), nelec, orbsym=solver.orbsym, wfnsym=solver.wfnsym)
					e_corr = None
				except Exception:
					e_corr = 0.0
				# perform calc
				if (e_corr is None):
					try:
						e, c = solver.kernel(h1e, h2e, len(_exp.cas_idx), nelec, ecore=e_core)
					except Exception as err:
						try:
							raise RuntimeError(('\nFCI Error :\n'
												'core_idx = {0:} , cas_idx = {1:}\n'
												'PySCF Error: {2:}\n\n').\
												format(_exp.core_idx, _exp.cas_idx, err))
						except Exception as err_2:
							sys.stderr.write(str(err_2))
							raise
					# calculate spin
					s, mult = solver.spin_square(c, len(_exp.cas_idx), nelec)
					# check for correct spin
					if (float(_mol.spin) - s > 1.0e-05):
						try:
							raise RuntimeError(('\nFCI Error : spin contamination\n\n'
												'2*S + 1 = {0:.3f}\n'
												'core_idx = {1:} , cas_idx = {2:}\n\n').\
												format(mult, _exp.core_idx, _exp.cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					# e_corr
					e_corr = e - _calc.energy['hf']
#				if (_exp.order < _exp.max_order): e_corr += np.float64(0.001) * np.random.random_sample()
				#
				return e_corr


		def sci(self, _mol, _calc, _exp, _mo, _base):
				""" sci calc """
				# init sci solver
				if (_mol.spin == 0):
					solver = fci.select_ci_spin0_symm.SCI(_mol)
				else:
					solver = fci.select_ci_symm.SCI(_mol)
				# settings
				solver.conv_tol = 1.0e-10
				solver.max_cycle = 500
				solver.max_space = 10
				solver.davidson_only = True
				# wfnsym
				solver.wfnsym = _calc.wfnsym
				# get integrals and core energy
				h1e, h2e, e_core = self.prepare(_mol, _calc, _exp, _mo)
				# electrons
				nelec = (_mol.nelec[0] - len(_exp.core_idx), _mol.nelec[1] - len(_exp.core_idx))
				# orbital symmetry
				solver.orbsym = symm.label_orb_symm(_mol, _mol.irrep_id, _mol.symm_orb, _mo[:, _exp.cas_idx])
				# fix spin if non-singlet
				if (_mol.spin > 0):
					sz = abs(nelec[0]-nelec[1]) * .5
					fci.addons.fix_spin(solver, ss=sz * (sz + 1.))
				# init guess (does it exist?)
				try:
					ci0 = fci.addons.symm_initguess(len(_exp.cas_idx), nelec, orbsym=solver.orbsym, wfnsym=solver.wfnsym)
					e_corr = None
				except Exception:
					e_corr = 0.0
				# perform calc
				if (e_corr is None):
					# calculate sci energy
					try:
						e, c = solver.kernel(h1e, h2e, len(_exp.cas_idx), nelec, ecore=e_core)
					except Exception as err:
						try:
							raise RuntimeError(('\nSCI Error :\n'
												'PySCF Error: {0:}\n\n').\
												format(err))
						except Exception as err_2:
							sys.stderr.write(str(err_2))
							raise
					# calculate spin
					s, mult = solver.spin_square(c, len(_exp.cas_idx), nelec)
					# check for correct spin
					if (float(_mol.spin) - s > 1.0e-05):
						try:
							raise RuntimeError(('\nSCI Error : spin contamination\n\n'
												'2*S + 1 = {0:.3f}\n\n').\
												format(mult))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					# e_corr
					e_corr = e - _calc.energy['hf']
				# sci dm
				if (_base and ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO'))):
					dm = solver.make_rdm1(c, len(_exp.cas_idx), nelec)
				else:
					dm = None
				#
				return e_corr, dm


		def ci(self, _mol, _calc, _exp, _mo, _base):
				""" cisd calc """
				# get integrals
				h1e, h2e, e_core = self.prepare(_mol, _calc, _exp, _mo)
				mol = gto.M(verbose=1)
				mol.incore_anyway = True
				mol.max_memory = _mol.max_memory
				if (_mol.spin == 0):
					hf = scf.RHF(mol)
				else:
					hf = scf.UHF(mol)
				hf.get_hcore = lambda *args: h1e
				hf._eri = h2e 
				# init ccsd
				if (_mol.spin == 0):
					cisd = ci.cisd.CISD(hf, mo_coeff=np.eye(len(_exp.cas_idx)), mo_occ=_calc.occup[_exp.cas_idx])
				else:
					cisd = ci.ucisd.UCISD(hf, mo_coeff=np.array((np.eye(len(_exp.cas_idx)), np.eye(len(_exp.cas_idx)))), \
											mo_occ=np.array((_calc.occup[_exp.cas_idx]>0, _calc.occup[_exp.cas_idx]==2), dtype=np.double))
				# settings
				cisd.conv_tol = 1.0e-10
				cisd.max_cycle = 500
				cisd.max_space = 10
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
						raise RuntimeError('\nCISD Error : no convergence\n\n')
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				# e_corr
				e_corr = cisd.e_corr
				# dm
				if (_base and ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO'))):
					dm = cisd.make_rdm1()
				else:
					dm = None
				#
				return e_corr, dm


		def cc(self, _mol, _calc, _exp, _mo, _base, _pt_corr=False):
				""" ccsd / ccsd(t) calc """
				# get integrals
				h1e, h2e, e_core = self.prepare(_mol, _calc, _exp, _mo)
				mol = gto.M(verbose=1)
				mol.incore_anyway = True
				mol.max_memory = _mol.max_memory
				if (_mol.spin == 0):
					hf = scf.RHF(mol)
				else:
					hf = scf.UHF(mol)
				hf.get_hcore = lambda *args: h1e
				hf._eri = h2e 
				# init ccsd
				if (_mol.spin == 0):
					ccsd = cc.ccsd.CCSD(hf, mo_coeff=np.eye(len(_exp.cas_idx)), mo_occ=_calc.occup[_exp.cas_idx])
				else:
					ccsd = cc.uccsd.UCCSD(hf, mo_coeff=np.array((np.eye(len(_exp.cas_idx)), np.eye(len(_exp.cas_idx)))), \
											mo_occ=np.array((_calc.occup[_exp.cas_idx]>0, _calc.occup[_exp.cas_idx]==2), dtype=np.double))
				# settings
				ccsd.conv_tol = 1.0e-10
				if (_base): ccsd.conv_tol_normt = 1.0e-10
				ccsd.max_cycle = 500
				ccsd.diis_space = 10
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
						raise RuntimeError('\nCCSD Error : no convergence\n\n')
					except Exception as err:
						sys.stderr.write(str(err))
						raise
				# e_corr
				e_corr = ccsd.e_corr
				# dm
				if (_base and (not _pt_corr) and ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO'))):
					ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
					dm = ccsd.make_rdm1()
				else:
					dm = None
				# calculate (t) correction
				if (_pt_corr): e_corr += ccsd.ccsd_t(eris=eris)
				#
				return e_corr, dm


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
					core_dm = np.dot(_orbs[:, _exp.core_idx], np.transpose(_orbs[:, _exp.core_idx])) * 2
					vj, vk = scf.hf.get_jk(_mol, core_dm)
					core_vhf = vj - vk * .5
					e_core = _mol.energy_nuc() + np.einsum('ij,ji', core_dm, _mol.hcore)
					e_core += np.einsum('ij,ji', core_dm, core_vhf) * .5
				else:
					e_core = _mol.energy_nuc()
					core_vhf = 0
				h1e_cas = reduce(np.dot, (np.transpose(_orbs[:, _exp.cas_idx]), \
										_mol.hcore + core_vhf, _orbs[:, _exp.cas_idx]))
				h2e_cas = ao2mo.incore.full(_mol.eri, _orbs[:, _exp.cas_idx])
				#
				return h1e_cas, h2e_cas, e_core


