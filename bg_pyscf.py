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
	from pyscf import gto, symm, scf, ao2mo, lo, ci, cc, mcscf, hci, fci
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
				# store e_tot, mo_occ, and orbsym
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


		def ref(self, _mol, _calc):
				""" reference calculation """
				# hf reference model
				if (_calc.exp_ref['METHOD'] == 'HF'):
					# number of electrons and orbitals
					_calc.no_act = _mol.nocc
					_calc.ne_act = int(np.sum(_calc.hf_mo_occ))
					if ('INPUT' in _calc.exp_ref):
						# number of electrons and orbitals
						if isinstance(_calc.exp_ref['INPUT'], dict):
							_calc.no_act = sum(_calc.exp_ref['INPUT'].values())
						elif isinstance(_calc.exp_ref['INPUT'], list):
							_calc.no_act = len(_calc.exp_ref['INPUT'])
						if isinstance(_calc.exp_ref['NELEC'], (tuple, list)):
							_calc.ne_act = _calc.exp_ref['NELEC'][0] + _calc.exp_ref['NELEC'][1]
						elif isinstance(_calc.exp_ref['NELEC'], int):
							_calc.ne_act = _calc.exp_ref['NELEC']
						# init model (no casci calc is performed at this stage)
						cas = mcscf.CASCI(_calc.hf, _calc.no_act, _calc.exp_ref['NELEC'])
						# set cas space
						if isinstance(_calc.exp_ref['INPUT'], dict):
							cas_space = np.array(mcscf.caslst_by_irrep(cas, _calc.hf.mo_coeff, \
													_calc.exp_ref['INPUT'], base=0))
						elif isinstance(_calc.exp_ref['INPUT'], list):
							cas_space = np.array(_calc.exp_ref['INPUT'])
						# number of active orbitals
						if (_calc.exp_type == 'occupied'):
							act_orbs = _mol.occ[np.where(np.in1d(_mol.occ, cas_space))]
						elif (_calc.exp_type == 'virtual'):
							act_orbs = _mol.virt[np.where(np.in1d(_mol.virt, cas_space))]
					else:
						# no cas space and number of active orbitals
						act_orbs = np.array([])
					# ref = hf
					ref_e_tot = _calc.hf.e_tot
					ref_mo_coeff = np.asarray(_calc.hf.mo_coeff, order='C')
				# casci/casscf reference model
				elif (_calc.exp_ref['METHOD'] in ['CASCI','CASSCF']):
					# number of electrons and orbitals
					if isinstance(_calc.exp_ref['INPUT'], dict):
						_calc.no_act = sum(_calc.exp_ref['INPUT'].values())
					elif isinstance(_calc.exp_ref['INPUT'], list):
						_calc.no_act = len(_calc.exp_ref['INPUT'])
					if isinstance(_calc.exp_ref['NELEC'], (tuple, list)):
						_calc.ne_act = _calc.exp_ref['NELEC'][0] + _calc.exp_ref['NELEC'][1]
					elif isinstance(_calc.exp_ref['NELEC'], int):
						_calc.ne_act = _calc.exp_ref['NELEC']
					# init model
					if (_calc.exp_ref['METHOD'] == 'CASSCF'):
						cas = mcscf.CASSCF(_calc.hf, _calc.no_act, _calc.exp_ref['NELEC'])
					elif (_calc.exp_ref['METHOD'] == 'CASCI'):
						cas = mcscf.CASCI(_calc.hf, _calc.no_act, _calc.exp_ref['NELEC'])
					# set cas space
					if isinstance(_calc.exp_ref['INPUT'], dict):
						cas_space = np.array(mcscf.caslst_by_irrep(cas, _calc.hf.mo_coeff, \
												_calc.exp_ref['INPUT'], base=0))
					elif isinstance(_calc.exp_ref['INPUT'], list):
						cas_space = np.array(_calc.exp_ref['INPUT'])
					# number of active orbitals
					if (_calc.exp_type == 'occupied'):
						act_orbs = _mol.occ[np.where(np.in1d(_mol.occ, cas_space))]
					elif (_calc.exp_type == 'virtual'):
						act_orbs = _mol.virt[np.where(np.in1d(_mol.virt, cas_space))]
					# select MOs
					mo = cas.sort_mo(cas_space, base=0)
					# frozen core
					cas.frozen = _mol.ncore
					# debug print
					if (_mol.verbose_prt):
						print('cas_space = {0:} , act_orbs = {1:} , no_act = {2:} , ne_act = {3:}'.\
								format(cas_space,act_orbs,_calc.no_act,_calc.ne_act))
					# perform cas calc
					cas.conv_tol = 1.0e-10
					cas.natorb = True
					ref_e_tot = cas.kernel(mo)[0]
					ref_mo_coeff = cas.mo_coeff
				#
				return act_orbs, ref_e_tot, ref_mo_coeff


		def trans_main(self, _mol, _calc, _exp):
				""" determine main transformation matrices """
				# set frozen list
				frozen = list(range(_mol.ncore)) if (_mol.spin == 0) else [list(range(_mol.ncore)),list(range(_mol.ncore))]
				# zeroth-order energy
				if (_calc.exp_base['METHOD'] is None):
					_calc.e_zero = 0.0
				elif (_calc.exp_base['METHOD'] == 'HCI'):
					# init solver
					hci_solver = hci.SCI(_mol)
					hci_solver.conv_tol = 1.0e-10
					hci_solver.max_cycle = 500
					hci_solver.max_space = 10
					hci_solver.ci_coeff_cutoff = 1e-3
					hci_solver.select_cutoff = 1e-3
					hci_solver.max_memory = _mol.max_memory
					# set core and cas spaces
					if (_calc.exp_type == 'occupied'):
						_exp.core_idx, _exp.cas_idx = self.core_cas_spaces(_mol, _exp, np.array(range(_mol.ncore, _mol.nocc)))
					if (_calc.exp_type == 'virtual'):
						_exp.core_idx, _exp.cas_idx = self.core_cas_spaces(_mol, _exp, np.array(range(_mol.nocc, _mol.norb)))
					# get integrals
					h1e, h2e = self.prepare(_mol, _calc, _exp, _calc.ref_mo_coeff)
					# nelec_cas
					nelec_cas = (_mol.nelec[0] - len(_exp.core_idx), _mol.nelec[1] - len(_exp.core_idx))
					# fix spin if non-singlet
					if (_mol.spin > 0):
						sz = abs(nelec_cas[0]-nelec_cas[1]) * .5
						fci.addons.fix_spin(hci_solver, ss=sz * (sz + 1.))
					try:
						e, c = hci_solver.kernel(h1e, h2e, len(_exp.cas_idx), nelec_cas, ecore=_exp.e_core)
					except Exception as err:
						try:
							raise RuntimeError(('\nHCI (main int-trans) Error :\n'
												'PySCF Error: {0:}\n\n').\
												format(err))
						except Exception as err_2:
							sys.stderr.write(str(err_2))
							raise
					# calculate spin
					s, mult = hci_solver.spin_square(c[0], len(_exp.cas_idx), nelec_cas)
					# check for correct spin
					if (int(round(s)) != _mol.spin):
						try:
							raise RuntimeError(('\nHCI (main int-trans) Error : wrong spin\n'
												'2*S + 1 = {0:.3f}\n\n').\
												format(mult))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					# e_zero
					_calc.e_zero = e[0] - _calc.ref_e_tot
				elif (_calc.exp_base['METHOD'] == 'CISD'):
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
					ccsd.conv_tol_normt = 1.0e-10
					ccsd.max_cycle = 500
					ccsd.diis_space = 10
					ccsd.mol.incore_anyway = True
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
							raise RuntimeError('\nCCSD-1 (main int-trans) Error : no convergence\n\n')
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					if ((_calc.exp_occ == 'NO') or (_calc.exp_virt == 'NO')): dm = ccsd.make_rdm1()
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
				# add (t) correction
				if (_calc.exp_base['METHOD'] == 'CCSD(T)'):
					if ((_calc.exp_occ == 'REF') and (_calc.exp_virt == 'REF')):
						_calc.e_zero += ccsd.ccsd_t(eris=eris)
					else:
						h1e = reduce(np.dot, (np.transpose(_calc.trans_mat), _mol.hcore, _calc.trans_mat))
						h2e = ao2mo.kernel(_mol, _calc.trans_mat)
						mol = gto.M(verbose=1)
						if (_mol.spin == 0):
							hf = scf.RHF(mol)
						else:
							hf = scf.UHF(mol)
						hf.get_hcore = lambda *args: h1e
						hf._eri = h2e 
						if (_mol.spin == 0):
							ccsd_2 = cc.ccsd.CCSD(hf, mo_coeff=np.eye(_mol.norb), mo_occ=_calc.hf_mo_occ)
						else:
							ccsd_2 = cc.uccsd.UCCSD(hf, mo_coeff=np.array((np.eye(_mol.norb), np.eye(_mol.norb))), \
													mo_occ=np.array((_calc.hf_mo_occ>0, _calc.hf_mo_occ==2), dtype=np.double))
						ccsd_2.max_cycle = 500
						ccsd_2.diis_space = 10
						ccsd_2.mol.incore_anyway = True
						ccsd_2.frozen = frozen
						eris = ccsd_2.ao2mo()
						for i in list(range(0, 12, 2)):
							ccsd_2.diis_start_cycle = i
							try:
								ccsd_2.kernel(eris=eris)
							except sp.linalg.LinAlgError: pass
							if (ccsd_2.converged): break
						if (not ccsd_2.converged):
							try:
								raise RuntimeError('\nCCSD-2 (main int-trans) Error : no convergence\n\n')
							except Exception as err:
								sys.stderr.write(str(err))
								raise
						_calc.e_zero += ccsd_2.ccsd_t(eris=eris)
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


		def core_cas_spaces(self, _mol, _exp, _tup):
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
						_exp.core_vhf = scf.hf.get_veff(_mol, core_dm)
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
					# initial guess
					nelec_cas = (_mol.nelec[0] - len(_exp.core_idx), _mol.nelec[1] - len(_exp.core_idx))
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
					try:
						e_cas, c_cas = solver_cas.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), \
															nelec_cas, ci0=hf_as_civec, orbsym=orbsym)
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
				if (_calc.exp_base['METHOD'] is None):
					e_corr = (e_cas + _exp.e_core) - _calc.ref_e_tot
#					if (_exp.order < _exp.max_order): e_corr += (e_cas + _exp.e_core) - _calc.ref_e_tot + 0.001 * np.random.random_sample()
				elif (_calc.exp_base['METHOD'] == 'HCI'):
					# init solver
					solver_base = hci.SCI()
					# fci settings
					solver_base.conv_tol = max(_exp.thres, 1.0e-10)
					solver_base.ci_coeff_cutoff = 1e-3
					solver_base.select_cutoff = 1e-3
					solver_base.max_cycle = 500
					solver_base.max_space = 10
					solver_base.max_memory = _mol.max_memory
					# fix spin if non-singlet
					if (_mol.spin > 0):
						sz = abs(nelec_cas[0]-nelec_cas[1]) * .5
						fci.addons.fix_spin(solver_base, ss=sz * (sz + 1.))
					try:
						e_base, c_base = solver_base.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx), nelec_cas)
					except Exception as err:
						try:
							raise RuntimeError(('\nCAS-HCI Error :\n'
												'core_idx = {0:} , cas_idx = {1:}\n'
												'PySCF Error: {2:}\n\n').\
												format(_exp.core_idx, _exp.cas_idx, err))
						except Exception as err_2:
							sys.stderr.write(str(err_2))
							raise
					# calculate spin
					base_s, base_mult = solver_base.spin_square(c_base[0], len(_exp.cas_idx), nelec_cas)
					# check for correct spin
					if (int(round(base_s)) != _mol.spin):
						try:
							raise RuntimeError(('\nCAS-HCI Error : wrong spin\n'
												'2*S + 1 = {0:.3f}\n'
												'core_idx = {1:} , cas_idx = {2:}\n\n').\
												format(base_mult, _exp.core_idx, _exp.cas_idx))
						except Exception as err:
							sys.stderr.write(str(err))
							raise
					e_corr = e_cas - e_base[0]
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
				cas_hf.e_tot = _calc.hf_e_tot - _e_core
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


