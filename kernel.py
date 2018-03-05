#!/usr/bin/env python
# -*- coding: utf-8 -*

""" kernel.py: kernel module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.10'
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


def hcore_eri(mol):
		""" get core hamiltonian and AO eris """
		hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
		eri = mol.intor('int2e_sph', aosym=4)
		return hcore, eri


def hf(mol, calc):
		""" hartree-fock calculation """
		# perform hf calc
		hf = scf.RHF(mol)
		hf.conv_tol = 1.0e-10
		hf.max_cycle = 500
		# fixed occupation
		hf.irrep_nelec = mol.irrep_nelec
		# perform hf calc
		for i in list(range(0, 12, 2)):
			hf.diis_start_cycle = i
			try:
				hf.kernel()
			except sp.linalg.LinAlgError: pass
			if hf.converged: break
		if not hf.converged:
			try:
				raise RuntimeError('\nHF Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# determine dimensions
		mol.norb, mol.nocc, mol.nvirt = _dim(hf, calc)
		# store energy, occupation, and orbsym
		e_hf = hf.e_tot
		occup = hf.mo_occ
		orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
		return hf, e_hf, occup, orbsym, np.asarray(hf.mo_coeff, order='C')


def _dim(hf, calc):
		""" determine dimensions """
		# occupied and virtual lists
		if calc.exp_type == 'occupied':
			occ = np.where(hf.mo_occ == 2.)[0]
			virt = np.where(hf.mo_occ < 2.)[0]
		elif calc.exp_type == 'virtual':
			occ = np.where(hf.mo_occ > 0.)[0]
			virt = np.where(hf.mo_occ == 0.)[0]
		# nocc, nvirt, and norb
		nocc = len(occ)
		nvirt = len(virt)
		norb = nocc + nvirt
		return norb, nocc, nvirt


def active(mol, calc):
		""" set active space """
		# reference and expansion spaces
		if calc.exp_type == 'occupied':
			ref_space = np.array(range(mol.nocc, mol.norb))
			exp_space = np.array(range(mol.ncore, mol.nocc))
		elif calc.exp_type == 'virtual':
			ref_space = np.array(range(mol.ncore, mol.nocc))
			exp_space = np.array(range(mol.nocc, mol.norb))
		# hf reference model
		if calc.exp_ref['METHOD'] == 'HF':
			# no active space
			no_act = len(ref_space)
		# casci/casscf reference model
		elif calc.exp_ref['METHOD'] in ['CASCI','CASSCF']:
			# active orbitals
			if calc.exp_type == 'occupied':
				no_act = np.count_nonzero(np.array(calc.exp_ref['ACTIVE']) < mol.nocc) + len(ref_space)
			elif calc.exp_type == 'virtual':
				no_act = np.count_nonzero(np.array(calc.exp_ref['ACTIVE']) >= mol.nocc) + len(ref_space)
			# sanity checks
			assert(np.count_nonzero(np.array(calc.exp_ref['ACTIVE']) < mol.ncore) == 0)
			assert(float(calc.exp_ref['NELEC'][0] + calc.exp_ref['NELEC'][1]) <= np.sum(calc.hf.mo_occ[calc.exp_ref['ACTIVE']]))
			# identical to hf ref?
			if no_act == len(ref_space):
				try:
					raise RuntimeError('\nCAS Error : choice of CAS returns HF solution\n\n')
				except Exception as err:
					sys.stderr.write(str(err))
					raise
		return ref_space, exp_space, no_act


def _mf(mol, calc, mo):
		""" calculate mean-field energy """
		mo_a = mo[:, np.where(calc.occup > 0.)[0]]
		mo_b = mo[:, np.where(calc.occup == 2.)[0]]
		dm_a = np.dot(mo_a, np.transpose(mo_a))
		dm_b = np.dot(mo_b, np.transpose(mo_b))
		dm = np.array((dm_a, dm_b))
		vj, vk = scf.hf.get_jk(mol, dm)
		vhf = vj[0] + vj[1] - vk
		e_mf = mol.energy_nuc()
		e_mf += np.einsum('ij,ij', mol.hcore.conj(), dm[0] + dm[1])
		e_mf += (np.einsum('ij,ji', vhf[0], dm[0]) + np.einsum('ij,ji', vhf[1], dm[1])) * .5
		return e_mf


def ref(mol, calc, exp):
		""" calculate reference energy and mo coefficients """
		# set core and cas spaces
		if calc.exp_type == 'occupied':
			exp.core_idx, exp.cas_idx = list(range(mol.nocc)), calc.ref_space.tolist()
		elif calc.exp_type == 'virtual':
			exp.core_idx, exp.cas_idx = list(range(mol.ncore)), calc.ref_space.tolist()
		# sort mo coefficients
		if calc.exp_ref['METHOD'] in ['CASCI','CASSCF']:
			# core region
			ncore_elec = mol.nelectron - (calc.exp_ref['NELEC'][0] + calc.exp_ref['NELEC'][1])
			assert(ncore_elec % 2 == 0)
			ncore_orb = ncore_elec // 2
			# divide into core-cas-virtual
			idx = np.asarray([i for i in range(mol.norb) if i not in calc.exp_ref['ACTIVE']])
			mo = np.hstack((calc.mo[:, idx[:ncore_orb]], calc.mo[:, calc.exp_ref['ACTIVE']], calc.mo[:, idx[ncore_orb:]]))
			calc.mo = np.asarray(mo, order='C')
			# set ref energies equal to hf energies
			e_ref = e_refbase = 0.0
			# casscf mo
			if calc.exp_ref['METHOD'] == 'CASSCF': calc.mo = _casscf(mol, calc, exp, calc.exp_model['METHOD'])
		else:
			# exp model
			e_ref = corr(mol, calc, exp, calc.exp_model['METHOD'])
			# exp base
			if calc.exp_base['METHOD'] is None:
				e_refbase = 0.0
			else:
				if np.abs(e_ref) > 1.0e-10:
					e_refbase = e_ref
				else:
					e_refbase = corr(mol, calc, exp, calc.exp_base['METHOD'])
		return e_ref + calc.energy['hf'], e_refbase + calc.energy['hf'], calc.mo


def corr(mol, calc, exp, method):
		""" calculate correlation energy """
		# fci calc
		if method == 'FCI':
			e_corr = _fci(mol, calc, exp, calc.mo, False)
		# sci base
		elif method == 'SCI':
			e_corr, _ = _sci(mol, calc, exp, calc.mo, False)
		# cisd calc
		elif method == 'CISD':
			e_corr, _ = _ci(mol, calc, exp, calc.mo, False)
		# ccsd / ccsd(t) calc
		elif method in ['CCSD','CCSD(T)']:
			e_corr, _ = _cc(mol, calc, exp, calc.mo, False, (method == 'CCSD(T)'))
		return e_corr


def base(mol, calc, exp):
		""" calculate base energy and mo coefficients """
		# set core and cas spaces
		exp.core_idx, exp.cas_idx = core_cas(mol, exp, calc.exp_space)
		# zeroth-order energy
		if calc.exp_base['METHOD'] is None:
			e_base = 0.0
		# cisd base
		elif calc.exp_base['METHOD'] == 'CISD':
			e_base, dm = _ci(mol, calc, exp, calc.mo, True)
			if mol.spin > 0 and dm is not None: dm = dm[0] + dm[1]
		# ccsd / ccsd(t) base
		elif calc.exp_base['METHOD'] in ['CCSD','CCSD(T)']:
			e_base, dm = _cc(mol, calc, exp, calc.mo, True, \
										(calc.exp_base['METHOD'] == 'CCSD(T)') and \
										((calc.exp_occ == 'REF') and (calc.exp_virt == 'REF')))
			if mol.spin > 0 and dm is not None: dm = dm[0] + dm[1]
		# sci base
		elif calc.exp_base['METHOD'] == 'SCI':
			e_base, dm = _sci(mol, calc, exp, calc.mo, True)
		# copy mo
		mo = np.copy(calc.mo)
		# occ-occ block (local or NOs)
		if calc.exp_occ != 'REF':
			if calc.exp_occ == 'NO':
				occup, no = symm.eigh(dm[:(mol.nocc-mol.ncore), :(mol.nocc-mol.ncore)], calc.orbsym[mol.ncore:mol.nocc])
				mo[:, mol.ncore:mol.nocc] = np.dot(calc.mo[:, mol.ncore:mol.nocc], no[:, ::-1])
			elif calc.exp_occ == 'PM':
				mo[:, mol.ncore:mol.nocc] = lo.PM(mol, calc.mo[:, mol.ncore:mol.nocc]).kernel()
			elif calc.exp_occ == 'FB':
				mo[:, mol.ncore:mol.nocc] = lo.Boys(mol, calc.mo[:, mol.ncore:mol.nocc]).kernel()
			elif calc.exp_occ in ['IBO-1','IBO-2']:
				iao = lo.iao.iao(mol, calc.mo[:, mol.core:mol.nocc])
				if calc.exp_occ == 'IBO-1':
					iao = lo.vec_lowdin(iao, calc.hf.get_ovlp())
					mo[:, mol.ncore:mol.nocc] = lo.ibo.ibo(mol, calc.mo[:, mol.ncore:mol.nocc], iao)
				elif calc.exp_occ == 'IBO-2':
					mo[:, mol.ncore:mol.nocc] = lo.ibo.PM(mol, calc.mo[:, mol.ncore:mol.nocc], iao).kernel()
		# virt-virt block (local or NOs)
		if calc.exp_virt != 'REF':
			if calc.exp_virt == 'NO':
				occup, no = symm.eigh(dm[-mol.nvirt:, -mol.nvirt:], calc.orbsym[mol.nocc:])
				mo[:, mol.nocc:] = np.dot(calc.mo[:, mol.nocc:], no[:, ::-1])
			elif calc.exp_virt == 'PM':
				mo[:, mol.nocc:] = lo.PM(mol, calc.mo[:, mol.nocc:]).kernel()
			elif calc.exp_virt == 'FB':
				mo[:, mol.nocc:] = lo.Boys(mol, calc.mo[:, mol.nocc:]).kernel()
		# (t) correction for NOs
		if calc.exp_occ == 'NO' or calc.exp_virt == 'NO':
			if calc.exp_base['METHOD'] == 'CCSD(T)':
				e_base, dm = _cc(mol, calc, exp, mo, False, True)
			elif calc.exp_base['METHOD'] == 'SCI':
				e_base, dm = _sci(mol, calc, exp, mo, False)
		return e_base, mo


def _casscf(mol, calc, exp, method):
		""" casscf calc """
		# casscf ref
		cas = mcscf.CASSCF(calc.hf, len(calc.exp_ref['ACTIVE']), calc.exp_ref['NELEC'])
		if abs(calc.exp_ref['NELEC'][0]-calc.exp_ref['NELEC'][1]) == 0:
			if method == 'FCI':
				cas.fcisolver = fci.direct_spin0_symm.FCI(mol)
			elif method == 'SCI':
				cas.fcisolver = fci.select_ci_spin0_symm.SCI(mol)
		else:
			if method == 'FCI':
				cas.fcisolver = fci.direct_spin1_symm.FCI(mol)
			elif method == 'SCI':
				cas.fcisolver = fci.select_ci_symm.SCI(mol)
		cas.fcisolver.conv_tol = 1.0e-10
		cas.conv_tol = 1.0e-10
		cas.max_stepsize = .01
		cas.max_cycle_micro = 1
		# wfnsym
		cas.fcisolver.wfnsym = calc.wfnsym
		# frozen
		cas.frozen = (mol.nelectron - (calc.exp_ref['NELEC'][0] + calc.exp_ref['NELEC'][1])) // 2
		# verbose print
		if mol.verbose: cas.verbose = 4
		# fix spin if non-singlet
		if mol.spin > 0:
			sz = abs(calc.exp_ref['NELEC'][0]-calc.exp_ref['NELEC'][1]) * .5
			cas.fix_spin_(ss=sz * (sz + 1.))
		# run casscf calc
		cas.kernel(calc.mo)
		if not cas.converged:
			try:
				raise RuntimeError('\nCASSCF Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# calculate spin
		s, mult = cas.fcisolver.spin_square(cas.ci, len(calc.exp_ref['ACTIVE']), calc.exp_ref['NELEC'])
		# check for correct spin
		if (mol.spin + 1) - mult > 1.0e-05:
			try:
				raise RuntimeError(('\nCASSCF Error : spin contamination\n'
									'2*S + 1 = {0:.3f}\n\n').\
									format(mult))
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# save mo
		mo = np.asarray(cas.mo_coeff, order='C')
		return mo


def _fci(mol, calc, exp, mo, base):
		""" fci calc """
		# init fci solver
		if mol.spin == 0:
			solver = fci.direct_spin0_symm.FCI(mol)
		else:
			solver = fci.direct_spin1_symm.FCI(mol)
		# settings
		solver.conv_tol = 1.0e-10
		solver.max_cycle = 500
		solver.max_space = 10
		solver.davidson_only = True
		# wfnsym
		solver.wfnsym = calc.wfnsym
		# get integrals and core energy
		h1e, h2e, e_core = _prepare(mol, calc, exp, mo)
		# electrons
		nelec = (mol.nelec[0] - len(exp.core_idx), mol.nelec[1] - len(exp.core_idx))
		# orbital symmetry
		solver.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo[:, exp.cas_idx])
		# fix spin if non-singlet
		if mol.spin > 0:
			sz = abs(nelec[0]-nelec[1]) * .5
			fci.addons.fix_spin(solver, ss=sz * (sz + 1.))
		# init guess (does it exist?)
		try:
			ci0 = fci.addons.symm_initguess(len(exp.cas_idx), nelec, orbsym=solver.orbsym, wfnsym=solver.wfnsym)
			e_corr = None
		except Exception:
			e_corr = 0.0
		# perform calc
		if e_corr is None:
			e, c = solver.kernel(h1e, h2e, len(exp.cas_idx), nelec, ecore=e_core)
			# calculate spin
			s, mult = solver.spin_square(c, len(exp.cas_idx), nelec)
			# check for correct spin
			if (mol.spin + 1) - mult > 1.0e-05:
				print('s = {0:} and mult = {1:}'.format(s, mult))
				try:
					raise RuntimeError(('\nFCI Error : spin contamination\n\n'
										'2*S + 1 = {0:.6f}\n'
										'core_idx = {1:} , cas_idx = {2:}\n\n').\
										format(mult, exp.core_idx, exp.cas_idx))
				except Exception as err:
					sys.stderr.write(str(err))
					raise
			# e_corr
			e_corr = e - calc.energy['hf']
#			if exp.order < exp.max_order: e_corr += np.float64(0.001) * np.random.random_sample()
		return e_corr


def _sci(mol, calc, exp, mo, base):
		""" sci calc """
		# init sci solver
		if mol.spin == 0:
			solver = fci.select_ci_spin0_symm.SCI(mol)
		else:
			solver = fci.select_ci_symm.SCI(mol)
		# settings
		solver.conv_tol = 1.0e-10
		solver.max_cycle = 500
		solver.max_space = 10
		solver.davidson_only = True
		# wfnsym
		solver.wfnsym = calc.wfnsym
		# get integrals and core energy
		h1e, h2e, e_core = _prepare(mol, calc, exp, mo)
		# electrons
		nelec = (mol.nelec[0] - len(exp.core_idx), mol.nelec[1] - len(exp.core_idx))
		# orbital symmetry
		solver.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo[:, exp.cas_idx])
		# fix spin if non-singlet
		if mol.spin > 0:
			sz = abs(nelec[0]-nelec[1]) * .5
			fci.addons.fix_spin(solver, ss=sz * (sz + 1.))
		# init guess (does it exist?)
		try:
			ci0 = fci.addons.symm_initguess(len(exp.cas_idx), nelec, orbsym=solver.orbsym, wfnsym=solver.wfnsym)
			e_corr = None
		except Exception:
			e_corr = 0.0
		# perform calc
		if e_corr is None:
			# calculate sci energy
			e, c = solver.kernel(h1e, h2e, len(exp.cas_idx), nelec, ecore=e_core)
			# calculate spin
			s, mult = solver.spin_square(c, len(exp.cas_idx), nelec)
			# check for correct spin
			if (mol.spin + 1) - mult > 1.0e-05:
				try:
					raise RuntimeError(('\nSCI Error : spin contamination\n\n'
										'2*S + 1 = {0:.3f}\n\n').\
										format(mult))
				except Exception as err:
					sys.stderr.write(str(err))
					raise
			# e_corr
			e_corr = e - calc.energy['hf']
		# sci dm
		if base:
			if calc.exp_occ == 'NO' or calc.exp_virt == 'NO':
				dm = solver.make_rdm1(c, len(exp.cas_idx), nelec)
		else:
			dm = None
		return e_corr, dm


def _ci(mol, calc, exp, mo, base):
		""" cisd calc """
		# get integrals
		h1e, h2e, e_core = _prepare(mol, calc, exp, mo)
		mol_tmp = gto.M(verbose=1)
		mol_tmp.incore_anyway = True
		mol_tmp.max_memory = mol.max_memory
		if mol.spin == 0:
			hf = scf.RHF(mol_tmp)
		else:
			hf = scf.UHF(mol_tmp)
		hf.get_hcore = lambda *args: h1e
		hf._eri = h2e 
		# init ccsd
		if mol.spin == 0:
			cisd = ci.cisd.CISD(hf, mo_coeff=np.eye(len(exp.cas_idx)), mo_occ=calc.occup[exp.cas_idx])
		else:
			cisd = ci.ucisd.UCISD(hf, mo_coeff=np.array((np.eye(len(exp.cas_idx)), np.eye(len(exp.cas_idx)))), \
									mo_occ=np.array((calc.occup[exp.cas_idx] > 0., calc.occup[exp.cas_idx] == 2.), dtype=np.double))
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
			if cisd.converged: break
		if not cisd.converged:
			try:
				raise RuntimeError('\nCISD Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# e_corr
		e_corr = cisd.e_corr
		# dm
		if base:
			if calc.exp_occ == 'NO' or calc.exp_virt == 'NO':
				dm = cisd.make_rdm1()
		else:
			dm = None
		return e_corr, dm


def _cc(mol, calc, exp, mo, base, pt=False):
		""" ccsd / ccsd(t) calc """
		# get integrals
		h1e, h2e, e_core = _prepare(mol, calc, exp, mo)
		mol_tmp = gto.M(verbose=1)
		mol_tmp.incore_anyway = True
		mol_tmp.max_memory = mol.max_memory
		if mol.spin == 0:
			hf = scf.RHF(mol_tmp)
		else:
			hf = scf.UHF(mol_tmp)
		hf.get_hcore = lambda *args: h1e
		hf._eri = h2e 
		# init ccsd
		if mol.spin == 0:
			ccsd = cc.ccsd.CCSD(hf, mo_coeff=np.eye(len(exp.cas_idx)), mo_occ=calc.occup[exp.cas_idx])
		else:
			ccsd = cc.uccsd.UCCSD(hf, mo_coeff=np.array((np.eye(len(exp.cas_idx)), np.eye(len(exp.cas_idx)))), \
									mo_occ=np.array((calc.occup[exp.cas_idx] > 0., calc.occup[exp.cas_idx] == 2.), dtype=np.double))
		# settings
		ccsd.conv_tol = 1.0e-10
		if base: ccsd.conv_tol_normt = 1.0e-10
		ccsd.max_cycle = 500
		ccsd.diis_space = 10
		eris = ccsd.ao2mo()
		# calculate ccsd energy
		for i in list(range(0, 12, 2)):
			ccsd.diis_start_cycle = i
			try:
				ccsd.kernel(eris=eris)
			except sp.linalg.LinAlgError: pass
			if ccsd.converged: break
		if not ccsd.converged:
			try:
				raise RuntimeError('\nCCSD Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# e_corr
		e_corr = ccsd.e_corr
		# dm
		if base and not pt:
			if calc.exp_occ == 'NO' or calc.exp_virt == 'NO':
				ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
				dm = ccsd.make_rdm1()
		else:
			dm = None
		# calculate (t) correction
		if pt:
			if mol.spin == 0:
				if ccsd.t1.shape[1] > 0:
					e_corr += ccsd.ccsd_t(eris=eris)
			else:
				if eris.focka.shape[0] - eris.nocca > 0:
					e_corr += ccsd.ccsd_t(eris=eris)
		return e_corr, dm


def core_cas(mol, exp, tup):
		""" define core and cas spaces """
		cas_idx = sorted(exp.incl_idx + sorted(tup.tolist()))
		core_idx = sorted(list(set(range(mol.nocc)) - set(cas_idx)))
		return core_idx, cas_idx


def _prepare(mol, calc, exp, orbs):
		""" generate input for correlated calculation """
		# extract cas integrals and calculate core energy
		if len(exp.core_idx) > 0:
			core_dm = np.dot(orbs[:, exp.core_idx], np.transpose(orbs[:, exp.core_idx])) * 2
			vj, vk = scf.hf.get_jk(mol, core_dm)
			core_vhf = vj - vk * .5
			e_core = mol.energy_nuc() + np.einsum('ij,ji', core_dm, mol.hcore)
			e_core += np.einsum('ij,ji', core_dm, core_vhf) * .5
		else:
			e_core = mol.energy_nuc()
			core_vhf = 0
		h1e_cas = reduce(np.dot, (np.transpose(orbs[:, exp.cas_idx]), \
								mol.hcore + core_vhf, orbs[:, exp.cas_idx]))
		h2e_cas = ao2mo.incore.full(mol.eri, orbs[:, exp.cas_idx])
		return h1e_cas, h2e_cas, e_core


