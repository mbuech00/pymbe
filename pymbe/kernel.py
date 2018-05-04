#!/usr/bin/env python
# -*- coding: utf-8 -*

""" kernel.py: kernel module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import numpy as np
import scipy as sp
from functools import reduce
from mpi4py import MPI
from pyscf import gto, symm, scf, ao2mo, lo, ci, cc, mcscf, fci


def ao_ints(mol, calc):
		""" get AO integrals """
		# core hamiltonian
		hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
		# electron repulsion ints
		eri = mol.intor('int2e_sph', aosym=4)
		# dipole integrals with gauge origin at (0,0,0)
		if 'dipmom' in calc.property:
			with mol.with_common_orig((0,0,0)):
				dipmom = mol.intor_symmetric('int1e_r', comp=3)
		else:
			dipmom = None
		return hcore, eri, dipmom


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
		# convergence check
		if not hf.converged:
			try:
				raise RuntimeError('\nHF Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# dipole moment
		tot_dipmom = hf.dip_moment(unit='au', verbose=0)
		# nuclear dipole moment
		charges = mol.atom_charges()
		coords  = mol.atom_coords()
		nuc_dipmom = np.einsum('i,ix->x', charges, coords)
		# electronic dipole moment
		elec_dipmom = nuc_dipmom - tot_dipmom
		# determine dimensions
		mol.norb, mol.nocc, mol.nvirt = _dim(hf, calc)
		# store energy, occupation, and orbsym
		e_hf = hf.e_tot
		occup = hf.mo_occ
		orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
		# wave function symmetry
		wfnsym = scf.hf_symm.get_wfnsym(hf, mo_coeff=hf.mo_coeff, mo_occ=hf.mo_occ)
		# sanity check
		if wfnsym != calc.wfnsym and calc.ref['METHOD'] == 'HF':
			try:
				raise RuntimeError('\nHF Error : wave function symmetry ({0:}) different from requested symmetry ({1:})\n\n'.\
									format(symm.irrep_id2name(mol.groupname, wfnsym), symm.irrep_id2name(mol.groupname, calc.wfnsym)))
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		return hf, np.asscalar(e_hf), elec_dipmom, occup, orbsym, np.asarray(hf.mo_coeff, order='C')


def _dim(hf, calc):
		""" determine dimensions """
		# occupied and virtual lists
		if calc.typ == 'occupied':
			occ = np.where(hf.mo_occ == 2.)[0]
			virt = np.where(hf.mo_occ < 2.)[0]
		elif calc.typ == 'virtual':
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
		if calc.typ == 'occupied':
			ref_space = np.array(range(mol.nocc, mol.norb))
			exp_space = np.array(range(mol.ncore, mol.nocc))
		elif calc.typ == 'virtual':
			ref_space = np.array(range(mol.ncore, mol.nocc))
			exp_space = np.array(range(mol.nocc, mol.norb))
		# hf reference model
		if calc.ref['METHOD'] == 'HF':
			# no active space
			ne_act = (0, 0)
			no_exp = no_act = 0
		# casci/casscf reference model
		elif calc.ref['METHOD'] in ['CASCI','CASSCF']:
			if calc.ref['ACTIVE'] == 'MANUAL':
				# active electrons
				ne_act = calc.ref['NELEC']
				# active orbitals
				no_act = len(calc.ref['SELECT'])
				# expansion space orbitals
				if calc.typ == 'occupied':
					no_exp = np.count_nonzero(np.array(calc.ref['SELECT']) < mol.nocc)
				elif calc.typ == 'virtual':
					no_exp = np.count_nonzero(np.array(calc.ref['SELECT']) >= mol.nocc)
				# sanity checks
				assert(np.count_nonzero(np.array(calc.ref['SELECT']) < mol.ncore) == 0)
				assert(float(ne_act[0] + ne_act[1]) <= np.sum(calc.hf.mo_occ[calc.ref['SELECT']]))
			else:
				from pyscf.mcscf import avas
				# avas
				no_avas, ne_avas = avas.avas(calc.hf, calc.ref['AO_LABELS'], canonicalize=True, \
												verbose=4 if mol.verbose else None, ncore=mol.ncore)[:2]
				# convert ne_avas to native python type
				ne_avas = np.asscalar(ne_avas)
				# active electrons
				ne_a = (ne_avas + mol.spin) // 2
				ne_b = ne_avas - ne_a
				ne_act = (ne_a, ne_b)
				# active orbitals
				no_act = no_avas
				# expansion space orbitals
				nocc_avas = ne_a
				nvirt_avas = no_act - nocc_avas
				if calc.typ == 'occupied':
					no_exp = nocc_avas
				elif calc.typ == 'virtual':
					no_exp = nvirt_avas
				# sanity checks
				assert(nocc_avas <= (mol.nocc - mol.ncore))
				assert(float(ne_act[0] + ne_act[1]) <= np.sum(calc.hf.mo_occ[mol.ncore:]))
			# identical to hf ref?
			if no_exp == 0:
				try:
					raise RuntimeError('\nCAS Error : choice of CAS returns HF solution\n\n')
				except Exception as err:
					sys.stderr.write(str(err))
					raise
			if mol.verbose:
				print(' ACTIVE: ne_act = {0:} , no_act = {1:} , no_exp = {2:}'.format(ne_act, no_act, no_exp))
		return ref_space, exp_space, no_exp, no_act, ne_act


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
		if calc.typ == 'occupied':
			exp.core_idx, exp.cas_idx = list(range(mol.nocc)), calc.ref_space.tolist()
		elif calc.typ == 'virtual':
			exp.core_idx, exp.cas_idx = list(range(mol.ncore)), calc.ref_space.tolist()
		# sort mo coefficients
		if calc.ref['METHOD'] in ['CASCI','CASSCF']:
			if calc.ref['ACTIVE'] == 'MANUAL':
				# inactive region
				inact_elec = mol.nelectron - (calc.ne_act[0] + calc.ne_act[1])
				assert(inact_elec % 2 == 0)
				inact_orb = inact_elec // 2
				# divide into inactive-active-virtual
				idx = np.asarray([i for i in range(mol.norb) if i not in calc.ref['SELECT']])
				mo = np.hstack((calc.mo[:, idx[:inact_orb]], calc.mo[:, calc.ref['SELECT']], calc.mo[:, idx[inact_orb:]]))
				calc.mo = np.asarray(mo, order='C')
			else:
				from pyscf.mcscf import avas
				calc.mo = avas.avas(calc.hf, calc.ref['AO_LABELS'], canonicalize=True, ncore=mol.ncore)[2]
			# set properties equal to hf values
			e_ref = e_ref_base = 0.0
			dipmom_ref = np.zeros(3, dtype=np.float64)
			# casscf mo
			if calc.ref['METHOD'] == 'CASSCF': calc.mo = _casscf(mol, calc, exp)
		else:
			# exp model
			e_ref, dipmom_ref = main(mol, calc, exp, calc.model['METHOD'])
			# exp base
			if calc.base['METHOD'] is None:
				e_ref_base = 0.0
			else:
				if np.abs(e_ref) < 1.0e-10:
					e_ref_base = e_ref
				else:
					e_ref_base = main(mol, calc, exp, calc.base['METHOD'])[0]
		if mol.verbose:
			print(' REF: core = {0:} , cas = {1:} , e_ref = {2:.10f} , e_ref_base = {3:.10f}'.format(exp.core_idx, exp.cas_idx, e_ref, e_ref_base))
		return e_ref + calc.property['energy']['hf'], e_ref_base + calc.property['energy']['hf'], dipmom_ref, calc.mo


def main(mol, calc, exp, method):
		""" main property function """
		# first-order properties
		if 'dipmom' in exp.property:
			prop = True
		else:
			prop = False
		# fci calc
		if method == 'FCI':
			e, dm = _fci(mol, calc, exp, prop)
		# sci base
		elif method == 'SCI':
			e, dm = _sci(mol, calc, exp, prop)
		# cisd calc
		elif method == 'CISD':
			e, dm = _ci(mol, calc, exp, prop)
		# ccsd / ccsd(t) calc
		elif method in ['CCSD','CCSD(T)']:
			e, dm = _cc(mol, calc, exp, prop, (method == 'CCSD(T)'))
		# calculate first-order properties
		if dm is not None:
			return e, _dipmom(mol.dipmom, calc.occup, exp.core_idx, exp.cas_idx, calc.mo, dm)
		else:
			return e, np.zeros(3, dtype=np.float64)


def _dipmom(ints, occup, core_idx, cas_idx, mo, cas_dm):
		""" calculate electronic dipole moment """
		# dm
		if core_idx.size == 0:
			dm = cas_dm
		else:
			dm = np.diag(occup)
			dm[cas_idx[:, None], cas_idx] = cas_dm
		elec_dipmom = np.empty(3, dtype=np.float64)
		for i in range(3):
			elec_dipmom[i] = np.trace(np.dot(dm, reduce(np.dot, (mo.T, ints[i], mo))))
		return elec_dipmom 


def base(mol, calc, exp):
		""" calculate base energy and mo coefficients """
		# set core and cas spaces
		exp.core_idx, exp.cas_idx = core_cas(mol, exp, calc.exp_space)
		# init dm
		dm = None
		# zeroth-order energy
		if calc.base['METHOD'] is None:
			e_base = 0.0
		# cisd base
		elif calc.base['METHOD'] == 'CISD':
			e_base, dm = _ci(mol, calc, exp, calc.occ == 'CISD' or calc.virt == 'CISD')
			if mol.spin > 0 and dm is not None: dm = dm[0] + dm[1]
		# ccsd / ccsd(t) base
		elif calc.base['METHOD'] in ['CCSD','CCSD(T)']:
			e_base, dm = _cc(mol, calc, exp, calc.occ == 'CCSD' or calc.virt == 'CCSD', \
										(calc.base['METHOD'] == 'CCSD(T)') and \
										((calc.occ == 'CAN') and (calc.virt == 'CAN')))
			if mol.spin > 0 and dm is not None: dm = dm[0] + dm[1]
		# sci base
		elif calc.base['METHOD'] == 'SCI':
			e_base, dm = _sci(mol, calc, exp, calc.occ == 'SCI' or calc.virt == 'SCI')
		# NOs
		if (calc.occ == 'CISD' or calc.virt == 'CISD') and dm is None:
			dm = _ci(mol, calc, exp, True)[1]
			if mol.spin > 0: dm = dm[0] + dm[1]
		elif (calc.occ == 'CCSD' or calc.virt == 'CCSD') and dm is None:
			dm = _cc(mol, calc, exp, True, False)[1]
			if mol.spin > 0: dm = dm[0] + dm[1]
		elif (calc.occ == 'SCI' or calc.virt == 'SCI') and dm is None:
			dm = _sci(mol, calc, exp, True)[1]
		# occ-occ block (local or NOs)
		if calc.occ != 'CAN':
			if calc.occ in ['CISD', 'CCSD', 'SCI']:
				occup, no = symm.eigh(dm[:(mol.nocc-mol.ncore), :(mol.nocc-mol.ncore)], calc.orbsym[mol.ncore:mol.nocc])
				calc.mo[:, mol.ncore:mol.nocc] = np.dot(calc.mo[:, mol.ncore:mol.nocc], no[:, ::-1])
			elif calc.occ == 'PM':
				calc.mo[:, mol.ncore:mol.nocc] = lo.PM(mol, calc.mo[:, mol.ncore:mol.nocc]).kernel()
			elif calc.occ == 'FB':
				calc.mo[:, mol.ncore:mol.nocc] = lo.Boys(mol, calc.mo[:, mol.ncore:mol.nocc]).kernel()
			elif calc.occ in ['IBO-1','IBO-2']:
				iao = lo.iao.iao(mol, calc.mo[:, mol.core:mol.nocc])
				if calc.occ == 'IBO-1':
					iao = lo.vec_lowdin(iao, calc.hf.get_ovlp())
					calc.mo[:, mol.ncore:mol.nocc] = lo.ibo.ibo(mol, calc.mo[:, mol.ncore:mol.nocc], iao)
				elif calc.occ == 'IBO-2':
					calc.mo[:, mol.ncore:mol.nocc] = lo.ibo.PM(mol, calc.mo[:, mol.ncore:mol.nocc], iao).kernel()
		# virt-virt block (local or NOs)
		if calc.virt != 'CAN':
			if calc.virt in ['CISD', 'CCSD', 'SCI']:
				occup, no = symm.eigh(dm[-mol.nvirt:, -mol.nvirt:], calc.orbsym[mol.nocc:])
				calc.mo[:, mol.nocc:] = np.dot(calc.mo[:, mol.nocc:], no[:, ::-1])
			elif calc.virt == 'PM':
				calc.mo[:, mol.nocc:] = lo.PM(mol, calc.mo[:, mol.nocc:]).kernel()
			elif calc.virt == 'FB':
				calc.mo[:, mol.nocc:] = lo.Boys(mol, calc.mo[:, mol.nocc:]).kernel()
		# extra calculation for non-invariant methods
		if calc.occ != 'CAN' or calc.virt != 'CAN':
			if calc.base['METHOD'] == 'CCSD(T)':
				e_base = _cc(mol, calc, exp, False, True)[0]
			elif calc.base['METHOD'] == 'SCI':
				e_base = _sci(mol, calc, exp, False)[0]
		return e_base


def _casscf(mol, calc, exp):
		""" casscf calc """
		# casscf ref
		cas = mcscf.CASSCF(calc.hf, calc.no_act, calc.ne_act)
		# fci solver
		if abs(calc.ne_act[0]-calc.ne_act[1]) == 0:
			cas.fcisolver = fci.direct_spin0_symm.FCI(mol)
		else:
			cas.fcisolver = fci.direct_spin1_symm.FCI(mol)
		cas.fcisolver.conv_tol = 1.0e-10
		cas.conv_tol = 1.0e-10
		cas.max_stepsize = .01
		cas.max_cycle_micro = 1
		cas.max_cycle_macro = 500
		cas.canonicalization = False
		# wfnsym
		cas.fcisolver.wfnsym = calc.wfnsym
		# frozen (inactive)
		cas.frozen = (mol.nelectron - (calc.ne_act[0] + calc.ne_act[1])) // 2
		# verbose print
		if mol.verbose: cas.verbose = 4
		# fix spin if non-singlet
		if mol.spin > 0:
			sz = abs(calc.ne_act[0]-calc.ne_act[1]) * .5
			cas.fix_spin_(ss=sz * (sz + 1.))
		# target state
		cas.state_specific_(state=calc.target)
		# run casscf calc
		cas.kernel(calc.mo)
		# convergence check
		if not cas.converged:
			try:
				raise RuntimeError('\nCASSCF Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# calculate spin
		s, mult = cas.fcisolver.spin_square(cas.ci, calc.no_act, calc.ne_act)
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
		fock_ao = cas.get_fock(cas.mo_coeff, cas.ci, None, None)
		fock = reduce(np.dot, (cas.mo_coeff.T, fock_ao, cas.mo_coeff))
		mo = np.empty_like(cas.mo_coeff)
		# core region
		if mol.ncore > 0:
			c = symm.eigh(fock[:mol.ncore, :mol.ncore], \
								cas.mo_coeff.orbsym[:mol.ncore])[1]
			mo[:, :mol.ncore] = np.dot(cas.mo_coeff[:, :mol.ncore], c)
		# inactive region (excl. core)
		if cas.frozen > mol.ncore:
			c = symm.eigh(fock[mol.ncore:cas.frozen, mol.ncore:cas.frozen], \
								cas.mo_coeff.orbsym[mol.ncore:cas.frozen])[1]
			mo[:, mol.ncore:cas.frozen] = np.dot(cas.mo_coeff[:, mol.ncore:cas.frozen], c)
		# active region
		mo[:, cas.frozen:(cas.frozen + calc.no_act)] = cas.mo_coeff[:, cas.frozen:(cas.frozen + calc.no_act)]
		# virtual region
		if mol.norb - (cas.frozen + calc.no_act) > 0:
			c = symm.eigh(fock[(cas.frozen + calc.no_act):, (cas.frozen + calc.no_act):], \
								cas.mo_coeff.orbsym[(cas.frozen + calc.no_act):])[1]
			mo[:, (cas.frozen + calc.no_act):] = np.dot(cas.mo_coeff[:, (cas.frozen + calc.no_act):], c)
		return mo


def _fci(mol, calc, exp, dens):
		""" fci calc """
		# no virtuals?
		if np.amin(calc.occup[exp.cas_idx]) == 2.0:
			return 0.0, None
		# init fci solver
		if mol.spin == 0:
			solver = fci.direct_spin0_symm.FCI(mol)
		else:
			solver = fci.direct_spin1_symm.FCI(mol)
		# settings
		solver.conv_tol = 1.0e-10
		solver.max_cycle = 500
		solver.max_space = 25
		solver.davidson_only = True
		# wfnsym
		solver.wfnsym = calc.wfnsym
		# target state
		solver.nroots = calc.target + 1
		# get integrals and core energy
		h1e, h2e = _prepare(mol, calc, exp)
		# electrons
		nelec = (mol.nelec[0] - len(exp.core_idx), mol.nelec[1] - len(exp.core_idx))
		# orbital symmetry
		solver.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo[:, exp.cas_idx])
		# fix spin if non-singlet
		if mol.spin > 0:
			sz = abs(nelec[0]-nelec[1]) * .5
			fci.addons.fix_spin(solver, ss=sz * (sz + 1.))
		# init guess (does it exist?)
		try:
			ci0 = fci.addons.symm_initguess(len(exp.cas_idx), nelec, orbsym=solver.orbsym, wfnsym=solver.wfnsym)
		except Exception:
			return 0.0, None
		# perform calc
		e, c = solver.kernel(h1e, h2e, len(exp.cas_idx), nelec, ecore=mol.e_core)
		# collect results
		if solver.nroots == 1:
			conv = solver.converged
			energy = e
			civec = c
		else:
			if len(solver.converged) == solver.nroots:
				conv = solver.converged[calc.target]
				energy = e[calc.target]
				civec = c[calc.target]
			else:
				return 0.0, None
		# convergence check
		if not conv:
			try:
				raise RuntimeError('\nFCI Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# calculate spin
		s, mult = solver.spin_square(civec, len(exp.cas_idx), nelec)
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
		e_corr = energy - calc.property['energy']['hf']
#		if exp.order < exp.max_order: e_corr += np.float64(0.001) * np.random.random_sample()
		# fci dm
		if dens:
			dm = solver.make_rdm1(c, len(exp.cas_idx), nelec)
		else:
			dm = None
		return e_corr, dm


def _sci(mol, calc, exp, dens):
		""" sci calc """
		# no virtuals?
		if np.amin(calc.occup[exp.cas_idx]) == 2.0:
			return 0.0, None
		# init sci solver
		if mol.spin == 0:
			solver = fci.select_ci_spin0_symm.SCI(mol)
		else:
			solver = fci.select_ci_symm.SCI(mol)
		# settings
		solver.conv_tol = 1.0e-10
		solver.max_cycle = 500
		solver.max_space = 25
		solver.davidson_only = True
		# wfnsym
		solver.wfnsym = calc.wfnsym
		# target state
		solver.nroots = calc.target + 1
		# get integrals and core energy
		h1e, h2e = _prepare(mol, calc, exp)
		# electrons
		nelec = (mol.nelec[0] - len(exp.core_idx), mol.nelec[1] - len(exp.core_idx))
		# orbital symmetry
		solver.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo[:, exp.cas_idx])
		# fix spin if non-singlet
		if mol.spin > 0:
			sz = abs(nelec[0]-nelec[1]) * .5
			fci.addons.fix_spin(solver, ss=sz * (sz + 1.))
		# init guess (does it exist?)
		try:
			ci0 = fci.addons.symm_initguess(len(exp.cas_idx), nelec, orbsym=solver.orbsym, wfnsym=solver.wfnsym)
		except Exception:
			return 0.0, None
		# perform calc
		e, c = solver.kernel(h1e, h2e, len(exp.cas_idx), nelec, ecore=mol.e_core)
		# collect results
		if solver.nroots == 1:
			conv = solver.converged
			energy = e
			civec = c
		else:
			if len(solver.converged) == solver.nroots:
				conv = solver.converged[calc.target]
				energy = e[calc.target]
				civec = c[calc.target]
			else:
				return 0.0, None
		# convergence check
		if not conv:
			try:
				raise RuntimeError('\nSCI Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# calculate spin
		s, mult = solver.spin_square(civec, len(exp.cas_idx), nelec)
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
		e_corr = energy - calc.property['energy']['hf']
		# sci dm
		if dens:
			dm = solver.make_rdm1(c, len(exp.cas_idx), nelec)
		else:
			dm = None
		return e_corr, dm


def _ci(mol, calc, exp, dens):
		""" cisd calc """
		# no virtuals?
		if np.amin(calc.occup[exp.cas_idx]) == 2.0:
			return 0.0, None
		# get integrals
		h1e, h2e = _prepare(mol, calc, exp)
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
		cisd.max_space = 25
		eris = cisd.ao2mo()
		# calculate cisd energy
		for i in range(5,-1,-1):
			cisd.level_shift = 1.0 / 10.0 ** (i)
			try:
				cisd.kernel(eris=eris)
			except sp.linalg.LinAlgError: pass
			if cisd.converged: break
		# convergence check
		if not cisd.converged:
			try:
				raise RuntimeError('\nCISD Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# e_corr
		e_corr = cisd.e_corr
		# dm
		if dens:
			dm = cisd.make_rdm1()
		else:
			dm = None
		return e_corr, dm


def _cc(mol, calc, exp, dens, pt=False):
		""" ccsd / ccsd(t) calc """
		# no virtuals?
		if np.amin(calc.occup[exp.cas_idx]) == 2.0:
			return 0.0, None
		# get integrals
		h1e, h2e = _prepare(mol, calc, exp)
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
		if dens: ccsd.conv_tol_normt = 1.0e-10
		ccsd.max_cycle = 500
		if exp.order > 0:
			# avoid async function execution
			ccsd.async_io = calc.async
			# avoid I/O
			if not calc.async: ccsd.incore_complete = True
		eris = ccsd.ao2mo()
		# calculate ccsd energy
		for i in list(range(0, 12, 2)):
			ccsd.diis_start_cycle = i
			try:
				ccsd.kernel(eris=eris)
			except sp.linalg.LinAlgError: pass
			if ccsd.converged: break
		# convergence check
		if not ccsd.converged:
			try:
				raise RuntimeError('\nCCSD Error : no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# e_corr
		e_corr = ccsd.e_corr
		# dm
		if dens and not pt:
			ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
			dm = ccsd.make_rdm1()
		else:
			dm = None
		# calculate (t) correction
		if pt:
			if np.amin(calc.occup[exp.cas_idx]) == 1.0:
				if len(np.where(calc.occup[exp.cas_idx] == 1.)[0]) >= 3:
					e_corr += ccsd.ccsd_t(eris=eris)
			else:
				e_corr += ccsd.ccsd_t(eris=eris)
		return e_corr, dm


def core_cas(mol, exp, tup):
		""" define core and cas spaces """
		cas_idx = np.asarray(sorted(exp.incl_idx + sorted(tup.tolist())))
		core_idx = np.asarray(sorted(list(set(range(mol.nocc)) - set(cas_idx))))
		return core_idx, cas_idx


def _prepare(mol, calc, exp):
		""" generate input for correlated calculation """
		# extract cas integrals and calculate core energy
		if mol.e_core is None or exp.typ == 'occupied':
			if len(exp.core_idx) > 0:
				core_dm = np.dot(calc.mo[:, exp.core_idx], np.transpose(calc.mo[:, exp.core_idx])) * 2
				vj, vk = scf.hf.get_jk(mol, core_dm)
				mol.core_vhf = vj - vk * .5
				mol.e_core = mol.energy_nuc() + np.einsum('ij,ji', core_dm, mol.hcore)
				mol.e_core += np.einsum('ij,ji', core_dm, mol.core_vhf) * .5
			else:
				mol.e_core = mol.energy_nuc()
				mol.core_vhf = 0
		h1e_cas = reduce(np.dot, (np.transpose(calc.mo[:, exp.cas_idx]), \
							mol.hcore + mol.core_vhf, calc.mo[:, exp.cas_idx]))
		h2e_cas = ao2mo.incore.full(mol.eri, calc.mo[:, exp.cas_idx])
		return h1e_cas, h2e_cas


