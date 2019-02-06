#!/usr/bin/env python
# -*- coding: utf-8 -*

""" kernel.py: kernel module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import os
import shutil
import copy
import numpy as np
import scipy as sp
from functools import reduce
from mpi4py import MPI
from pyscf import gto, symm, scf, ao2mo, lo, ci, cc, mcscf, fci

import tools


SPIN_TOL = 1.0e-05


def ao_ints(mol, calc):
		""" get AO integrals """
		# core hamiltonian and electron repulsion ints
		if mol.atom: # ab initio hamiltonian
			hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
			if mol.cart:
				eri = mol.intor('int2e_cart', aosym=4)
			else:
				eri = mol.intor('int2e_sph', aosym=4)
		else: # model hamiltonian
			hcore = _hubbard_h1(mol)
			eri = _hubbard_eri(mol)
		# dipole integrals with gauge origin at origo
		if calc.target in ['dipole', 'trans']:
			with mol.with_common_origin([0.0, 0.0, 0.0]):
				dipole = mol.intor_symmetric('int1e_r', comp=3)
		else:
			dipole = None
		return hcore, eri, dipole


def _hubbard_h1(mol):
		""" set hubbard hopping hamiltonian """
		# 1d
		if mol.dim == 1:
			# init
			n = mol.nsites
			t = mol.t
			h1 = np.zeros([n]*2, dtype=np.float64)
			# adjacent neighbours
			for i in range(n-1):
				h1[i, i+1] = h1[i+1, i] = -t
			# pbc
			if mol.pbc:
				h1[-1, 0] = h1[0, -1] = -t
		# 2d
		elif mol.dim == 2:
			# init
			n = int(np.sqrt(mol.nsites))
			t = mol.t
			h1 = np.zeros([n**2]*2, dtype=np.float64)
			# adjacent neighbours - sideways
			for i in range(n**2):
				if i % n == 0:
					h1[i, i+1] = -t
				elif i % n == n-1:
					h1[i, i-1] = -t
				else:
					h1[i, i-1] = h1[i, i+1] = -t
			# adjacent neighbours - up-down
			for i in range(n**2):
				if i < n:
					h1[i, i+n] = -t
				elif i >= n**2 - n:
					h1[i, i-n] = -t
				else:
					h1[i, i-n] = h1[i, i+n] = -t
			# pbc
			if mol.pbc:
				# sideways
				for i in range(n):
					h1[i*n, i*n+(n-1)] = h1[i*n+(n-1), i*n] = -t
				# up-down
				for i in range(n):
					h1[i, n*(n-1)+i] = h1[n*(n-1)+i, i] = -t
		return h1


def _hubbard_eri(mol):
		""" set hubbard two-electron hamiltonian """
		# init
		n = mol.nsites
		u = mol.u
		eri = np.zeros([n]*4, dtype=np.float64)
		for i in range(n):
			eri[i,i,i,i] = u
		return eri


def hf(mol, calc):
		""" hartree-fock calculation """
		# perform restricted hf calc
		mol_hf = mol.copy()
		mol_hf.build(0, 0, symmetry = mol.hf_sym)
		hf = scf.RHF(mol_hf)
		# debug print
		if mol.debug >= 1:
			hf.verbose = 4
		hf.init_guess = mol.hf_init_guess
		hf.conv_tol = 1.0e-09
		hf.max_cycle = 1000
		if mol.atom: # ab initio hamiltonian
			hf.irrep_nelec = mol.irrep_nelec
		else: # model hamiltonian
			hf.get_ovlp = lambda *args: np.eye(mol.nsites)
			hf.get_hcore = lambda *args: mol.hcore 
			hf._eri = mol.eri
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
		if calc.target == 'dipole':
			dm = hf.make_rdm1()
			elec_dipole = np.einsum('xij,ji->x', mol.dipole, dm)
			elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > 1.0e-15 else 0.0 for i in range(elec_dipole.size)])
		else:
			elec_dipole = None
		# determine dimensions
		norb, nocc, nvirt = _dim(hf, calc)
		# store energy, occupation, and orbsym
		e_hf = hf.e_tot
		occup = hf.mo_occ
		if mol.atom:
			orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
		else:
			orbsym = np.zeros(hf.mo_energy.size, dtype=np.int)
		# debug print of orbital energies
		if mol.debug >= 1:
			if mol.symmetry:
				gpname = mol.symmetry
			else:
				gpname = 'C1'
			print('\n HF:  mo   symmetry    energy')
			for i in range(hf.mo_energy.size):
				print('     {:>3d}   {:>5s}     {:>7.3f}'.format(i, symm.addons.irrep_id2name(gpname, orbsym[i]), hf.mo_energy[i]))
			print('\n')
		return nocc, nvirt, norb, hf, np.asscalar(e_hf), elec_dipole, occup, orbsym, hf.mo_energy, np.asarray(hf.mo_coeff, order='C')


def _dim(hf, calc):
		""" determine dimensions """
		# occupied and virtual lists
		occ = np.where(hf.mo_occ > 0.)[0]
		virt = np.where(hf.mo_occ == 0.)[0]
		# nocc, nvirt, and norb
		nocc = occ.size
		nvirt = virt.size
		norb = nocc + nvirt
		return norb, nocc, nvirt


def ref_mo(mol, calc):
		""" determine reference mo coefficients """
		# sort mo coefficients
		mo_energy = calc.mo_energy
		mo_coeff = calc.mo_coeff
		if calc.ref['active'] == 'manual':
			# electrons
			nelec = calc.ref['nelec']
			tools.assertion(np.sum(nelec) > 0, 'no electrons in the reference space')
			# active orbs
			if isinstance(calc.ref['select'], dict):
				cas = mcscf.CASSCF(calc.hf, np.sum(list(calc.ref['select'].values())), nelec)
				calc.ref['select'] = mcscf.caslst_by_irrep(cas, calc.mo_coeff, calc.ref['select'], base=0)
			calc.ref['select'] = np.asarray(calc.ref['select'])
			# inactive orbitals
			inact_elec = mol.nelectron - (nelec[0] + nelec[1])
			tools.assertion(inact_elec % 2 == 0, 'odd number of inactive electrons')
			inact_orbs = inact_elec // 2
			# active orbitals
			act_orbs = calc.ref['select'].size
			# virtual orbitals
			virt_orbs = mol.norb - inact_orbs - act_orbs
			# divide into inactive-active-virtual
			idx = np.asarray([i for i in range(mol.norb) if i not in calc.ref['select']])
			mo_coeff = np.concatenate((mo_coeff[:, idx[:inact_orbs]], mo_coeff[:, calc.ref['select']], mo_coeff[:, idx[inact_orbs:]]), axis=1)
			if mol.atom:
				calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)
		# reference and expansion spaces
		ref_space = np.arange(inact_orbs, inact_orbs+act_orbs)
		exp_space = np.append(np.arange(mol.ncore, inact_orbs), np.arange(inact_orbs+act_orbs, mol.norb))
		# casci or casscf
		if calc.ref['method'] == 'casci':
			mo_energy = np.concatenate((mo_energy[idx[:inact_orbs]], mo_energy[calc.ref['select']], mo_energy[idx[inact_orbs:]]))
		elif calc.ref['method'] == 'casscf':
			# casscf quantities
			mo_energy, mo_coeff = _casscf(mol, calc, mo_coeff, ref_space, nelec)
		if mol.debug >= 1:
			print('\n reference nelec  = {:}'.format(nelec))
			print(' reference space  = {:}'.format(ref_space))
			print(' expansion space  = {:}\n'.format(exp_space))
		return mo_energy, np.asarray(mo_coeff, order='C'), nelec, ref_space, exp_space


def ref_prop(mol, calc, exp):
		""" calculate reference space properties """
		# set core and cas spaces
		exp.core_idx, exp.cas_idx = tools.core_cas(mol, calc.ref_space, np.array([], dtype=np.int32))
		# closed-shell HF exception
		if np.all(calc.occup[calc.ref_space] == 2.):
			if calc.target in ['energy', 'excitation']:
				ref = 0.0
			else:
				ref = np.zeros(3, dtype=np.float64)
		else:
			# exp model
			ref = main(mol, calc, exp, calc.model['method'])
			if calc.base['method'] is not None:
				ref -= main(mol, calc, exp, calc.base['method'])
		return ref


def main(mol, calc, exp, method):
		""" main prop function """
		# fci calc
		if method == 'fci':
			res_tmp = _fci(mol, calc, exp.core_idx, exp.cas_idx)
		# cisd calc
		elif method == 'cisd':
			res_tmp = _ci(mol, calc, exp.core_idx, exp.cas_idx, exp.order)
		# ccsd / ccsd(t) calc
		elif method in ['ccsd','ccsd(t)']:
			res_tmp = _cc(mol, calc, exp.core_idx, exp.cas_idx, exp.order, method == 'ccsd(t)')
		if calc.target in ['energy', 'excitation']:
			res = res_tmp[calc.target]
		# return first-order properties
		elif calc.target == 'dipole':
			res = _dipole(mol, calc, exp, res_tmp['rdm1'])
		elif calc.target == 'trans':
			res = _trans(mol, calc, exp, res_tmp['t_rdm1'], \
							res_tmp['hf_weight'][0], res_tmp['hf_weight'][1])
		return res


def _dipole(mol, calc, exp, cas_rdm1, trans=False):
		""" calculate electronic (transition) dipole moment """
		# init (transition) rdm1
		if trans:
			rdm1 = np.zeros([mol.norb, mol.norb], dtype=np.float64)
		else:
			rdm1 = np.diag(calc.occup)
		# insert correlated subblock
		rdm1[exp.cas_idx[:, None], exp.cas_idx] = cas_rdm1
		# ao representation
		rdm1 = np.einsum('pi,ij,qj->pq', calc.mo_coeff, rdm1, calc.mo_coeff)
		# compute elec_dipole
		elec_dipole = np.einsum('xij,ji->x', mol.dipole, rdm1)
		# remove noise
		elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > 1.0e-15 else 0.0 for i in range(elec_dipole.size)])
		# 'correlation' dipole
		if not trans:
			elec_dipole -= calc.prop['hf']['dipole']
		return elec_dipole


def _trans(mol, calc, exp, cas_t_rdm1, hf_weight_gs, hf_weight_ex):
		""" calculate electronic transition dipole moment """
		return _dipole(mol, calc, exp, cas_t_rdm1, True) * np.sign(hf_weight_gs) * np.sign(hf_weight_ex)


def base(mol, calc):
		""" calculate base energy and mo coefficients """
		# set core and cas spaces
		core_idx, cas_idx = tools.core_cas(mol, np.arange(mol.ncore), np.arange(mol.ncore, mol.norb))
		# init rdm1
		rdm1 = None
		# no base
		if calc.base['method'] is None:
			base = 0.0
		# cisd base
		elif calc.base['method'] == 'cisd':
			res = _ci(mol, calc, core_idx, cas_idx, 0)
			base = res['energy']
			if 'rdm1' in res:
				rdm1 = res['rdm1']
				if mol.spin > 0:
					rdm1 = rdm1[0] + rdm1[1]
		# ccsd / ccsd(t) base
		elif calc.base['method'] in ['ccsd','ccsd(t)']:
			res = _cc(mol, calc, core_idx, cas_idx, 0, \
						calc.base['method'] == 'ccsd(t)' and (calc.orbs['occ'] == 'can' and calc.orbs['virt'] == 'can'))
			base = res['energy']
			if 'rdm1' in res:
				rdm1 = res['rdm1']
				if mol.spin > 0:
					rdm1 = rdm1[0] + rdm1[1]
		# NOs
		if (calc.orbs['occ'] == 'cisd' or calc.orbs['virt'] == 'cisd') and rdm1 is None:
			res = _ci(mol, calc, core_idx, cas_idx, 0)
			rdm1 = res['rdm1']
			if mol.spin > 0:
				rdm1 = rdm1[0] + rdm1[1]
		elif (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd') and rdm1 is None:
			res = _cc(mol, calc, core_idx, cas_idx, 0, False)
			rdm1 = res['rdm1']
			if mol.spin > 0:
				rdm1 = rdm1[0] + rdm1[1]
		# occ-occ block (local or NOs)
		if calc.orbs['occ'] != 'can':
			if calc.orbs['occ'] in ['cisd', 'ccsd']:
				occup, no = symm.eigh(rdm1[:(mol.nocc-mol.ncore), :(mol.nocc-mol.ncore)], calc.orbsym[mol.ncore:mol.nocc])
				calc.mo_coeff[:, mol.ncore:mol.nocc] = np.einsum('ip,pj->ij', calc.mo_coeff[:, mol.ncore:mol.nocc], no[:, ::-1])
			elif calc.orbs['occ'] == 'pm':
				calc.mo_coeff[:, mol.ncore:mol.nocc] = lo.PM(mol, calc.mo_coeff[:, mol.ncore:mol.nocc]).kernel()
			elif calc.orbs['occ'] == 'fb':
				calc.mo_coeff[:, mol.ncore:mol.nocc] = lo.Boys(mol, calc.mo_coeff[:, mol.ncore:mol.nocc]).kernel()
			elif calc.orbs['occ'] in ['ibo-1','ibo-2']:
				iao = lo.iao.iao(mol, calc.mo_coeff[:, mol.ncore:mol.nocc])
				if calc.orbs['occ'] == 'ibo-1':
					iao = lo.vec_lowdin(iao, calc.hf.get_ovlp())
					calc.mo_coeff[:, mol.ncore:mol.nocc] = lo.ibo.ibo(mol, calc.mo_coeff[:, mol.ncore:mol.nocc], iao)
				elif calc.orbs['occ'] == 'ibo-2':
					calc.mo_coeff[:, mol.ncore:mol.nocc] = lo.ibo.PM(mol, calc.mo_coeff[:, mol.ncore:mol.nocc], iao).kernel()
		# virt-virt block (local or NOs)
		if calc.orbs['virt'] != 'can':
			if calc.orbs['virt'] in ['cisd', 'ccsd']:
				occup, no = symm.eigh(rdm1[-mol.nvirt:, -mol.nvirt:], calc.orbsym[mol.nocc:])
				calc.mo_coeff[:, mol.nocc:] = np.einsum('ip,pj->ij', calc.mo_coeff[:, mol.nocc:], no[:, ::-1])
			elif calc.orbs['virt'] == 'pm':
				calc.mo_coeff[:, mol.nocc:] = lo.PM(mol, calc.mo_coeff[:, mol.nocc:]).kernel()
			elif calc.orbs['virt'] == 'fb':
				calc.mo_coeff[:, mol.nocc:] = lo.Boys(mol, calc.mo_coeff[:, mol.nocc:]).kernel()
		# extra calculation for non-invariant ccsd(t)
		if calc.base['method'] == 'ccsd(t)' and (calc.orbs['occ'] != 'can' or calc.orbs['virt'] != 'can'):
			res = _cc(mol, calc, core_idx, cas_idx, 0, True)
			base = res['energy']
		# update orbsym
		if mol.atom:
			calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
		return base


def _casscf(mol, calc, mo_coeff, ref_space, nelec):
		""" casscf calc """
		# casscf ref
		cas = mcscf.CASSCF(calc.hf, ref_space.size, nelec)
		# fci solver
		cas.conv_tol = 1.0e-10
		cas.max_cycle_macro = 500
		# frozen (inactive)
		cas.frozen = mol.ncore
		# debug print
		if mol.debug >= 1:
			cas.verbose = 4
		# fcisolver
		if np.abs(nelec[0]-nelec[1]) == 0:
			if mol.symmetry:
				fcisolver = fci.direct_spin0_symm.FCI(mol)
			else:
				fcisolver = fci.direct_spin0.FCI(mol)
		else:
			if mol.symmetry:
				fcisolver = fci.direct_spin1_symm.FCI(mol)
			else:
				fcisolver = fci.direct_spin1.FCI(mol)
		# conv_tol
		fcisolver.conv_tol = max(calc.thres['init'], 1.0e-10)
		# orbital symmetry
		fcisolver.orbsym = calc.orbsym[ref_space]
		# wfnsym
		fcisolver.wfnsym = calc.ref['wfnsym'][0]
		# set solver
		cas.fcisolver = fcisolver
		# state-averaged calculation
		if len(calc.ref['wfnsym']) > 1:
			# weights
			weights = np.array((1 / len(calc.ref['wfnsym']),) * len(calc.ref['wfnsym']), dtype=np.float64)
			# are all states of same symmetry?
			if len(set(calc.ref['wfnsym'])) == 1:
				# state average
				cas.state_average_(weights)
			else:
				# nroots for first fcisolver
				fcisolver.nroots = np.count_nonzero(np.asarray(calc.ref['wfnsym']) == list(set(calc.ref['wfnsym']))[0])
				# init list of fcisolvers
				fcisolvers = [fcisolver]
				# loop over symmetries
				for i in range(1, len(set(calc.ref['wfnsym']))):
					# copy fcisolver
					fcisolver_ = copy.copy(fcisolver)
					# wfnsym for fcisolver_
					fcisolver_.wfnsym = list(set(calc.ref['wfnsym']))[i]
					# nroots for fcisolver_
					fcisolver_.nroots = np.count_nonzero(np.asarray(calc.ref['wfnsym']) == list(set(calc.ref['wfnsym']))[i])
					# append to fcisolvers
					fcisolvers.append(fcisolver_)
				# state average
				mcscf.state_average_mix_(cas, fcisolvers, weights)
		# hf starting guess
		if calc.ref['hf_guess']:
			na = fci.cistring.num_strings(ref_space.size, nelec[0])
			nb = fci.cistring.num_strings(ref_space.size, nelec[1])
			ci0 = np.zeros((na, nb))
			ci0[0, 0] = 1
		else:
			ci0 = None
		# run casscf calc
		cas.kernel(mo_coeff, ci0=ci0)
		if len(calc.ref['wfnsym']) == 1:
			c = [cas.ci]
		else:
			c = cas.ci
		# multiplicity check
		for root in range(len(c)):
			s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)
			if np.abs((mol.spin + 1) - mult) > SPIN_TOL:
				# fix spin by applyting level shift
				sz = np.abs(nelec[0]-nelec[1]) * 0.5
				cas.fix_spin_(shift=0.25, ss=sz * (sz + 1.))
				# run casscf calc
				cas.kernel(mo_coeff, ci0=ci0)
				if len(calc.ref['wfnsym']) == 1:
					c = [cas.ci]
				else:
					c = cas.ci
				# verify correct spin
				for root in range(len(c)):
					s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)
					tools.assertion(np.abs((mol.spin + 1) - mult) < SPIN_TOL, \
									'spin contamination for root entry = {:} , 2*S + 1 = {:.6f}'. \
										format(root, mult))
		# convergence check
		if not cas.converged:
			try:
				raise RuntimeError('\nCASSCF Error: no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# debug print of orbital energies
		if mol.atom:
			orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, cas.mo_coeff)
		else:
			orbsym = np.zeros(cas.mo_energy.size, dtype=np.int)
		if mol.debug >= 1:
			if mol.symmetry:
				gpname = mol.symmetry
			else:
				gpname = 'C1'
			print('\n CASSCF:  mo   symmetry    energy')
			for i in range(cas.mo_energy.size):
				print('         {:>3d}   {:>5s}     {:>7.3f}'.format(i, symm.addons.irrep_id2name(gpname, orbsym[i]), cas.mo_energy[i]))
			print('\n')
		return cas.mo_energy, np.asarray(cas.mo_coeff, order='C')


def _fci(mol, calc, core_idx, cas_idx):
		""" fci calc """
		# electrons
		nelec = (np.count_nonzero(calc.occup[cas_idx] > 0.), np.count_nonzero(calc.occup[cas_idx] > 1.))
		# init fci solver
		if mol.spin == 0:
			solver = fci.direct_spin0_symm.FCI(mol)
		else:
			solver = fci.direct_spin1_symm.FCI(mol)
		# settings
		solver.conv_tol = max(calc.thres['init'], 1.0e-10)
		if calc.target in ['dipole', 'trans']:
			solver.conv_tol *= 1.0e-04
			solver.lindep = solver.conv_tol * 1.0e-01
		solver.max_cycle = 5000
		solver.max_space = 25
		solver.davidson_only = True
		solver.pspace_size = 0
		# debug print
		if mol.debug >= 3:
			solver.verbose = 10
		# wfnsym
		solver.wfnsym = calc.state['wfnsym']
		# get integrals and core energy
		h1e, h2e = _prepare(mol, calc, core_idx, cas_idx)
		# orbital symmetry
		solver.orbsym = calc.orbsym[cas_idx]
		# hf starting guess
		if calc.extra['hf_guess']:
			na = fci.cistring.num_strings(cas_idx.size, nelec[0])
			nb = fci.cistring.num_strings(cas_idx.size, nelec[1])
			ci0 = np.zeros((na, nb))
			ci0[0, 0] = 1
		else:
			ci0 = None
		# number of roots
		solver.nroots = calc.state['root'] + 1
		# interface
		def _fci_kernel():
				""" interface to solver.kernel """
				# perform calc
				e, c = solver.kernel(h1e, h2e, cas_idx.size, nelec, ecore=mol.e_core, \
										orbsym=solver.orbsym, ci0=ci0)
				# collect results
				if solver.nroots == 1:
					return [e], [c]
				else:
					return [e[0], e[-1]], [c[0], c[-1]]
		# perform calc
		energy, civec = _fci_kernel()
		# multiplicity check
		for root in range(len(civec)):
			s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
			if np.abs((mol.spin + 1) - mult) > SPIN_TOL:
				# fix spin by applyting level shift
				sz = np.abs(nelec[0]-nelec[1]) * 0.5
				solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.))
				# perform calc
				energy, civec = _fci_kernel()
				# verify correct spin
				for root in range(len(civec)):
					s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
					tools.assertion(np.abs((mol.spin + 1) - mult) < SPIN_TOL, \
									'spin contamination for root entry = {:} , 2*S + 1 = {:.6f} , '
									'core_idx = {:} , cas_idx = {:}'. \
										format(root, mult, core_idx, cas_idx))
		# convergence check
		if solver.nroots == 1:
			tools.assertion(solver.converged, 'state 0 not converged , core_idx = {:} , cas_idx = {:}'. \
								format(core_idx, cas_idx))
		else:
			if calc.target == 'excitation':
				for root in [0, solver.nroots-1]:
					tools.assertion(solver.converged[root], 'state {:} not converged , core_idx = {:} , cas_idx = {:}'. \
										format(root, core_idx, cas_idx))
			else:
				tools.assertion(solver.converged[solver.nroots-1], 'state {:} not converged , core_idx = {:} , cas_idx = {:}'. \
										format(solver.nroots-1, core_idx, cas_idx))
		res = {}
		# e_corr
		if calc.target == 'energy':
			res['energy'] = energy[-1] - calc.prop['hf']['energy']
		if calc.target == 'excitation':
			res['excitation'] = energy[-1] - energy[0]
		# fci rdm1 and t_rdm1
		if calc.target == 'dipole':
			res['rdm1'] = solver.make_rdm1(civec[-1], cas_idx.size, nelec)
		if calc.target == 'trans':
			res['t_rdm1'] = solver.trans_rdm1(civec[0], civec[-1], cas_idx.size, nelec)
			res['hf_weight'] = [civec[i][0, 0] for i in range(2)]
		return res


def _ci(mol, calc, core_idx, cas_idx, order):
		""" cisd calc """
		# get integrals
		h1e, h2e = _prepare(mol, calc, core_idx, cas_idx)
		mol_tmp = gto.M(verbose=0)
		mol_tmp.incore_anyway = mol.incore_anyway
		mol_tmp.max_memory = mol.max_memory
		if mol.spin == 0:
			hf = scf.RHF(mol_tmp)
		else:
			hf = scf.UHF(mol_tmp)
		hf.get_hcore = lambda *args: h1e
		hf._eri = h2e 
		# init ccsd
		if mol.spin == 0:
			cisd = ci.cisd.CISD(hf, mo_coeff=np.eye(cas_idx.size), mo_occ=calc.occup[cas_idx])
		else:
			cisd = ci.ucisd.UCISD(hf, mo_coeff=np.array((np.eye(cas_idx.size), np.eye(cas_idx.size))), \
									mo_occ=np.array((calc.occup[cas_idx] > 0., calc.occup[cas_idx] == 2.), dtype=np.double))
		# settings
		cisd.conv_tol = max(calc.thres['init'], 1.0e-10)
		cisd.max_cycle = 500
		cisd.max_space = 25
		eris = cisd.ao2mo()
		# calculate cisd energy
		cisd.kernel(eris=eris)
		# e_corr
		res = {'energy': cisd.e_corr}
		# rdm1
		if order == 0 and (calc.orbs['occ'] == 'cisd' or calc.orbs['virt'] == 'cisd'):
			res['rdm1'] = cisd.make_rdm1()
		return res


def _cc(mol, calc, core_idx, cas_idx, order, pt=False):
		""" ccsd / ccsd(t) calc """
		# get integrals
		h1e, h2e = _prepare(mol, calc, core_idx, cas_idx)
		mol_tmp = gto.M(verbose=0)
		mol_tmp.incore_anyway = mol.incore_anyway
		mol_tmp.max_memory = mol.max_memory
		if mol.spin == 0:
			hf = scf.RHF(mol_tmp)
		else:
			hf = scf.UHF(mol_tmp)
		hf.get_hcore = lambda *args: h1e
		hf._eri = h2e 
		# init ccsd
		if mol.spin == 0:
			ccsd = cc.ccsd.CCSD(hf, mo_coeff=np.eye(cas_idx.size), mo_occ=calc.occup[cas_idx])
		else:
			ccsd = cc.uccsd.UCCSD(hf, mo_coeff=np.array((np.eye(cas_idx.size), np.eye(cas_idx.size))), \
									mo_occ=np.array((calc.occup[cas_idx] > 0., calc.occup[cas_idx] == 2.), dtype=np.double))
		# settings
		ccsd.conv_tol = max(calc.thres['init'], 1.0e-10)
		if order == 0 and (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd') and not pt:
			ccsd.conv_tol_normt = ccsd.conv_tol
		ccsd.max_cycle = 500
		# avoid async function execution if requested
		ccsd.async_io = calc.misc['async']
		# avoid I/O if not async
		if not calc.misc['async']: ccsd.incore_complete = True
		eris = ccsd.ao2mo()
		# calculate ccsd energy
		for i in list(range(0, 12, 2)):
			ccsd.diis_start_cycle = i
			try:
				ccsd.kernel(eris=eris)
			except sp.linalg.LinAlgError:
				pass
			if ccsd.converged: break
		# convergence check
		if not ccsd.converged:
			try:
				raise RuntimeError('\nCCSD Error: no convergence\n\n')
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# e_corr
		res = {'energy': ccsd.e_corr}
		# rdm1
		if order == 0 and (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd') and not pt:
			ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
			res['rdm1'] = ccsd.make_rdm1()
		# calculate (t) correction
		if pt:
			if np.amin(calc.occup[cas_idx]) == 1.0:
				if np.where(calc.occup[cas_idx] == 1.)[0].size >= 3:
					res['energy'] += ccsd.ccsd_t(eris=eris)
			else:
				res['energy'] += ccsd.ccsd_t(eris=eris)
		return res


def _prepare(mol, calc, core_idx, cas_idx):
		""" generate input for correlated calculation """
		# extract cas integrals and calculate core energy
		if core_idx.size > 0:
			core_dm = np.einsum('ip,jp->ij', calc.mo_coeff[:, core_idx], calc.mo_coeff[:, core_idx]) * 2
			vj, vk = scf.hf.dot_eri_dm(mol.eri, core_dm)
			mol.core_vhf = vj - vk * .5
			mol.e_core = mol.energy_nuc() + np.einsum('ij,ji', core_dm, mol.hcore)
			mol.e_core += np.einsum('ij,ji', core_dm, mol.core_vhf) * .5
		else:
			mol.e_core = mol.energy_nuc()
			mol.core_vhf = 0
		h1e_cas = np.einsum('pi,pq,qj->ij', calc.mo_coeff[:, cas_idx], mol.hcore + mol.core_vhf, calc.mo_coeff[:, cas_idx])
		h2e_cas = ao2mo.incore.full(mol.eri, calc.mo_coeff[:, cas_idx])
		return h1e_cas, h2e_cas


