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
		if calc.target['dipole'] or calc.target['trans']:
			with mol.with_common_orig((0,0,0)):
				dipole = mol.intor_symmetric('int1e_r', comp=3)
		else:
			dipole = None
		return hcore, eri, dipole


def hf(mol, calc):
		""" hartree-fock calculation """
		# perform restricted hf calc
		hf = scf.RHF(mol)
		hf.conv_tol = 1.0e-10
		hf.max_cycle = 500
		# fixed occupation
		hf.irrep_nelec = mol.irrep_nelec
		# perform hf calc
		hf.kernel()
		# dipole moment
		tot_dipole = hf.dip_moment(unit='au', verbose=0)
		# nuclear dipole moment
		charges = mol.atom_charges()
		coords  = mol.atom_coords()
		nuc_dipole = np.einsum('i,ix->x', charges, coords)
		# electronic dipole moment
		elec_dipole = nuc_dipole - tot_dipole
		elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > 1.0e-15 else 0.0 for i in range(elec_dipole.size)])
		# determine dimensions
		mol.norb, mol.nocc, mol.nvirt = _dim(hf, calc)
		# store energy, occupation, and orbsym
		e_hf = hf.e_tot
		occup = hf.mo_occ
		orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
		# wave function symmetry
		wfnsym = scf.hf_symm.get_wfnsym(hf, mo_coeff=hf.mo_coeff, mo_occ=hf.mo_occ)
		# sanity check
		if wfnsym != calc.state['wfnsym'] and calc.ref['method'] == 'hf':
			try:
				raise RuntimeError('\nhf Error : wave function symmetry ({0:}) different from requested symmetry ({1:})\n\n'.\
									format(symm.irrep_id2name(mol.groupname, wfnsym), symm.irrep_id2name(mol.groupname, calc.state['wfnsym'])))
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		return hf, np.asscalar(e_hf), elec_dipole, occup, orbsym, np.asarray(hf.mo_coeff, order='C')


def _dim(hf, calc):
		""" determine dimensions """
		# occupied and virtual lists
		if calc.model['type'] == 'occ':
			occ = np.where(hf.mo_occ == 2.)[0]
			virt = np.where(hf.mo_occ < 2.)[0]
		elif calc.model['type'] == 'virt':
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
		if calc.model['type'] == 'occ':
			ref_space = np.array(range(mol.nocc, mol.norb))
			exp_space = np.array(range(mol.ncore, mol.nocc))
		elif calc.model['type'] == 'virt':
			ref_space = np.array(range(mol.ncore, mol.nocc))
			exp_space = np.array(range(mol.nocc, mol.norb))
		# hf reference model
		if calc.ref['method'] == 'hf':
			# no active space
			ne_act = (0, 0)
			no_exp = no_act = 0
		# casci/casscf reference model
		elif calc.ref['method'] in ['casci','casscf']:
			if calc.ref['active'] == 'manual':
				# active electrons
				ne_act = calc.ref['nelec']
				# active orbs
				no_act = len(calc.ref['select'])
				# expansion space orbs
				if calc.model['type'] == 'occ':
					no_exp = np.count_nonzero(np.array(calc.ref['select']) < mol.nocc)
				elif calc.model['type'] == 'virt':
					no_exp = np.count_nonzero(np.array(calc.ref['select']) >= mol.nocc)
				# sanity checks
				assert np.count_nonzero(np.array(calc.ref['select']) < mol.ncore) == 0
				assert float(ne_act[0] + ne_act[1]) <= np.sum(calc.hf.mo_occ[calc.ref['select']])
			else:
				from pyscf.mcscf import avas
				# avas
				no_avas, ne_avas = avas.avas(calc.hf, calc.ref['ao_labels'], canonicalize=True, \
												verbose=4 if mol.debug else None, ncore=mol.ncore)[:2]
				# convert ne_avas to native python type
				ne_avas = np.asscalar(ne_avas)
				# active electrons
				ne_a = (ne_avas + mol.spin) // 2
				ne_b = ne_avas - ne_a
				ne_act = (ne_a, ne_b)
				# active orbs
				no_act = no_avas
				# expansion space orbs
				nocc_avas = ne_a
				nvirt_avas = no_act - nocc_avas
				if calc.model['type'] == 'occ':
					no_exp = nocc_avas
				elif calc.model['type'] == 'virt':
					no_exp = nvirt_avas
				# sanity checks
				assert nocc_avas <= (mol.nocc - mol.ncore)
				assert float(ne_act[0] + ne_act[1]) <= np.sum(calc.hf.mo_occ[mol.ncore:])
			# identical to hf ref?
			if no_exp == 0:
				try:
					raise RuntimeError('\nCAS Error : choice of CAS returns hf solution\n\n')
				except Exception as err:
					sys.stderr.write(str(err))
					raise
			if mol.debug:
				print(' active: ne_act = {0:} , no_act = {1:} , no_exp = {2:}'.format(ne_act, no_act, no_exp))
		return ref_space, exp_space, no_exp, no_act, ne_act


def ref(mol, calc, exp):
		""" calculate reference energy and mo coefficients """
		# set core and cas spaces
		if calc.model['type'] == 'occ':
			exp.core_idx, exp.cas_idx = np.arange(mol.nocc), calc.ref_space
		elif calc.model['type'] == 'virt':
			exp.core_idx, exp.cas_idx = np.arange(mol.ncore), calc.ref_space
		# sort mo coefficients
		if calc.ref['method'] in ['casci','casscf']:
			if calc.ref['active'] == 'manual':
				# inactive region
				inact_elec = mol.nelectron - (calc.ne_act[0] + calc.ne_act[1])
				assert inact_elec % 2 == 0
				inact_orb = inact_elec // 2
				# divide into inactive-active-virtual
				idx = np.asarray([i for i in range(mol.norb) if i not in calc.ref['select']])
				mo = np.hstack((calc.mo[:, idx[:inact_orb]], calc.mo[:, calc.ref['select']], calc.mo[:, idx[inact_orb:]]))
				calc.mo = np.asarray(mo, order='C')
			else:
				from pyscf.mcscf import avas
				calc.mo = avas.avas(calc.hf, calc.ref['ao_labels'], canonicalize=True, ncore=mol.ncore)[2]
			# set properties equal to hf values
			ref = {'energy': [0.0 for i in range(calc.nroots)], 'base': 0.0}
			if calc.target['dipole']:
				ref['dipole'] = [np.zeros(3, dtype=np.float64) for i in range(calc.nroots)]
			if calc.target['trans']:
				ref['trans'] = [np.zeros(3, dtype=np.float64) for i in range(calc.nroots-1)]
			# casscf mo
			if calc.ref['method'] == 'casscf': calc.mo = _casscf(mol, calc, exp)
		else:
			if mol.spin == 0:
				# set properties equal to hf values
				ref = {'energy': [0.0 for i in range(calc.nroots)], 'base': 0.0}
				if calc.target['dipole']:
					ref['dipole'] = [np.zeros(3, dtype=np.float64) for i in range(calc.nroots)]
				if calc.target['trans']:
					ref['trans'] = [np.zeros(3, dtype=np.float64) for i in range(calc.nroots-1)]
			else:
				# exp model
				res = main(mol, calc, exp, calc.model['method'])
				# e_ref
				ref = {'energy': [res['energy'][i] for i in range(calc.nroots)]}
				# dipole_ref
				if calc.target['dipole']:
					ref['dipole'] = [res['dipole'][i] for i in range(calc.nroots)]
				# trans_dipole_ref
				if calc.target['trans']:
					ref['trans'] = [res['trans'][i] for i in range(calc.nroots-1)]
				# e_ref_base
				if calc.base['method'] is None:
					ref['base'] = 0.0
				else:
					if np.abs(ref['e_ref']) < 1.0e-10:
						ref['base'] = ref['energy'][0]
					else:
						res = main(mol, calc, exp, calc.base['method'])
						ref['base'] = res['energy'][0]
		if mol.debug:
			string = '\n REF: core = {:} , cas = {:}\n'
			form = (exp.core_idx.tolist(), exp.cas_idx.tolist())
			string += '      ground state energy = {:.4e} , ground state base energy = {:.4e}\n'
			form += (ref['energy'][0], ref['base'],)
			if calc.nroots > 1:
				for i in range(1, calc.nroots):
					string += '      excitation energy for root {:} = {:.4f}\n'
					form += (i, ref['energy'][i],)
			if calc.target['dipole']:
				for i in range(calc.nroots):
					string += '      dipole moment for root {:} = ({:.4f}, {:.4f}, {:.4f})\n'
					if calc.prot['specific']:
						form += (calc.state['root'], *ref['dipole'][i],)
					else:
						form += (i, *ref['dipole'][i],)
			if calc.target['trans']:
				for i in range(1, calc.nroots):
					string += '      transition dipole moment for excitation {:} > {:} = ({:.4f}, {:.4f}, {:.4f})\n'
					if calc.prot['specific']:
						form += (0, calc.state['root'], *ref['trans'][i-1],)
					else:
						form += (0, i, *ref['trans'][i-1],)
			print(string.format(*form))
		return ref, calc.mo


def main(mol, calc, exp, method):
		""" main prop function """
		# fci calc
		if method == 'fci':
			res_tmp = _fci(mol, calc, exp)
		# cisd calc
		elif method == 'cisd':
			res_tmp = _ci(mol, calc, exp)
		# ccsd / ccsd(t) calc
		elif method in ['ccsd','ccsd(t)']:
			res_tmp = _cc(mol, calc, exp, method == 'ccsd(t)')
		# return correlation energy
		res = {'energy': res_tmp['energy']}
		# return first-order properties
		if calc.target['dipole']:
			res['dipole'] = [_dipole(mol, calc, exp, res_tmp['rdm1'][i]) for i in range(calc.nroots)]
			if calc.nroots > 1:
				res['dipole'][1:] = [res['dipole'][i] - res['dipole'][0] for i in range(1, calc.nroots)]
		if calc.target['trans']:
			res['trans'] = [_trans(mol, calc, exp, res_tmp['t_rdm1'][i], \
									res_tmp['hf_weight'][0], res_tmp['hf_weight'][i+1]) for i in range(calc.nroots-1)]
		return res


def _dipole(mol, calc, exp, cas_rdm1, trans=False):
		""" calculate electronic (transition) dipole moment """
		# init (transition) rdm1
		if not trans:
			rdm1 = np.diag(calc.occup)
		else:
			rdm1 = np.zeros([mol.norb, mol.norb], dtype=np.float64)
		# insert correlated subblock
		rdm1[exp.cas_idx[:, None], exp.cas_idx] = cas_rdm1
		# init elec_dipole
		elec_dipole = np.empty(3, dtype=np.float64)
		for i in range(3):
			# mo ints
			mo_ints = np.einsum('pi,pq,qj->ij', calc.mo, mol.dipole[i], calc.mo)
			# elec dipole
			elec_dipole[i] = np.einsum('ij,ij->', rdm1, mo_ints)
		# remove noise
		elec_dipole = np.array([elec_dipole[i] if np.abs(elec_dipole[i]) > 1.0e-15 else 0.0 for i in range(elec_dipole.size)])
		# 'correlation' dipole
		if not trans:
			elec_dipole -= calc.prop['hf']['dipole']
		return elec_dipole


def _trans(mol, calc, exp, cas_t_rdm1, hf_weight_gs, hf_weight_ex):
		""" calculate electronic transition dipole moment """
		return _dipole(mol, calc, exp, cas_t_rdm1, True) * np.sign(hf_weight_gs) * np.sign(hf_weight_ex)


def base(mol, calc, exp):
		""" calculate base energy and mo coefficients """
		# set core and cas spaces
		exp.core_idx, exp.cas_idx = core_cas(mol, exp, calc.exp_space)
		# init rdm1
		rdm1 = None
		# zeroth-order energy
		if calc.base['method'] is None:
			base = {'energy': 0.0}
		# cisd base
		elif calc.base['method'] == 'cisd':
			res = _ci(mol, calc, exp)
			base = {'energy': res['energy'][0]}
			if 'rdm1' in res:
				rdm1 = res['rdm1']
				if mol.spin > 0:
					rdm1 = rdm1[0] + rdm1[1]
		# ccsd / ccsd(t) base
		elif calc.base['method'] in ['ccsd','ccsd(t)']:
			res = _cc(mol, calc, exp, calc.base['method'] == 'ccsd(t)')
			base = {'energy': res['energy'][0]}
			if 'rdm1' in res:
				rdm1 = res['rdm1']
				if mol.spin > 0:
					rdm1 = rdm1[0] + rdm1[1]
		# NOs
		if (calc.orbs['occ'] == 'cisd' or calc.orbs['virt'] == 'cisd') and rdm1 is None:
			res = _ci(mol, calc, exp)
			rdm1 = res['rdm1']
			if mol.spin > 0:
				rdm1 = rdm1[0] + rdm1[1]
		elif (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd') and rdm1 is None:
			res = _cc(mol, calc, exp, False)
			rdm1 = res['rdm1']
			if mol.spin > 0:
				rdm1 = rdm1[0] + rdm1[1]
		# occ-occ block (local or NOs)
		if calc.orbs['occ'] != 'can':
			if calc.orbs['occ'] in ['cisd', 'ccsd']:
				occup, no = symm.eigh(rdm1[:(mol.nocc-mol.ncore), :(mol.nocc-mol.ncore)], calc.orbsym[mol.ncore:mol.nocc])
				calc.mo[:, mol.ncore:mol.nocc] = np.einsum('ip,pj->ij', calc.mo[:, mol.ncore:mol.nocc], no[:, ::-1])
			elif calc.orbs['occ'] == 'pm':
				calc.mo[:, mol.ncore:mol.nocc] = lo.pm(mol, calc.mo[:, mol.ncore:mol.nocc]).kernel()
			elif calc.orbs['occ'] == 'fb':
				calc.mo[:, mol.ncore:mol.nocc] = lo.Boys(mol, calc.mo[:, mol.ncore:mol.nocc]).kernel()
			elif calc.orbs['occ'] in ['ibo-1','ibo-2']:
				iao = lo.iao.iao(mol, calc.mo[:, mol.core:mol.nocc])
				if calc.orbs['occ'] == 'ibo-1':
					iao = lo.vec_lowdin(iao, calc.hf.get_ovlp())
					calc.mo[:, mol.ncore:mol.nocc] = lo.ibo.ibo(mol, calc.mo[:, mol.ncore:mol.nocc], iao)
				elif calc.orbs['occ'] == 'ibo-2':
					calc.mo[:, mol.ncore:mol.nocc] = lo.ibo.pm(mol, calc.mo[:, mol.ncore:mol.nocc], iao).kernel()
		# virt-virt block (local or NOs)
		if calc.orbs['virt'] != 'can':
			if calc.orbs['virt'] in ['cisd', 'ccsd']:
				occup, no = symm.eigh(rdm1[-mol.nvirt:, -mol.nvirt:], calc.orbsym[mol.nocc:])
				calc.mo[:, mol.nocc:] = np.einsum('ip,pj->ij', calc.mo[:, mol.nocc:], no[:, ::-1])
			elif calc.orbs['virt'] == 'pm':
				calc.mo[:, mol.nocc:] = lo.pm(mol, calc.mo[:, mol.nocc:]).kernel()
			elif calc.orbs['virt'] == 'fb':
				calc.mo[:, mol.nocc:] = lo.Boys(mol, calc.mo[:, mol.nocc:]).kernel()
		# extra calculation for non-invariant ccsd(t)
		if calc.base['method'] == 'ccsd(t)' and (calc.orbs['occ'] != 'can' or calc.orbs['virt'] != 'can'):
			res = _cc(mol, calc, exp, True)
			base['energy'] = res['energy'][0]
		return base


def _casscf(mol, calc, exp):
		""" casscf calc """
		# casscf ref
		cas = mcscf.CASSCF(calc.hf, calc.no_act, calc.ne_act)
		# fci solver
		if abs(calc.ne_act[0]-calc.ne_act[1]) == 0:
			cas.fcisolver = fci.direct_spin0_symm.FCI(mol)
		else:
			cas.fcisolver = fci.direct_spin1_symm.FCI(mol)
		cas.fcisolver.conv_tol = calc.thres['init']
		cas.conv_tol = 1.0e-10
		cas.max_stepsize = .01
		cas.max_cycle_macro = 500
		cas.canonicalization = False
		# wfnsym
		cas.fcisolver.wfnsym = calc.state['wfnsym']
		# frozen (inactive)
		cas.frozen = (mol.nelectron - (calc.ne_act[0] + calc.ne_act[1])) // 2
		# debug print
		if mol.debug: cas.verbose = 4
		# fix spin if non-singlet
		if mol.spin > 0:
			sz = abs(calc.ne_act[0]-calc.ne_act[1]) * .5
			cas.fix_spin_(ss=sz * (sz + 1.))
		# state-specific or state-averaged calculation
		if calc.nroots > 1:
			if calc.prot['specific']:
				# state-specific
				cas.state_specific_(state=calc.state['root'])
			else:
				# state-averaged
				weights = np.ones(calc.nroots, dtype=np.float64) / calc.nroots
				cas.state_average_(weights)
		# run casscf calc
		cas.kernel(calc.mo)
		# calculate spin
		s, mult = cas.fcisolver.spin_square(cas.ci, calc.no_act, calc.ne_act)
		# check for correct spin
		if (mol.spin + 1) - mult > 1.0e-05:
			try:
				raise RuntimeError(('\ncasscf Error : spin contamination\n'
									'2*S + 1 = {0:.3f}\n\n').\
									format(mult))
			except Exception as err:
				sys.stderr.write(str(err))
				raise
		# save mo
		fock_ao = cas.get_fock(cas.mo_coeff, cas.ci, None, None)
		fock = np.einsum('pi,pq,qj->ij', cas.mo_coeff, fock_ao, cas.mo_coeff)
		mo = np.empty_like(cas.mo_coeff)
		# core region
		if mol.ncore > 0:
			c = symm.eigh(fock[:mol.ncore, :mol.ncore], \
								cas.mo_coeff.orbsym[:mol.ncore])[1]
			mo[:, :mol.ncore] = np.einsum('ip,pj->ij', cas.mo_coeff[:, :mol.ncore], c)
		# inactive region (excl. core)
		if cas.frozen > mol.ncore:
			c = symm.eigh(fock[mol.ncore:cas.frozen, mol.ncore:cas.frozen], \
								cas.mo_coeff.orbsym[mol.ncore:cas.frozen])[1]
			mo[:, mol.ncore:cas.frozen] = np.einsum('ip,pj->ij', cas.mo_coeff[:, mol.ncore:cas.frozen], c)
		# active region
		mo[:, cas.frozen:(cas.frozen + calc.no_act)] = cas.mo_coeff[:, cas.frozen:(cas.frozen + calc.no_act)]
		# virtual region
		if mol.norb - (cas.frozen + calc.no_act) > 0:
			c = symm.eigh(fock[(cas.frozen + calc.no_act):, (cas.frozen + calc.no_act):], \
								cas.mo_coeff.orbsym[(cas.frozen + calc.no_act):])[1]
			mo[:, (cas.frozen + calc.no_act):] = np.einsum('ip,pj->ij', cas.mo_coeff[:, (cas.frozen + calc.no_act):], c)
		return mo


def _fci(mol, calc, exp):
		""" fci calc """
		# init fci solver
		if mol.spin == 0:
			solver = fci.direct_spin0_symm.FCI(mol)
		else:
			solver = fci.direct_spin1_symm.FCI(mol)
		# settings
		solver.conv_tol = calc.thres['init']
		if calc.target['dipole'] or calc.target['trans']:
			solver.conv_tol_residual = calc.thres['init']
		solver.max_cycle = 500
		solver.max_space = 25
		solver.davidson_only = True
		# wfnsym
		solver.wfnsym = calc.state['wfnsym']
		# number of roots
		solver.nroots = calc.state['root'] + 1
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
			raise RuntimeError('\nfci Error : no initial guess found\n\n')
		# perform calc
		e, c = solver.kernel(h1e, h2e, len(exp.cas_idx), nelec, ecore=mol.e_core)
		# collect results
		if solver.nroots == 1:
			energy = [e]
			civec = [c]
		else:
			if calc.prot['specific']:
				energy = [e[0], e[calc.state['root']]]
				civec = [c[0], c[calc.state['root']]]
			else:
				energy = e
				civec = c
		# sanity check
		for i in range(calc.nroots):
			# calculate spin
			s, mult = solver.spin_square(civec[i], len(exp.cas_idx), nelec)
			# check for correct spin
			assert (mol.spin + 1) - mult < 1.0e-05, ('\nfci Error : spin contamination for root = {0:}\n\n'
													'2*S + 1 = {1:.6f}\n'
													'core_idx = {2:} , cas_idx = {3:}\n\n').\
													format(i, mult, exp.core_idx, exp.cas_idx)
		# e_corr
		res = {'energy': [energy[0] - calc.prop['hf']['energy']]}
#		if exp.order < exp.max_order: e['e_corr'] += np.float64(0.001) * np.random.random_sample()
		# e_exc
		if calc.nroots > 1:
			for i in range(1, calc.nroots):
				res['energy'].append(energy[i] - energy[0])
		# fci rdm1 and t_rdm1
		if calc.target['dipole']:
			res['rdm1'] = [solver.make_rdm1(civec[i], len(exp.cas_idx), nelec) for i in range(calc.nroots)]
		if calc.target['trans']:
			res['t_rdm1'] = [solver.trans_rdm1(civec[0], civec[i], len(exp.cas_idx), nelec) for i in range(1, calc.nroots)]
			res['hf_weight'] = [civec[i][0, 0] for i in range(calc.nroots)]
		return res


def _ci(mol, calc, exp):
		""" cisd calc """
		# get integrals
		h1e, h2e = _prepare(mol, calc, exp)
		mol_tmp = gto.M(verbose=1)
		mol_tmp.incore_anyway = True
		mol_tmp.max_memory = mol.max_memory
		if mol.spin == 0:
			hf = scf.Rhf(mol_tmp)
		else:
			hf = scf.Uhf(mol_tmp)
		hf.get_hcore = lambda *args: h1e
		hf._eri = h2e 
		# init ccsd
		if mol.spin == 0:
			cisd = ci.cisd.CISD(hf, mo_coeff=np.eye(len(exp.cas_idx)), mo_occ=calc.occup[exp.cas_idx])
		else:
			cisd = ci.ucisd.UCISD(hf, mo_coeff=np.array((np.eye(len(exp.cas_idx)), np.eye(len(exp.cas_idx)))), \
									mo_occ=np.array((calc.occup[exp.cas_idx] > 0., calc.occup[exp.cas_idx] == 2.), dtype=np.double))
		# settings
		cisd.conv_tol = calc.thres['init']
		cisd.max_cycle = 500
		cisd.max_space = 25
		eris = cisd.ao2mo()
		# calculate cisd energy
		cisd.kernel(eris=eris)
		# e_corr
		res = {'energy': [cisd.e_corr]}
		# rdm1
		if exp.order == 0 and (calc.orbs['occ'] == 'cisd' or calc.orbs['virt'] == 'cisd'):
			res['rdm1'] = cisd.make_rdm1()
		return res


def _cc(mol, calc, exp, pt=False):
		""" ccsd / ccsd(t) calc """
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
		ccsd.conv_tol = calc.thres['init']
		if exp.order == 0 and (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd'):
			ccsd.conv_tol_normt = calc.thres['init']
		ccsd.max_cycle = 500
		if exp.order > 0:
			# avoid async function execution if requested
			ccsd.async_io = calc.misc['async']
			# avoid I/O if not async
			if not calc.misc['async']: ccsd.incore_complete = True
		eris = ccsd.ao2mo()
		# calculate ccsd energy
		ccsd.kernel(eris=eris)
		# e_corr
		res = {'energy': [ccsd.e_corr]}
		# rdm1
		if exp.order == 0 and (calc.orbs['occ'] == 'ccsd' or calc.orbs['virt'] == 'ccsd'):
			ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
			res['rdm1'] = ccsd.make_rdm1()
		# calculate (t) correction
		if pt:
			if np.amin(calc.occup[exp.cas_idx]) == 1.0:
				if len(np.where(calc.occup[exp.cas_idx] == 1.)[0]) >= 3:
					res['energy'][0] += ccsd.ccsd_t(eris=eris)
			else:
				res['energy'][0] += ccsd.ccsd_t(eris=eris)
		return res


def core_cas(mol, exp, tup):
		""" define core and cas spaces """
		cas_idx = np.asarray(sorted(exp.incl_idx + sorted(tup.tolist())))
		core_idx = np.asarray(sorted(list(set(range(mol.nocc)) - set(cas_idx))))
		return core_idx, cas_idx


def _prepare(mol, calc, exp):
		""" generate input for correlated calculation """
		# extract cas integrals and calculate core energy
		if mol.e_core is None or exp.model['type'] == 'occ':
			if len(exp.core_idx) > 0:
				core_dm = np.einsum('ip,jp->ij', calc.mo[:, exp.core_idx], calc.mo[:, exp.core_idx]) * 2
				vj, vk = scf.hf.get_jk(mol, core_dm)
				mol.core_vhf = vj - vk * .5
				mol.e_core = mol.energy_nuc() + np.einsum('ij,ji', core_dm, mol.hcore)
				mol.e_core += np.einsum('ij,ji', core_dm, mol.core_vhf) * .5
			else:
				mol.e_core = mol.energy_nuc()
				mol.core_vhf = 0
		h1e_cas = np.einsum('pi,pq,qj->ij', calc.mo[:, exp.cas_idx], mol.hcore + mol.core_vhf, calc.mo[:, exp.cas_idx])
		h2e_cas = ao2mo.incore.full(mol.eri, calc.mo[:, exp.cas_idx])
		return h1e_cas, h2e_cas


