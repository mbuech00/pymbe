#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_pyscf.py: pyscf-related routines for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from pyscf import gto, scf, ao2mo, cc, fci


class PySCFCls():
		""" pyscf class """
		def hf_calc(mol):
				""" underlying hf calculation """
				# perform hf calc
				hf = scf.RHF(mol)
				hf.conv_tol = 1.0e-10
				hf.kernel()
				# determine dimensions
				norb = hf.mo_coeff.shape[1]
				nocc = int(hf.mo_occ.sum()) // 2 - mol.ncore
				nvirt = norb - nocc
				#
				return hf, norb, nocc, nvirt


		def int_trans(mol, natural):
				""" integral transformation """
				if (natural):
					# calculate ccsd density matrix
					ccsd = cc.CCSD(mol.hf)
					ccsd.kernel()
					dm = ccsd.make_rdm1()
					occup, no = sp.linalg.eigh(dm)
					no = hf.mo_coeff.dot(no)
					trans_mat = no[:,::-1]
				else:
					trans_mat = mol.hf.mo_coeff
				# transform 1- and 2-electron integrals
				h1e = reduce(np.dot, (trans_mat.T, mol.hf.get_hcore(), trans_mat))
				h2e = ao2mo.kernel(mol, trans_mat) # with four-fold permutation symmetry
				h2e = ao2mo.restore(1, h2e, trans_mat.shape[1]) # remove symmetry
				#
				return h1e, h2e


		def corr_calc(mol, h1e, h2e, core_idx, cas_idx, model):
				""" correlated calculation """
				if (model == 'ccsd'):
					ccsd = cc.CCSD(mol.hf)
					ccsd.frozen = sorted(list(set(range(mol.norb))-set(exp.cas_idx)))
					# ccsd calculation
					ccsd.kernel()
					energy = ccsd.e_corr
				else:
					# extract 1- and 2-electron integrals
					if (len(core_idx) > 0):
						vhf_core = np.einsum('iipq->pq', h2e[core_idx][:,core_idx]) * 2
						vhf_core -= np.einsum('piiq->pq', h2e[:,core_idx][:,:,core_idx])
						h1e_cas = (h1e + vhf_core)[cas_idx][:,cas_idx]
					else:
						h1e_cas = h1e[cas_idx][:,cas_idx]
					h2e_cas = h2e[cas_idx][:,cas_idx][:,:,cas_idx][:,:,:,cas_idx]
					# set core energy
					if (len(core_idx) > 0):
						e_core = h1e[core_idx][:,core_idx].trace() * 2 + \
									vhf_core[core_idx][:,core_idx].trace() + mol.energy_nuc()
					else:
						e_core = mol.energy_nuc()
					# init fci solver
					if (mol.spin == 0):
						fcisolver = fci.direct_spin0.FCI()
					else:
						fcisolver = fci.direct_spin1.FCI()
					fcisolver.conv_tol = 1.0e-08
					fcisolver.max_cycle = 100
					fcisolver.max_memory = mol.max_memory
					# casci calculation
					casci = fcisolver.kernel(h1e_cas, h2e_cas, len(cas_idx),
												mol.nelec - 2 * len(core_idx), ecore=e_core)
					energy = casci[0] - mol.hf.e_tot
				#
				return energy


		def corr_input(incl_idx, frozen_idx, tup):
				""" generate input for casci calculation """
				cas_idx = sorted(incl_idx+tup.tolist())
				core_idx = sorted(frozen_idx+list(set(range(nocc))-set(cas_idx)))
				#
				return cas_idx, core_idx


