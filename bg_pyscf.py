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

import numpy as np
import scipy as sp
from pyscf import scf, ao2mo, cc, fci


class PySCFCls():
		""" pyscf class """
		def hf_calc(self, _mol):
				""" underlying hf calculation """
				# perform hf calc
				hf = scf.RHF(_mol)
				hf.conv_tol = 1.0e-10
				hf.kernel()
				# determine dimensions
				norb = hf.mo_coeff.shape[1]
				nocc = int(hf.mo_occ.sum()) // 2 - _mol.ncore
				nvirt = norb - nocc
				#
				return hf, norb, nocc, nvirt


		def int_trans(self, _mol, _natural):
				""" integral transformation """
				if (_natural):
					# calculate ccsd density matrix
					ccsd = cc.CCSD(_mol.hf)
					ccsd.kernel()
					dm = ccsd.make_rdm1()
					occup, no = sp.linalg.eigh(dm)
					no = _mol.hf.mo_coeff.dot(no)
					trans_mat = no[:,::-1]
				else:
					trans_mat = _mol.hf.mo_coeff
				# transform 1- and 2-electron integrals
				h1e = reduce(np.dot, (trans_mat.T, _mol.hf.get_hcore(), trans_mat))
				h2e = ao2mo.kernel(_mol, trans_mat) # with four-fold permutation symmetry
				h2e = ao2mo.restore(1, h2e, trans_mat.shape[1]) # remove symmetry
				#
				return h1e, h2e


		def corr_calc(self, _mol, _calc, _exp):
				""" correlated calculation """
				if (_calc.exp_model == 'CCSD'):
					ccsd = cc.CCSD(_mol.hf)
					ccsd.frozen = sorted(list(set(range(_mol.norb))-set(_exp.cas_idx)))
					# ccsd calculation
					ccsd.kernel()
					energy = ccsd.e_corr
				else:
					# extract 1- and 2-electron integrals
					if (len(_exp.core_idx) > 0):
						vhf_core = np.einsum('iipq->pq', _calc.h2e[_exp.core_idx][:,_exp.core_idx]) * 2
						vhf_core -= np.einsum('piiq->pq', _calc.h2e[:,_exp.core_idx][:,:,_exp.core_idx])
						h1e_cas = (_calc.h1e + vhf_core)[_exp.cas_idx][:,_exp.cas_idx]
					else:
						h1e_cas = _calc.h1e[_exp.cas_idx][:,_exp.cas_idx]
					h2e_cas = _calc.h2e[_exp.cas_idx][:,_exp.cas_idx][:,:,_exp.cas_idx][:,:,:,_exp.cas_idx]
					# set core energy
					if (len(_exp.core_idx) > 0):
						e_core = h1e[_exp.core_idx][:,_exp.core_idx].trace() * 2 + \
									vhf_core[_exp.core_idx][:,_exp.core_idx].trace() + \
									_mol.energy_nuc()
					else:
						e_core = _mol.energy_nuc()
					# init fci solver
					if (_mol.spin == 0):
						fcisolver = fci.direct_spin0.FCI()
					else:
						fcisolver = fci.direct_spin1.FCI()
					fcisolver.conv_tol = 1.0e-08
					fcisolver.max_cycle = 100
					fcisolver.max_memory = _mol.max_memory
					# casci calculation
					casci = fcisolver.kernel(h1e_cas, h2e_cas, len(_exp.cas_idx),
												_mol.nelec - 2 * len(_exp.core_idx), ecore=e_core)
					energy = casci[0] - _mol.hf.e_tot
				#
				return energy


		def corr_input(self, _mol, _exp, _tup):
				""" generate input for casci calculation """
				cas_idx = sorted(_exp.incl_idx + _tup.tolist())
				core_idx = sorted(_exp.frozen_idx + list(set(range(_mol.nocc))-set(cas_idx)))
				#
				return cas_idx, core_idx


