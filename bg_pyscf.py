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
from functools import reduce
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
				nocc = int(hf.mo_occ.sum()) // 2
				nvirt = norb - nocc
				# duplicate total hf energy
				e_hf = hf.e_tot
				#
				return hf, e_hf, norb, nocc, nvirt


		def int_trans(self, _mol, _calc):
				""" integral transformation """
				if (_calc.exp_virt == 'natural'):
					# calculate ccsd density matrix
					ccsd = cc.CCSD(_mol.hf)
					ccsd.kernel()
					dm = ccsd.make_rdm1()
					occup, no = sp.linalg.eigh(dm[_mol.nocc:,_mol.nocc:])
					mo_coeff_virt = np.dot(_mol.hf.mo_coeff[:,_mol.nocc:], no[:,::-1])
					trans_mat = _mol.hf.mo_coeff
					trans_mat[:,_mol.nocc:] = mo_coeff_virt
				else:
					trans_mat = _mol.hf.mo_coeff
				# transform 1- and 2-electron integrals
				h1e = reduce(np.dot, (np.transpose(trans_mat), _mol.hf.get_hcore(), trans_mat))
				h2e = ao2mo.kernel(_mol, trans_mat) # with four-fold permutation symmetry
				h2e = ao2mo.restore(1, h2e, _mol.norb) # remove symmetry
				#
				return h1e, h2e


		def corr_input(self, _mol, _exp, _tup):
				""" generate input for correlated calculation """
				# generate orbital lists
				cas_idx = sorted(_exp.incl_idx + _tup.tolist())
				core_idx = sorted(_exp.frozen_idx + list(set(range(_mol.nocc)) - set(cas_idx)))
				cas_idx = sorted(list(set(cas_idx) - set(core_idx)))
				# extract core and cas integrals and calculated core energy (not needed for cc calc)
				if (_cas.exp_type == 'FCI'):
					if (len(_exp.core_idx) > 0):
						vhf_core = np.einsum('iipq->pq', _mol.h2e[core_idx][:,core_idx]) * 2
						vhf_core -= np.einsum('piiq->pq', _mol.h2e[:,core_idx][:,:,core_idx])
						h1e_cas = (_mol.h1e + vhf_core)[cas_idx][:,cas_idx]
					else:
						h1e_cas = _mol.h1e[cas_idx][:,cas_idx]
					h2e_cas = _mol.h2e[cas_idx][:,cas_idx][:,:,cas_idx][:,:,:,cas_idx]
					# set core energy
					if (len(core_idx) > 0):
						e_core = _mol.h1e[core_idx][:,core_idx].trace() * 2 + \
									vhf_core[core_idx][:,core_idx].trace() + \
									_mol.energy_nuc()
					else:
						e_core = _mol.energy_nuc()
				else:
					h1e_cas = None
					h2e_cas = None
					e_core = None
				#
				return cas_idx, core_idx, h1e_cas, h2e_cas, e_core


		def corr_calc(self, _mol, _calc, _exp):
				""" correlated calculation """
				if (_calc.exp_model == 'CCSD'):
					# ccsd calculation
					ccsd = cc.CCSD(_mol.hf)
					ccsd.frozen = sorted(list(set(range(_mol.norb))-set(_exp.cas_idx)))
					ccsd.kernel()
					e_corr = ccsd.e_corr
				else:
					# init fci solver
					if (_mol.spin == 0):
						fcisolver = fci.direct_spin0.FCI()
					else:
						fcisolver = fci.direct_spin1.FCI()
					fcisolver.conv_tol = 1.0e-08
					fcisolver.max_cycle = 100
					fcisolver.max_memory = _mol.max_memory
					# casci calculation
					casci = fcisolver.kernel(_exp.h1e_cas, _exp.h2e_cas, len(_exp.cas_idx),
												_mol.nelectron - 2 * len(_exp.core_idx), ecore=_exp.e_core)
					e_corr = casci[0] - _mol.e_hf
				#
				return e_corr


