#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
mbe testing module
"""

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import pytest
import numpy as np
from mpi4py import MPI
from typing import Union

from mbe import main, _inc, _sum
from kernel import hf, ref_mo, ints, dipole_ints, ref_prop, hubbard_h1e, \
                   hubbard_eri
from tools import n_tuples, occ_prune, virt_prune, tuples, hash_2d
from system import MolCls
from parallel import MPICls
from calculation import CalcCls
from expansion import ExpCls


test_cases_main = [
    ('h2o', 1, -249055688365223385, 9199082625845137542, -0.0374123708341898, -0.0018267714680604286, np.array([-0.0374123708341898]), 10, 9, 11, np.array([-0.004676546354273725]), np.array([0.0018267714680604286]), np.array([0.010886635891736773])),
    ('h2o', 2, 8509729643108359722, 8290417336063232159, -0.11605435599270209, -0.0001698239845069338, np.array([-0.11605435599270209]), np.array([65]), np.array([57]), np.array([69]), np.array([-0.004144798428310789]), np.array([0.0001698239845069338]), np.array([0.009556269221292268]))
]


test_cases_sum = [
    ('energy', np.array([1, 7, 8], dtype=np.int64), 1.2177665733781107),
    ('dipole', np.array([1, 7, 8], dtype=np.int64), np.array([1.21776657, 1.21776657, 1.21776657], dtype=np.float64)),
    ('excitation', np.array([1, 7, 8], dtype=np.int64), 1.2177665733781107),
    ('trans', np.array([1, 7, 8], dtype=np.int64), np.array([1.21776657, 1.21776657, 1.21776657], dtype=np.float64)),
    ('energy', np.array([1, 7, 8, 9], dtype=np.int64), 2.7229882355444195),
    ('dipole', np.array([1, 7, 8, 9], dtype=np.int64), np.array([2.72298824, 2.72298824, 2.72298824], dtype=np.float64)),
    ('excitation', np.array([1, 7, 8, 9], dtype=np.int64), 2.7229882355444195),
    ('trans', np.array([1, 7, 8, 9], dtype=np.int64), np.array([2.72298824, 2.72298824, 2.72298824], dtype=np.float64))
]


@pytest.mark.parametrize(argnames='mol, order, ref_hashes_sum, ref_hashes_amax, ref_inc_sum, ref_inc_amax, ref_tot, ref_mean_ndets, ref_min_ndets, ref_max_ndets, ref_mean_inc, ref_min_inc, ref_max_inc', \
                         argvalues=test_cases_main, \
                         ids=['-'.join([case[0], str(case[1])]) for case in test_cases_main], \
                         indirect=['mol'])
def test_main(mol: MolCls, order: int, ref_hashes_sum: int, \
              ref_hashes_amax: int, ref_inc_sum: float, ref_inc_amax: float, \
              ref_tot: np.ndarray, ref_mean_ndets: int, ref_min_ndets: int, \
              ref_max_ndets: int, ref_mean_inc: float, ref_min_inc: float, \
              ref_max_inc: float):
        """
        this function tests main
        """
        mpi = MPICls()

        calc = CalcCls(mol.ncore, mol.nelectron, mol.groupname)

        calc.target_mbe = 'energy'
        calc.misc['rst'] = False

        mol.nocc, mol.nvirt, mol.norb, calc.hf, calc.prop['hf']['energy'], \
        calc.prop['hf']['dipole'], calc.occup, calc.orbsym, \
        calc.mo_coeff = hf(mol, calc.hf_ref)

        calc.mo_coeff, calc.nelec, calc.ref_space = ref_mo(mol, calc.mo_coeff, calc.occup, calc.orbsym, \
                                                           calc.orbs, calc.ref, calc.model, calc.hf)

        mol.hcore, mol.vhf, mol.eri = ints(mol, calc.mo_coeff, mpi.global_master, mpi.local_master, \
                                           mpi.global_comm, mpi.local_comm, mpi.master_comm, mpi.num_masters)

        mol.dipole_ints = dipole_ints(mol, calc.mo_coeff)

        calc.prop['ref'][calc.target_mbe] = ref_prop(mol, calc.occup, calc.target_mbe, \
                                                     calc.orbsym, calc.model['hf_guess'], \
                                                     calc.ref_space, calc.model, calc.orbs['type'], \
                                                     calc.state, calc.prop['hf']['energy'], \
                                                     calc.prop['hf']['dipole'], calc.base['method'])

        exp = ExpCls(mol, calc)

        hashes = []
        inc = []

        for exp.order in range(1, order+1):

            exp.n_tuples['inc'].append(n_tuples(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc], \
                                            exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]], \
                                            occ_prune(calc.occup, calc.ref_space), \
                                            virt_prune(calc.occup, calc.ref_space), exp.order))

            hashes_win, inc_win, tot, mean_ndets, min_ndets, max_ndets, mean_inc, min_inc, max_inc = main(mpi, mol, calc, exp)

            hashes.append(np.ndarray(buffer=hashes_win, dtype=np.int64, shape=(exp.n_tuples['inc'][exp.order-1],)))

            inc.append(np.ndarray(buffer=inc_win, dtype=np.float64, shape=(exp.n_tuples['inc'][exp.order-1], 1)))

            exp.prop[calc.target_mbe]['hashes'].append(hashes_win)

            exp.prop[calc.target_mbe]['inc'].append(inc_win)

            exp.mean_ndets.append(mean_ndets)
            exp.min_ndets.append(min_ndets)
            exp.max_ndets.append(max_ndets)

            exp.mean_inc.append(mean_inc)
            exp.min_inc.append(min_inc)
            exp.max_inc.append(max_inc)

        assert isinstance(hashes_win, MPI.Win)
        assert isinstance(inc_win, MPI.Win)
        assert np.sum(hashes[-1]) == ref_hashes_sum
        assert np.amax(hashes[-1]) == ref_hashes_amax
        assert np.sum(inc[-1]) == pytest.approx(ref_inc_sum)
        assert np.amax(inc[-1]) == pytest.approx(ref_inc_amax)
        assert tot == pytest.approx(ref_tot)
        assert mean_ndets == ref_mean_ndets
        assert min_ndets == ref_min_ndets
        assert max_ndets == ref_max_ndets
        assert mean_inc == pytest.approx(ref_mean_inc)
        assert min_inc == pytest.approx(ref_min_inc)
        assert max_inc == pytest.approx(ref_max_inc)


def test_inc():
        """
        this function tests _inc
        """
        n = 4
        model = {'method': 'fci', 'cc_backend': 'pyscf', 'solver': 'pyscf_spin0', 'hf_guess': True}
        prop = {'hf': {'energy': 0., 'dipole': None}, 'ref': {'energy': 0.}}
        state = {'wfnsym': 'A', 'root': 0}
        occup = np.array([2.] * (n // 2) + [0.] * (n // 2), dtype=np.float64)
        orbsym = np.zeros(n, dtype=np.int64)
        h1e_cas, h2e_cas = hubbard_h1e((1, n), False), hubbard_eri((1, n), 2.)
        core_idx, cas_idx = np.array([], dtype=np.int64), np.arange(n, dtype=np.int64)

        e, ndets, n_elec = _inc(model, None, 'can', 0, occup, 'energy', state, 'c1', orbsym,
                                prop, 0, h1e_cas, h2e_cas, core_idx, cas_idx, 0, None)
        
        assert e == pytest.approx(-2.875942809005048)
        assert ndets == 36
        assert n_elec == (2, 2)
        

@pytest.mark.parametrize(argnames='target_mbe, tup, ref_res', argvalues=test_cases_sum, \
                         ids=[case[0] + '-' + str(case[1]) for case in test_cases_sum])
def test_sum(target_mbe: str, tup: np.ndarray, ref_res: Union[float, np.ndarray]):
        """
        this function tests _sum
        """
        exp_space = [np.arange(10, dtype=np.int64), np.array([1, 2, 3, 4, 5, 7, 8, 9], dtype=np.int64)]
        nocc = 3
        min_order = 2
        ref_occ = False
        ref_virt = False
        hashes = []
        exp_occ = exp_space[0][exp_space[0] < nocc]
        exp_virt = exp_space[0][nocc <= exp_space[0]]
        hashes.append(hash_2d(np.array([tup for tup in tuples(exp_occ, exp_virt, ref_occ, ref_virt, 2)], dtype=np.int64)))
        hashes[0].sort()
        exp_occ = exp_space[1][exp_space[1] < nocc]
        exp_virt = exp_space[1][nocc <= exp_space[1]]
        hashes.append(hash_2d(np.array([tup for tup in tuples(exp_occ, exp_virt, ref_occ, ref_virt, 3)], dtype=np.int64)))
        hashes[1].sort()
        inc = []
        np.random.seed(1)
        inc.append(np.random.rand(21))
        np.random.seed(2)
        inc.append(np.random.rand(36))

        res = _sum(nocc, target_mbe, min_order, tup.size, inc, hashes, ref_occ, ref_virt, tup)

        assert res == pytest.approx(ref_res)
