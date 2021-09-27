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
from typing import Union

from mbe import _inc, _sum
from kernel import hubbard_h1e, hubbard_eri
from tools import tuples, hash_2d


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
