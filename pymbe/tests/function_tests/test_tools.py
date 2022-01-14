#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools testing module
"""

from __future__ import annotations

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import pytest
import numpy as np
from typing import TYPE_CHECKING
from types import GeneratorType

from pymbe.tools import time_str, fsum, hash_2d, hash_1d, hash_lookup, tuples, \
                        start_idx, _comb_idx, _idx, n_tuples, cas, core_cas, \
                        _cas_idx_cart, _coor_to_idx, idx_tril, pi_space, \
                        _pi_orbs, pi_prune, occ_prune, virt_prune, nelec, \
                        ndets, mat_idx, near_nbrs, natural_keys, _convert, \
                        intervals, inc_dim, inc_shape

if TYPE_CHECKING:

    from typing import Union, Tuple, List, Optional


test_cases_fsum = [
    ('1d', np.arange(10., dtype=np.float64), 45.),
    ('2d', np.arange(4. ** 2, dtype=np.float64).reshape(4, 4), np.array([24., 28., 32., 36.], dtype=np.float64))
]

test_cases_hash_lookup = [
    ('present', np.array([1, 3, 5, 7, 9], dtype=np.int64), True),
    ('absent', np.array([1, 3, 5, 7, 11], dtype=np.int64), False)
]

test_cases_tuples = [
    ('no_ref_space', np.array([], dtype=np.int64), 18),
    ('ref_space', np.array([3, 4], dtype=np.int64), 20)
]

test_cases_start_idx = [
    ('all', np.array([1, 2], dtype=np.int64), np.array([6, 7, 12], dtype=np.int64), (2, 3, 1)),
    ('occ', np.array([0, 1, 2], dtype=np.int64), None, (3, 0, 0)),
    ('virt', None, np.array([6, 9, 12], dtype=np.int64), (3, -1, 2))
]

test_cases_comb_idx = [
    (np.array([1, 2, 6, 7], dtype=np.int64), 12.),
    (np.array([1, 2], dtype=np.int64), 5.),
    (np.array([5, 7], dtype=np.int64), 13.)
]

test_cases_idx = [
    (1, 3.),
    (2, 12.),
    (3, 19.)
]

test_cases_n_tuples = [
    ('empty', False, False, 1460500),
    ('ref_occ', True, False, 2118508),
    ('ref_virt', False, True, 1460752),
    ('ref_occ_virt', True, True, 2118760)
]

test_cases_pi_prune = [
    ('3_tot_2_pi', np.array([0, 1, 2], dtype=np.int64), True),
    ('7_tot_4_pi', np.array([0, 1, 2, 4, 5, 6, 7], dtype=np.int64), True),
    ('4_tot_3_pi', np.array([0, 1, 2, 4], dtype=np.int64), False),
    ('5_tot_3_pi', np.array([0, 1, 2, 5, 6], dtype=np.int64), False)
]

test_cases_occ_prune = [
    ('occ', np.arange(2, 7, dtype=np.int64), True),
    ('no_occ', np.arange(3, 7, dtype=np.int64), False)
]

test_cases_virt_prune = [
    ('virt', np.arange(1, 4, dtype=np.int64), True),
    ('no_virt', np.arange(1, 3, dtype=np.int64), False)
]

test_cases_nelec = [
    ('2_elecs', np.array([2, 4], dtype=np.int64), (1, 1)),
    ('no_elecs', np.array([3, 4], dtype=np.int64), (0, 0))
]

test_cases_ndets = [
    ('no_args', np.arange(1, 5, dtype=np.int64), None, None, 36),
    ('ref_space', np.arange(1, 7, dtype=np.int64), np.array([1, 2], dtype=np.int64), None, 4900),
    ('ref_space-no_elec', np.arange(1, 7, 2, dtype=np.int64), np.array([1, 3], dtype=np.int64), (1, 1), 100)
]

test_cases_mat_idx = [
    (6, 4, 4, (1, 2)),
    (9, 8, 2, (4, 1))
]

test_cases_near_nbrs = [
    ((1, 2), 4, 4, [(0, 2), (2, 2), (1, 3), (1, 1)]),
    ((4, 1), 8, 2, [(3, 1), (5, 1), (4, 0), (4, 0)])
]

test_cases_natural_keys = [
    ('str', 'mbe_test_string', ['mbe_test_string']),
    ('str+int', 'mbe_test_string_1', ['mbe_test_string_', 1, ''])
]

test_cases_convert = [
    ('str', 'string', str),
    ('int', '1', int)
]

test_cases_inc_dim = [
    ('energy', 1),
    ('dipole', 3)
]

test_cases_inc_shape = [
    ('energy', 5, 1, (5, )),
    ('dipole', 5, 3, (5, 3))
]


def test_time_str() -> None:
        """
        this function tests time_str
        """
        assert time_str(3742.4) == '1h 2m 22.40s'


@pytest.mark.parametrize(argnames='a, ref_sum', \
                         argvalues=[case[1:] for case in test_cases_fsum], \
                         ids=[case[0] for case in test_cases_fsum])
def test_fsum(a: np.ndarray, ref_sum: float) -> None:
        """
        this function tests fsum
        """
        assert fsum(a) == pytest.approx(ref_sum)
        

def test_hash_2d() -> None:
        """
        this function tests hash_2d
        """
        hash_array = hash_2d(np.arange(4 * 4, dtype=np.int64).reshape(4, 4))

        assert (hash_array == np.array([-2930228190932741801, \
                                         1142744019865853604, \
                                        -8951855736587463849, \
                                         4559082070288058232], dtype=np.int64)).all()


def test_hash_1d() -> None:
        """
        this function tests hash_1d
        """
        hash = np.arange(5, dtype=np.int64)

        assert hash_1d(hash) == 1974765062269638978


@pytest.mark.parametrize(argnames='b, present', \
                         argvalues=[case[1:] for case in test_cases_hash_lookup], \
                         ids=[case[0] for case in test_cases_hash_lookup])
def test_hash_lookup(b: np.ndarray, present: bool) -> None:
        """
        this function tests hash_lookup
        """
        a = np.arange(10, dtype=np.int64)

        if present:
            assert (hash_lookup(a, b) == b).all()
        else:
            assert hash_lookup(a, b) is None


@pytest.mark.parametrize(argnames='ref_space, ref_n_tuples', \
                         argvalues=[case[1:] for case in test_cases_tuples], \
                         ids=[case[0] for case in test_cases_tuples])
def test_tuples(ref_space: np.ndarray, ref_n_tuples: int) -> None:
        """
        this function tests tuples
        """
        nocc = 4
        order = 3
        occup = np.array([2.] * 4 + [0.] * 4, dtype=np.float64)
        exp_space = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)

        gen = tuples(exp_space[exp_space < nocc], exp_space[nocc <= exp_space],
                     virt_prune(occup, ref_space), occ_prune(occup, ref_space), 
                     order)

        assert isinstance(gen, GeneratorType)
        assert sum(1 for _ in gen) == ref_n_tuples
        

@pytest.mark.parametrize(argnames='tup_occ, tup_virt, ref_start_idx', \
                         argvalues=[case[1:] for case in test_cases_start_idx], \
                         ids=[case[0] for case in test_cases_start_idx])
def test_start_idx(tup_occ: Optional[np.ndarray], \
                   tup_virt: Optional[np.ndarray], \
                   ref_start_idx: Tuple[int, int, int]) -> None:
        """
        this function tests start_idx
        """
        occ_space = np.array([0, 1, 2, 5], dtype=np.int64)
        virt_space = np.array([6, 7, 9, 12], dtype=np.int64)

        assert start_idx(occ_space, virt_space, tup_occ, tup_virt) == ref_start_idx


@pytest.mark.parametrize(argnames='tup, ref_idx', \
                         argvalues=test_cases_comb_idx, \
                         ids=[str(case[0]) for case in test_cases_comb_idx])
def test_comb_idx(tup: np.ndarray, ref_idx: float) -> None:
        """
        this function tests _comb_idx
        """
        space = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)

        assert _comb_idx(space, tup) == ref_idx


@pytest.mark.parametrize(argnames='order, ref_idx', \
                         argvalues=test_cases_idx, \
                         ids=[case[0] for case in test_cases_idx])
def test_idx(order: int, ref_idx: float) -> None:
        """
        this function tests _idx
        """
        space = np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)

        assert _idx(space, 5, order) == ref_idx


@pytest.mark.parametrize(argnames='ref_occ, ref_virt, ref_n_tuples', \
                         argvalues=[case[1:] for case in test_cases_n_tuples], \
                         ids=[case[0] for case in test_cases_n_tuples])
def test_n_tuples(ref_occ: bool, ref_virt: bool, ref_n_tuples: int) -> None:
        """
        this function tests n_tuples
        """
        order = 5
        occ_space = np.arange(10, dtype=np.int64)
        virt_space = np.arange(10, 50, dtype=np.int64)

        assert n_tuples(occ_space, virt_space, ref_occ, ref_virt, order) == ref_n_tuples


def test_cas() -> None:
        """
        this function tests cas
        """
        assert (cas(np.array([7, 13], dtype=np.int64), np.arange(5, dtype=np.int64)) == np.array([0, 1, 2, 3, 4, 7, 13], dtype=np.int64)).all()

    
def test_core_cas() -> None:
        """
        this function tests core_cas
        """
        core_idx, cas_idx = core_cas(8, np.arange(3, 5, dtype=np.int64), np.array([9, 21], dtype=np.int64))

        assert (core_idx == np.array([0, 1, 2, 5, 6, 7], dtype=np.int64)).all()
        assert (cas_idx == np.array([ 3, 4, 9, 21], dtype=np.int64)).all()


def test_cas_idx_cart() -> None:
        """
        this function tests _cas_idx_cart
        """
        assert (_cas_idx_cart(np.arange(0, 10, 3, dtype=np.int64)) == np.array([[0, 0], [0, 3], [0, 6], [0, 9], [3, 0], [3, 3], [3, 6], [3, 9],
                                                                [6, 0], [6, 3], [6, 6], [6, 9], [9, 0], [9, 3], [9, 6], [9, 9]], dtype=np.int64)).all()
        
        
def test_coor_to_idx() -> None:
        """
        this function tests _coor_to_idx
        """
        assert _coor_to_idx((4, 9)) == 49


def test_idx_tril() -> None:
        """
        this function tests idx_tril
        """
        assert (idx_tril(np.arange(2, 14, 3, dtype=np.int64)) == np.array([ 5, 17, 20, 38, 41, 44, 68, 71, 74, 77], dtype=np.int64)).all()
        

def test_pi_space() -> None:
        """
        this function tests pi_space
        """
        orbsym_dooh = np.array([14, 15, 5, 2, 3, 5,  0, 11, 10, 
                                 7,  6, 5, 3, 2, 0, 14, 15,  5], dtype=np.int64)
        exp_space = np.arange(18, dtype=np.int64)

        pi_pairs, pi_hashes = pi_space('Dooh', orbsym_dooh, exp_space)

        assert (pi_pairs == np.array([12, 13, 7, 8, 3, 4, 0, 1, 9, 10, 15, 16], dtype=np.int64)).all()
        assert (pi_hashes == np.array([-8471304755370577665, -7365615264797734692, 
                                       -3932386661120954737, -3821038970866580488,
                                         758718848004794914,  7528999078095043310], dtype=np.int64)).all()


def test_pi_orbs() -> None:
        """
        this function tests _pi_orbs
        """
        pi_orbs = _pi_orbs(np.array([1, 2, 4, 5], dtype=np.int64), 
                           np.arange(8, dtype=np.int64))

        assert (pi_orbs == np.array([1, 2, 4, 5], dtype=np.int64)).all()


@pytest.mark.parametrize(argnames='tup, ref_bool', \
                         argvalues=[case[1:] for case in test_cases_pi_prune], \
                         ids=[case[0] for case in test_cases_pi_prune])
def test_pi_prune(tup: np.ndarray, ref_bool: bool) -> None:
        """
        this function tests pi_prune
        """
        pi_space = np.array([1, 2, 4, 5], dtype=np.int64)
        pi_hashes = np.sort(np.array([-2163557957507198923, 1937934232745943291], dtype=np.int64))

        if ref_bool:
            assert pi_prune(pi_space, pi_hashes, tup)
        else:
            assert not pi_prune(pi_space, pi_hashes, tup)


@pytest.mark.parametrize(argnames='tup, ref_bool', \
                         argvalues=[case[1:] for case in test_cases_occ_prune], \
                         ids=[case[0] for case in test_cases_occ_prune])
def test_occ_prune(tup: np.ndarray, ref_bool: bool) -> None:
        """
        this function tests occ_prune
        """
        occup = np.array([2.] * 3 + [0.] * 4, dtype=np.float64)

        if ref_bool:
            assert occ_prune(occup, tup)
        else:
            assert not occ_prune(occup, tup)
        

@pytest.mark.parametrize(argnames='tup, ref_bool', \
                         argvalues=[case[1:] for case in test_cases_virt_prune], \
                         ids=[case[0] for case in test_cases_virt_prune])
def test_virt_prune(tup: np.ndarray, ref_bool: bool) -> None:
        """
        this function tests virt_prune
        """
        occup = np.array([2.] * 3 + [0.] * 4, dtype=np.float64)

        if ref_bool:
            assert virt_prune(occup, tup)
        else:
            assert not virt_prune(occup, tup)
        

@pytest.mark.parametrize(argnames='tup, ref_nelec', \
                         argvalues=[case[1:] for case in test_cases_nelec], \
                         ids=[case[0] for case in test_cases_nelec])
def test_nelec(tup: np.ndarray, ref_nelec: Tuple[int, int]) -> None:
        """
        this function tests nelec
        """
        occup = np.array([2.] * 3 + [0.] * 4, dtype=np.float64)

        assert nelec(occup, tup) == ref_nelec


@pytest.mark.parametrize(argnames='cas_idx, ref_space, n_elec, ref_ndets', \
                         argvalues=[case[1:] for case in test_cases_ndets], \
                         ids=[case[0] for case in test_cases_ndets])
def test_ndets(cas_idx: np.ndarray, ref_space: np.ndarray, \
               n_elec: Tuple[int, int], ref_ndets: int) -> None:
        """
        this function tests ndets
        """
        occup = np.array([2.] * 3 + [0.] * 4, dtype=np.float64)

        assert ndets(occup, cas_idx, ref_space=ref_space, n_elec=n_elec) == ref_ndets


@pytest.mark.parametrize(argnames='site_idx, nx, ny, ref_idx_tup', \
                         argvalues=test_cases_mat_idx, \
                         ids=[str(case[3]) for case in test_cases_mat_idx])
def test_mat_idx(site_idx: int, nx: int, ny: int, \
                 ref_idx_tup: Tuple[int, int]) -> None:
        """
        this function tests mat_idx
        """
        assert mat_idx(site_idx, nx, ny) == ref_idx_tup


@pytest.mark.parametrize(argnames='site_xy, nx, ny, ref_nbrs', \
                         argvalues=test_cases_near_nbrs, \
                         ids=[str(case[0]) for case in test_cases_near_nbrs])
def test_near_nbrs(site_xy: Tuple[int, int], nx: int, ny: int, \
                   ref_nbrs: List[Tuple[int, int]]) -> None:
        """
        this function tests near_nbrs
        """
        assert near_nbrs(site_xy, nx, ny) == ref_nbrs
        

@pytest.mark.parametrize(argnames='test_string, ref_keys', \
                         argvalues=[case[1:] for case in test_cases_natural_keys], \
                         ids=[case[0] for case in test_cases_natural_keys])
def test_natural_keys(test_string: str, \
                      ref_keys: List[Union[int, str]]) -> None:
        """
        this function tests natural_keys
        """
        assert natural_keys(test_string) == ref_keys


@pytest.mark.parametrize(argnames='test_string, ref_type', \
                         argvalues=[case[1:] for case in test_cases_convert], \
                         ids=[case[0] for case in test_cases_convert])
def test_convert(test_string: str, ref_type: type) -> None:
        """
        this function tests _convert
        """
        assert isinstance(_convert(test_string), ref_type)
        

def test_intervals() -> None:
        """
        this function tests intervals
        """
        assert [i for i in intervals(np.array([0, 1, 2, 5, 7, 8, 10, 11, 12, 13], dtype=np.int64))] == [[0, 2], [5], [7, 8], [10, 13]]
        

@pytest.mark.parametrize(argnames='target, ref_dim', \
                         argvalues=test_cases_inc_dim, \
                         ids=[case[0] for case in test_cases_inc_dim])
def test_inc_dim(target: str, ref_dim: int) -> None:
        """
        this function tests inc_dim
        """
        assert inc_dim(target) == ref_dim


@pytest.mark.parametrize(argnames='n, dim, ref_shape', \
                         argvalues=[case[1:] for case in test_cases_inc_shape], \
                         ids=[case[0] for case in test_cases_inc_shape])
def test_inc_shape(n: int, dim: int, \
                   ref_shape: Union[Tuple[int], Tuple[int, int]]) -> None:
        """
        this function tests inc_dim
        """
        assert inc_shape(n, dim) == ref_shape
