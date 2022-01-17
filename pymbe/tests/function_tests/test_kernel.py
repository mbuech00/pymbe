#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
kernel testing module
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

from pymbe.kernel import e_core_h1e, main, _dipole, _trans, _fci, _cc

if TYPE_CHECKING:

    from pyscf import gto, scf
    from typing import List, Tuple, Union, Dict


test_cases_main = [
    ('h2o', 'fci', 'energy', 'pyscf', 0, -0.014121462439533161),
    ('hubbard', 'fci', 'energy', 'pyscf', 0, -2.8759428090050676),
    ('h2o', 'ccsd', 'energy', 'pyscf', 0, -0.014118607610972691),
    ('h2o', 'ccsd', 'energy', 'ecc', 0, -0.014118607610972691),
    ('h2o', 'ccsd', 'energy', 'ncc', 0, -0.014118607610972691),
    ('h2o', 'fci', 'dipole', 'pyscf', 0, np.array([0., 0., -7.97786374e-03], dtype=np.float64)),
    ('h2o', 'ccsd', 'dipole', 'pyscf', 0, np.array([0., 0., -8.05218072e-03], dtype=np.float64)),
    ('h2o', 'fci', 'excitation', 'pyscf', 1, 1.314649936052632),
    ('hubbard', 'fci', 'excitation', 'pyscf', 1, 1.850774199956839),
    ('h2o', 'fci', 'trans', 'pyscf', 1, np.array([0., 0., -2.64977135e-01], dtype=np.float64)),
]

test_cases_fci = [
    ('h2o', 'energy', 0, {'energy': -0.014121462439547372}),
    ('hubbard', 'energy', 0, {'energy': -2.875942809005066}),
    ('h2o', 'dipole', 0, {'rdm1_sum': 9.978231697964103, 'rdm1_amax': 2.}),
    ('hubbard', 'dipole', 0, {'rdm1_sum': 7.416665666590797, 'rdm1_amax': 1.}),
    ('h2o', 'excitation', 1, {'excitation': 1.314649936052632}),
    ('hubbard', 'excitation', 1, {'excitation': 1.8507741999568346}),
    ('h2o', 'trans', 1, {'t_rdm1': 0., 'hf_weight_sum': 0.9918466871769327}),
    ('hubbard', 'trans', 1, {'t_rdm1': 0., 'hf_weight_sum': -0.0101664409948010})
]

test_cases_cc = [
    ('h2o', 'ccsd', 'energy', 'pyscf', {'energy': -0.014118607610972705}),
    ('h2o', 'ccsd(t)', 'energy', 'pyscf', {'energy': -0.01412198067950329}),
    ('h2o', 'ccsd', 'dipole', 'pyscf', {'rdm1_sum': 9.978003347693397, 'rdm1_amax': 2.}),
    ('h2o', 'ccsd(t)', 'dipole', 'pyscf', {'rdm1_sum': 9.978193957339084, 'rdm1_amax': 2.}),
    ('h2o', 'ccsd', 'energy', 'ecc', {'energy': -0.014118607610972705}),
    ('h2o', 'ccsd(t)', 'energy', 'ecc', {'energy': -0.01412198067950329}),
    ('h2o', 'ccsdt', 'energy', 'ecc', {'energy': -0.014122626346599783}),
    ('h2o', 'ccsd', 'energy', 'ncc', {'energy': -0.014118607610972705}),
    ('h2o', 'ccsd(t)', 'energy', 'ncc', {'energy': -0.01412198067950329}),
    ('h2o', 'ccsdt', 'energy', 'ncc', {'energy': -0.014122626346599783}),
    ('h2o', 'ccsdtq', 'energy', 'ncc', {'energy': -0.014121463191623542}),
]


def test_e_core_h1e() -> None:
        """
        this function tests e_core_h1e
        """
        e_nuc = 0.
        np.random.seed(1234)
        hcore = np.random.rand(6, 6)
        np.random.seed(1234)
        vhf = np.random.rand(3, 6, 6)
        core_idx = np.array([0], dtype=np.int64)
        cas_idx = np.array([2, 4, 5], dtype=np.int64)
        e_core, h1e_cas = e_core_h1e(e_nuc, hcore, vhf, core_idx, cas_idx)

        assert e_core == pytest.approx(0.5745583511366769)
        assert h1e_cas == pytest.approx(np.array([[0.74050151, 1.00616633, 0.02753690], \
                                                  [0.79440516, 0.63367224, 1.13619731], \
                                                  [1.60429528, 1.40852194, 1.40916262]], dtype=np.float64))


@pytest.mark.parametrize(argnames='system, method, target, cc_backend, root, ref_res', \
                         argvalues=test_cases_main, \
                         ids=['-'.join([item for item in case[0:4] if item]) for case in test_cases_main], \
                         indirect=['system'])
def test_main(system: str, mol: gto.Mole, hf: scf.RHF, \
              ints: Tuple[np.ndarray, np.ndarray], orbsym: List[int],  \
              dipole_quantities: Tuple[np.ndarray, np.ndarray], method: str, \
              target: str, cc_backend: str, root: int, \
              ref_res: Union[float, np.ndarray]) -> None:
        """
        this function tests main
        """
        if system == 'h2o':

            occup = hf.mo_occ
            point_group = 'C2v'
            hf_energy = hf.e_tot
            e_core = mol.energy_nuc()
            core_idx = np.array([], dtype=np.int64)
            cas_idx = np.array([0, 1, 2, 3, 4, 7, 9], dtype=np.int64)
            cas_idx_tril = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, \
                                     13, 14, 28, 29, 30, 31, 32, 35, 45, 46, \
                                     47, 48, 49, 52, 54], dtype=np.int64)

        elif system == 'hubbard':

            occup = np.array([2.] * 3 + [0.] * 3, dtype=np.float64)
            point_group = 'C1'
            hf_energy = 0.
            e_core = 0.
            core_idx = np.array([0], dtype=np.int64)
            cas_idx = np.arange(1, 5, dtype=np.int64)
            cas_idx_tril = np.array([2, 4, 5, 7, 8, 9, 11, 12, 13, 14], dtype=np.int64)
            
        h1e, h2e = ints
        h1e_cas = h1e[cas_idx[:, None], cas_idx]
        h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
        n_elec = (np.count_nonzero(occup[cas_idx] > 0.), np.count_nonzero(occup[cas_idx] > 1.))

        if target == 'energy':

            dipole_ints = None
            hf_prop = hf_energy

        elif target == 'excitation':

            dipole_ints = None
            hf_prop = 0.

        elif target == 'dipole':

            dipole_ints, hf_prop = dipole_quantities

        elif target == 'trans':

            dipole_ints, _ = dipole_quantities
            hf_prop = np.zeros(3, dtype=np.float64)

        res = main(method, cc_backend, 'pyscf_spin0', 'can', 0, occup, target, \
                   0, point_group, orbsym, True, root, hf_prop, e_core, \
                   h1e_cas, h2e_cas, core_idx, cas_idx, n_elec, 0, dipole_ints)

        assert res == pytest.approx(ref_res)
        

def test_dipole() -> None:
        """
        this function tests _dipole
        """
        occup = np.array([2.] * 3 + [0.] * 3, dtype=np.float64)
        hf_dipole = np.zeros(3, dtype=np.float64)
        cas_idx = np.arange(1, 5, dtype=np.int64)
        np.random.seed(1234)
        dipole_ints = np.random.rand(3, 6, 6)
        np.random.seed(1234)
        cas_rdm1 = np.random.rand(cas_idx.size, cas_idx.size)
        dipole = _dipole(dipole_ints, occup, cas_idx, cas_rdm1, 
                         hf_dipole=hf_dipole)

        assert dipole == pytest.approx(np.array([5.90055525, 5.36437348, 6.40001788], dtype=np.float64))
        

def test_trans() -> None:
        """
        this function tests _trans
        """
        occup = np.array([2.] * 3 + [0.] * 3, dtype=np.float64)
        cas_idx = np.arange(1, 5, dtype=np.int64)
        np.random.seed(1234)
        dipole_ints = np.random.rand(3, 6, 6)
        np.random.seed(1234)
        cas_rdm1 = np.random.rand(cas_idx.size, cas_idx.size)
        trans = _trans(dipole_ints, occup, cas_idx, cas_rdm1, .9, .4)

        assert trans == pytest.approx(np.array([5.51751635, 4.92678927, 5.45675281], dtype=np.float64))


@pytest.mark.parametrize(argnames='system, target, root, ref', \
                         argvalues=test_cases_fci, \
                         ids=['-'.join(case[0:2]) for case in test_cases_fci], \
                         indirect=['system'])
def test_fci(system: str, mol: gto.Mole, hf: scf.RHF, \
             ints: Tuple[np.ndarray, np.ndarray], orbsym: List[int], \
             target: str, root: int, ref: Dict[str, Union[float, int]]) -> None:
        """
        this function tests _fci
        """
        if system == 'h2o':

            hf_energy = hf.e_tot
            e_core = mol.energy_nuc()
            cas_idx = np.array([0, 1, 2, 3, 4, 7, 9], dtype=np.int64)
            cas_idx_tril = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, \
                                     13, 14, 28, 29, 30, 31, 32, 35, 45, 46, \
                                     47, 48, 49, 52, 54], dtype=np.int64)
            occup = hf.mo_occ

        elif system == 'hubbard':

            hf_energy = 0.
            e_core = 0.
            cas_idx = np.arange(1, 5, dtype=np.int64)
            cas_idx_tril = np.array([2, 4, 5, 7, 8, 9, 11, 12, 13, 14], dtype=np.int64)
            occup = np.array([2.] * 3 + [0.] * 3, dtype=np.float64)
            
        h1e, h2e = ints
        h1e_cas = h1e[cas_idx[:, None], cas_idx]
        h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
        n_elec = (np.count_nonzero(occup[cas_idx] > 0.), np.count_nonzero(occup[cas_idx] > 1.))

        res = _fci('pyscf_spin0', 0, target, 0, orbsym, True, root, hf_energy, \
                   e_core, h1e_cas, h2e_cas, occup, cas_idx, n_elec, 0)

        if target == 'energy':
            assert res['energy'] == pytest.approx(ref['energy'])
        elif target == 'dipole':
            assert np.sum(res['rdm1']) == pytest.approx(ref['rdm1_sum'])
            assert np.amax(res['rdm1']) == pytest.approx(ref['rdm1_amax'], rel=1e-5, abs=1e-12)
        elif target == 'excitation':
            assert res['excitation'] == pytest.approx(ref['excitation'])
        elif target == 'trans':
            assert np.trace(res['t_rdm1']) == pytest.approx(ref['t_rdm1'])
            assert np.sum(res['hf_weight']) == pytest.approx(ref['hf_weight_sum'])


@pytest.mark.parametrize(argnames='system, method, target, cc_backend, ref', \
                         argvalues=test_cases_cc, \
                         ids=['-'.join(case[0:3]) + '-' + case[3] for case in test_cases_cc], \
                         indirect=['system'])
def test_cc(hf: scf.RHF, orbsym: List[int], \
            ints: Tuple[np.ndarray, np.ndarray], method: str, target: str, 
            cc_backend: str, ref: Dict[str, float]) -> None:
        """
        this function tests _cc
        """
        core_idx = np.array([], dtype=np.int64)
        cas_idx = np.array([0, 1, 2, 3, 4, 7, 9], dtype=np.int64)
        n_elec = (np.count_nonzero(hf.mo_occ[cas_idx] > 0.), np.count_nonzero(hf.mo_occ[cas_idx] > 1.))
        h1e, h2e = ints
        h1e_cas = h1e[cas_idx[:, None], cas_idx]
        cas_idx_tril = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, \
                                 13, 14, 28, 29, 30, 31, 32, 35, 45, 46, \
                                 47, 48, 49, 52, 54], dtype=np.int64)
        h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]

        res = _cc(0, hf.mo_occ, core_idx, cas_idx, method, cc_backend, n_elec,
                  'can', 'C2v', orbsym, h1e_cas, h2e_cas, True, 
                  target == 'dipole', 0)

        if target == 'energy':
            assert res['energy'] == pytest.approx(ref['energy'])
        elif target == 'dipole':
            assert np.sum(res['rdm1']) == pytest.approx(ref['rdm1_sum'])
            assert np.amax(res['rdm1']) == pytest.approx(ref['rdm1_amax'], rel=1e-4, abs=1e-12)
