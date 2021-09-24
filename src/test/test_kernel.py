#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
kernel testing module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import pytest
from _pytest.fixtures import SubRequest
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, symm, ao2mo
from typing import List, Tuple, Union, Optional
from warnings import catch_warnings, simplefilter

from kernel import ints, _ao_ints, gauge_origin, dipole_ints, e_core_h1e, \
                   hubbard_h1e, hubbard_eri, hf as kernel_hf, _dim, ref_mo, \
                   ref_prop, main, _dipole, _trans, base, _casscf, _fci, _cc
from tools import idx_tril, nelec
from system import MolCls


test_cases_ints = [('h2o', 'rnd')]

test_cases_ao_ints = [
    ('h2o', (13, 13), (13, 13, 13, 13)), 
    ('hubbard', (6, 6), (6, 6, 6, 6)),
]

test_cases_gauge_origin = [
    ('h2o', 'zero', np.zeros(3, dtype=np.float64)),
    ('h2o', 'charge', np.array([ 0.,  0., -0.01730611])),
]

test_cases_dipole_ints = [('h2o', 'rnd')]

test_cases_hubbard_h1e = [
    ((1, 4), False, np.array([[ 0., -1.,  0.,  0.],
                              [-1.,  0., -1.,  0.],
                              [ 0., -1.,  0., -1.],
                              [ 0.,  0., -1.,  0.]])),
    ((1, 4), True, np.array([[ 0., -1.,  0.,  -1.],
                             [-1.,  0., -1.,  0.],
                             [ 0., -1.,  0., -1.],
                             [-1.,  0., -1.,  0.]])),
    ((2, 2), False, np.array([[ 0., -1., -1.,  0.],
                              [-1.,  0.,  0., -1.],
                              [-1.,  0.,  0., -1.],
                              [ 0., -1., -1.,  0.]])),
    ((2, 3), False, np.array([[ 0., -1.,  0.,  0., -1.,  0.],
                              [-1.,  0., -1., -1.,  0., -1.],
                              [ 0., -1.,  0.,  0., -1.,  0.],
                              [ 0., -1.,  0.,  0., -1.,  0.],
                              [-1.,  0., -1., -1.,  0., -1.],
                              [ 0., -1.,  0.,  0., -1.,  0.]])),
]

test_cases_hf = [
    ('h2o', 'h2o', False, False, False, -75.9838464521063, np.array([0., 0., 8.64255793e-01]), False, True),
    ('h2o', 'h2o', True, False, False, -75.9838464521063, np.array([0., 0., 8.64255793e-01]), False, True),
    ('h2o', 'h2o', False, False, True, -76.03260101758543, np.array([7.81125165e-13, -4.52461299e-13, 8.62876951e-01]), False, False),
    ('h2o', 'h2o', False, True, False, -75.9838464521063, np.array([0., 0., 8.64255793e-01]), True, True),
    ('h2o', 'h2o', True, True, False, -75.9838464521063, np.array([0., 0., 8.64255793e-01]), False, True),
    ('h2o', 'h2o', False, True, True, -76.03260101758543, np.array([7.81125165e-13, -4.52461299e-13, 8.62876951e-01]), False, False)
]

test_cases_dim = [
    ('closed-shell', np.array([2.] * 4 + [0.] * 6), (10, 4, 6)),
    ('open-shell', np.array([2.] * 4 + [1.] + [0.] * 6), (11, 5, 6)),
]

test_cases_ref_mo = [
    ('c2', 'casci', 'can', [i for i in range(2, 6)], True, True, (4, 4)),
    ('c2', 'casscf', 'can', [4, 5, 7, 8], False, False, (2, 2)),
    ('c2', 'casci', 'ccsd', [i for i in range(2, 6)], False, True, (4, 4)),
    ('c2', 'casci', 'local', [i for i in range(2, 6)], False, True, (4, 4))
]

test_cases_ref_prop = [
    ('h2o', 'fci', None, 'energy', 'pyscf', 0, -0.03769780809258805),
    ('h2o', 'ccsd', None, 'energy', 'pyscf', 0, -0.03733551374348559),
    ('h2o', 'fci', 'ccsd', 'energy', 'pyscf', 0, -0.00036229313775759664),
    ('h2o', 'ccsd(t)', 'ccsd', 'energy', 'pyscf', 0, -0.0003336954549769955),
    ('h2o', 'fci', None, 'dipole', 'pyscf', 0, np.array([0., 0., -0.02732937])),
    ('h2o', 'ccsd', None, 'dipole', 'pyscf', 0, np.array([0.,  0., -2.87487935e-02])),
    ('h2o', 'fci', 'ccsd', 'dipole', 'pyscf', 0, np.array([0., 0., 1.41941689e-03])),
    ('h2o', 'ccsd(t)', 'ccsd', 'dipole', 'pyscf', 0, np.array([0., 0., 1.47038530e-03])),
    ('h2o', 'fci', None, 'excitation', 'pyscf', 1, 0.7060145137233889),
    ('h2o', 'fci', None, 'trans', 'pyscf', 1, np.array([0., 0., 0.72582795])),
    ('h2o', 'ccsd', None, 'energy', 'ecc', 0, -0.03733551374348559),
    ('h2o', 'fci', 'ccsd', 'energy', 'ecc', 0, -0.0003622938195746786),
    ('h2o', 'ccsd(t)', 'ccsd', 'energy', 'ecc', 0, -0.0003336954549769955),
    ('h2o', 'ccsd', None, 'energy', 'ncc', 0, -0.03733551374348559),
    ('h2o', 'fci', 'ccsd', 'energy', 'ncc', 0, -0.0003622938195746786),
    ('h2o', 'ccsd(t)', 'ccsd', 'energy', 'ncc', 0, -0.0003336954549769955),
]

test_cases_main = [
    ('h2o', 'fci', 'energy', 'pyscf', 0, -0.014121462439533161, 133),
    ('hubbard', 'fci', 'energy', 'pyscf', 0, -2.8759428090050676, 36),
    ('h2o', 'ccsd', 'energy', 'pyscf', 0, -0.014118607610972691, 441),
    ('h2o', 'ccsd', 'energy', 'ecc', 0, -0.014118607610972691, 441),
    ('h2o', 'ccsd', 'energy', 'ncc', 0, -0.014118607610972691, 441),
    ('h2o', 'fci', 'dipole', 'pyscf', 0, np.array([0., 0., -7.97781259e-03]), 133),
    ('h2o', 'ccsd', 'dipole', 'pyscf', 0, np.array([0., 0., -8.05212961e-03]), 441),
    ('h2o', 'fci', 'excitation', 'pyscf', 1, 1.314649936052632, 133),
    ('hubbard', 'fci', 'excitation', 'pyscf', 1, 1.850774199956839, 36),
    ('h2o', 'fci', 'trans', 'pyscf', 1, np.array([0., 0., -0.26497816]), 133),
]

test_cases_base = [
    ('h2o', 'ccsd', 'energy', 'pyscf', -0.13432841702437032, np.zeros(3, dtype=np.float64)),
    ('h2o', 'ccsd', 'dipole', 'pyscf', -0.13432841702437032, np.array([0., 0., -4.31202762e-02])),
    ('h2o', 'ccsd', 'energy', 'ecc', -0.13432841702437032, np.zeros(3, dtype=np.float64)),
    ('h2o', 'ccsd', 'energy', 'ncc', -0.13432841702437032, np.zeros(3, dtype=np.float64))
]

test_cases_casscf = [
    ('c2', ['Ag'], [1.], False, 2.2922857024683, 6.528333586540256),
    ('c2', ['Ag'], [1.], True, 2.2922857024683, 6.528333586540256),
    ('c2', ['Ag', 'Ag', 'Ag', 'B1g'], [.25, .25, .25, .25], False, 2.700100458554667, 6.437087455128202),
]

test_cases_fci = [
    ('h2o', 'energy', 0, -0.014121462439547372, 133, None, None, None, None, None),
    ('hubbard', 'energy', 0, -3.0307832514917887, 400, None, None, None, None, None),
    ('h2o', 'dipole', 0, -0.014121462439547372, 133, 9.978231697964103, 2., None, None, None),
    ('hubbard', 'dipole', 0, -3.0307832514917887, 400, 11.474010037419035, 1., None, None, None),
    ('h2o', 'excitation', 1, 1.3005284736130989,  133, None, None, 1.314649936052632, None, None),
    ('hubbard', 'excitation', 1, -2.8722307292538862,  400, None, None, 0.15855252223787852, None, None),
    ('h2o', 'trans', 1, 1.3005284736130989, 133, None, None, None, 0., 0.),
    ('hubbard', 'trans', 1, -2.8722307292542144, 400, None, None, None, 0., 0.)
]

test_cases_cc = [
    ('h2o', 'ccsd', False, 'pyscf',  -0.014118607610972705, None, None),
    ('h2o', 'ccsd(t)', False, 'pyscf',  -0.01412198067950329, None, None),
    ('h2o', 'ccsd', True, 'pyscf', -0.014118607610972705, 9.978003347693397, 2.),
    ('h2o', 'ccsd(t)', True, 'pyscf', -0.01412198067950329, 9.978193957339084, 2.),
    ('h2o', 'ccsd', False, 'ecc', -0.014118607610972705, None, None),
    ('h2o', 'ccsd(t)', False, 'ecc', -0.01412198067950329, None, None),
    ('h2o', 'ccsdt', False, 'ecc', -0.014122626346599783, None, None),
    ('h2o', 'ccsd', False, 'ncc', -0.014118607610972705, None, None),
    ('h2o', 'ccsd(t)', False, 'ncc', -0.01412198067950329, None, None),
    ('h2o', 'ccsdt', False, 'ncc', -0.014122626346599783, None, None),
    ('h2o', 'ccsdtq', False, 'ncc', -0.014121463191623542, None, None),
]


@pytest.fixture
def mol(request: SubRequest) -> MolCls:
        """
        this fixture constructs the mol object for use in most tests
        """
        if request.param == 'h2o':

            mol = gto.Mole()
            mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943', \
                      basis='631g', symmetry='C2v', verbose=0)

            mol.norb = mol.nao_nr()
            mol.nocc = mol.nelectron // 2
            mol.nvirt = mol.norb - mol.nocc
            mol.ncore = 1
            mol.e_nuc = mol.energy_nuc()
            mol.x2c = False
            mol.gauge_origin = np.zeros(3)
            mol.debug = 0

        elif request.param == 'c2':

            mol = gto.Mole()
            mol.build(atom='C 0. 0. 0.625; C 0. 0. -0.625', basis='631g', \
                      symmetry='D2h', verbose=0)

            mol.norb = mol.nao_nr()
            mol.nocc = mol.nelectron // 2
            mol.nvirt = mol.norb - mol.nocc
            mol.ncore = 2
            mol.e_nuc = mol.energy_nuc()
            mol.x2c = False
            mol.gauge_origin = np.zeros(3)
            mol.debug = 0

        elif request.param == 'hubbard':

            mol = gto.M()

            mol.matrix = (1, 6)
            mol.n = 1.
            mol.u = 2.
            mol.pbc = True

        mol.system = request.param

        return mol


@pytest.fixture
def hf(mol: MolCls) -> scf.RHF:
        """
        this fixture constructs the hf object and executes a hf calculation
        """
        if mol.system in ['h2o', 'c2']:

            hf = scf.RHF(mol)
            with catch_warnings():
                simplefilter("ignore")
                hf.kernel()
        
        elif mol.system == 'hubbard':

            hf = None

        return hf


@pytest.fixture
def mo_coeff(request: SubRequest, mol: MolCls) -> np.ndarray:
        """
        this fixture constructs mo coefficients
        """
        if request.param == 'h2o':

            hf_ref = {'init_guess': 'minao', 'symmetry': mol.symmetry,
                      'irrep_nelec': {'A1': 6, 'B1': 2, 'B2': 2}, 
                      'newton': False}
            _, _, _, _, _, _, _, _, mo_coeff = kernel_hf(mol, hf_ref)

        elif request.param == 'rnd':

            np.random.seed(1234)
            mo_coeff = np.random.rand(mol.norb, mol.norb)

        return mo_coeff


@pytest.mark.parametrize(argnames='mol, mo_coeff', argvalues=test_cases_ints, \
                         ids=[case[0] for case in test_cases_ints], \
                         indirect=True)
def test_ints(mol: MolCls, mo_coeff: np.ndarray):
        """
        this function tests ints
        """
        hcore_win, vhf_win, eri_win = ints(mol, mo_coeff, True, True, 
                                           MPI.COMM_WORLD, MPI.COMM_WORLD, 
                                           MPI.COMM_WORLD, 1)

        assert isinstance(hcore_win, MPI.Win)
        assert isinstance(vhf_win, MPI.Win)
        assert isinstance(eri_win, MPI.Win)


@pytest.mark.parametrize(argnames='mol, req_hcore, req_eri', \
                         argvalues=test_cases_ao_ints, \
                         ids=[case[0] for case in test_cases_ao_ints], \
                         indirect=['mol'])
def test_ao_ints(mol: MolCls, req_hcore: Tuple[int, int], \
                 req_eri: Tuple[int, int, int, int]):
        """
        this function tests ao_ints
        """
        hcore, eri = _ao_ints(mol)

        assert hcore.shape == req_hcore
        assert eri.shape == req_eri


@pytest.mark.parametrize(argnames='mol, gauge, req_gauge_origin', \
                         argvalues=test_cases_gauge_origin, \
                         ids=['-'.join(case[0:2]) for case in test_cases_gauge_origin], \
                         indirect=['mol'])
def test_gauge_origin(mol: MolCls, gauge: str, req_gauge_origin: np.ndarray):
        """
        this function tests gauge_origin
        """
        mol.gauge = gauge

        assert gauge_origin(mol) == pytest.approx(req_gauge_origin)


@pytest.mark.parametrize(argnames='mol, mo_coeff', \
                         argvalues=test_cases_dipole_ints, \
                         ids=[case[0] for case in test_cases_dipole_ints], \
                         indirect=True)
def test_dipole_ints(mol: MolCls, mo_coeff: np.ndarray):
        """
        this function tests dipole_ints
        """
        ints = dipole_ints(mol, mo_coeff)

        assert ints.shape == (3, 13, 13)


def test_e_core_h1e():
        """
        this function tests e_core_h1e
        """
        e_nuc = 0.
        np.random.seed(1234)
        hcore = np.random.rand(6, 6)
        np.random.seed(1234)
        vhf = np.random.rand(3, 6, 6)
        core_idx = np.array([0])
        cas_idx = np.array([2, 4, 5])
        e_core, h1e_cas = e_core_h1e(e_nuc, hcore, vhf, core_idx, cas_idx)

        assert e_core == pytest.approx(0.5745583511366769)
        assert h1e_cas == pytest.approx(np.array([[0.74050151, 1.00616633, 0.02753690], \
                                                  [0.79440516, 0.63367224, 1.13619731], \
                                                  [1.60429528, 1.40852194, 1.40916262]]))


@pytest.mark.parametrize(argnames='matrix, pbc, req_h1e', \
                         argvalues=test_cases_hubbard_h1e, \
                         ids=[str(case[0]).replace(' ', '') + ('-pbc' if case[1] else '') for case in test_cases_hubbard_h1e])
def test_hubbard_h1e(matrix: Tuple[int, int], pbc: bool, req_h1e: np.ndarray):
        """
        this function tests hubbard_h1e
        """
        h1e = hubbard_h1e(matrix, pbc)

        assert h1e == pytest.approx(req_h1e)


def test_hubbard_eri():
        """
        this function tests hubbard_eri
        """
        matrix = (1, 2)
        eri = hubbard_eri(matrix, 2.)

        assert eri == pytest.approx(np.array([[[[2., 0.], [0., 0.]],
                                               [[0., 0.], [0., 0.]]],
                                              [[[0., 0.], [0., 0.]],
                                               [[0., 0.], [0., 2.]]]]))


@pytest.mark.parametrize(argnames='mol, mo_coeff, newton, symmetry, x2c, req_e_hf, req_dipole, mo_coeff_eq, rdm1_eq', \
                         argvalues=test_cases_hf, \
                         ids=[case[0] + ('-sym' if case[3] else '') + ('-newton' if case[2] else '') + ('-x2c' if case[4] else '') for case in test_cases_hf], \
                         indirect=['mol', 'mo_coeff'])
def test_hf(mol: MolCls, mo_coeff: np.ndarray, newton: bool, symmetry: bool, \
            x2c: bool, req_e_hf: float, req_dipole: np.ndarray, \
            mo_coeff_eq: bool, rdm1_eq: bool):
        """
        this function tests hf
        """
        req_mo_coeff = mo_coeff

        hf_ref = {'init_guess': 'minao', 'symmetry': mol.symmetry,
                  'irrep_nelec': {'A1': 6, 'B1': 2, 'B2': 2}, 'newton': False}
        if newton:
            hf_ref['newton'] = True
        if not symmetry:
            hf_ref['symmetry'] = 'C1'
            hf_ref['irrep_nelec'] = {'A': 10}
        mol.x2c = x2c

        nocc, nvirt, norb, _, e_hf, dipole, occup, orbsym, mo_coeff = kernel_hf(mol, hf_ref)

        rdm1 = scf.hf.make_rdm1(mo_coeff, occup)
        req_rdm1 = scf.hf.make_rdm1(req_mo_coeff, occup)

        assert nocc == mol.nocc
        assert nvirt == mol.nvirt
        assert norb == mol.norb
        assert e_hf == pytest.approx(req_e_hf)
        assert dipole == pytest.approx(req_dipole, rel=1e-5, abs=1e-12)
        assert occup == pytest.approx(np.array([2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0.]))
        assert orbsym == pytest.approx(np.array([0, 0, 2, 0, 3, 0, 2, 2, 3, 0, 0, 2, 0]))
        assert mo_coeff == pytest.approx(req_mo_coeff) if mo_coeff_eq else mo_coeff != pytest.approx(req_mo_coeff)
        assert rdm1 == pytest.approx(req_rdm1, rel=1e-5, abs=1e-12) if rdm1_eq else rdm1 != pytest.approx(req_rdm1, rel=1e-5, abs=1e-12)


@pytest.mark.parametrize(argnames='mo_occ, req_dims', \
                         argvalues=[case[1:] for case in test_cases_dim], \
                         ids=[case[0] for case in test_cases_dim])
def test_dim(mo_occ: np.ndarray, req_dims: Tuple[int, int, int]):
        """
        this function tests _dim
        """
        dims = _dim(mo_occ)

        assert dims == pytest.approx(req_dims)


@pytest.mark.parametrize(argnames='mol, method, orb_type, select, mo_coeff_eq, rdm1_eq, req_act_n_elec', \
                         argvalues=test_cases_ref_mo, ids=['-'.join(case[0:3]) for case in test_cases_ref_mo], \
                         indirect=['mol'])
def test_ref_mo(mol: MolCls, hf: scf.RHF, method: str, orb_type: str, \
                select: List[int], mo_coeff_eq: bool, rdm1_eq: bool, \
                req_act_n_elec: Tuple[int, int]):
        """
        this function tests ref_mo
        """
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        model = {'method': 'fci', 'solver': 'pyscf_spin0'}
        ref = {'method': method, 'hf_guess': True, 'active': 'manual',
               'select': select,
               'wfnsym': ['Ag'], 'weights': [1.]}
        orbs = {'type': orb_type}

        mo_coeff, act_n_elec, ref_space = ref_mo(mol, hf.mo_coeff, hf.mo_occ, orbsym,
                                                 orbs, ref, model, hf)

        rdm1 = scf.hf.make_rdm1(mo_coeff, hf.mo_occ)
        hf_rdm1 = scf.hf.make_rdm1(hf.mo_coeff, hf.mo_occ)

        assert mo_coeff == pytest.approx(hf.mo_coeff) if mo_coeff_eq else mo_coeff != pytest.approx(hf.mo_coeff)
        assert rdm1 == pytest.approx(hf_rdm1) if rdm1_eq else rdm1 != pytest.approx(hf_rdm1)
        assert act_n_elec == req_act_n_elec
        assert ref_space == pytest.approx(np.array(select))


@pytest.mark.parametrize(argnames='mol, method, base_method, target_mbe, cc_backend, root, req_res', \
                         argvalues=test_cases_ref_prop, \
                         ids=['-'.join([item for item in case[0:5] if item]) for case in test_cases_ref_prop], \
                         indirect=['mol'])
def test_ref_prop(mol: MolCls, method: str, base_method: Optional[str], \
                  target_mbe: str, cc_backend: str, root: int, \
                  req_res: Union[float, np.ndarray]):
        """
        this function tests ref_prop
        """
        hf_ref = {'irrep_nelec': {}, 'init_guess': 'h1e', 'symmetry': mol.symmetry, 'newton': False}

        mol.nocc, mol.nvirt, mol.norb, _, e_hf, dipole_hf, occup, orbsym, mo_coeff = kernel_hf(mol, hf_ref)

        mol.dipole_ints = dipole_ints(mol, mo_coeff)

        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)

        mol.hcore, mol.vhf, mol.eri = ints(mol, mo_coeff, True, True,
                                           MPI.COMM_WORLD, MPI.COMM_WORLD, 
                                           MPI.COMM_WORLD, 1)

        ref_space = np.array([0, 1, 2, 3, 4, 6, 8, 10])
        state = {'root': root, 'wfnsym': 'A1'}
        model = {'method': method, 'cc_backend': cc_backend, 'solver': 'pyscf_spin0'}

        res = ref_prop(mol, occup, target_mbe, orbsym, True, ref_space, model, 'can', state, e_hf, dipole_hf, base_method)

        assert res == pytest.approx(req_res)


@pytest.mark.parametrize(argnames='mol, method, target_mbe, cc_backend, root, req_res, req_ndets', \
                         argvalues=test_cases_main, \
                         ids=['-'.join([item for item in case[0:4] if item]) for case in test_cases_main], \
                         indirect=['mol'])
def test_main(mol: MolCls, hf: scf.RHF, method: bool, target_mbe: bool, \
              cc_backend: str, root: int, req_res: Union[float, np.ndarray], \
              req_ndets: int):
        """
        this function tests main
        """
        if mol.system == 'h2o':

            occup = hf.mo_occ
            state_wfnsym = 'A1'
            point_group = 'C2v'
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
            e_hf = hf.e_tot
            e_core = mol.e_nuc
            hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
            h1e = np.einsum('pi,pq,qj->ij', hf.mo_coeff, hcore_ao, hf.mo_coeff)
            eri_ao = mol.intor('int2e_sph', aosym=4)
            h2e = ao2mo.incore.full(eri_ao, hf.mo_coeff)
            core_idx = np.array([])
            cas_idx = np.array([0, 1, 2, 3, 4, 7, 9])
            h1e_cas = h1e[cas_idx[:, None], cas_idx]
            cas_idx_tril = idx_tril(cas_idx)
            h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
            n_elec = nelec(hf.mo_occ, cas_idx)
            mol.dipole_ints = dipole_ints(mol, hf.mo_coeff)
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
            dipole_hf = np.einsum('xij,ji->x', ao_dip, hf.make_rdm1())

        elif mol.system == 'hubbard':

            occup = np.array([2.] * 3 + [0.] * 3)
            state_wfnsym = 'A'
            point_group = 'C1'
            orbsym = np.zeros(6, dtype=np.int64)
            e_hf = 0.
            e_core = 0.
            h1e = hubbard_h1e((1, 6), True)
            h2e = hubbard_eri((1, 6), 2.)
            h2e = ao2mo.restore(4, h2e, 6)
            core_idx = np.array([0])
            cas_idx = np.arange(1, 5)
            h1e_cas = h1e[cas_idx[:, None], cas_idx]
            cas_idx_tril = idx_tril(cas_idx)
            h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
            n_elec = nelec(occup, cas_idx)
            mol.dipole_ints = None
            dipole_hf = None

        res, ndets = main(method, cc_backend, 'pyscf_spin0', 'can', 0, occup, \
                           target_mbe, state_wfnsym, point_group, orbsym, \
                           True, root, e_hf, e_core, h1e_cas, h2e_cas, \
                           core_idx, cas_idx, n_elec, 0, mol.dipole_ints, \
                           dipole_hf)

        assert res == pytest.approx(req_res)
        assert ndets == req_ndets
        

def test_dipole():
        """
        this function tests _dipole
        """
        occup = np.array([2.] * 3 + [0.] * 3)
        hf_dipole = np.zeros(3, dtype=np.float64)
        cas_idx = np.arange(1, 5)
        np.random.seed(1234)
        dipole_ints = np.random.rand(3, 6, 6)
        np.random.seed(1234)
        cas_rdm1 = np.random.rand(cas_idx.size, cas_idx.size)
        dipole = _dipole(dipole_ints, occup, hf_dipole, cas_idx, cas_rdm1)

        assert dipole == pytest.approx(np.array([5.90055525, 5.36437348, 6.40001788]))
        

def test_trans():
        """
        this function tests _trans
        """
        occup = np.array([2.] * 3 + [0.] * 3)
        hf_dipole = np.zeros(3, dtype=np.float64)
        cas_idx = np.arange(1, 5)
        np.random.seed(1234)
        dipole_ints = np.random.rand(3, 6, 6)
        np.random.seed(1234)
        cas_rdm1 = np.random.rand(cas_idx.size, cas_idx.size)
        trans = _trans(dipole_ints, occup, hf_dipole, cas_idx, cas_rdm1, .9, .4)

        assert trans == pytest.approx(np.array([5.51751635, 4.92678927, 5.45675281]))


@pytest.mark.parametrize(argnames='mol, method, target_mbe, cc_backend, req_energy, req_dipole', \
                         argvalues=test_cases_base, ids=['-'.join(case[0:4]) for case in test_cases_base], \
                         indirect=['mol'])
def test_base(mol: MolCls, method: str, target_mbe: str, cc_backend: str, \
              req_energy: float, req_dipole: np.ndarray):
        """
        this function tests base
        """
        hf_ref = {'irrep_nelec': {}, 'init_guess': 'h1e', 'symmetry': mol.symmetry, 'newton': True}

        nocc, nvirt, norb, _, _, dipole_hf, occup, _, mo_coeff = kernel_hf(mol, hf_ref)

        mol.dipole_ints = dipole_ints(mol, mo_coeff)

        mol.hcore, mol.vhf, mol.eri = ints(mol, mo_coeff, True, True, MPI.COMM_WORLD, \
                                           MPI.COMM_WORLD, MPI.COMM_WORLD, 1)

        mol.system = {'charge': mol.charge, 'spin': mol.spin, 'basis': mol.basis, \
                      'frozen': True}

        e, dipole = base(mol, 'can', occup, mo_coeff, target_mbe, method, cc_backend, dipole_hf)

        assert nocc == mol.nocc
        assert nvirt == mol.nvirt
        assert norb == mol.norb
        assert e == pytest.approx(req_energy)
        assert dipole == pytest.approx(req_dipole)


@pytest.mark.parametrize(argnames='mol, wfnsym, weights, hf_guess, req_sum, req_amax', \
                         argvalues=test_cases_casscf, \
                         ids=['-'.join([case[0]] + ["{:02}{}".format(weight, wfnsym) for weight, wfnsym in zip(case[2], case[1])] + (['hf_guess'] if case[3] else [])) for case in test_cases_casscf], \
                         indirect=['mol'])
def test_casscf(mol: MolCls, hf: scf.RHF, wfnsym: List[str], \
                weights: List[float], hf_guess: bool, req_sum: float, \
                req_amax: float):
        """
        this function tests _casscf
        """
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)

        mo_coeff = _casscf(mol, 'pyscf_spin0', wfnsym, weights, orbsym, hf_guess,
                           hf, hf.mo_coeff, np.arange(2, 10), (4, 4))

        assert np.sum(mo_coeff) == pytest.approx(req_sum)
        assert np.amax(mo_coeff) == pytest.approx(req_amax)


@pytest.mark.parametrize(argnames='mol, target_mbe, root, req_energy, req_n_dets, req_rdm1_sum, req_rdm1_amax, req_excitation, req_t_rdm1, req_hf_weight_sum', \
                         argvalues=test_cases_fci, \
                         ids=['-'.join(case[0:2]) for case in test_cases_fci], \
                         indirect=['mol'])
def test_fci(mol: MolCls, hf: scf.RHF, target_mbe: str, root: int, \
             req_energy: float, req_n_dets: int, \
             req_rdm1_sum: Optional[float], req_rdm1_amax: Optional[float], \
             req_excitation: Optional[float], req_t_rdm1: Optional[float], \
             req_hf_weight_sum: Optional[float]):
        """
        this function tests _fci
        """
        if mol.system == 'h2o':

            wfnsym = 'A1'
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
            e_hf = hf.e_tot
            e_core = mol.e_nuc
            hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
            h1e = np.einsum('pi,pq,qj->ij', hf.mo_coeff, hcore_ao, hf.mo_coeff)
            eri_ao = mol.intor('int2e_sph', aosym=4)
            h2e = ao2mo.incore.full(eri_ao, hf.mo_coeff)
            cas_idx = np.array([0, 1, 2, 3, 4, 7, 9])
            h1e_cas = h1e[cas_idx[:, None], cas_idx]
            cas_idx_tril = idx_tril(cas_idx)
            h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
            occup = hf.mo_occ
            n_elec = nelec(occup, cas_idx)

        elif mol.system == 'hubbard':

            wfnsym = 'A'
            orbsym = np.zeros(8, dtype=np.int64)
            e_hf = 0.
            e_core = 0.
            h1e = hubbard_h1e((2, 4), True)
            h2e = hubbard_eri((2, 4), 2.)
            h2e = ao2mo.restore(4, h2e, 8)
            cas_idx = np.arange(1, 7)
            h1e_cas = h1e[cas_idx[:, None], cas_idx]
            cas_idx_tril = idx_tril(cas_idx)
            h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
            occup = np.array([2.] * 4 + [0.] * 4)
            n_elec = nelec(occup, cas_idx)

        res = _fci('pyscf_spin0', 0, target_mbe, wfnsym, orbsym, True, root, e_hf, e_core,
                    h1e_cas, h2e_cas, occup, np.array([]), cas_idx, n_elec, 0)

        assert res['energy'] == pytest.approx(req_energy)
        assert res['n_dets'] == req_n_dets
        if req_rdm1_sum:
            assert np.sum(res['rdm1']) == pytest.approx(req_rdm1_sum)
        if req_rdm1_amax:
            assert np.amax(res['rdm1']) == pytest.approx(req_rdm1_amax, rel=1e-5, abs=1e-12)
        if req_excitation:
            assert res['excitation'] == pytest.approx(req_excitation)
        if req_t_rdm1:
            assert np.trace(res['t_rdm1']) == pytest.approx(req_t_rdm1)
        if req_hf_weight_sum:
            assert np.sum(res['hf_weight']) == pytest.approx(req_hf_weight_sum)


@pytest.mark.parametrize(argnames='mol, method, rdm1, cc_backend, req_energy, req_rdm1_sum, req_rdm1_amax', \
                         argvalues=test_cases_cc, \
                         ids=['-'.join(case[0:2]) + ('-rdm1' if case[2] else '') + '-' + case[3] for case in test_cases_cc], \
                         indirect=['mol'])
def test_cc(mol: MolCls, hf: scf.RHF, method: str, rdm1: bool, \
            cc_backend: str, req_energy: float, req_rdm1_sum: Optional[float], \
            req_rdm1_amax: Optional[float]):
        """
        this function tests _cc
        """
        core_idx = np.array([])
        cas_idx = np.array([0, 1, 2, 3, 4, 7, 9])
        n_elec = nelec(hf.mo_occ, cas_idx)
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
        h1e = np.einsum('pi,pq,qj->ij', hf.mo_coeff, hcore_ao, hf.mo_coeff)
        eri_ao = mol.intor('int2e_sph', aosym=4)
        h2e = ao2mo.incore.full(eri_ao, hf.mo_coeff)
        h1e_cas = h1e[cas_idx[:, None], cas_idx]
        cas_idx_tril = idx_tril(cas_idx)
        h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]

        res = _cc(0, hf.mo_occ, core_idx, cas_idx, method, cc_backend=cc_backend,
                  n_elec=n_elec, orb_type='can', point_group='C2v', orbsym=orbsym, h1e=h1e_cas, h2e=h2e_cas, hf=hf, rdm1=rdm1)

        assert res['energy'] == pytest.approx(req_energy)
        if req_rdm1_sum:
            assert np.sum(res['rdm1']) == pytest.approx(req_rdm1_sum)
        if req_rdm1_amax:
            assert np.amax(res['rdm1']) == pytest.approx(req_rdm1_amax, rel=1e-4, abs=1e-12)
