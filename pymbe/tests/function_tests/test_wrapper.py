#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
wrapper testing module
"""

from __future__ import annotations

__author__ = "Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import pytest
import numpy as np
from pyscf import scf, symm
from typing import TYPE_CHECKING

from pymbe.wrapper import (
    ints as wrapper_ints,
    _ao_ints,
    dipole_ints,
    _hubbard_h1e,
    _hubbard_eri,
    hf as wrapper_hf,
    _dim,
    ref_mo,
    ref_prop,
    base,
    _casscf,
    linear_orbsym,
)

if TYPE_CHECKING:

    from _pytest.fixtures import SubRequest
    from pyscf import gto
    from typing import List, Tuple, Union, Optional, Dict, Any


test_cases_ints = [("h2o", "rnd")]

test_cases_ao_ints = [
    (
        "h2o",
        -241.43071004923337,
        2.4535983000672585,
        518.7407144449278,
        4.7804457081113805,
    ),
    (
        "hubbard",
        -12.0,
        0.0,
        12.0,
        2.0,
    ),
]

test_cases_dipole_ints = [("h2o", "rnd")]

test_cases_hubbard_h1e = [
    (
        (1, 4),
        False,
        np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, -1.0],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    (
        (1, 4),
        True,
        np.array(
            [
                [0.0, -1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    (
        (2, 2),
        False,
        np.array(
            [
                [0.0, -1.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    (
        (2, 3),
        False,
        np.array(
            [
                [0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
                [-1.0, 0.0, -1.0, -1.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
                [-1.0, 0.0, -1.0, -1.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
]

test_cases_hf = [
    (
        "h2o",
        "energy",
        "h2o",
        False,
        False,
        -75.9838464521063,
        True,
        True,
    ),
    (
        "h2o",
        "dipole",
        "h2o",
        False,
        False,
        np.array([0.0, 0.0, 8.64255793e-01], dtype=np.float64),
        True,
        True,
    ),
    (
        "h2o",
        "energy",
        "h2o",
        True,
        False,
        -75.9838464521063,
        False,
        True,
    ),
    (
        "h2o",
        "dipole",
        "h2o",
        True,
        False,
        np.array([0.0, 0.0, 8.64255793e-01], dtype=np.float64),
        False,
        True,
    ),
    (
        "h2o",
        "energy",
        "h2o",
        False,
        True,
        -76.03260101758543,
        False,
        False,
    ),
    (
        "h2o",
        "dipole",
        "h2o",
        False,
        True,
        np.array([0.0, 0.0, 8.62876951e-01], dtype=np.float64),
        False,
        False,
    ),
]

test_cases_dim = [
    (
        "closed-shell",
        np.array([2.0] * 4 + [0.0] * 6, dtype=np.float64),
        (10, 4, 6),
    ),
    (
        "open-shell",
        np.array([2.0] * 4 + [1.0] + [0.0] * 6, dtype=np.float64),
        (11, 5, 6),
    ),
]

test_cases_ref_mo = [
    (
        "c2",
        "ccsd",
        np.arange(2, 6, dtype=np.int64),
        {},
        False,
        True,
    ),
    (
        "c2",
        "local",
        np.arange(2, 6, dtype=np.int64),
        {},
        False,
        True,
    ),
    (
        "c2",
        "casscf",
        np.array([4, 5, 7, 8], dtype=np.int64),
        {"wfnsym": ["Ag"], "weights": [1.0]},
        False,
        False,
    ),
]

test_cases_casscf = [
    (
        "c2",
        ["Ag"],
        [1.0],
        False,
        2.2922857024683,
        6.528333586540256,
    ),
    (
        "c2",
        ["Ag"],
        [1.0],
        True,
        2.2922857024683,
        6.528333586540256,
    ),
    (
        "c2",
        ["Ag", "Ag", "Ag", "B1g"],
        [0.25, 0.25, 0.25, 0.25],
        False,
        3.042674374623752,
        6.377238595069308,
    ),
]

test_cases_ref_prop = [
    (
        "h2o",
        "fci",
        None,
        "energy",
        "pyscf",
        0,
        -0.03769780809258805,
    ),
    (
        "h2o",
        "ccsd",
        None,
        "energy",
        "pyscf",
        0,
        -0.03733551374348559,
    ),
    (
        "h2o",
        "fci",
        "ccsd",
        "energy",
        "pyscf",
        0,
        -0.00036229313775759664,
    ),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "energy",
        "pyscf",
        0,
        -0.0003336954549769955,
    ),
    (
        "h2o",
        "fci",
        None,
        "dipole",
        "pyscf",
        0,
        np.array([0.0, 0.0, -0.02732937], dtype=np.float64),
    ),
    (
        "h2o",
        "ccsd",
        None,
        "dipole",
        "pyscf",
        0,
        np.array([0.0, 0.0, -2.87487935e-02], dtype=np.float64),
    ),
    (
        "h2o",
        "fci",
        "ccsd",
        "dipole",
        "pyscf",
        0,
        np.array([0.0, 0.0, 1.41941689e-03], dtype=np.float64),
    ),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "dipole",
        "pyscf",
        0,
        np.array([0.0, 0.0, 1.47038530e-03], dtype=np.float64),
    ),
    (
        "h2o",
        "fci",
        None,
        "excitation",
        "pyscf",
        1,
        0.7060145137233889,
    ),
    (
        "h2o",
        "fci",
        None,
        "trans",
        "pyscf",
        1,
        np.array([0.0, 0.0, 0.72582795], dtype=np.float64),
    ),
    (
        "h2o",
        "ccsd",
        None,
        "energy",
        "ecc",
        0,
        -0.03733551374348559,
    ),
    (
        "h2o",
        "fci",
        "ccsd",
        "energy",
        "ecc",
        0,
        -0.0003622938195746786,
    ),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "energy",
        "ecc",
        0,
        -0.0003336954549769955,
    ),
    (
        "h2o",
        "ccsd",
        None,
        "energy",
        "ncc",
        0,
        -0.03733551374348559,
    ),
    (
        "h2o",
        "fci",
        "ccsd",
        "energy",
        "ncc",
        0,
        -0.0003622938195746786,
    ),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "energy",
        "ncc",
        0,
        -0.0003336954549769955,
    ),
]

test_cases_base = [
    (
        "h2o",
        "ccsd",
        "energy",
        "pyscf",
        -0.13432841702437032,
    ),
    (
        "h2o",
        "ccsd",
        "dipole",
        "pyscf",
        np.array([0.0, 0.0, -4.31213133e-02], dtype=np.float64),
    ),
    (
        "h2o",
        "ccsd",
        "energy",
        "ecc",
        -0.13432841702437032,
    ),
    (
        "h2o",
        "ccsd",
        "energy",
        "ncc",
        -0.13432841702437032,
    ),
]

test_cases_linear_orbsym = [("c2", "c2")]


@pytest.fixture
def mo_coeff(request: SubRequest, hf: scf.RHF, norb: int) -> np.ndarray:
    """
    this fixture constructs mo coefficients
    """
    if request.param in ["h2o", "c2"]:

        mo_coeff = hf.mo_coeff

    elif request.param == "rnd":

        np.random.seed(1234)
        mo_coeff = np.random.rand(norb, norb)

    return mo_coeff


@pytest.mark.parametrize(
    argnames="system, mo_coeff",
    argvalues=test_cases_ints,
    ids=[case[0] for case in test_cases_ints],
    indirect=True,
)
def test_ints(mol: gto.Mole, mo_coeff: np.ndarray, norb: int, nocc: int) -> None:
    """
    this function tests ints
    """
    hcore, vhf, eri = wrapper_ints(mol, mo_coeff, norb, nocc)

    assert np.sum(hcore) == pytest.approx(-12371.574250637233)
    assert np.amax(hcore) == pytest.approx(-42.09685184826769)
    assert np.sum(vhf) == pytest.approx(39687.423264678)
    assert np.amax(vhf) == pytest.approx(95.00353546601883)
    assert np.sum(eri) == pytest.approx(381205.21288377955)
    assert np.amax(eri) == pytest.approx(149.4981150522994)


@pytest.mark.parametrize(
    argnames="system, ref_hcore_sum, ref_hcore_amax, ref_eri_sum, ref_eri_amax",
    argvalues=test_cases_ao_ints,
    ids=[case[0] for case in test_cases_ao_ints],
    indirect=["system"],
)
def test_ao_ints(
    mol: gto.Mole,
    ref_hcore_sum: float,
    ref_hcore_amax: float,
    ref_eri_sum: float,
    ref_eri_amax: float,
) -> None:
    """
    this function tests _ao_ints
    """
    hcore, eri = _ao_ints(mol, False, 2.0, (1, 6), True)

    assert np.sum(hcore) == pytest.approx(ref_hcore_sum)
    assert np.amax(hcore) == pytest.approx(ref_hcore_amax)
    assert np.sum(eri) == pytest.approx(ref_eri_sum)
    assert np.amax(eri) == pytest.approx(ref_eri_amax)


@pytest.mark.parametrize(
    argnames="system, mo_coeff",
    argvalues=test_cases_dipole_ints,
    ids=[case[0] for case in test_cases_dipole_ints],
    indirect=True,
)
def test_dipole_ints(mol: gto.Mole, mo_coeff: np.ndarray) -> None:
    """
    this function tests dipole_ints
    """
    ints = dipole_ints(mol, mo_coeff, np.zeros(3, dtype=np.float64))

    assert np.sum(ints) == pytest.approx(1455.7182550859516)
    assert np.amax(ints) == pytest.approx(9.226332432385433)


@pytest.mark.parametrize(
    argnames="matrix, pbc, ref_h1e",
    argvalues=test_cases_hubbard_h1e,
    ids=[
        str(case[0]).replace(" ", "") + ("-pbc" if case[1] else "")
        for case in test_cases_hubbard_h1e
    ],
)
def test_hubbard_h1e(matrix: Tuple[int, int], pbc: bool, ref_h1e: np.ndarray) -> None:
    """
    this function tests hubbard_h1e
    """
    h1e = _hubbard_h1e(matrix, pbc)

    assert (h1e == ref_h1e).all()


def test_hubbard_eri() -> None:
    """
    this function tests hubbard_eri
    """
    ref = np.array(
        [
            [[[2.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]],
        ],
        dtype=np.float64,
    )

    matrix = (1, 2)
    eri = _hubbard_eri(matrix, 2.0)

    assert (eri == ref).all()


@pytest.mark.parametrize(
    argnames="system, target, mo_coeff, newton, x2c, ref_hf_prop, mo_coeff_eq, rdm1_eq",
    argvalues=test_cases_hf,
    ids=[
        "-".join(case[0:2])
        + ("-sym" if case[3] else "")
        + ("-newton" if case[2] else "")
        + ("-x2c" if case[4] else "")
        for case in test_cases_hf
    ],
    indirect=["system", "mo_coeff"],
)
def test_hf(
    mol: gto.Mole,
    target: str,
    nocc: int,
    norb: int,
    mo_coeff: np.ndarray,
    newton: bool,
    x2c: bool,
    ref_hf_prop: Union[float, np.ndarray],
    mo_coeff_eq: bool,
    rdm1_eq: bool,
) -> None:
    """
    this function tests hf
    """
    ref_nocc = nocc
    ref_nvirt = norb - nocc
    ref_norb = norb
    ref_mo_coeff = mo_coeff

    nocc, nvirt, norb, _, hf_prop, occup, orbsym, mo_coeff = wrapper_hf(
        mol, target=target, newton=newton, x2c=x2c
    )

    rdm1 = scf.hf.make_rdm1(mo_coeff, occup)
    ref_rdm1 = scf.hf.make_rdm1(ref_mo_coeff, occup)

    assert nocc == ref_nocc
    assert nvirt == ref_nvirt
    assert norb == ref_norb
    assert hf_prop == pytest.approx(ref_hf_prop, rel=1e-5, abs=1e-11)
    assert (occup == np.array([2.0] * 5 + [0.0] * 8, dtype=np.float64)).all()
    assert (
        orbsym == np.array([0, 0, 2, 0, 3, 0, 2, 2, 3, 0, 0, 2, 0], dtype=np.float64)
    ).all()
    assert (
        mo_coeff == pytest.approx(ref_mo_coeff)
        if mo_coeff_eq
        else mo_coeff != pytest.approx(ref_mo_coeff)
    )
    assert (
        rdm1 == pytest.approx(ref_rdm1, rel=1e-5, abs=1e-12)
        if rdm1_eq
        else rdm1 != pytest.approx(ref_rdm1, rel=1e-5, abs=1e-12)
    )


@pytest.mark.parametrize(
    argnames="mo_occ, ref_dims",
    argvalues=[case[1:] for case in test_cases_dim],
    ids=[case[0] for case in test_cases_dim],
)
def test_dim(mo_occ: np.ndarray, ref_dims: Tuple[int, int, int]) -> None:
    """
    this function tests _dim
    """
    dims = _dim(mo_occ)

    assert dims == ref_dims


@pytest.mark.parametrize(
    argnames="system, orb_type, ref_space, casscf_kwargs, mo_coeff_eq, rdm1_eq",
    argvalues=test_cases_ref_mo,
    ids=["-".join(case[0:2]) for case in test_cases_ref_mo],
    indirect=["system"],
)
def test_ref_mo(
    mol: gto.Mole,
    hf: scf.RHF,
    orbsym: np.ndarray,
    norb: int,
    ncore: int,
    nocc: int,
    orb_type: str,
    ref_space: np.ndarray,
    casscf_kwargs: Dict[str, Any],
    mo_coeff_eq: bool,
    rdm1_eq: bool,
) -> None:
    """
    this function tests ref_mo
    """
    mo_coeff, orbsym = ref_mo(
        orb_type,
        mol,
        hf,
        hf.mo_coeff,
        hf.mo_occ,
        orbsym,
        norb,
        ncore,
        nocc,
        norb - nocc,
        ref_space,
        **casscf_kwargs,
    )

    rdm1 = scf.hf.make_rdm1(mo_coeff, hf.mo_occ)
    hf_rdm1 = scf.hf.make_rdm1(hf.mo_coeff, hf.mo_occ)

    assert (
        mo_coeff == pytest.approx(hf.mo_coeff)
        if mo_coeff_eq
        else mo_coeff != pytest.approx(hf.mo_coeff)
    )
    assert rdm1 == pytest.approx(hf_rdm1) if rdm1_eq else rdm1 != pytest.approx(hf_rdm1)


@pytest.mark.parametrize(
    argnames="system, wfnsym, weights, hf_guess, ref_sum, ref_amax",
    argvalues=test_cases_casscf,
    ids=[
        "-".join(
            [case[0]]
            + [f"{weight:02}{wfnsym}" for weight, wfnsym in zip(case[2], case[1])]
            + (["hf_guess"] if case[3] else [])
        )
        for case in test_cases_casscf
    ],
    indirect=["system"],
)
def test_casscf(
    mol: gto.Mole,
    hf: scf.RHF,
    orbsym: np.ndarray,
    wfnsym: List[str],
    weights: List[float],
    hf_guess: bool,
    ncore: int,
    ref_sum: float,
    ref_amax: float,
) -> None:
    """
    this function tests _casscf
    """
    wfnsym_int = []
    for i in range(len(wfnsym)):
        wfnsym_int.append(symm.addons.irrep_name2id(mol.groupname, wfnsym[i]))

    mo_coeff = _casscf(
        mol,
        "pyscf_spin0",
        wfnsym_int,
        weights,
        orbsym,
        hf_guess,
        hf,
        hf.mo_coeff,
        np.arange(2, 10, dtype=np.int64),
        (4, 4),
        ncore,
    )

    assert np.sum(mo_coeff) == pytest.approx(ref_sum)
    assert np.amax(mo_coeff) == pytest.approx(ref_amax)


@pytest.mark.parametrize(
    argnames="system, method, base_method, target, cc_backend, root, ref_res",
    argvalues=test_cases_ref_prop,
    ids=[
        "-".join([item for item in case[0:5] if item]) for case in test_cases_ref_prop
    ],
    indirect=["system"],
)
def test_ref_prop(
    mol: gto.Mole,
    hf: scf.RHF,
    ints: Tuple[np.ndarray, np.ndarray],
    vhf: np.ndarray,
    dipole_quantities: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    nocc: int,
    method: str,
    base_method: Optional[str],
    target: str,
    cc_backend: str,
    root: int,
    ref_res: Union[float, np.ndarray],
) -> None:
    """
    this function tests ref_prop
    """
    hcore, eri = ints

    ref_space = np.array([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64)

    kwargs = {}

    if target == "energy":

        kwargs["hf_prop"] = hf.e_tot

    elif target == "dipole":

        kwargs["dipole_ints"], kwargs["hf_prop"] = dipole_quantities

    elif target == "trans":

        kwargs["dipole_ints"], _ = dipole_quantities

    res = ref_prop(
        mol,
        hcore,
        vhf,
        eri,
        hf.mo_occ,
        orbsym,
        nocc,
        ref_space,
        method=method,
        base_method=base_method,
        cc_backend=cc_backend,
        fci_state_root=root,
        target=target,
        **kwargs,
    )

    assert res == pytest.approx(ref_res)


@pytest.mark.parametrize(
    argnames="system, method, target, cc_backend, ref",
    argvalues=test_cases_base,
    ids=["-".join(case[0:4]) for case in test_cases_base],
    indirect=["system"],
)
def test_base(
    mol: gto.Mole,
    hf: scf.RHF,
    orbsym: np.ndarray,
    norb: int,
    ncore: int,
    nocc: int,
    dipole_quantities: Tuple[np.ndarray, np.ndarray],
    method: str,
    target: str,
    cc_backend: str,
    ref: Union[float, np.ndarray],
) -> None:
    """
    this function tests base
    """
    dipole_kwargs = {}

    if target == "dipole":

        _, dipole_kwargs["hf_prop"] = dipole_quantities
        dipole_kwargs["gauge_origin"] = np.zeros(3, dtype=np.float64)

    base_prop = base(
        method,
        mol,
        hf,
        hf.mo_coeff,
        hf.mo_occ,
        orbsym,
        norb,
        ncore,
        nocc,
        cc_backend=cc_backend,
        target=target,
        **dipole_kwargs,
    )

    assert base_prop == pytest.approx(ref)


@pytest.mark.parametrize(
    argnames="system, mo_coeff",
    argvalues=test_cases_linear_orbsym,
    ids=[case[0] for case in test_cases_linear_orbsym],
    indirect=True,
)
def test_linear_orbsym(mol: gto.Mole, mo_coeff: np.ndarray) -> None:
    """
    this function tests linear_orbsym
    """
    ref = np.array(
        [0, 5, 0, 5, 6, 7, 0, 2, 3, 5, 0, 6, 7, 0, 2, 3, 5, 5], dtype=np.int64
    )

    orbsym_parent = linear_orbsym(mol, mo_coeff)

    assert (orbsym_parent == ref).all()
