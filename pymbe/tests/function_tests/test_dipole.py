#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
dipole moment testing module
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
from typing import TYPE_CHECKING

from pymbe.dipole import DipoleExpCls

if TYPE_CHECKING:

    from mpi4py import MPI
    from pyscf import scf
    from typing import Tuple, Optional

    from pymbe.pymbe import MBE

test_cases_ref_prop = [
    (
        "h2o",
        "fci",
        None,
        "pyscf",
        0,
        np.array([0.0, 0.0, -0.02732937], dtype=np.float64),
    ),
    (
        "h2o",
        "ccsd",
        None,
        "pyscf",
        0,
        np.array([0.0, 0.0, -2.87487935e-02], dtype=np.float64),
    ),
    (
        "h2o",
        "fci",
        "ccsd",
        "pyscf",
        0,
        np.array([0.0, 0.0, 1.41941689e-03], dtype=np.float64),
    ),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "pyscf",
        0,
        np.array([0.0, 0.0, 1.47038530e-03], dtype=np.float64),
    ),
]

test_cases_kernel = [
    ("h2o", "fci", "pyscf", np.array([0.0, 0.0, -1.02539807e-03], dtype=np.float64)),
    ("h2o", "ccsd", "pyscf", np.array([0.0, 0.0, -1.02539807e-03], dtype=np.float64)),
]

test_cases_fci_kernel = [
    ("h2o", np.array([0.0, 0.0, -1.02539807e-03], dtype=np.float64)),
]

test_cases_cc_kernel = [
    (
        "h2o",
        "ccsd",
        "pyscf",
        np.array([0.0, 0.0, -1.02539807e-03], dtype=np.float64),
    ),
    (
        "h2o",
        "ccsd(t)",
        "pyscf",
        np.array([0.0, 0.0, -1.02539807e-03], dtype=np.float64),
    ),
]


@pytest.fixture
def exp(mbe: MBE, dipole_quantities: Tuple[np.ndarray, np.ndarray]):
    """
    this fixture constructs a DipoleExpCls object
    """
    mbe.dipole_ints, _ = dipole_quantities
    exp = DipoleExpCls(mbe)
    exp.target = "dipole"

    return exp


@pytest.mark.parametrize(
    argnames="system, method, base_method, cc_backend, root, ref_res",
    argvalues=test_cases_ref_prop,
    ids=[
        "-".join([item for item in case[0:4] if item]) for case in test_cases_ref_prop
    ],
    indirect=["system"],
)
def test_ref_prop(
    mbe: MBE,
    exp: DipoleExpCls,
    ints_win: Tuple[MPI.Win, MPI.Win, MPI.Win],
    orbsym: np.ndarray,
    method: str,
    base_method: Optional[str],
    cc_backend: str,
    root: int,
    ref_res: np.ndarray,
) -> None:
    """
    this function tests ref_prop
    """
    exp.method = method
    exp.cc_backend = cc_backend
    exp.orbsym = orbsym
    exp.fci_state_root = root
    exp.hcore, exp.eri, exp.vhf = ints_win
    exp.ref_space = np.array([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64)
    exp.base_method = base_method

    res = exp._ref_prop(mbe.mpi)

    assert res == pytest.approx(ref_res)


@pytest.mark.parametrize(
    argnames="system, method, cc_backend, ref_res",
    argvalues=test_cases_kernel,
    ids=["-".join([item for item in case[0:3] if item]) for case in test_cases_kernel],
    indirect=["system"],
)
def test_kernel(
    exp: DipoleExpCls,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    method: str,
    cc_backend: str,
    ref_res: np.ndarray,
) -> None:
    """
    this function tests _kernel
    """
    exp.orbsym = orbsym

    occup = hf.mo_occ

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    nelec = np.array(
        [
            np.count_nonzero(occup[cas_idx] > 0.0),
            np.count_nonzero(occup[cas_idx] > 1.0),
        ]
    )

    res = exp._kernel(method, 0.0, h1e_cas, h2e_cas, core_idx, cas_idx, nelec)

    assert res == pytest.approx(ref_res)


@pytest.mark.parametrize(
    argnames="system, ref",
    argvalues=test_cases_fci_kernel,
    ids=[case[0] for case in test_cases_fci_kernel],
    indirect=["system"],
)
def test_fci_kernel(
    exp: DipoleExpCls,
    system: str,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    ref: np.ndarray,
) -> None:
    """
    this function tests _fci_kernel
    """
    exp.orbsym = orbsym

    if system == "h2o":

        occup = hf.mo_occ

    elif system == "hubbard":

        occup = np.array([2.0] * 3 + [0.0] * 3, dtype=np.float64)

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    nelec = np.array(
        [
            np.count_nonzero(occup[cas_idx] > 0.0),
            np.count_nonzero(occup[cas_idx] > 1.0),
        ]
    )

    res = exp._fci_kernel(0.0, h1e_cas, h2e_cas, core_idx, cas_idx, nelec)

    assert res == pytest.approx(ref)


@pytest.mark.parametrize(
    argnames="system, method, cc_backend, ref",
    argvalues=test_cases_cc_kernel,
    ids=["-".join(case[0:3]) for case in test_cases_cc_kernel],
    indirect=["system"],
)
def test_cc_kernel(
    exp: DipoleExpCls,
    hf: scf.RHF,
    orbsym: np.ndarray,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    method: str,
    cc_backend: str,
    ref: np.ndarray,
) -> None:
    """
    this function tests _cc_kernel
    """
    exp.orbsym = orbsym

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    nelec = np.array(
        [
            np.count_nonzero(hf.mo_occ[cas_idx] > 0.0),
            np.count_nonzero(hf.mo_occ[cas_idx] > 1.0),
        ]
    )

    res = exp._cc_kernel(method, core_idx, cas_idx, nelec, h1e_cas, h2e_cas, False)

    assert res == pytest.approx(ref)
