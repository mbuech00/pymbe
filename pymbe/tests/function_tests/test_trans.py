#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
transition dipole moment testing module
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

from pymbe.trans import TransExpCls

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
        1,
        np.array([0.0, 0.0, 0.72582795], dtype=np.float64),
    ),
]

test_cases_kernel = [
    ("h2o", "fci", "pyscf", 1, np.array([0.0, 0.0, 4.37279173e-02], dtype=np.float64)),
]

test_cases_fci_kernel = [
    ("h2o", 1, np.array([0.0, 0.0, 4.37279173e-02], dtype=np.float64)),
]


@pytest.fixture
def exp(mbe: MBE, dipole_quantities: Tuple[np.ndarray, np.ndarray]):
    """
    this fixture constructs a TransExpCls object
    """
    mbe.dipole_ints, _ = dipole_quantities
    exp = TransExpCls(mbe)
    exp.target = "trans"

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
    exp: TransExpCls,
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
    argnames="system, method, cc_backend, root, ref_res",
    argvalues=test_cases_kernel,
    ids=["-".join([item for item in case[0:3] if item]) for case in test_cases_kernel],
    indirect=["system"],
)
def test_kernel(
    exp: TransExpCls,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    method: str,
    cc_backend: str,
    root: int,
    ref_res: np.ndarray,
) -> None:
    """
    this function tests _kernel
    """
    exp.orbsym = orbsym
    exp.fci_state_root = root

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
    argnames="system, root, ref",
    argvalues=test_cases_fci_kernel,
    ids=[case[0] for case in test_cases_fci_kernel],
    indirect=["system"],
)
def test_fci_kernel(
    exp: TransExpCls,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    root: int,
    ref: np.ndarray,
) -> None:
    """
    this function tests _fci_kernel
    """
    exp.orbsym = orbsym
    exp.fci_state_root = root

    occup = hf.mo_occ

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
