#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
excitation energy testing module
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

from pymbe.excitation import ExcExpCls

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
        0.7060145137233889,
        (1.0, 1.0),
        (0.9856854425080487, 0.4710257978333213),
    ),
]

test_cases_kernel = [
    ("h2o", "fci", "pyscf", 1, 1.3506589063482437),
    ("hubbard", "fci", "pyscf", 1, 1.850774199956839),
]

test_cases_fci_kernel = [
    (
        "h2o",
        1,
        1.35065890634786,
        (1.0, 1.0),
        (0.9979706719230536, 0.4951968133086011),
    ),
    (
        "hubbard",
        1,
        1.8507741999568346,
        (1.0, 1.0),
        (0.14031179591440077, 0.2135931586339437),
    ),
]


@pytest.fixture
def exp(mbe: MBE):
    """
    this fixture constructs a ExcExpCls object
    """
    exp = ExcExpCls(mbe)
    exp.target = "excitation"

    return exp


@pytest.mark.parametrize(
    argnames="system, method, base_method, cc_backend, root, ref_res, ref_civec_sum, "
    "ref_civec_amax",
    argvalues=test_cases_ref_prop,
    ids=[
        "-".join([item for item in case[0:4] if item]) for case in test_cases_ref_prop
    ],
    indirect=["system"],
)
def test_ref_prop(
    mbe: MBE,
    exp: ExcExpCls,
    ints: Tuple[np.ndarray, np.ndarray],
    vhf: np.ndarray,
    orbsym: np.ndarray,
    method: str,
    base_method: Optional[str],
    cc_backend: str,
    root: int,
    ref_res: float,
    ref_civec_sum: Tuple[float, float],
    ref_civec_amax: Tuple[float, float],
) -> None:
    """
    this function tests ref_prop
    """
    exp.method = method
    exp.cc_backend = cc_backend
    exp.orbsym = orbsym
    exp.fci_state_root = root
    exp.hcore, exp.eri = ints
    exp.vhf = vhf
    exp.ref_space = np.array([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64)
    exp.ref_nelec = np.array(
        [
            np.count_nonzero(exp.occup[exp.ref_space] > 0.0),
            np.count_nonzero(exp.occup[exp.ref_space] > 1.0),
        ],
    )
    exp.base_method = base_method

    res, civec = exp._ref_prop(mbe.mpi)

    assert res == pytest.approx(ref_res)
    assert np.sum(civec[0] ** 2) == pytest.approx(ref_civec_sum[0])
    assert np.amax(civec[0] ** 2) == pytest.approx(ref_civec_amax[0])
    assert np.sum(civec[1] ** 2) == pytest.approx(ref_civec_sum[1])
    assert np.amax(civec[1] ** 2) == pytest.approx(ref_civec_amax[1])


@pytest.mark.parametrize(
    argnames="system, method, cc_backend, root, ref_res",
    argvalues=test_cases_kernel,
    ids=["-".join([item for item in case[0:3] if item]) for case in test_cases_kernel],
    indirect=["system"],
)
def test_kernel(
    exp: ExcExpCls,
    system: str,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    method: str,
    cc_backend: str,
    root: int,
    ref_res: float,
) -> None:
    """
    this function tests _kernel
    """
    exp.orbsym = orbsym
    exp.fci_state_root = root

    if system == "h2o":
        occup = hf.mo_occ

    elif system == "hubbard":
        occup = np.array([2.0] * 3 + [0.0] * 3, dtype=np.float64)
        exp.hf_guess = False

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    nelec = np.array(
        [
            np.count_nonzero(occup[cas_idx] > 0.0),
            np.count_nonzero(occup[cas_idx] > 1.0),
        ]
    )

    res = exp._kernel(
        method, 0.0, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, ref_guess=False
    )

    assert res == pytest.approx(ref_res)


@pytest.mark.parametrize(
    argnames="system, root, ref, ref_civec_sum, ref_civec_amax",
    argvalues=test_cases_fci_kernel,
    ids=[case[0] for case in test_cases_fci_kernel],
    indirect=["system"],
)
def test_fci_kernel(
    exp: ExcExpCls,
    system: str,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    root: int,
    ref: float,
    ref_civec_sum: Tuple[float, float],
    ref_civec_amax: Tuple[float, float],
) -> None:
    """
    this function tests _fci_kernel
    """
    exp.orbsym = orbsym
    exp.fci_state_root = root

    if system == "h2o":
        occup = hf.mo_occ

    elif system == "hubbard":
        occup = np.array([2.0] * 3 + [0.0] * 3, dtype=np.float64)
        exp.hf_guess = False

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    nelec = np.array(
        [
            np.count_nonzero(occup[cas_idx] > 0.0),
            np.count_nonzero(occup[cas_idx] > 1.0),
        ]
    )

    res, civec = exp._fci_kernel(0.0, h1e_cas, h2e_cas, core_idx, cas_idx, nelec, False)

    assert res == pytest.approx(ref)
    assert np.sum(civec[0] ** 2) == pytest.approx(ref_civec_sum[0])
    assert np.amax(civec[0] ** 2) == pytest.approx(ref_civec_amax[0])
    assert np.sum(civec[1] ** 2) == pytest.approx(ref_civec_sum[1])
    assert np.amax(civec[1] ** 2) == pytest.approx(ref_civec_amax[1])
