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


@pytest.fixture
def exp(mbe: MBE):
    """
    this fixture constructs a DipoleExpCls object
    """
    exp = DipoleExpCls(mbe)
    exp.target = "dipole"

    return exp


@pytest.mark.parametrize(
    argnames="system, method, base_method, cc_backend, root, ref_res",
    argvalues=test_cases_ref_prop,
    ids=[
        "-".join([item for item in case[0:3] if item]) + "-dipole-" + case[3]
        for case in test_cases_ref_prop
    ],
    indirect=["system"],
)
def test_ref_prop(
    mbe: MBE,
    exp: DipoleExpCls,
    ints: Tuple[np.ndarray, np.ndarray],
    vhf: np.ndarray,
    dipole_quantities: Tuple[np.ndarray, np.ndarray],
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
    hcore, eri = ints

    exp.method = method
    exp.cc_backend = cc_backend
    exp.orbsym = orbsym
    exp.fci_state_root = root
    exp.dipole_ints, exp.hf_prop = dipole_quantities
    exp.ref_space = np.array([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64)
    exp.base_method = base_method

    res = exp._ref_prop(hcore, eri, vhf, mbe.mpi)

    assert res == pytest.approx(ref_res)
