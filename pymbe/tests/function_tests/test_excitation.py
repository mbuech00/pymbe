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
    argnames="system, method, base_method, cc_backend, root, ref_res",
    argvalues=test_cases_ref_prop,
    ids=[
        "-".join([item for item in case[0:4] if item]) for case in test_cases_ref_prop
    ],
    indirect=["system"],
)
def test_ref_prop(
    mbe: MBE,
    exp: ExcExpCls,
    ints_win: Tuple[MPI.Win, MPI.Win, MPI.Win],
    orbsym: np.ndarray,
    method: str,
    base_method: Optional[str],
    cc_backend: str,
    root: int,
    ref_res: float,
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
