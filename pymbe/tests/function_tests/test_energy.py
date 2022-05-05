#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
energy testing module
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
import scipy.special as sc
from mpi4py import MPI
from typing import TYPE_CHECKING

from pymbe.energy import EnergyExpCls

if TYPE_CHECKING:

    from typing import List, Tuple, Optional
    from pyscf import scf

    from pymbe.pymbe import MBE

test_cases_mbe = [
    (
        "h2o",
        1,
        -249055688365223385,
        9199082625845137542,
        -0.0374123708341898,
        -0.0018267714680604286,
        -0.0374123708341898,
        -0.004676546354273725,
        0.0018267714680604286,
        0.010886635891736773,
    ),
    (
        "h2o",
        2,
        8509729643108359722,
        8290417336063232159,
        -0.11605435599270209,
        -0.0001698239845069338,
        -0.15346672682557028,
        -0.004144798428310789,
        0.0001698239845069338,
        0.009556269221292268,
    ),
]

test_cases_ref_prop = [
    (
        "h2o",
        "fci",
        None,
        "pyscf",
        0,
        -0.03769780809258805,
    ),
    (
        "h2o",
        "ccsd",
        None,
        "pyscf",
        0,
        -0.03733551374348559,
    ),
    (
        "h2o",
        "fci",
        "ccsd",
        "pyscf",
        0,
        -0.00036229313775759664,
    ),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "pyscf",
        0,
        -0.0003336954549769955,
    ),
    (
        "h2o",
        "ccsd",
        None,
        "ecc",
        0,
        -0.03733551374348559,
    ),
    (
        "h2o",
        "fci",
        "ccsd",
        "ecc",
        0,
        -0.0003622938195746786,
    ),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "ecc",
        0,
        -0.0003336954549769955,
    ),
    (
        "h2o",
        "ccsd",
        None,
        "ncc",
        0,
        -0.03733551374348559,
    ),
    (
        "h2o",
        "fci",
        "ccsd",
        "ncc",
        0,
        -0.0003622938195746786,
    ),
    (
        "h2o",
        "ccsd(t)",
        "ccsd",
        "ncc",
        0,
        -0.0003336954549769955,
    ),
]


@pytest.fixture
def exp(mbe: MBE):
    """
    this fixture constructs a EnergyExpCls object
    """
    exp = EnergyExpCls(mbe)
    exp.target = "energy"

    return exp


@pytest.mark.parametrize(
    argnames="system, order, ref_hashes_sum, ref_hashes_amax, ref_inc_sum, "
    "ref_inc_amax, ref_mbe_tot_prop, ref_mean_inc, ref_min_inc, ref_max_inc",
    argvalues=test_cases_mbe,
    ids=["-".join([case[0], str(case[1])]) for case in test_cases_mbe],
    indirect=["system"],
)
def test_mbe(
    mbe: MBE,
    exp: EnergyExpCls,
    hf: scf.RHF,
    nocc: int,
    order: int,
    ref_hashes_sum: int,
    ref_hashes_amax: int,
    ref_inc_sum: float,
    ref_inc_amax: float,
    ref_mbe_tot_prop: np.ndarray,
    ref_mean_inc: float,
    ref_min_inc: float,
    ref_max_inc: float,
) -> None:
    """
    this function tests _mbe
    """
    exp.hf_prop = hf.e_tot

    hashes: List[np.ndarray] = []
    inc: List[np.ndarray] = []

    for exp.order in range(1, order + 1):

        n_tuples = 0.0

        for k in range(1, exp.order):
            n_tuples += sc.binom(
                exp.exp_space[-1][exp.exp_space[-1] < nocc].size, k
            ) * sc.binom(
                exp.exp_space[-1][nocc <= exp.exp_space[-1]].size, exp.order - k
            )

        n_tuples += sc.binom(
            exp.exp_space[-1][exp.exp_space[-1] < nocc].size, exp.order
        )
        n_tuples += sc.binom(
            exp.exp_space[-1][nocc <= exp.exp_space[-1]].size, exp.order
        )

        exp.n_tuples["inc"].append(int(n_tuples))

        exp._mbe(mbe.mpi)

        hashes.append(
            np.ndarray(
                buffer=exp.hashes[-1].Shared_query(0)[0],  # type: ignore
                dtype=np.int64,
                shape=(exp.n_tuples["inc"][exp.order - 1],),
            )
        )

        inc.append(
            np.ndarray(
                buffer=exp.incs[-1].Shared_query(0)[0],  # type: ignore
                dtype=np.float64,
                shape=(exp.n_tuples["inc"][exp.order - 1], 1),
            )
        )

        exp.hashes.append(exp.hashes[-1])

        exp.incs.append(exp.incs[-1])

        exp.mean_inc.append(exp.mean_inc[-1])
        exp.min_inc.append(exp.min_inc[-1])
        exp.max_inc.append(exp.max_inc[-1])

    assert isinstance(exp.hashes[-1], MPI.Win)
    assert isinstance(exp.incs[-1], MPI.Win)
    assert np.sum(hashes[-1]) == ref_hashes_sum
    assert np.amax(hashes[-1]) == ref_hashes_amax
    assert np.sum(inc[-1]) == pytest.approx(ref_inc_sum)
    assert np.amax(inc[-1]) == pytest.approx(ref_inc_amax)
    assert exp.mbe_tot_prop[-1] == pytest.approx(ref_mbe_tot_prop)
    assert exp.mean_inc[-1] == pytest.approx(ref_mean_inc)
    assert exp.min_inc[-1] == pytest.approx(ref_min_inc)
    assert exp.max_inc[-1] == pytest.approx(ref_max_inc)


@pytest.mark.parametrize(
    argnames="system",
    argvalues=["h2o"],
    indirect=["system"],
)
def test_purge(mbe: MBE, exp: EnergyExpCls) -> None:
    """
    this function tests _purge
    """
    ref_hashes = [
        np.array(
            [
                -6318372561352273418,
                -5475322122992870313,
                -1752257283205524125,
                -669804309911520350,
                1455941523185766351,
                6981656516950638826,
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                -8862568739552411231,
                -7925134385272954056,
                -7216722148388372205,
                -6906205837173860435,
                -4310406760124882618,
                -4205406112023021717,
                -3352798558434503475,
                366931854209709639,
                6280027850766028273,
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                -9111224886591032877,
                -6640293625692100246,
                -4012521487842354405,
                -2930228190932741801,
                2993709457496479298,
            ],
            dtype=np.int64,
        ),
    ]

    ref_inc = [
        np.array([1.0, 2.0, 4.0, 5.0, 6.0, 8.0], dtype=np.float64),
        np.array([1.0, 2.0, 4.0, 5.0, 8.0, 9.0, 10.0, 11.0, 15.0], dtype=np.float64),
        np.array([2.0, 4.0, 5.0, 7.0, 12.0], dtype=np.float64),
    ]

    exp.nocc = 3
    exp.occup = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    exp.exp_space = [np.array([0, 1, 2, 3, 5], dtype=np.int64)]
    exp.screen_orbs = np.array([4], dtype=np.int64)
    exp.order = 4
    exp.min_order = 2
    exp.n_tuples = {"inc": [9, 18, 15]}

    start_hashes = [
        np.array(
            [
                -6318372561352273418,
                -5475322122992870313,
                -2211238527921376434,
                -1752257283205524125,
                -669804309911520350,
                1455941523185766351,
                2796798554289973955,
                6981656516950638826,
                7504768460337078519,
            ]
        ),
        np.array(
            [
                -8862568739552411231,
                -7925134385272954056,
                -7370655119274612396,
                -7216722148388372205,
                -6906205837173860435,
                -6346674104600383423,
                -6103692259034244091,
                -4310406760124882618,
                -4205406112023021717,
                -3352798558434503475,
                366931854209709639,
                680656656239891583,
                3949415985151233945,
                4429162622039029653,
                6280027850766028273,
                7868645139422709341,
                8046408145842912366,
                8474590989972277172,
            ]
        ),
        np.array(
            [
                -9191542714830049336,
                -9111224886591032877,
                -8945201412191574338,
                -6640293625692100246,
                -4012521487842354405,
                -3041224019630807622,
                -2930228190932741801,
                -864833587293421682,
                775579459894020071,
                1344711228121337165,
                2515975357592924865,
                2993709457496479298,
                4799605789003109011,
                6975445416347248252,
                7524854823186007981,
            ]
        ),
    ]

    hashes: List[np.ndarray] = []
    inc: List[np.ndarray] = []

    for k in range(0, 3):

        hashes_win = MPI.Win.Allocate_shared(
            8 * exp.n_tuples["inc"][k], 8, comm=mbe.mpi.local_comm  # type: ignore
        )
        buf = hashes_win.Shared_query(0)[0]
        hashes.append(
            np.ndarray(
                buffer=buf,  # type: ignore
                dtype=np.int64,
                shape=(exp.n_tuples["inc"][k],),
            )
        )
        hashes[-1][:] = start_hashes[k]
        exp.hashes.append(hashes_win)

        inc_win = MPI.Win.Allocate_shared(
            8 * exp.n_tuples["inc"][k], 8, comm=mbe.mpi.local_comm  # type: ignore
        )
        buf = inc_win.Shared_query(0)[0]
        inc.append(
            np.ndarray(
                buffer=buf,  # type: ignore
                dtype=np.float64,
                shape=(exp.n_tuples["inc"][k],),
            )
        )
        inc[-1][:] = np.arange(1, exp.n_tuples["inc"][k] + 1, dtype=np.float64)
        exp.incs.append(inc_win)

    exp._purge(mbe.mpi)

    purged_hashes: List[np.ndarray] = []
    purged_inc: List[np.ndarray] = []

    for k in range(0, 3):

        buf = exp.hashes[k].Shared_query(0)[0]
        purged_hashes.append(
            np.ndarray(
                buffer=buf,  # type: ignore
                dtype=np.int64,
                shape=(exp.n_tuples["inc"][k],),
            )
        )

        buf = exp.incs[k].Shared_query(0)[0]
        purged_inc.append(
            np.ndarray(
                buffer=buf,  # type: ignore
                dtype=np.float64,
                shape=(exp.n_tuples["inc"][k],),
            )
        )

    assert exp.n_tuples["inc"] == [6, 9, 5]
    assert (purged_hashes[0] == ref_hashes[0]).all()
    assert (purged_hashes[1] == ref_hashes[1]).all()
    assert (purged_hashes[2] == ref_hashes[2]).all()
    assert (purged_inc[0] == ref_inc[0]).all()
    assert (purged_inc[1] == ref_inc[1]).all()
    assert (purged_inc[2] == ref_inc[2]).all()


@pytest.mark.parametrize(
    argnames="system, method, base_method, cc_backend, root, ref_res",
    argvalues=test_cases_ref_prop,
    ids=[
        "-".join([item for item in case[0:3] if item]) + "-energy-" + case[3]
        for case in test_cases_ref_prop
    ],
    indirect=["system"],
)
def test_ref_prop(
    mbe: MBE,
    exp: EnergyExpCls,
    hf: scf.RHF,
    ints: Tuple[np.ndarray, np.ndarray],
    vhf: np.ndarray,
    orbsym: np.ndarray,
    method: str,
    base_method: Optional[str],
    cc_backend: str,
    root: int,
    ref_res: float,
) -> None:
    """
    this function tests _ref_prop
    """
    hcore, eri = ints

    exp.method = method
    exp.cc_backend = cc_backend
    exp.orbsym = orbsym
    exp.fci_state_root = root
    exp.hf_prop = hf.e_tot
    exp.ref_space = np.array([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64)
    exp.base_method = base_method

    res = exp._ref_prop(hcore, eri, vhf, mbe.mpi)

    assert res == pytest.approx(ref_res)
