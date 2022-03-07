#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
kernel testing module
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

from pymbe.kernel import e_core_h1e, main_kernel, dipole_kernel, fci_kernel, cc_kernel

if TYPE_CHECKING:

    from pyscf import gto, scf
    from typing import Tuple, Union, Dict


test_cases_main = [
    (
        "h2o",
        "fci",
        "energy",
        "pyscf",
        0,
        {"energy": -0.00627368491326763},
    ),
    (
        "hubbard",
        "fci",
        "energy",
        "pyscf",
        0,
        {"energy": -2.8759428090050676},
    ),
    (
        "h2o",
        "ccsd",
        "energy",
        "pyscf",
        0,
        {"energy": -0.006273684840715439},
    ),
    (
        "h2o",
        "ccsd",
        "energy",
        "ecc",
        0,
        {"energy": -0.00627368488758955},
    ),
    (
        "h2o",
        "ccsd",
        "energy",
        "ncc",
        0,
        {"energy": -0.006273684885561386},
    ),
    (
        "h2o",
        "fci",
        "dipole",
        "pyscf",
        0,
        {"rdm1_sum": 9.996892589445256, "rdm1_amax": 2.0},
    ),
    (
        "h2o",
        "ccsd",
        "dipole",
        "pyscf",
        0,
        {"rdm1_sum": 9.996892588820488, "rdm1_amax": 2.0},
    ),
    (
        "h2o",
        "fci",
        "excitation",
        "pyscf",
        1,
        {"excitation": 1.3506589063482437},
    ),
    (
        "hubbard",
        "fci",
        "excitation",
        "pyscf",
        1,
        {"excitation": 1.850774199956839},
    ),
    (
        "h2o",
        "fci",
        "trans",
        "pyscf",
        1,
        {
            "t_rdm1_sum": 1.4605282782265316,
            "t_rdm1_amax": 1.405776909336337,
            "t_rdm1_trace": 0.0,
        },
    ),
]

test_cases_fci_kernel = [
    (
        "h2o",
        "energy",
        0,
        {"energy": -0.00627368491326763},
    ),
    (
        "hubbard",
        "energy",
        0,
        {"energy": -2.875942809005066},
    ),
    (
        "h2o",
        "dipole",
        0,
        {"rdm1_sum": 9.996892589445256, "rdm1_amax": 2.0},
    ),
    (
        "hubbard",
        "dipole",
        0,
        {"rdm1_sum": 7.416665666590797, "rdm1_amax": 1.0},
    ),
    (
        "h2o",
        "excitation",
        1,
        {"excitation": 1.35065890634786},
    ),
    (
        "hubbard",
        "excitation",
        1,
        {"excitation": 1.8507741999568346},
    ),
    (
        "h2o",
        "trans",
        1,
        {
            "t_rdm1_sum": 1.4605282782265316,
            "t_rdm1_amax": 1.405776909336337,
            "t_rdm1_trace": 0.0,
        },
    ),
    (
        "hubbard",
        "trans",
        1,
        {
            "t_rdm1_sum": -0.4308198845268202,
            "t_rdm1_amax": 0.3958365757196145,
            "t_rdm1_trace": 0.0,
        },
    ),
]

test_cases_cc_kernel = [
    (
        "h2o",
        "ccsd",
        "energy",
        "pyscf",
        {"energy": -0.0062736848407002966},
    ),
    (
        "h2o",
        "ccsd(t)",
        "energy",
        "pyscf",
        {"energy": -0.0062736848407002966},
    ),
    (
        "h2o",
        "ccsd",
        "dipole",
        "pyscf",
        {"rdm1_sum": 9.996892588820488, "rdm1_amax": 2.0},
    ),
    (
        "h2o",
        "ccsd(t)",
        "dipole",
        "pyscf",
        {"rdm1_sum": 9.996892588820348, "rdm1_amax": 2.0},
    ),
    (
        "h2o",
        "ccsd",
        "energy",
        "ecc",
        {"energy": -0.00627368488758955},
    ),
    (
        "h2o",
        "ccsd(t)",
        "energy",
        "ecc",
        {"energy": -0.006273684887573003},
    ),
    (
        "h2o",
        "ccsdt",
        "energy",
        "ecc",
        {"energy": -0.00627368488758955},
    ),
    (
        "h2o",
        "ccsd",
        "energy",
        "ncc",
        {"energy": -0.006273684885561386},
    ),
    (
        "h2o",
        "ccsd(t)",
        "energy",
        "ncc",
        {"energy": -0.006273684885577932},
    ),
    (
        "h2o",
        "ccsdt",
        "energy",
        "ncc",
        {"energy": -0.006273684885561386},
    ),
    (
        "h2o",
        "ccsdtq",
        "energy",
        "ncc",
        {"energy": -0.006273684885577932},
    ),
]


def test_e_core_h1e() -> None:
    """
    this function tests e_core_h1e
    """
    e_nuc = 0.0
    np.random.seed(1234)
    hcore = np.random.rand(6, 6)
    np.random.seed(1234)
    vhf = np.random.rand(3, 6, 6)
    core_idx = np.array([0], dtype=np.int64)
    cas_idx = np.array([2, 4, 5], dtype=np.int64)
    e_core, h1e_cas = e_core_h1e(e_nuc, hcore, vhf, core_idx, cas_idx)

    assert e_core == pytest.approx(0.5745583511366769)
    assert h1e_cas == pytest.approx(
        np.array(
            [
                [0.74050151, 1.00616633, 0.02753690],
                [0.79440516, 0.63367224, 1.13619731],
                [1.60429528, 1.40852194, 1.40916262],
            ],
            dtype=np.float64,
        )
    )


@pytest.mark.parametrize(
    argnames="system, method, target, cc_backend, root, ref_res",
    argvalues=test_cases_main,
    ids=["-".join([item for item in case[0:4] if item]) for case in test_cases_main],
    indirect=["system"],
)
def test_kernel_main(
    system: str,
    mol: gto.Mole,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    dipole_quantities: Tuple[np.ndarray, np.ndarray],
    method: str,
    target: str,
    cc_backend: str,
    root: int,
    ref_res: Dict[str, Union[float, int]],
) -> None:
    """
    this function tests main_kernel
    """
    if system == "h2o":

        occup = hf.mo_occ
        point_group = "C2v"
        hf_energy = hf.e_tot
        e_core = mol.energy_nuc()

    elif system == "hubbard":

        occup = np.array([2.0] * 3 + [0.0] * 3, dtype=np.float64)
        point_group = "C1"
        hf_energy = 0.0
        e_core = 0.0

    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    n_elec = (
        np.count_nonzero(occup[cas_idx] > 0.0),
        np.count_nonzero(occup[cas_idx] > 1.0),
    )

    if target == "energy":

        hf_prop = np.array([hf_energy], dtype=np.float64)

    elif target == "excitation":

        hf_prop = np.array([0.0], dtype=np.float64)

    elif target == "dipole":

        _, hf_prop = dipole_quantities

    elif target == "trans":

        hf_prop = np.zeros(3, dtype=np.float64)

    res = main_kernel(
        method,
        cc_backend,
        "pyscf_spin0",
        "can",
        0,
        occup,
        target,
        0,
        point_group,
        orbsym,
        True,
        root,
        hf_prop,
        e_core,
        h1e_cas,
        h2e_cas,
        core_idx,
        cas_idx,
        n_elec,
        0,
    )

    if target == "energy":
        assert res["energy"] == pytest.approx(ref_res["energy"])
    elif target == "dipole":
        assert np.sum(res["rdm1"]) == pytest.approx(ref_res["rdm1_sum"])
        assert np.amax(res["rdm1"]) == pytest.approx(ref_res["rdm1_amax"])
    elif target == "excitation":
        assert res["excitation"] == pytest.approx(ref_res["excitation"])
    elif target == "trans":
        assert np.sum(res["t_rdm1"]) == pytest.approx(ref_res["t_rdm1_sum"])
        assert np.amax(res["t_rdm1"]) == pytest.approx(ref_res["t_rdm1_amax"])
        assert np.trace(res["t_rdm1"]) == pytest.approx(ref_res["t_rdm1_trace"])


def test_dipole() -> None:
    """
    this function tests dipole_kernel
    """
    occup = np.array([2.0] * 3 + [0.0] * 3, dtype=np.float64)
    hf_dipole = np.zeros(3, dtype=np.float64)
    cas_idx = np.arange(1, 5, dtype=np.int64)
    np.random.seed(1234)
    dipole_ints = np.random.rand(3, 6, 6)
    np.random.seed(1234)
    cas_rdm1 = np.random.rand(cas_idx.size, cas_idx.size)
    dipole = dipole_kernel(dipole_ints, occup, cas_idx, cas_rdm1, hf_dipole=hf_dipole)

    assert dipole == pytest.approx(
        np.array([5.90055525, 5.36437348, 6.40001788], dtype=np.float64)
    )


@pytest.mark.parametrize(
    argnames="system, target, root, ref",
    argvalues=test_cases_fci_kernel,
    ids=["-".join(case[0:2]) for case in test_cases_fci_kernel],
    indirect=["system"],
)
def test_fci(
    system: str,
    mol: gto.Mole,
    hf: scf.RHF,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    orbsym: np.ndarray,
    target: str,
    root: int,
    ref: Dict[str, float],
) -> None:
    """
    this function tests fci_kernel
    """
    if system == "h2o":

        hf_energy = np.array([hf.e_tot], dtype=np.float64)
        e_core = mol.energy_nuc()
        occup = hf.mo_occ

    elif system == "hubbard":

        hf_energy = np.array([0.0], dtype=np.float64)
        e_core = 0.0
        occup = np.array([2.0] * 3 + [0.0] * 3, dtype=np.float64)

    _, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    n_elec = (
        np.count_nonzero(occup[cas_idx] > 0.0),
        np.count_nonzero(occup[cas_idx] > 1.0),
    )

    res = fci_kernel(
        "pyscf_spin0",
        0,
        target,
        0,
        orbsym,
        True,
        root,
        hf_energy,
        e_core,
        h1e_cas,
        h2e_cas,
        occup,
        cas_idx,
        n_elec,
        0,
    )

    if target == "energy":
        assert res["energy"] == pytest.approx(ref["energy"])
    elif target == "dipole":
        assert np.sum(res["rdm1"]) == pytest.approx(ref["rdm1_sum"])
        assert np.amax(res["rdm1"]) == pytest.approx(ref["rdm1_amax"])
    elif target == "excitation":
        assert res["excitation"] == pytest.approx(ref["excitation"])
    elif target == "trans":
        assert np.sum(res["t_rdm1"]) == pytest.approx(ref["t_rdm1_sum"])
        assert np.amax(res["t_rdm1"]) == pytest.approx(ref["t_rdm1_amax"])
        assert np.trace(res["t_rdm1"]) == pytest.approx(ref["t_rdm1_trace"])


@pytest.mark.parametrize(
    argnames="system, method, target, cc_backend, ref",
    argvalues=test_cases_cc_kernel,
    ids=["-".join(case[0:3]) + "-" + case[3] for case in test_cases_cc_kernel],
    indirect=["system"],
)
def test_cc(
    hf: scf.RHF,
    orbsym: np.ndarray,
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ints_cas: Tuple[np.ndarray, np.ndarray],
    method: str,
    target: str,
    cc_backend: str,
    ref: Dict[str, float],
) -> None:
    """
    this function tests cc_kernel
    """
    core_idx, cas_idx, _ = indices

    h1e_cas, h2e_cas = ints_cas

    n_elec = (
        np.count_nonzero(hf.mo_occ[cas_idx] > 0.0),
        np.count_nonzero(hf.mo_occ[cas_idx] > 1.0),
    )

    res = cc_kernel(
        0,
        hf.mo_occ,
        core_idx,
        cas_idx,
        method,
        cc_backend,
        n_elec,
        "can",
        "C2v",
        orbsym,
        h1e_cas,
        h2e_cas,
        True,
        target,
        0,
    )

    if target == "energy":
        assert res["energy"] == pytest.approx(ref["energy"])
    elif target == "dipole":
        assert np.sum(res["rdm1"]) == pytest.approx(ref["rdm1_sum"])
        assert np.amax(res["rdm1"]) == pytest.approx(ref["rdm1_amax"])
