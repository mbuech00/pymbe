#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
system testing module
"""

from __future__ import annotations

__author__ = "Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import pytest
import importlib
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

examples = [
    ["h2o", "ccsd", "energy", "can", "vac"],
    ["h2o", "ccsd", "energy", "can", "occ"],
    ["h2o", "ccsd", "energy", "can", "mixed"],
    ["h2o", "ccsd", "dipole", "can", "occ"],
    ["h2o", "ccsd", "dipole", "can", "mixed"],
    ["h2o", "ccsd", "rdm12", "can", "occ"],
    ["h2o", "ccsd", "rdm12", "can", "mixed"],
    ["h2o", "ccsd(t)", "energy", "can", "vac"],
    ["h2o", "ccsd(t)", "energy", "can", "occ"],
    ["h2o", "ccsd(t)", "energy", "can", "mixed"],
    ["h2o", "ccsd(t)", "dipole", "can", "occ"],
    ["h2o", "ccsd(t)", "dipole", "can", "mixed"],
    ["h2o", "ccsd(t)", "rdm12", "can", "occ"],
    ["h2o", "ccsd(t)", "rdm12", "can", "mixed"],
    ["h2o", "ccsdt", "energy", "can", "vac"],
    ["h2o", "ccsdt", "energy", "can", "occ"],
    ["h2o", "ccsdt", "energy", "can", "mixed"],
    ["h2o", "ccsdtq", "energy", "can", "occ"],
    ["h2o", "ccsdtq", "energy", "can", "mixed"],
    ["h2o", "fci", "energy", "can", "vac"],
    ["h2o", "fci", "energy", "can", "occ"],
    ["h2o", "fci", "energy", "can", "mixed"],
    ["h2o", "fci", "energy", "can", "vac", "ccsd"],
    ["h2o", "fci", "energy", "can", "occ", "ccsd"],
    ["h2o", "fci", "energy", "can", "mixed", "ccsd"],
    ["h2o", "fci", "energy", "can", "vac", "ccsd(t)"],
    ["h2o", "fci", "energy", "can", "occ", "ccsd(t)"],
    ["h2o", "fci", "energy", "can", "mixed", "ccsd(t)"],
    ["h2o", "fci", "energy", "can", "vac", "ccsdt"],
    ["h2o", "fci", "energy", "can", "occ", "ccsdt"],
    ["h2o", "fci", "energy", "can", "mixed", "ccsdt"],
    ["h2o", "fci", "energy", "can", "occ", "ccsdtq"],
    ["h2o", "fci", "energy", "can", "mixed", "ccsdtq"],
    ["h2o", "fci", "energy", "ccsd", "vac"],
    ["h2o", "fci", "energy", "ccsd", "occ"],
    ["h2o", "fci", "energy", "ccsd", "mixed"],
    ["h2o", "fci", "energy", "ccsd(t)", "vac"],
    ["h2o", "fci", "energy", "ccsd(t)", "occ"],
    ["h2o", "fci", "energy", "ccsd(t)", "mixed"],
    ["h2o", "fci", "energy", "casscf", "mixed"],
    ["h2o", "fci", "dipole", "can", "vac"],
    ["h2o", "fci", "dipole", "can", "occ"],
    ["h2o", "fci", "dipole", "can", "mixed"],
    ["h2o", "fci", "dipole", "can", "vac", "ccsd"],
    ["h2o", "fci", "dipole", "can", "occ", "ccsd"],
    ["h2o", "fci", "dipole", "can", "mixed", "ccsd"],
    ["h2o", "fci", "dipole", "can", "vac", "ccsd(t)"],
    ["h2o", "fci", "dipole", "can", "occ", "ccsd(t)"],
    ["h2o", "fci", "dipole", "can", "mixed", "ccsd(t)"],
    ["h2o", "fci", "dipole", "ccsd", "vac"],
    ["h2o", "fci", "dipole", "ccsd", "occ"],
    ["h2o", "fci", "dipole", "ccsd", "mixed"],
    ["h2o", "fci", "dipole", "ccsd(t)", "vac"],
    ["h2o", "fci", "dipole", "ccsd(t)", "occ"],
    ["h2o", "fci", "dipole", "ccsd(t)", "mixed"],
    ["h2o", "fci", "excitation", "can", "mixed"],
    ["h2o", "fci", "trans", "can", "mixed"],
    ["h2o", "fci", "rdm12", "can", "vac"],
    ["h2o", "fci", "rdm12", "can", "occ"],
    ["h2o", "fci", "rdm12", "can", "mixed"],
    ["h2o", "fci", "sa_rdm12", "can", "mixed"],
    ["h2o", "fci", "sa_genfock", "can", "mixed"],
    ["h2o", "fci", "genfock", "can", "vac"],
    ["h2o", "fci", "genfock", "can", "occ"],
    ["h2o", "fci", "genfock", "can", "mixed"],
    ["ch2", "ccsd", "energy", "can", "vac"],
    ["ch2", "ccsd", "energy", "can", "occ"],
    ["ch2", "ccsd", "energy", "can", "mixed"],
    ["ch2", "ccsd", "dipole", "can", "vac"],
    ["ch2", "ccsd", "dipole", "can", "occ"],
    ["ch2", "ccsd", "dipole", "can", "mixed"],
    ["ch2", "ccsd(t)", "energy", "can", "vac"],
    ["ch2", "ccsd(t)", "energy", "can", "occ"],
    ["ch2", "ccsd(t)", "energy", "can", "mixed"],
    ["ch2", "ccsd(t)", "dipole", "can", "vac"],
    ["ch2", "ccsd(t)", "dipole", "can", "occ"],
    ["ch2", "ccsd(t)", "dipole", "can", "mixed"],
    ["ch2", "fci", "energy", "can", "vac"],
    ["ch2", "fci", "energy", "can", "occ"],
    ["ch2", "fci", "energy", "can", "mixed"],
    ["ch2", "fci", "energy", "can", "vac", "ccsd"],
    ["ch2", "fci", "energy", "can", "occ", "ccsd"],
    ["ch2", "fci", "energy", "can", "mixed", "ccsd"],
    ["ch2", "fci", "energy", "can", "vac", "ccsd(t)"],
    ["ch2", "fci", "energy", "can", "occ", "ccsd(t)"],
    ["ch2", "fci", "energy", "can", "mixed", "ccsd(t)"],
    ["ch2", "fci", "energy", "ccsd", "vac"],
    ["ch2", "fci", "energy", "ccsd", "occ"],
    ["ch2", "fci", "energy", "ccsd", "mixed"],
    ["ch2", "fci", "energy", "ccsd(t)", "vac"],
    ["ch2", "fci", "energy", "ccsd(t)", "occ"],
    ["ch2", "fci", "energy", "ccsd(t)", "mixed"],
    ["ch2", "fci", "energy", "casscf", "occ"],
    ["ch2", "fci", "energy", "casscf", "mixed"],
    ["ch2", "fci", "dipole", "can", "vac"],
    ["ch2", "fci", "dipole", "can", "occ"],
    ["ch2", "fci", "dipole", "can", "mixed"],
    ["ch2", "fci", "dipole", "ccsd", "vac"],
    ["ch2", "fci", "dipole", "ccsd", "occ"],
    ["ch2", "fci", "dipole", "ccsd", "mixed"],
    ["ch2", "fci", "dipole", "ccsd(t)", "vac"],
    ["ch2", "fci", "dipole", "ccsd(t)", "occ"],
    ["ch2", "fci", "dipole", "ccsd(t)", "mixed"],
    ["ch2", "fci", "excitation", "can", "mixed"],
    ["ch2", "fci", "trans", "can", "mixed"],
    ["c2", "fci", "energy", "can", "mixed", "pi_prune"],
    ["c2", "fci", "energy", "sa_casscf", "mixed"],
]


@pytest.mark.parametrize(
    argnames="example",
    argvalues=examples,
    ids=["-".join([string for string in example if string]) for example in examples],
)
def test_system(example: List[str]) -> None:
    """
    this test system tests pymbe by executing all example scripts
    """
    string = (
        "pymbe.examples."
        + ".".join([string for string in example])
        + "."
        + "_".join([string for string in example])
    )

    example_module = importlib.import_module(string)

    prop = example_module.mbe_example(rst=False)

    ref_path = (
        Path(__file__).parent
        / "../../examples"
        / "/".join([string for string in example])
    )

    if example[2] in ["energy", "excitation", "dipole"]:
        assert prop == pytest.approx(np.load(ref_path / "ref.npy"))

    elif example[2] == "trans":
        assert prop == pytest.approx(np.load(ref_path / "ref.npy"), rel=1e-5, abs=1e-12)

    elif example[2] == "rdm12":
        assert prop[0] == pytest.approx(
            np.load(ref_path / "ref_rdm1.npy"), rel=1e-5, abs=1e-12
        )
        assert prop[1] == pytest.approx(
            np.load(ref_path / "ref_rdm2.npy"), rel=1e-5, abs=1e-8
        )

    elif example[2] == "genfock":
        assert prop[0] == pytest.approx(np.load(ref_path / "ref_energy.npy"))
        assert prop[1] == pytest.approx(
            np.load(ref_path / "ref_gen_fock.npy"), rel=1e-5, abs=1e-11
        )
