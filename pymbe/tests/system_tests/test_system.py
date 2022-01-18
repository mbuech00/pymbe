#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
system testing module
"""

from __future__ import annotations

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import pytest
import os
import shutil
import importlib
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from typing import List, Union, Generator

examples = [
    (['h2o', 'ccsd', 'energy', 'can', 'vac', ''], -76.125789),
    (['h2o', 'ccsd', 'energy', 'can', 'occ', ''], -76.118485),
    (['h2o', 'ccsd', 'energy', 'can', 'mixed', ''], -76.118197),
    (['h2o', 'ccsd', 'dipole', 'can', 'occ', ''], np.array([0.000000, 0.000000, -0.992200])),
    (['h2o', 'ccsd', 'dipole', 'can', 'mixed', ''], np.array([0.000000, 0.000000, -0.994236])),
    (['h2o', 'ccsd(t)', 'energy', 'can', 'vac', ''], -76.126560),
    (['h2o', 'ccsd(t)', 'energy', 'can', 'occ', ''], -76.119334),
    (['h2o', 'ccsd(t)', 'energy', 'can', 'mixed', ''], -76.119171),
    (['h2o', 'ccsd(t)', 'dipole', 'can', 'occ', ''], np.array([0.000000, 0.000000, -0.991222])),
    (['h2o', 'ccsd(t)', 'dipole', 'can', 'mixed', ''], np.array([0.000000, 0.000000, -0.993518])),
    (['h2o', 'ccsdt', 'energy', 'can', 'vac', ''], -76.126706),
    (['h2o', 'ccsdt', 'energy', 'can', 'occ', ''], -76.119441),
    (['h2o', 'ccsdt', 'energy', 'can', 'mixed', ''], -76.119249),
    (['h2o', 'ccsdtq', 'energy', 'can', 'occ', ''], -76.119941),
    (['h2o', 'ccsdtq', 'energy', 'can', 'mixed', ''], -76.119676),
    (['h2o', 'fci', 'energy', 'can', 'vac', ''], -76.126916),
    (['h2o', 'fci', 'energy', 'can', 'occ', ''], -76.119953),
    (['h2o', 'fci', 'energy', 'can', 'mixed', ''], -76.119686),
    (['h2o', 'fci', 'energy', 'can', 'vac', 'ccsd'], -76.119101),
    (['h2o', 'fci', 'energy', 'can', 'occ', 'ccsd'], -76.119675),
    (['h2o', 'fci', 'energy', 'can', 'mixed', 'ccsd'], -76.119664),
    (['h2o', 'fci', 'energy', 'can', 'vac', 'ccsd(t)'], -76.119410),
    (['h2o', 'fci', 'energy', 'can', 'occ', 'ccsd(t)'], -76.119705),
    (['h2o', 'fci', 'energy', 'can', 'mixed', 'ccsd(t)'], -76.119665),
    (['h2o', 'fci', 'energy', 'can', 'vac', 'ccsdt'], -76.119433),
    (['h2o', 'fci', 'energy', 'can', 'occ', 'ccsdt'], -76.119692),
    (['h2o', 'fci', 'energy', 'can', 'mixed', 'ccsdt'], -76.119675),
    (['h2o', 'fci', 'energy', 'can', 'occ', 'ccsdtq'], -76.119675),
    (['h2o', 'fci', 'energy', 'can', 'mixed', 'ccsdtq'], -76.119676),
    (['h2o', 'fci', 'energy', 'ccsd', 'vac', ''], -76.122613),
    (['h2o', 'fci', 'energy', 'ccsd', 'occ', ''], -76.119751),
    (['h2o', 'fci', 'energy', 'ccsd', 'mixed', ''], -76.119674),
    (['h2o', 'fci', 'energy', 'ccsd(t)', 'vac', ''], -76.122913),
    (['h2o', 'fci', 'energy', 'ccsd(t)', 'occ', ''], -76.119757),
    (['h2o', 'fci', 'energy', 'ccsd(t)', 'mixed', ''], -76.119674),
    (['h2o', 'fci', 'energy', 'casscf', 'mixed', ''], -76.119677),
    (['h2o', 'fci', 'dipole', 'can', 'vac', ''], np.array([0.000000, 0.000000, -0.988700])),
    (['h2o', 'fci', 'dipole', 'can', 'occ', ''], np.array([0.000000, 0.000000, -0.989914])),
    (['h2o', 'fci', 'dipole', 'can', 'mixed', ''], np.array([0.000000, 0.000000, -0.992742])),
    (['h2o', 'fci', 'dipole', 'can', 'vac', 'ccsd'], np.array([0.000000, 0.000000, -0.998640])),
    (['h2o', 'fci', 'dipole', 'can', 'occ', 'ccsd'], np.array([0.000000, 0.000000, -0.990961])),
    (['h2o', 'fci', 'dipole', 'can', 'mixed', 'ccsd'], np.array([0.000000, 0.000000, -0.992605])),
    (['h2o', 'fci', 'dipole', 'can', 'vac', 'ccsd(t)'], np.array([0.000000, 0.000000, -0.993047])),
    (['h2o', 'fci', 'dipole', 'can', 'occ', 'ccsd(t)'], np.array([0.000000, 0.000000, -0.991431])),
    (['h2o', 'fci', 'dipole', 'can', 'mixed', 'ccsd(t)'], np.array([0.000000, 0.000000, -0.992624])),
    (['h2o', 'fci', 'dipole', 'ccsd', 'vac', ''], np.array([0.000000, 0.000000, -1.002677])),
    (['h2o', 'fci', 'dipole', 'ccsd', 'occ', ''], np.array([0.000000, 0.000000, -0.991435])),
    (['h2o', 'fci', 'dipole', 'ccsd', 'mixed', ''], np.array([0.000000, 0.000000, -0.992545])),
    (['h2o', 'fci', 'dipole', 'ccsd(t)', 'vac', ''], np.array([0.000000, 0.000000, -1.002059])),
    (['h2o', 'fci', 'dipole', 'ccsd(t)', 'occ', ''], np.array([0.000000, 0.000000, -0.991315])),
    (['h2o', 'fci', 'dipole', 'ccsd(t)', 'mixed', ''], np.array([0.000000, 0.000000, -0.992541])),
    (['h2o', 'fci', 'excitation', 'can', 'mixed', ''], 4.05755e-01),
    (['h2o', 'fci', 'trans', 'can', 'mixed', ''], np.array([0.000000, 0.000000, 0.654025])),
    (['c2', 'fci', 'energy', 'can', 'mixed', 'pi_prune'], -75.626379),
    (['c2', 'fci', 'energy', 'sa_casscf', 'mixed', ''], -75.625784),
    (['1d_hubbard', 'fci', 'energy', 'can', 'vac', ''], -8.607500),
    (['1d_hubbard', 'fci', 'energy', 'local', 'vac', ''], -8.608025)
]


@pytest.mark.parametrize(argnames='example, ref', \
                         argvalues=examples, \
                         ids=['-'.join([string for string in example[0] if string]) for example in examples])
def test_system(example: List[str], ref: Union[float, np.ndarray]) -> None:
        """
        this test system tests pymbe by executing all example scripts
        """
        string = 'pymbe.examples.' + \
                 '.'.join([string for string in example if string]) + '.' + \
                 '_'.join([string for string in example if string])

        example_module = importlib.import_module(string)

        mbe = example_module.mbe_example()

        if example[2] == 'energy':

            tot_prop = (mbe.hf_prop + mbe.base_prop + mbe.ref_prop + mbe.exp.prop['energy']['tot'][-1]).item()

        elif example[2] == 'dipole':

            tot_prop = mbe.nuc_dipole - (mbe.hf_prop + mbe.base_prop + mbe.ref_prop + mbe.exp.prop['dipole']['tot'][-1])

        elif example[2] == 'excitation':

            tot_prop = (mbe.ref_prop + mbe.exp.prop['excitation']['tot'][-1]).item()

        elif example[2] == 'trans':

            tot_prop = mbe.ref_prop + mbe.exp.prop['trans']['tot'][-1]

        assert tot_prop == pytest.approx(ref)
