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
import importlib
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from typing import List, Union

examples = [
    (['h2o', 'ccsd', 'energy', 'can', 'vac', ''], -76.125789),
    (['h2o', 'ccsd', 'energy', 'can', 'occ', ''], -76.118485),
    (['h2o', 'ccsd', 'energy', 'can', 'mixed', ''], -76.118197),
    (['h2o', 'ccsd', 'dipole', 'can', 'occ', ''], np.array([0., 0., -0.99220032])),
    (['h2o', 'ccsd', 'dipole', 'can', 'mixed', ''], np.array([0., 0., -0.99423554])),
    (['h2o', 'ccsd(t)', 'energy', 'can', 'vac', ''], -76.126560),
    (['h2o', 'ccsd(t)', 'energy', 'can', 'occ', ''], -76.119334),
    (['h2o', 'ccsd(t)', 'energy', 'can', 'mixed', ''], -76.119171),
    (['h2o', 'ccsd(t)', 'dipole', 'can', 'occ', ''], np.array([0., 0., -0.99122177])),
    (['h2o', 'ccsd(t)', 'dipole', 'can', 'mixed', ''], np.array([0., 0., -0.99351814])),
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
    (['h2o', 'fci', 'energy', 'ccsd', 'vac', ''], -76.122925),
    (['h2o', 'fci', 'energy', 'ccsd', 'occ', ''], -76.119751),
    (['h2o', 'fci', 'energy', 'ccsd', 'mixed', ''], -76.119673),
    (['h2o', 'fci', 'energy', 'ccsd(t)', 'vac', ''], -76.123228),
    (['h2o', 'fci', 'energy', 'ccsd(t)', 'occ', ''], -76.119757),
    (['h2o', 'fci', 'energy', 'ccsd(t)', 'mixed', ''], -76.119674),
    (['h2o', 'fci', 'energy', 'casscf', 'mixed', ''], -76.119677),
    (['h2o', 'fci', 'dipole', 'can', 'vac', ''], np.array([0., 0., -0.98870004])),
    (['h2o', 'fci', 'dipole', 'can', 'occ', ''], np.array([0., 0., -0.98991406])),
    (['h2o', 'fci', 'dipole', 'can', 'mixed', ''], np.array([0., 0., -0.99274173])),
    (['h2o', 'fci', 'dipole', 'can', 'vac', 'ccsd'], np.array([0., 0., -0.99864001])),
    (['h2o', 'fci', 'dipole', 'can', 'occ', 'ccsd'], np.array([0., 0., -0.99096101])),
    (['h2o', 'fci', 'dipole', 'can', 'mixed', 'ccsd'], np.array([0., 0., -0.99260507])),
    (['h2o', 'fci', 'dipole', 'can', 'vac', 'ccsd(t)'], np.array([0., 0., -0.99304722])),
    (['h2o', 'fci', 'dipole', 'can', 'occ', 'ccsd(t)'], np.array([0., 0., -0.99143106])),
    (['h2o', 'fci', 'dipole', 'can', 'mixed', 'ccsd(t)'], np.array([0., 0., -0.99262394])),
    (['h2o', 'fci', 'dipole', 'ccsd', 'vac', ''], np.array([0., 0., -1.0006686])),
    (['h2o', 'fci', 'dipole', 'ccsd', 'occ', ''], np.array([0., 0., -0.99143464])),
    (['h2o', 'fci', 'dipole', 'ccsd', 'mixed', ''], np.array([0., 0., -0.99254520])),
    (['h2o', 'fci', 'dipole', 'ccsd(t)', 'vac', ''], np.array([0., 0., -1.0002103])),
    (['h2o', 'fci', 'dipole', 'ccsd(t)', 'occ', ''], np.array([0., 0., -0.99131529])),
    (['h2o', 'fci', 'dipole', 'ccsd(t)', 'mixed', ''], np.array([0., 0., -0.99254059])),
    (['h2o', 'fci', 'excitation', 'can', 'mixed', ''], 4.05755e-01),
    (['h2o', 'fci', 'trans', 'can', 'mixed', ''], np.array([0., 0., 0.65402456])),
    (['ch2', 'ccsd', 'energy', 'can', 'vac', ''], -38.980256),
    (['ch2', 'ccsd', 'energy', 'can', 'occ', ''], -38.979269),
    (['ch2', 'ccsd', 'energy', 'can', 'mixed', ''], -38.979216),
    (['ch2', 'ccsd(t)', 'energy', 'can', 'vac', ''], -38.980723),
    (['ch2', 'ccsd(t)', 'energy', 'can', 'occ', ''], -38.979818),
    (['ch2', 'ccsd(t)', 'energy', 'can', 'mixed', ''], -38.979822),
    (['ch2', 'fci', 'energy', 'can', 'vac', ''], -38.980902),
    (['ch2', 'fci', 'energy', 'can', 'occ', ''], -38.980082),
    (['ch2', 'fci', 'energy', 'can', 'mixed', ''], -38.980075),
    (['ch2', 'fci', 'energy', 'can', 'vac', 'ccsd'], -38.979801),
    (['ch2', 'fci', 'energy', 'can', 'occ', 'ccsd'], -38.980029),
    (['ch2', 'fci', 'energy', 'can', 'mixed', 'ccsd'], -38.980069),
    (['ch2', 'fci', 'energy', 'can', 'vac', 'ccsd(t)'], -38.979971),
    (['ch2', 'fci', 'energy', 'can', 'occ', 'ccsd(t)'], -38.980068),
    (['ch2', 'fci', 'energy', 'can', 'mixed', 'ccsd(t)'], -38.980066),
    (['ch2', 'fci', 'energy', 'ccsd', 'vac', ''], -38.981114),
    (['ch2', 'fci', 'energy', 'ccsd', 'occ', ''], -38.980064),
    (['ch2', 'fci', 'energy', 'ccsd', 'mixed', ''], -38.980060),
    (['ch2', 'fci', 'energy', 'ccsd(t)', 'vac', ''], -38.981102),
    (['ch2', 'fci', 'energy', 'ccsd(t)', 'occ', ''], -38.980065),
    (['ch2', 'fci', 'energy', 'ccsd(t)', 'mixed', ''], -38.980060),
    (['ch2', 'fci', 'energy', 'casscf', 'occ', ''], -38.980082),
    (['ch2', 'fci', 'energy', 'casscf', 'mixed', ''], -38.980059),
    (['ch2', 'fci', 'dipole', 'can', 'vac', ''], np.array([0., 0.22544755, 0.])),
    (['ch2', 'fci', 'dipole', 'can', 'occ', ''], np.array([0., 0.22875506, 0.])),
    (['ch2', 'fci', 'dipole', 'can', 'mixed', ''], np.array([0., 0.22910716, 0.])),
    (['ch2', 'fci', 'dipole', 'ccsd', 'vac', ''], np.array([0., 0.22972397, 0.])),
    (['ch2', 'fci', 'dipole', 'ccsd', 'occ', ''], np.array([0., 0.22905094, 0.])),
    (['ch2', 'fci', 'dipole', 'ccsd', 'mixed', ''], np.array([0., 0.22912233, 0.])),
    (['ch2', 'fci', 'dipole', 'ccsd(t)', 'vac', ''], np.array([0., 0.22977128, 0.])),
    (['ch2', 'fci', 'dipole', 'ccsd(t)', 'occ', ''], np.array([0., 0.22904178, 0.])),
    (['ch2', 'fci', 'dipole', 'ccsd(t)', 'mixed', ''], np.array([0., 0.22912217, 0.])),
    (['ch2', 'fci', 'excitation', 'can', 'mixed', ''], 3.42034e-01),
    (['ch2', 'fci', 'trans', 'can', 'mixed', ''], np.array([0., 0.24206675, 0.])),
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

        prop = example_module.mbe_example(rst=False)

        assert prop == pytest.approx(ref)
