#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
system testing module
"""

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import pytest

from system import _set_ncore, MolCls


@pytest.mark.parametrize(argnames='mol', argvalues=['hcl'], indirect=['mol'])
def test_set_ncore(mol: MolCls):
        """
        this function tests _set_ncore
        """
        assert _set_ncore(mol.natm, mol.atom_charge) == 5
