#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
test configuration module
"""

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import pytest
from _pytest.fixtures import SubRequest
import numpy as np
from pyscf import gto, scf
from warnings import catch_warnings, simplefilter

from system import MolCls


@pytest.fixture
def mol(request: SubRequest) -> MolCls:
        """
        this fixture constructs the mol object for testing
        """
        if request.param == 'h2o':

            mol = gto.Mole()
            mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943', \
                      basis='631g', symmetry='C2v', verbose=0)

            mol.norb = mol.nao_nr()
            mol.nocc = mol.nelectron // 2
            mol.nvirt = mol.norb - mol.nocc
            mol.ncore = 1
            mol.e_nuc = mol.energy_nuc()
            mol.x2c = False
            mol.gauge_origin = np.zeros(3, dtype=np.int64)
            mol.debug = 0

        elif request.param == 'c2':

            mol = gto.Mole()
            mol.build(atom='C 0. 0. 0.625; C 0. 0. -0.625', basis='631g', \
                      symmetry='D2h', verbose=0)

            mol.norb = mol.nao_nr()
            mol.nocc = mol.nelectron // 2
            mol.nvirt = mol.norb - mol.nocc
            mol.ncore = 2
            mol.e_nuc = mol.energy_nuc()
            mol.x2c = False
            mol.gauge_origin = np.zeros(3, dtype=np.int64)
            mol.debug = 0

        elif request.param == 'hcl':

            mol = gto.Mole()
            mol.build(atom='H 0 0 0; Cl 0 0 1.', basis='631g', \
                      symmetry='C2v', verbose=0)
            mol.norb = mol.nao_nr()
            mol.nocc = mol.nelectron // 2
            mol.nvirt = mol.norb - mol.nocc
            mol.ncore = 5
            mol.e_nuc = mol.energy_nuc()
            mol.x2c = False
            mol.gauge_origin = np.zeros(3, dtype=np.int64)
            mol.debug = 0

        elif request.param == 'hubbard':

            mol = gto.M()

            mol.matrix = (1, 6)
            mol.n = 1.
            mol.u = 2.
            mol.pbc = True

        mol.system = request.param

        return mol


@pytest.fixture
def hf(mol: MolCls) -> scf.RHF:
        """
        this fixture constructs the hf object and executes a hf calculation
        """
        if mol.system in ['h2o', 'c2']:

            hf = scf.RHF(mol)
            with catch_warnings():
                simplefilter("ignore")
                hf.kernel()
        
        elif mol.system == 'hubbard':

            hf = None

        return hf
