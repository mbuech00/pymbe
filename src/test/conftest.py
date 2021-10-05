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
from mpi4py import MPI
from pyscf import gto, scf, symm, ao2mo
from typing import List, Tuple
from warnings import catch_warnings, simplefilter

from system import MolCls


@pytest.fixture
def mol(request: SubRequest) -> MolCls:
        """
        this fixture constructs the mol object
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
            hf.conv_tol = 1.e-10
            with catch_warnings():
                simplefilter("ignore")
                hf.kernel()
        
        elif mol.system == 'hubbard':

            hf = None

        return hf


@pytest.fixture
def ints(mol: MolCls, hf: scf.RHF) -> Tuple[np.ndarray, np.ndarray]:
        """
        this fixture constructs hcore and eri integrals
        """
        if mol.system == 'h2o':

            hcore_ao = hf.get_hcore()
            hcore = np.einsum('pi,pq,qj->ij', hf.mo_coeff, hcore_ao, hf.mo_coeff)
            eri_ao = mol.intor('int2e_sph', aosym=4)
            eri = ao2mo.incore.full(eri_ao, hf.mo_coeff)

        elif mol.system == 'hubbard':

            hcore = np.zeros([6] * 2, dtype=np.float64)
            for i in range(5):
                hcore[i, i+1] = hcore[i+1, i] = -1.
            hcore[-1, 0] = hcore[0, -1] = -1.

            eri = np.zeros([6] * 4, dtype=np.float64)
            for i in range(6):
                eri[i,i,i,i] = 2.
            eri = ao2mo.restore(4, eri, 6)

        return hcore, eri


@pytest.fixture
def vhf(mol: MolCls, ints: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        this fixture constructs vhf
        """
        eri = ints[1]
        eri = ao2mo.restore(1, eri, mol.norb)
        vhf = np.empty((mol.nocc, mol.norb, mol.norb), dtype=np.float64)
        for i in range(mol.nocc):
            idx = np.asarray([i])
            vhf[i] = np.einsum('pqrs->rs', eri[idx[:, None], idx, :, :]) * 2.
            vhf[i] -= np.einsum('pqrs->ps', eri[:, idx[:, None], idx, :]) * 2. * .5

        return vhf


@pytest.fixture
def ints_win(mol: MolCls, hf: scf.RHF, ints: Tuple[np.ndarray, np.ndarray], vhf: np.ndarray) -> Tuple[MPI.Win, MPI.Win, MPI.Win]:
        """
        this fixture constructs MPI windows for hcore, eri and vhf integrals
        """
        hcore_win = MPI.Win.Allocate_shared(8 * mol.norb**2, 8, comm=MPI.COMM_WORLD)
        buf = hcore_win.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        eri_win = MPI.Win.Allocate_shared(8 * (mol.norb * (mol.norb + 1) // 2) ** 2, 8, comm=MPI.COMM_WORLD)
        buf = eri_win.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

        hcore[:], eri[:] = ints


        vhf_win = MPI.Win.Allocate_shared(8 * mol.nocc*mol.norb**2, 8, comm=MPI.COMM_WORLD)
        buf = vhf_win.Shared_query(0)[0]
        vhf_arr = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        vhf_arr[:] = vhf

        return hcore_win, eri_win, vhf_win


@pytest.fixture
def orbsym(mol: MolCls, hf: scf.RHF) -> List[int]:
        """
        this fixture determines orbital symmetries
        """
        if mol.system in ['h2o', 'c2']:

            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        
        elif mol.system == 'hubbard':

            orbsym = np.zeros(6, dtype=np.int64)

        return orbsym


@pytest.fixture
def dipole_quantities(mol: MolCls, hf: scf.RHF) -> Tuple[np.ndarray, np.ndarray]:
        """
        this fixture determines the hf dipole moment and dipole integrals
        """
        if mol.system == 'h2o':

            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
            dipole_ints = np.einsum('pi,xpq,qj->xij', hf.mo_coeff, ao_dip, hf.mo_coeff)
            dipole_hf = np.einsum('xij,ji->x', ao_dip, hf.make_rdm1())

        elif mol.system == 'hubbard':

            dipole_ints = None
            dipole_hf = None

        return dipole_ints, dipole_hf
