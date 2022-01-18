#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
test configuration module
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
import numpy as np
from mpi4py import MPI
from pyscf import gto, scf, symm, ao2mo
from typing import TYPE_CHECKING
from warnings import catch_warnings, simplefilter

if TYPE_CHECKING:

    from _pytest.fixtures import SubRequest
    from _pytest._code.code import ExceptionInfo
    from typing import List, Tuple


def pytest_keyboard_interrupt(excinfo: ExceptionInfo) -> None:
        """
        this function replaces the pytest keyboard interrupt teardown function
        and ensures any remaining files are removed
        """
        shutil.rmtree(os.getcwd()+'/rst', ignore_errors=True)


@pytest.fixture
def system(request: SubRequest) -> str:
        """
        this fixture stores the system string for other fixtures to access
        """
        return request.param


@pytest.fixture
def mol(system: str) -> gto.Mole:
        """
        this fixture constructs the mol object
        """
        if system == 'h2o':

            mol = gto.Mole()
            mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943', \
                      basis='631g', symmetry='C2v', verbose=0)

        elif system == 'c2':

            mol = gto.Mole()
            mol.build(atom='C 0. 0. 0.625; C 0. 0. -0.625', basis='631g', \
                      symmetry='D2h', verbose=0)

        elif system == 'hcl':

            mol = gto.Mole()
            mol.build(atom='H 0 0 0; Cl 0 0 1.', basis='631g', \
                      symmetry='C2v', verbose=0)

        elif system == 'hubbard':

            mol = gto.M()

        return mol


@pytest.fixture
def ncore(system: str) -> int:
        """
        this fixture sets the number of core orbitals
        """
        if system == 'h2o':

            ncore = 1

        elif system == 'c2':

            ncore = 2

        return ncore


@pytest.fixture
def nocc(mol: gto.Mole) -> int:
        """
        this fixture extracts the number of occupied orbitals from the mol 
        object
        """
        return mol.nelectron // 2


@pytest.fixture
def norb(mol: gto.Mole) -> int:
        """
        this fixture extracts the number of orbitals from the mol object
        """
        return mol.nao_nr().item()


@pytest.fixture
def hf(system: str, mol: gto.Mole) -> scf.RHF:
        """
        this fixture constructs the hf object and executes a hf calculation
        """
        if system in ['h2o', 'c2']:

            hf = scf.RHF(mol)
            hf.conv_tol = 1.e-10
            with catch_warnings():
                simplefilter("ignore")
                hf.kernel()
        
        elif system == 'hubbard':

            hf = None

        return hf


@pytest.fixture
def mo_coeff(request: SubRequest, hf: scf.RHF, norb: int) -> np.ndarray:
        """
        this fixture constructs mo coefficients
        """
        if request.param == 'h2o':

            mo_coeff = hf.mo_coeff

        elif request.param == 'rnd':

            np.random.seed(1234)
            mo_coeff = np.random.rand(norb, norb)

        return mo_coeff


@pytest.fixture
def ints(system: str, mol: gto.Mole, \
         hf: scf.RHF) -> Tuple[np.ndarray, np.ndarray]:
        """
        this fixture constructs hcore and eri integrals
        """
        if system == 'h2o':

            hcore_ao = hf.get_hcore()
            hcore = np.einsum('pi,pq,qj->ij', hf.mo_coeff, hcore_ao, hf.mo_coeff)
            eri_ao = mol.intor('int2e_sph', aosym=4)
            eri = ao2mo.incore.full(eri_ao, hf.mo_coeff)

        elif system == 'hubbard':

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
def vhf(ints: Tuple[np.ndarray, np.ndarray], norb: int, \
        nocc: int) -> np.ndarray:
        """
        this fixture constructs vhf
        """
        eri = ints[1]
        eri = ao2mo.restore(1, eri, norb)
        vhf = np.empty((nocc, norb, norb), dtype=np.float64)
        for i in range(nocc):
            idx = np.asarray([i])
            vhf[i] = np.einsum('pqrs->rs', eri[idx[:, None], idx, :, :]) * 2.
            vhf[i] -= np.einsum('pqrs->ps', eri[:, idx[:, None], idx, :]) * 2. * .5

        return vhf


@pytest.fixture
def ints_win(norb: int, nocc: int, ints: Tuple[np.ndarray, np.ndarray], \
             vhf: np.ndarray) -> Tuple[MPI.Win, MPI.Win, MPI.Win]:
        """
        this fixture constructs MPI windows for hcore, eri and vhf integrals
        """
        hcore_win = MPI.Win.Allocate_shared(8 * norb**2, 8, comm=MPI.COMM_WORLD)
        buf = hcore_win.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(norb,) * 2)

        eri_win = MPI.Win.Allocate_shared(8 * (norb * (norb + 1) // 2) ** 2, 8, comm=MPI.COMM_WORLD)
        buf = eri_win.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(norb * (norb + 1) // 2,) * 2)

        hcore[:], eri[:] = ints

        vhf_win = MPI.Win.Allocate_shared(8 * nocc*norb**2, 8, comm=MPI.COMM_WORLD)
        buf = vhf_win.Shared_query(0)[0]
        vhf_arr = np.ndarray(buffer=buf, dtype=np.float64, shape=(nocc, norb, norb))

        vhf_arr[:] = vhf

        return hcore_win, eri_win, vhf_win


@pytest.fixture
def orbsym(system: str, mol: gto.Mole, hf: scf.RHF) -> List[int]:
        """
        this fixture determines orbital symmetries
        """
        if system in ['h2o', 'c2']:

            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        
        elif system == 'hubbard':

            orbsym = np.zeros(6, dtype=np.int64)

        return orbsym


@pytest.fixture
def dipole_quantities(system: str, mol: gto.Mole, \
                      hf: scf.RHF) -> Tuple[np.ndarray, np.ndarray]:
        """
        this fixture determines the hf dipole moment and dipole integrals
        """
        if system == 'h2o':

            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
            dipole_ints = np.einsum('pi,xpq,qj->xij', hf.mo_coeff, ao_dip, hf.mo_coeff)
            dipole_hf = np.einsum('xij,ji->x', ao_dip, hf.make_rdm1())

        elif system == 'hubbard':

            dipole_ints = None
            dipole_hf = None

        return dipole_ints, dipole_hf
        

@pytest.fixture
def exp() -> ExpCls:
        """
        this fixture constructs a dummy ExpCls object
        """
        exp = ExpCls()
        return exp


class ExpCls:
        """
        this class is a dummy ExpCls class
        """
