#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
mbe testing module
"""

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import pytest
import numpy as np
import scipy.special as sc
from mpi4py import MPI
from pyscf import scf
from typing import List, Tuple, Union

from mbe import main, _inc, _sum
from system import MolCls
from parallel import MPICls
from calculation import CalcCls
from expansion import ExpCls


test_cases_main = [
    ('h2o', 1, -249055688365223385, 9199082625845137542, -0.0374123708341898, -0.0018267714680604286, np.array([-0.0374123708341898]), np.array([10]), np.array([9]), np.array([11]), np.array([-0.004676546354273725]), np.array([0.0018267714680604286]), np.array([0.010886635891736773])),
    ('h2o', 2, 8509729643108359722, 8290417336063232159, -0.11605435599270209, -0.0001698239845069338, np.array([-0.11605435599270209]), np.array([65]), np.array([57]), np.array([69]), np.array([-0.004144798428310789]), np.array([0.0001698239845069338]), np.array([0.009556269221292268]))
]

test_cases_sum = [
    ('energy', np.array([1, 7, 8], dtype=np.int64), 1.2177665733781107),
    ('dipole', np.array([1, 7, 8], dtype=np.int64), np.array([1.21776657, 1.21776657, 1.21776657], dtype=np.float64)),
    ('excitation', np.array([1, 7, 8], dtype=np.int64), 1.2177665733781107),
    ('trans', np.array([1, 7, 8], dtype=np.int64), np.array([1.21776657, 1.21776657, 1.21776657], dtype=np.float64)),
    ('energy', np.array([1, 7, 8, 9], dtype=np.int64), 2.7229882355444195),
    ('dipole', np.array([1, 7, 8, 9], dtype=np.int64), np.array([2.72298824, 2.72298824, 2.72298824], dtype=np.float64)),
    ('excitation', np.array([1, 7, 8, 9], dtype=np.int64), 2.7229882355444195),
    ('trans', np.array([1, 7, 8, 9], dtype=np.int64), np.array([2.72298824, 2.72298824, 2.72298824], dtype=np.float64))
]


@pytest.fixture
def objects(mol: MolCls, hf: scf.RHF, \
            ints_win: Tuple[MPI.Win, MPI.Win, MPI.Win], \
            dipole_quantities: Tuple[np.ndarray, np.ndarray], \
            orbsym: List[int]) -> Tuple[MPICls, CalcCls, ExpCls]:
        """
        this fixture constructs the mpi, calc and exp objects for testing
        """
        mol.hcore, mol.eri, mol.vhf = ints_win

        calc = CalcCls(mol.ncore, mol.nelectron, mol.groupname)
        calc.target_mbe = 'energy'
        calc.misc['rst'] = False
        mol.dipole_ints, calc.prop['hf']['dipole'] = dipole_quantities
        calc.prop['hf']['energy'] = hf.e_tot
        calc.occup = hf.mo_occ
        calc.orbsym = orbsym
        calc.mo_coeff = hf.mo_coeff
        calc.ref_space = np.asarray(calc.ref['select'], dtype=np.int64)
        calc.prop['ref'][calc.target_mbe] = 0.

        exp = ExpCls(mol, calc)
        mpi = MPICls()

        return mpi, calc, exp


@pytest.mark.parametrize(argnames='mol, order, ref_hashes_sum, ref_hashes_amax, ref_inc_sum, ref_inc_amax, ref_tot, ref_mean_ndets, ref_min_ndets, ref_max_ndets, ref_mean_inc, ref_min_inc, ref_max_inc', \
                         argvalues=test_cases_main, \
                         ids=['-'.join([case[0], str(case[1])]) for case in test_cases_main], \
                         indirect=['mol'])
def test_main(mol: MolCls, objects: Tuple[MPICls, CalcCls, ExpCls], \
              order: int, ref_hashes_sum: int, ref_hashes_amax: int, \
              ref_inc_sum: float, ref_inc_amax: float, ref_tot: np.ndarray, \
              ref_mean_ndets: int, ref_min_ndets: int, ref_max_ndets: int, \
              ref_mean_inc: float, ref_min_inc: float, ref_max_inc: float):
        """
        this function tests main
        """
        mpi, calc, exp = objects

        hashes = []
        inc = []

        for exp.order in range(1, order+1):

            n_tuples = 0.

            for k in range(1, exp.order):
                n_tuples += sc.binom(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc].size, k) * sc.binom(exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]].size, exp.order - k)

            n_tuples += sc.binom(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc].size, exp.order)
            n_tuples += sc.binom(exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]].size, exp.order)

            exp.n_tuples['inc'].append(int(n_tuples))

            hashes_win, inc_win, tot, mean_ndets, min_ndets, max_ndets, \
            mean_inc, min_inc, max_inc = main(mpi, mol, calc, exp)

            hashes.append(np.ndarray(buffer=hashes_win, dtype=np.int64, shape=(exp.n_tuples['inc'][exp.order-1],)))

            inc.append(np.ndarray(buffer=inc_win, dtype=np.float64, shape=(exp.n_tuples['inc'][exp.order-1], 1)))

            exp.prop[calc.target_mbe]['hashes'].append(hashes_win)

            exp.prop[calc.target_mbe]['inc'].append(inc_win)

            exp.mean_ndets.append(mean_ndets)
            exp.min_ndets.append(min_ndets)
            exp.max_ndets.append(max_ndets)

            exp.mean_inc.append(mean_inc)
            exp.min_inc.append(min_inc)
            exp.max_inc.append(max_inc)

        assert isinstance(hashes_win, MPI.Win)
        assert isinstance(inc_win, MPI.Win)
        assert np.sum(hashes[-1]) == ref_hashes_sum
        assert np.amax(hashes[-1]) == ref_hashes_amax
        assert np.sum(inc[-1]) == pytest.approx(ref_inc_sum)
        assert np.amax(inc[-1]) == pytest.approx(ref_inc_amax)
        assert tot == pytest.approx(ref_tot)
        assert mean_ndets == ref_mean_ndets
        assert min_ndets == ref_min_ndets
        assert max_ndets == ref_max_ndets
        assert mean_inc == pytest.approx(ref_mean_inc)
        assert min_inc == pytest.approx(ref_min_inc)
        assert max_inc == pytest.approx(ref_max_inc)


@pytest.mark.parametrize(argnames='mol', argvalues=['hubbard'], \
                         indirect=['mol'])
def test_inc(ints: Tuple[np.ndarray, np.ndarray]):
        """
        this function tests _inc
        """
        n = 6
        model = {'method': 'fci', 'cc_backend': 'pyscf', 'solver': 'pyscf_spin0', 'hf_guess': True}
        prop = {'hf': {'energy': 0., 'dipole': None}, 'ref': {'energy': 0.}}
        state = {'wfnsym': 'A', 'root': 0}
        occup = np.array([2.] * (n // 2) + [0.] * (n // 2), dtype=np.float64)
        orbsym = np.zeros(n, dtype=np.int64)
        h1e_cas, h2e_cas = ints
        core_idx, cas_idx = np.array([], dtype=np.int64), np.arange(n, dtype=np.int64)

        e, ndets, n_elec = _inc(model, None, 'can', 0, occup, 'energy', state, \
                                'c1', orbsym, prop, 0, h1e_cas, h2e_cas, \
                                core_idx, cas_idx, 0, None)
        
        assert e == pytest.approx(-5.409456845093448)
        assert ndets == 400
        assert n_elec == (3, 3)
        

@pytest.mark.parametrize(argnames='target_mbe, tup, ref_res', argvalues=test_cases_sum, \
                         ids=[case[0] + '-' + str(case[1]) for case in test_cases_sum])
def test_sum(target_mbe: str, tup: np.ndarray, \
             ref_res: Union[float, np.ndarray]):
        """
        this function tests _sum
        """
        nocc = 3
        min_order = 2
        inc = []
        np.random.seed(1)
        inc.append(np.random.rand(21))
        np.random.seed(2)
        inc.append(np.random.rand(36))
        hashes = []
        hashes.append(np.array([-9202428759661734630, -8712053614062906321, -6318372561352273418,
                                -5704331169117380813, -5475322122992870313, -2395101507181501705,
                                -2361262697551529625, -2211238527921376434, -2140115254313904493,
                                -1769792267912035584, -1752257283205524125,  -669804309911520350,
                                 1455941523185766351,  2212326720080977450,  2248846252070972957,
                                 2796798554289973955,  3935864756934676997,  4352052642437003428,
                                 6981656516950638826,  7504768460337078519,  9123845761190921543], dtype=np.int64))
        hashes.append(np.array([-9198944415400131734, -8972810312738912477, -8808983030330342215,
                                -7973821167853631741, -7695363886579078617, -7618341867291768848,
                                -7370655119274612396, -7109478362892924185, -6971905576144995277,
                                -6346674104600383423, -6154667499349861323, -6103692259034244091,
                                -5798288073144894648, -4310406760124882618, -4205406112023021717,
                                -3498426837436738930, -3479492968911590647, -3352798558434503475,
                                -2873481280266186796, -2629798042131054482, -1829605908125193058,
                                -1795423808173281765, -1405348565896712846,  -898637747284230767,
                                 -775474974383967908,  2332618250368229543,  2744080864795541539,
                                 3231441095074161519,  3949415985151233945,  5616657939615333112,
                                 5991752789726291576,  6280027850766028273,  6686650809470102352,
                                 8474590989972277172,  8862583647964226937,  8967033778183145807], dtype=np.int64))
        ref_occ = False
        ref_virt = False

        res = _sum(nocc, target_mbe, min_order, tup.size, inc, hashes, \
                   ref_occ, ref_virt, tup)

        assert res == pytest.approx(ref_res)
