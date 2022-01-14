#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
purge testing module
"""

from __future__ import annotations

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import pytest
import numpy as np
from mpi4py import MPI
from typing import TYPE_CHECKING

from pymbe.purge import main
from pymbe.parallel import MPICls


if TYPE_CHECKING:

    from pymbe.tests.function_tests.conftest import ExpCls


def test_main(exp: ExpCls) -> None:
        """
        this function tests main
        """
        mpi = MPICls()

        exp.target = 'energy'

        exp.nocc = 3

        exp.occup = np.array([2., 2., 2., 0., 0., 0.], dtype=np.float64)

        exp.ref_space = np.array([], dtype=np.int64)

        exp.exp_space = [np.array([0, 1, 2, 3, 5], dtype=np.int64)]

        exp.prop = {exp.target: {'inc': [], 'hashes': []}}

        exp.time = {'purge': []}

        exp.screen_orbs = np.array([4], dtype=np.int64)

        exp.order = 4
        exp.min_order = 2
        
        exp.n_tuples = {'inc': [9, 18, 15]}

        start_hashes = [np.array([-6318372561352273418, -5475322122992870313, -2211238527921376434, 
                                  -1752257283205524125,  -669804309911520350,  1455941523185766351, 
                                   2796798554289973955,  6981656516950638826,  7504768460337078519]),
                        np.array([-8862568739552411231, -7925134385272954056, -7370655119274612396, 
                                  -7216722148388372205, -6906205837173860435, -6346674104600383423,
                                  -6103692259034244091, -4310406760124882618, -4205406112023021717,
                                  -3352798558434503475,   366931854209709639,   680656656239891583,
                                   3949415985151233945,  4429162622039029653,  6280027850766028273,
                                   7868645139422709341,  8046408145842912366,  8474590989972277172]),
                        np.array([-9191542714830049336, -9111224886591032877, -8945201412191574338,
                                  -6640293625692100246, -4012521487842354405, -3041224019630807622,
                                  -2930228190932741801,  -864833587293421682,   775579459894020071,
                                   1344711228121337165,  2515975357592924865,  2993709457496479298,
                                   4799605789003109011,  6975445416347248252,  7524854823186007981])]

        hashes = []
        inc = []

        for k in range(0, 3):

            hashes_win = MPI.Win.Allocate_shared(8 * exp.n_tuples['inc'][k], 8, comm=mpi.local_comm)
            buf = hashes_win.Shared_query(0)[0]
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['inc'][k],)))
            hashes[-1][:] = start_hashes[k]
            exp.prop[exp.target]['hashes'].append(hashes_win)

            inc_win = MPI.Win.Allocate_shared(8 * exp.n_tuples['inc'][k], 8, comm=mpi.local_comm)
            buf = inc_win.Shared_query(0)[0]
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples['inc'][k],)))
            inc[-1][:] = np.arange(1, exp.n_tuples['inc'][k]+1, dtype=np.float64)
            exp.prop[exp.target]['inc'].append(inc_win)

        exp.prop[exp.target], exp.n_tuples = main(mpi, exp)

        purged_hashes = []
        purged_inc = []

        for k in range(0, 3):

            purged_hashes.append(np.ndarray(buffer=exp.prop[exp.target]['hashes'][k], dtype=np.int64, shape=(exp.n_tuples['inc'][k],)))

            purged_inc.append(np.ndarray(buffer=exp.prop[exp.target]['inc'][k], dtype=np.float64, shape=(exp.n_tuples['inc'][k],)))

        assert exp.n_tuples['inc'] == [6, 9, 5]
        assert (purged_hashes[0] == np.array([-6318372561352273418, -5475322122992870313, -1752257283205524125,
                                               -669804309911520350,  1455941523185766351,  6981656516950638826], dtype=np.int64)).all()
        assert (purged_hashes[1] == np.array([-8862568739552411231, -7925134385272954056, -7216722148388372205,
                                              -6906205837173860435, -4310406760124882618, -4205406112023021717,
                                              -3352798558434503475,   366931854209709639,  6280027850766028273], dtype=np.int64)).all()
        assert (purged_hashes[2] == np.array([-9111224886591032877, -6640293625692100246, -4012521487842354405,
                                              -2930228190932741801,  2993709457496479298], dtype=np.int64)).all()
        assert (purged_inc[0] == np.array([1., 2., 4., 5., 6., 8.], dtype=np.float64)).all()
        assert (purged_inc[1] == np.array([ 1., 2., 4., 5., 8., 9., 10., 11., 15.], dtype=np.float64)).all()
        assert (purged_inc[2] == np.array([ 2., 4., 5., 7., 12.], dtype=np.float64)).all()
