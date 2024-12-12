#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tuple generator based on filter module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"


import numpy as np
from typing import List, Tuple, Optional, Generator
from itertools import combinations
from pymbe.tools import valid_tup, _start_idx

def genFilt(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    nocc: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
    thres: float,
    norb: np.ndarray,
    pair_contribs: np.ndarray,
    start_tup: Optional[np.ndarray] = None,
    start_idx: int = 0,
) -> Generator[Tuple[np.ndarray, Optional[List[np.ndarray]]], None, None]:
    """
    this function is the the new generator for tuples based on the numerical overlaps
    """
    print("Debug: genFilt was called")
    # check if only single orbitals in expansion space
    if orb_clusters is None:
        # occupied and virtual expansion spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # start tuples
        start_tup_occ: Optional[np.ndarray]
        start_tup_virt: Optional[np.ndarray]
        if start_tup is not None:
            start_tup_occ = start_tup[start_tup < nocc]
            start_tup_virt = start_tup[nocc <= start_tup]
            if start_tup_occ.size == 0:
                start_tup_occ = None
            if start_tup_virt.size == 0:
                start_tup_virt = None
        else:
            start_tup_occ = start_tup_virt = None
        nocc_start, occ_start, virt_start = _start_idx(
            occ_space, virt_space, start_tup_occ, start_tup_virt
        )

        pair_contribs = np.power(pair_contribs, 1 / (order - 1))
        
        # pair_contribs = np.power(pair_contribs, 1/(order-1)) 
        
        # list of occupied orbitals and sorted pair contributions wrt other occupied orbitals
        sort_pair_occ_contribs: List[Tuple[np.ndarray, np.ndarray]] = []
        for orb in range(nocc):
            sort_idx = np.argsort(pair_contribs[orb + 1 : nocc, orb])[::-1]
            sort_pair_occ_contribs.append(
                (
                    np.array([(orb + idx + 1).item() for idx in sort_idx], dtype=np.int64),
                    np.array(
                        [pair_contribs[orb + 1 : nocc, orb][idx] for idx in sort_idx],
                        dtype=np.float64,
                    ),
                )
            )

        # list of all orbitals and sorted pair contributions wrt (other) virtual orbitals
        sort_pair_virt_contribs: List[Tuple[np.ndarray, np.ndarray]] = []
        for orb in range(norb):
            sort_idx = np.argsort(pair_contribs[max(orb + 1, nocc) :, orb])[::-1]
            sort_pair_virt_contribs.append(
                (
                    np.array([max(orb + 1, nocc) + idx for idx in sort_idx], dtype=np.int64),
                    np.array(
                        [pair_contribs[max(orb + 1, nocc) :, orb][idx] for idx in sort_idx],
                        dtype=np.float64,
                    ),
                )
            )

        #for tup_nocc in range(1,order):
        for tup_nocc in range(nocc_start, order + 1):
            # check if occupation produces valid increment
            if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
                for tup in tuples(sort_pair_occ_contribs, sort_pair_virt_contribs, pair_contribs, order, tup_nocc, nocc, thres):
                    # prod = np.prod(pair_contribs[tup.reshape(-1, 1), tup][np.triu_indices(order, 1)])
                    # tup_list.append((tuple(int(x) for x in tup), prod))
                    yield np.array(tup, dtype=np.int64), None

def tuples(
    sort_pair_occ_contribs: List[Tuple[np.ndarray, np.ndarray]],
    sort_pair_virt_contribs: List[Tuple[np.ndarray, np.ndarray]],
    pair_contribs: np.ndarray,
    order: int,
    tup_nocc: int,
    nocc: int,
    thres: float,
):
    # single-orbital tuples
    if order == 1:
        if tup_nocc == 0:
            for orb_idx in range(nocc, norb):
                yield np.array([orb_idx], dtype=np.int64)
        elif tup_nocc == 1:
            for orb_idx in range(nocc):
                yield np.array([orb_idx], dtype=np.int64)
    # multiple-orbital tuples
    else:
        # only virtual orbitals
        if tup_nocc == 0:
            for orb_idx in range(nocc, norb):
                for tup, _ in backtrack(
                    [orb_idx], 1.0, sort_pair_virt_contribs, pair_contribs, order, thres
                ):
                    yield np.array(tup, dtype=np.int64)
        # only occupied orbitals
        elif tup_nocc == order:
            for orb_idx in range(nocc):
                for tup, _ in backtrack(
                    [orb_idx], 1.0, sort_pair_occ_contribs, pair_contribs, order, thres
                ):
                    yield np.array(tup, dtype=np.int64)
        # occupied and virtual orbitals
        else:
            # single occupied orbital
            if tup_nocc == 1:
                for orb_idx in range(nocc):
                    for tup, _ in backtrack(
                        [orb_idx], 1.0, sort_pair_virt_contribs, pair_contribs, order, thres
                    ):
                        yield np.array(tup, dtype=np.int64)
            # multiple occupied orbitals
            else:
                for orb_idx in range(nocc):
                    for tup_occ, orb_prod in backtrack(
                        [orb_idx], 1.0, sort_pair_occ_contribs, pair_contribs, tup_nocc, thres
                    ):
                        for tup, _ in backtrack(
                            tup_occ,
                            orb_prod,
                            sort_pair_virt_contribs,
                            pair_contribs,
                            order,
                            thres,
                        ):
                            yield np.array(tup, dtype=np.int64)


def backtrack(
    curr_tup: List[int],
    curr_prod: float,
    sorted_pair_contribs: List[Tuple[np.ndarray, np.ndarray]],
    pair_contribs: np.ndarray,
    order: int,
    thres: float, 
):
    # get maximum index below which product is below threshold, no orbital lower in
    # the array can produce a product larger than this
    max_idx = sorted_pair_contribs[curr_tup[-1]][1].size - np.searchsorted(
        sorted_pair_contribs[curr_tup[-1]][1][::-1], thres / curr_prod, side="right"
    )

    # return if no orbital pair produces a product above threshold
    if max_idx == 0:
        return

    # check if tuple construction is finished
    if len(curr_tup) + 1 == order:
        # add all pair contributions
        orb_prods = curr_prod * np.prod(
            pair_contribs[
                sorted_pair_contribs[curr_tup[-1]][0][:max_idx].reshape(-1, 1), curr_tup
            ],
            axis=1,
        )

        # loop over orbitals that produce product above threshold
        for idx in np.where(orb_prods > thres)[0]:
            # add next orbital to tuple and yield
            yield curr_tup + [sorted_pair_contribs[curr_tup[-1]][0][idx]], orb_prods[
                idx
            ]

    else:
        # only get orbitals which can still produce valid tuples above the threshold
        valid_orb_idx = sorted_pair_contribs[curr_tup[-1]][0][:max_idx][
            sorted_pair_contribs[curr_tup[-1]][0][:max_idx]
            < len(sorted_pair_contribs) - order + len(curr_tup) + 1
        ]

        # add all pair contributions
        orb_prods = curr_prod * np.prod(
            pair_contribs[valid_orb_idx.reshape(-1, 1), curr_tup],
            axis=1,
        )

        # loop over orbitals that produce product above threshold
        for idx in np.where(orb_prods > thres)[0]:
            # add next orbital to tuple and go to next recursion
            yield from backtrack(
                curr_tup + [valid_orb_idx[idx]],
                orb_prods[idx],
                sorted_pair_contribs,
                pair_contribs,
                order,
                thres,
            )

    # go to previous recursion function
    return

