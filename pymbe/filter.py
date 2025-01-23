#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
filter module
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
from pymbe.tools import valid_tup, start_indices


def tuples_filtered(
    orb_space: np.ndarray,
    orb_clusters: Optional[List[np.ndarray]],
    nocc: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
    thres: float,
    pair_contribs: np.ndarray,
    start_tup: Optional[np.ndarray] = None,
) -> Generator[Tuple[np.ndarray, Optional[List[np.ndarray]]], None, None]:
    """
    this function is the generator for tuples with pair importance above a given
    threshold
    """
    # check if only single orbitals in expansion space
    if orb_clusters is None:
        # occupied and virtual expansion spaces
        occ_space = orb_space[orb_space < nocc]
        virt_space = orb_space[nocc <= orb_space]

        # start tuples
        nocc_start, occ_start, virt_start = start_indices(
            occ_space, virt_space, nocc, start_tup
        )

        # get results for the pair contributions
        (sort_pair_occ_contribs, sort_pair_virt_contribs, exp_exp_matrix, exp_nocc) = (
            norm_pair_contribs(orb_space, nocc, order, pair_contribs)
        )

        # loop over number of occupied orbitals
        for tup_nocc in range(nocc_start, order + 1):
            # check if occupation produces valid increment
            if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
                for tup in tuples_filtered_with_nocc(
                    sort_pair_occ_contribs,
                    sort_pair_virt_contribs,
                    exp_exp_matrix,
                    order,
                    tup_nocc,
                    exp_nocc,
                    thres,
                    orb_space,
                    occ_start,
                    virt_start,
                ):
                    yield tup, None


def norm_pair_contribs(
    orb_space: np.ndarray,
    nocc: int,
    order: int,
    pair_contribs: np.ndarray,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    int,
]:
    """
    this function normalizesand sorts orbital pair contributions
    """
    # normalization of pair contributions
    if order != 1:
        pair_contribs = np.power(pair_contribs, 1 / (order - 1))

    # pair contribution submatrix for expansion space
    exp_pair_contribs = pair_contribs[orb_space.reshape(-1, 1), orb_space]

    # number of occupied orbitals in expansion space
    exp_nocc = np.count_nonzero(orb_space < nocc)

    # list of occupied orbitals and sorted pair contributions wrt other occupied
    # orbitals
    sort_pair_occ_contribs: List[Tuple[np.ndarray, np.ndarray]] = []
    for orb in range(len(orb_space)):
        sort_idx = np.argsort(exp_pair_contribs[orb + 1 : exp_nocc, orb])[::-1]
        sort_pair_occ_contribs.append(
            (
                np.array([(orb + idx + 1).item() for idx in sort_idx], dtype=np.int64),
                np.array(
                    [
                        exp_pair_contribs[orb + 1 : exp_nocc, orb][idx]
                        for idx in sort_idx
                    ],
                    dtype=np.float64,
                ),
            )
        )

    # list of all orbitals and sorted pair contributions wrt other virtual orbitals
    sort_pair_virt_contribs: List[Tuple[np.ndarray, np.ndarray]] = []
    for orb in range(len(orb_space)):
        sort_idx = np.argsort(exp_pair_contribs[max(orb + 1, exp_nocc) :, orb])[::-1]
        sort_pair_virt_contribs.append(
            (
                np.array(
                    [max(orb + 1, exp_nocc) + idx for idx in sort_idx], dtype=np.int64
                ),
                np.array(
                    [
                        exp_pair_contribs[max(orb + 1, exp_nocc) :, orb][idx]
                        for idx in sort_idx
                    ],
                    dtype=np.float64,
                ),
            )
        )

    return sort_pair_occ_contribs, sort_pair_virt_contribs, exp_pair_contribs, exp_nocc


def tuples_filtered_with_nocc(
    sort_pair_occ_contribs: List[Tuple[np.ndarray, np.ndarray]],
    sort_pair_virt_contribs: List[Tuple[np.ndarray, np.ndarray]],
    pair_contribs: np.ndarray,
    order: int,
    tup_nocc: int,
    exp_nocc: int,
    thres: float,
    orb_space: np.ndarray,
    occ_start: int = 0,
    virt_start: int = 0,
) -> Generator[np.ndarray, None, None]:
    """
    this function generates tuples for different orbital combinations
    """
    # single-orbital tuples
    if order == 1:
        if tup_nocc == 0:
            for orb_idx in range(exp_nocc + virt_start, orb_space.size):
                mapped_tup = int(orb_space[orb_idx])
                yield np.array([mapped_tup], dtype=np.int64)
        elif tup_nocc == 1:
            for orb_idx in range(occ_start, exp_nocc):
                yield np.array([orb_space[orb_idx]], dtype=np.int64)
    # multiple-orbital tuples
    else:
        # only virtual orbitals
        if tup_nocc == 0:
            for orb_idx in range(exp_nocc + virt_start, orb_space.size):
                for tup, _ in backtrack(
                    [orb_idx], 1.0, sort_pair_virt_contribs, pair_contribs, order, thres
                ):
                    yield orb_space[tup]
        # only occupied orbitals
        elif tup_nocc == order:
            for orb_idx in range(occ_start, exp_nocc):
                for tup, _ in backtrack(
                    [orb_idx], 1.0, sort_pair_occ_contribs, pair_contribs, order, thres
                ):
                    yield orb_space[tup]
        # occupied and virtual orbitals
        else:
            # single occupied orbital
            if tup_nocc == 1:
                for orb_idx in range(occ_start, exp_nocc):
                    for tup, _ in backtrack(
                        [orb_idx],
                        1.0,
                        sort_pair_virt_contribs,
                        pair_contribs,
                        order,
                        thres,
                    ):
                        yield orb_space[tup]
            # multiple occupied orbitals
            else:
                # for orb_idx in range(nocc):
                for orb_idx in range(occ_start, exp_nocc):
                    for tup_occ, orb_prod in backtrack(
                        [orb_idx],
                        1.0,
                        sort_pair_occ_contribs,
                        pair_contribs,
                        tup_nocc,
                        thres,
                    ):
                        for tup, _ in backtrack(
                            tup_occ,
                            orb_prod,
                            sort_pair_virt_contribs,
                            pair_contribs,
                            order,
                            thres,
                        ):
                            yield orb_space[tup]


def backtrack(
    curr_tup: List[int],
    curr_prod: float,
    sorted_pair_contribs: List[Tuple[np.ndarray, np.ndarray]],
    pair_contribs: np.ndarray,
    order: int,
    thres: float,
) -> Generator[Tuple[List[int], float], None, None]:
    """
    this function recursively generates tuples with pair importance above threshold
    """
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
            overlap = orb_prods[idx]
            # add next orbital to tuple and yield
            yield curr_tup + [sorted_pair_contribs[curr_tup[-1]][0][idx]], overlap

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
