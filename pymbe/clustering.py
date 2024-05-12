#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
clustering module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from math import exp, log
from itertools import combinations
from pyscf import symm
from typing import TYPE_CHECKING

from pymbe.logger import logger


if TYPE_CHECKING:
    from numpy.random import Generator
    from typing import List, Tuple


# seed for random number generators
SEED = 42

# sample size to determine starting temperature
SAMPLE_SIZE = 1000

# target accept probability to determine starting temperature
TARGET_ACCEPT_PROB = 0.99

# maximum iterations after which the optimization is stopped if the score function does
# not improve
MAX_ITERATIONS = 100000


def cluster_driver(
    max_cluster_size: int,
    max_order: int,
    orb_pairs: np.ndarray,
    exp_space: np.ndarray,
    nocc: int,
    point_group: str,
    orbsym: np.ndarray,
) -> List[np.ndarray]:
    """
    this function is the mbe main function
    """
    # number of occupied and virtual orbitals in expansion space
    exp_nocc = (exp_space < nocc).sum()

    # get number of clusters
    ncluster = -(exp_space.size // -max_cluster_size)

    # get cluster size
    cluster_size = -(exp_space.size // -ncluster)

    # number of larger and smaller clusters
    ncluster_large = exp_space.size - ncluster * (cluster_size - 1)
    ncluster_small = ncluster - ncluster_large

    # maximum number of occupied orbitals per cluster
    cluster_nocc = -(exp_nocc // -ncluster)

    # number of clusters with occupation cluster_nocc
    ncluster_high = exp_nocc - ncluster * (cluster_nocc - 1)

    # clusters with cluster_size and cluster_nocc
    ncluster_large_high = min(ncluster_large, ncluster_high)

    # clusters with cluster_size and cluster_nocc - 1
    ncluster_large_low = max(ncluster_large - ncluster_high, 0)

    # clusters with cluster_size - 1 and cluster_nocc
    if ncluster_large_low == 0:
        ncluster_small_high = ncluster_high - ncluster_large_high
    else:
        ncluster_small_high = 0

    # clusters with cluster_size - 1 and cluster_nocc - 1
    ncluster_small_low = ncluster_small - ncluster_small_high

    # different cluster types
    cluster_types: Tuple[Tuple[int, int, int], ...] = (
        (cluster_size, cluster_nocc, ncluster_large_high),
        (cluster_size, cluster_nocc - 1, ncluster_large_low),
        (cluster_size - 1, cluster_nocc, ncluster_small_high),
        (cluster_size - 1, cluster_nocc - 1, ncluster_small_low),
    )
    cluster_types = tuple(
        (size, nocc, nclusters)
        for size, nocc, nclusters in cluster_types
        if nclusters > 0
    )

    # simulated annealing to determine optimal orbital clusters
    exp_clusters = _simulated_annealing(
        orb_pairs, cluster_types, ncluster, exp_nocc, exp_space.size, exp_space
    )

    # log orbital clusters
    symm_header = "Cluster symmetries"
    orb_header = "Cluster orbitals"
    symm_len = len(symm_header)
    orb_len = len(orb_header)
    for cluster in exp_clusters:
        symm_len = max(
            symm_len,
            len(
                ", ".join(
                    [
                        symm.addons.irrep_id2name(point_group, orb)
                        for orb in orbsym[cluster]
                    ]
                )
            ),
        )
        orb_len = max(orb_len, len(", ".join(map(str, cluster))))
    logger.info2(" " + (19 + orb_len + symm_len) * "-")
    logger.info2(
        f"  Cluster No. | {orb_header:^{orb_len}} | {symm_header:^{symm_len}} "
    )
    logger.info2(" " + (19 + orb_len + symm_len) * "-")
    for cluster_idx, cluster in enumerate(exp_clusters):
        symm_str = ", ".join(
            [symm.addons.irrep_id2name(point_group, orb) for orb in orbsym[cluster]]
        )
        orb_str = ", ".join(map(str, cluster))
        logger.info2(
            f"  {cluster_idx:11} | {orb_str:>{orb_len}} | " f"{symm_str:>{symm_len}} "
        )
    logger.info2(" " + (19 + orb_len + symm_len) * "-")

    return exp_clusters


def _simulated_annealing(
    orb_pairs: np.ndarray,
    cluster_types: Tuple[Tuple[int, int, int], ...],
    tot_ncluster: int,
    nocc: int,
    norb: int,
    exp_space: np.ndarray,
) -> List[np.ndarray]:
    """
    this function performs simulated annealing to determine the optimal cluster types
    """
    # initialize random number generator
    rng = np.random.default_rng(seed=SEED)

    # generate number of occupied and virtual orbitals for every cluster
    clusters = np.empty((2, tot_ncluster), dtype=np.int64)
    idx = 0
    for cluster_size, cluster_nocc, nclusters in cluster_types:
        clusters[0, idx : idx + nclusters] = cluster_nocc
        clusters[1, idx : idx + nclusters] = cluster_size - cluster_nocc
        idx += nclusters

    # index for all orbitals per cluster
    cluster_idx = np.cumsum(clusters, axis=1)

    # total number of pairs per cluster
    cluster_npairs = np.empty((2, tot_ncluster), np.int64)
    for idx in range(clusters.shape[1]):
        cluster_npairs[:, idx] = clusters[:, idx] * np.sum(
            clusters[:, idx + 1 :], axis=1
        )

    # get number of pairs for every orbital per cluster
    cluster_npairs_per_orb = np.floor_divide(
        cluster_npairs,
        clusters,
        out=np.zeros_like(clusters),
        where=clusters != 0,
    )

    # index for all pairs per cluster
    cluster_pair_idx = np.cumsum(cluster_npairs, axis=1)

    # determine number of possible swaps
    nocc_swaps = cluster_pair_idx[0, -1]
    nvirt_swaps = cluster_pair_idx[1, -1]
    ntot_swaps = nocc_swaps + nvirt_swaps

    # intialize samples to determine starting temperature
    samples = np.empty((SAMPLE_SIZE, 2), dtype=np.float64)

    # generate samples to determine starting temperature
    for sample_idx in range(SAMPLE_SIZE):
        # initialize difference
        diff = 0.0

        # search for sample that is not a local maximum
        while diff >= 0.0:
            # generate sample
            sample = np.concatenate(
                (rng.permutation(nocc), rng.permutation(np.arange(nocc, norb)))
            )

            # find lower neighboring point
            nneighbor = 0
            while diff >= 0.0 and nneighbor < int(
                log(0.01) / log((ntot_swaps - 1) / ntot_swaps)
            ):
                # get neighboring point
                orb1, orb2 = _gen_neighbor(
                    rng,
                    cluster_idx,
                    cluster_pair_idx,
                    cluster_npairs_per_orb,
                    nocc,
                    ntot_swaps,
                    nocc_swaps,
                )

                # evaluate candidate point
                diff = _score_diff(
                    orb_pairs, sample, (orb1, orb2), cluster_idx, clusters, nocc
                )

                # increment number of neighbors
                nneighbor += 1

            # get score of sample
            samples[sample_idx, :] = _score(orb_pairs, sample, cluster_types, nocc)

            # get score of neighboring point
            samples[sample_idx, 1] += diff

    # intitialize variables
    starting_temp = 1.0
    accept_prob = 0.0

    # determine optimal starting temperature
    while abs(accept_prob - TARGET_ACCEPT_PROB) > 1e-3:
        # calculate current acceptance probability
        accept_prob = np.sum(np.exp(samples[:, 1] / starting_temp)) / np.sum(
            np.exp(samples[:, 0] / starting_temp)
        )
        starting_temp *= log(accept_prob) / log(TARGET_ACCEPT_PROB)

    # set starting temperature
    logger.info(
        f" Starting simulated annealing for orbital clustering with starting"
        f" temperature {starting_temp:2.3f}"
    )

    # generate an initial point
    start = np.concatenate(
        (
            rng.permutation(nocc),
            rng.permutation(np.arange(nocc, norb)),
        )
    )

    # set current working solution to start
    curr = start
    curr_eval = _score(orb_pairs, curr, cluster_types, nocc)

    # best working solution
    best, best_eval = curr, curr_eval

    logger.info(" " + 27 * "-")
    logger.info(f"  Iteration | Score funtion ")
    logger.info(" " + 27 * "-")
    logger.info(f"         0  |   {best_eval:5.3e}  ")

    # run the algorithm
    iteration = 1
    last_iteration = 0
    alpha = 1e-5 ** (1 / MAX_ITERATIONS)
    while iteration - last_iteration < MAX_ITERATIONS:
        # get neighboring point
        orb1, orb2 = _gen_neighbor(
            rng,
            cluster_idx,
            cluster_pair_idx,
            cluster_npairs_per_orb,
            nocc,
            ntot_swaps,
            nocc_swaps,
        )

        # evaluate candidate point
        diff = _score_diff(orb_pairs, curr, (orb1, orb2), cluster_idx, clusters, nocc)

        # calculate temperature for current epoch
        t = starting_temp * alpha**iteration

        # check for new best solution
        if curr_eval + diff > best_eval:
            # store new best point
            best = curr.copy()
            best[orb1], best[orb2] = best[orb2], best[orb1]
            best_eval = curr_eval + diff

            # set last iteration that better solution was found
            last_iteration = iteration

            # report progress
            logger.info(f"   {iteration:7d}  |   {best_eval:5.3e}  ")

        # check if new point should be kept either because it is better or because
        # metropolis acceptance criterion is fulfilled
        if diff > 0 or rng.uniform() < exp(diff / t):
            # store the new current point
            curr[orb1], curr[orb2] = curr[orb2], curr[orb1]
            curr_eval = curr_eval + diff

        # increment iteration
        iteration += 1

    logger.info(" " + 27 * "-" + "\n")

    # extract clusters
    exp_clusters = []
    occ_idx = 0
    virt_idx = nocc
    for cluster_nocc, cluster_nvirt in clusters.T:
        exp_clusters.append(
            np.sort(
                exp_space[
                    np.concatenate(
                        (
                            best[occ_idx : occ_idx + cluster_nocc],
                            best[virt_idx : virt_idx + cluster_nvirt],
                        )
                    )
                ]
            )
        )
        occ_idx += cluster_nocc
        virt_idx += cluster_nvirt

    return exp_clusters


def _score_diff(
    orb_pairs: np.ndarray,
    curr: np.ndarray,
    swap: Tuple[int, int],
    cluster_idx: np.ndarray,
    clusters: np.ndarray,
    nocc: int,
) -> float:
    """
    this function calculates the score difference when two orbitals are swapped
    """
    # initialize difference
    diff = 0.0

    # determine clusters that swapped orbitals are located in
    if swap[0] < nocc:
        cluster1 = np.searchsorted(cluster_idx[0], swap[0], side="right")
        cluster2 = np.searchsorted(cluster_idx[0], swap[1], side="right")
    else:
        cluster1 = np.searchsorted(cluster_idx[1], swap[0] - nocc, side="right")
        cluster2 = np.searchsorted(cluster_idx[1], swap[1] - nocc, side="right")

    # shift for first cluster
    if cluster1 > 0:
        nocc_shift = cluster_idx[0, cluster1 - 1]
        nvirt_shift = nocc + cluster_idx[1, cluster1 - 1]
    else:
        nocc_shift = 0
        nvirt_shift = nocc

    # add and subtract contributions from first cluster
    for norb in range(clusters[0, cluster1]):
        idx = nocc_shift + norb
        if idx != swap[0]:
            diff -= orb_pairs[curr[swap[0]], curr[idx]]
            diff += orb_pairs[curr[swap[1]], curr[idx]]
    for norb in range(clusters[1, cluster1]):
        idx = nvirt_shift + norb
        if idx != swap[0]:
            diff -= orb_pairs[curr[swap[0]], curr[idx]]
            diff += orb_pairs[curr[swap[1]], curr[idx]]

    # shift for first cluster
    if cluster2 > 0:
        nocc_shift = cluster_idx[0, cluster2 - 1]
        nvirt_shift = nocc + cluster_idx[1, cluster2 - 1]
    else:
        nocc_shift = 0
        nvirt_shift = nocc

    # add and subtract contributions from second cluster
    for norb in range(clusters[0, cluster2]):
        idx = nocc_shift + norb
        if idx != swap[1]:
            diff -= orb_pairs[curr[swap[1]], curr[idx]]
            diff += orb_pairs[curr[swap[0]], curr[idx]]
    for norb in range(clusters[1, cluster2]):
        idx = nvirt_shift + norb
        if idx != swap[1]:
            diff -= orb_pairs[curr[swap[1]], curr[idx]]
            diff += orb_pairs[curr[swap[0]], curr[idx]]

    return diff


def _score(
    orb_pairs: np.ndarray,
    curr: np.ndarray,
    cluster_types: Tuple[Tuple[int, int, int], ...],
    nocc: int,
) -> float:
    """
    this function calculates the score for a given set of orbitals
    """
    energy = 0.0
    occ_idx = 0
    virt_idx = 0
    for cluster_size, cluster_nocc, nclusters in cluster_types:
        cluster_nvirt = cluster_size - cluster_nocc
        for _ in range(nclusters):
            for pair in combinations(
                np.concatenate(
                    (
                        curr[occ_idx : occ_idx + cluster_nocc],
                        curr[nocc + virt_idx : nocc + virt_idx + cluster_nvirt],
                    )
                ),
                2,
            ):
                energy += orb_pairs[pair[0], pair[1]]
            occ_idx += cluster_nocc
            virt_idx += cluster_nvirt
    return energy


def _gen_neighbor(
    rng: Generator,
    cluster_idx: np.ndarray,
    cluster_pair_idx: np.ndarray,
    cluster_npairs_per_orb: np.ndarray,
    nocc: int,
    ntot_swaps: int,
    nocc_swaps: int,
) -> Tuple[int, int]:
    """
    this function generates a random neighbor from a given point
    """
    # generate random integer
    rand_int = rng.integers(ntot_swaps)

    # determine if occupied or virtual orbitals are swapped
    if rand_int < nocc_swaps:
        occ_idx = 0
    else:
        occ_idx = 1
        rand_int -= nocc_swaps

    # get position of random integer in first cluster
    cluster1 = np.searchsorted(cluster_pair_idx[occ_idx], rand_int, side="right")

    # shift remaining integer
    if cluster1 > 0:
        rand_int -= cluster_pair_idx[occ_idx, cluster1 - 1]

    # get orbital in first cluster
    orb1 = rand_int // cluster_npairs_per_orb[occ_idx, cluster1]

    # shift first orbital
    if cluster1 > 0:
        orb1 += cluster_idx[occ_idx, cluster1 - 1]

    # get second orbital
    orb2 = (
        cluster_idx[occ_idx, cluster1]
        + rand_int % cluster_npairs_per_orb[occ_idx, cluster1]
    )

    # shift virtual orbitals
    if occ_idx == 1:
        orb1 += nocc
        orb2 += nocc

    return orb1, orb2
