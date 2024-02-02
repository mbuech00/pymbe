#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
screening module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from math import comb
from scipy import optimize
from numpy.polynomial.polynomial import Polynomial
from typing import TYPE_CHECKING

from pymbe.logger import logger

from pymbe.tools import (
    n_tuples,
    get_ncluster_blks,
    single_cluster_n_tuples_with_nocc,
    valid_tup,
)


if TYPE_CHECKING:
    from typing import List, Dict, Tuple, Optional


def fixed_screen(
    screen_func: str,
    screen_start: int,
    screen_perc: float,
    screen: Dict[str, np.ndarray],
    exp_clusters: List[np.ndarray],
    exp_space: np.ndarray,
    order: int,
) -> List[int]:
    """
    this function returns indices to be removed from fixed screening
    """
    thres = 1.0 if order < screen_start else screen_perc
    nscreen = exp_space.size - int(thres * exp_space.size)
    if screen_func == "rnd":
        rng = np.random.default_rng()
        norb = 0
        remove_idx = []
        while norb < nscreen:
            idx = rng.integers(len(exp_clusters))
            norb += exp_clusters[idx].size
            remove_idx.append(idx)

    else:
        cluster_screen = [
            np.abs(screen[screen_func][cluster[0]]) for cluster in exp_clusters
        ]
        # stable sorting algorithm is important to ensure that the same
        # clusters are screened away every time in case of equality
        cluster_significance = np.argsort(cluster_screen, kind="stable")

        norb = 0
        remove_idx = []
        for idx in cluster_significance:
            if norb >= nscreen:
                break
            norb += exp_clusters[idx].size
            remove_idx.append(idx)

    return remove_idx


def adaptive_screen(
    mbe_tot_error: float,
    screen_thres: float,
    screen: List[Dict[str, np.ndarray]],
    exp_clusters: List[np.ndarray],
    exp_space: np.ndarray,
    exp_single_orbs: bool,
    nocc: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    curr_order: int,
    conv_tol: float,
) -> Tuple[Optional[int], float]:
    # define maximum possible order
    max_order = exp_space.size

    # check if expansion has ended
    if curr_order >= max_order:
        return None, mbe_tot_error

    # define maximum cluster size
    max_cluster_size = max(cluster.size for cluster in exp_clusters)

    # define allowed error
    error_thresh = screen_thres - mbe_tot_error

    # initialize array for estimated quantities
    est_error = np.zeros((len(exp_clusters), max_order - curr_order), dtype=np.float64)
    est_rel_factor = np.zeros_like(est_error)
    est_mean_norm_abs_inc = np.zeros_like(est_error)

    # initialize array for cluster contribution errors
    tot_error = np.zeros(len(exp_clusters), dtype=np.float64)

    # initialize array for error difference to threshold
    error_diff = np.zeros_like(tot_error)

    # loop over clusters
    for cluster_idx in range(len(exp_clusters)):
        # get mean absolute increments and relative factor for
        # cluster
        mean_norm_abs_inc = np.array(
            [screen["mean_norm_abs_inc"][cluster_idx] for screen in screen],
            dtype=np.float64,
        )
        rel_factor = np.array(
            [screen["rel_factor"][cluster_idx] for screen in screen],
            dtype=np.float64,
        )

        # log transform mean absolute increments
        log_mean_norm_abs_inc = np.log(mean_norm_abs_inc[mean_norm_abs_inc > 0.0])

        # get corresponding relative factors
        rel_factor = rel_factor[mean_norm_abs_inc > 0.0]

        # get number of clusters for fit
        nclusters = np.argwhere(mean_norm_abs_inc > 0.0).reshape(-1) + 1

        # require at least 3 points to fit
        if nclusters.size > 2:
            # fit logarithmic mean absolute increment
            (opt_slope, opt_zero), cov = np.polyfit(
                nclusters, log_mean_norm_abs_inc, 1, cov=True
            )
            err_slope, err_zero = np.sqrt(np.diag(cov))
            opt_slope += 2 * err_slope
            opt_zero += 2 * err_zero

            # assume mean absolute increment does not decrease
            mean_norm_abs_inc_fit = Polynomial([opt_zero, opt_slope])

            # define fitting function for relative factor
            def rel_factor_fit(x, half, slope):
                return 1.0 / (1.0 + ((x - nclusters[0]) / half) ** slope)

            # fit relative factor
            if np.count_nonzero(rel_factor < 0.5) > 2:
                (opt_half, opt_slope), cov = optimize.curve_fit(
                    rel_factor_fit,
                    nclusters,
                    rel_factor,
                    bounds=([0.5, 1.0], [max_order + 1, np.inf]),
                    maxfev=1000000,
                )
                err_half, err_slope = np.sqrt(np.diag(cov))

                opt_half = min(opt_half + 2 * err_half, max_order + 1)
                opt_slope = max(opt_slope - 2 * err_slope, 1.0)
            else:
                opt_half = opt_slope = 0.0
                rel_factor_fit = lambda *args: 1.0

            # initialize number of tuples for cluster for remaining
            # orders
            ntup_all_cluster = 0

            # initialize number of total tuples for remaining orders
            ntup_all_total = 0

            # loop over remaining orders
            for order_idx, order in enumerate(range(curr_order + 1, max_order + 1)):
                # get total number of tuples at this order
                ntup_all_total += n_tuples(
                    exp_space,
                    exp_clusters if not exp_single_orbs else None,
                    nocc,
                    ref_nelec,
                    ref_nhole,
                    vanish_exc,
                    order,
                )

                # estimate the relative factor for this order
                est_rel_factor[cluster_idx, order_idx] = rel_factor_fit(
                    order, opt_half, opt_slope
                )

                # initialize number of tuples for cluster for this order
                ntup_order_cluster = 0

                # get cluster size and number of virtual orbitals in cluster
                cluster_size = exp_clusters[cluster_idx].size
                cluster_nvirt = (nocc <= exp_clusters[cluster_idx]).sum()

                # get number of expansion space clusters for every cluster size and
                # number of virtual orbitals with current cluster removed
                if not exp_single_orbs:
                    size_nvirt_blks = get_ncluster_blks(
                        exp_clusters[:cluster_idx] + exp_clusters[cluster_idx + 1 :],
                        nocc,
                    )

                # loop over a certain number of clusters
                for ncluster in range(1, order + 1):
                    # loop over occupations
                    for tup_nocc in range(order + 1):
                        # check if tuple is valid for chosen method
                        if valid_tup(
                            ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc
                        ):
                            # get number of tuples for this cluster for a given order and
                            # occupation
                            ntup = single_cluster_n_tuples_with_nocc(
                                exp_space,
                                size_nvirt_blks if not exp_single_orbs else None,
                                cluster_size,
                                cluster_nvirt,
                                nocc,
                                ncluster,
                                order,
                                tup_nocc,
                            )

                            # add to number of tuples for cluster for this order
                            ntup_order_cluster += ntup

                            # calculate the error for this order
                            if ntup > 0:
                                est_error[cluster_idx, order_idx] += (
                                    est_rel_factor[cluster_idx, order_idx]
                                    * ntup
                                    * np.exp(mean_norm_abs_inc_fit(ncluster))
                                    * comb(order, tup_nocc) ** 2
                                )

                # check if cluster produces increments for this order
                if ntup_order_cluster > 0:
                    # add to number of tuples for cluster for all remaining orders
                    ntup_all_cluster += ntup_order_cluster

                    # estimate the mean absolute increment for all increments at this order
                    est_mean_norm_abs_inc[cluster_idx, order_idx] = (
                        est_error[cluster_idx, order_idx] / ntup_order_cluster
                    )

                    # add to total error
                    tot_error[cluster_idx] += est_error[cluster_idx, order_idx]

                    # stop if the last few orders contribute less than 1%
                    if (
                        np.sum(
                            est_error[
                                cluster_idx,
                                max(0, order_idx - max_cluster_size + 1) : order_idx
                                + 1,
                            ]
                        )
                        / tot_error[cluster_idx]
                        < 0.01
                    ):
                        break

            # calculate difference to allowed error
            error_diff[cluster_idx] = (
                ntup_all_cluster / ntup_all_total
            ) * error_thresh - tot_error[cluster_idx]

    # get index in expansion space for minimum cluster contribution
    min_idx = int(np.argmax(error_diff))

    # check if maximum order mean aboslute increment contribution for minimum error
    # cluster comes close to convergence threshold
    if 0.0 < max(est_mean_norm_abs_inc[min_idx, :]) < 1e1 * conv_tol:
        # log screening
        if exp_clusters[min_idx].size == 1:
            logger.info2(
                f" Orbital {exp_clusters[min_idx].item()} is screened away due to the "
                "majority of increments getting close to convergence\n criterium"
            )
        else:
            cluster_str = np.array2string(exp_clusters[min_idx], separator=", ")
            logger.info2(
                f" Orbital cluster {cluster_str} is screened away due to the majority "
                "of increments getting close to convergence\n criterium"
            )

        return min_idx, mbe_tot_error

    # screen cluster away if contribution is smaller than threshold
    elif error_diff[min_idx] > 0.0:
        # log screening
        if exp_clusters[min_idx].size == 1:
            logger.info2(
                f" Orbital {exp_clusters[min_idx].item()} is screened away (Error "
                f"= {tot_error[min_idx]:>10.4e})"
            )
        else:
            cluster_str = np.array2string(exp_clusters[min_idx], separator=", ")
            logger.info2(
                f" Orbital cluster {cluster_str} is screened away (Error = "
                f"{tot_error[min_idx]:>10.4e})"
            )
        logger.info2(" " + 70 * "-")
        logger.info2(
            "  Order | Est. relative factor | Est. mean abs. increment | Est. error"
        )
        logger.info2(" " + 70 * "-")
        for order, factor, mean_norm_abs_inc, error in zip(
            range(curr_order + 1, max_order + 1),
            est_rel_factor[min_idx],
            est_mean_norm_abs_inc[min_idx],
            est_error[min_idx],
        ):
            if error == 0.0:
                break
            logger.info2(
                f"  {order:5} |      {factor:>10.4e}      |        "
                f"{mean_norm_abs_inc:>10.4e}        | {error:>10.4e}"
            )
        logger.info2(" " + 70 * "-" + "\n")

        # add screened cluster contribution to error
        mbe_tot_error += tot_error[min_idx]

        return min_idx, mbe_tot_error

    # cluster with minimum contribution is not screened away
    else:
        return None, mbe_tot_error
