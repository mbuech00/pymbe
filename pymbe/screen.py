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
from math import floor, log10, ceil
from scipy import stats
from typing import TYPE_CHECKING

from pymbe.logger import logger

from pymbe.tools import n_tuples, n_tuples_predictors


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
    screen: Dict[str, np.ndarray],
    bins: np.ndarray,
    ntot_bins: np.ndarray,
    signs: np.ndarray,
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
    est_mean_inc = np.zeros_like(est_error)

    # initialize array for cluster contribution errors
    tot_error = np.zeros(len(exp_clusters), dtype=np.float64)

    # initialize array for error difference to threshold
    error_diff = np.zeros_like(tot_error)

    # loop over clusters
    for cluster_idx, cluster_orbs in enumerate(exp_clusters):
        # get mask number of clusters with more than single sample
        mask = screen["inc_count"][:, cluster_orbs[0]] > 1

        # get corresponding number of clusters
        nclusters = np.arange(1, screen["inc_count"].shape[0] + 1, dtype=np.float64)[
            mask
        ]

        # calculate logarithm of mean increment magnitude
        mean = np.log(
            screen["inc_sum"][mask, cluster_orbs[0]]
            / screen["inc_count"][mask, cluster_orbs[0]]
        )

        # calculate variance of the logarithm of mean increment magnitude
        mean_variance = (
            screen["log_inc_sum2"][mask, cluster_orbs[0]]
            - 2 * mean * screen["log_inc_sum"][mask, cluster_orbs[0]]
        ) / screen["inc_count"][mask, cluster_orbs[0]] + mean**2
        mean_variance /= screen["inc_count"][mask, cluster_orbs[0]]

        # calculate weights as reciprocal of standard error of the mean
        weights = np.sqrt(
            np.divide(
                1,
                mean_variance,
                out=np.zeros_like(mean_variance),
                where=mean_variance != 0,
            )
        )

        # require at least 3 points to fit and ensure all
        # (screening exponent + 1)-orbital correlations have been calculated
        if (
            nclusters.size >= 3
            and curr_order
            >= sum(
                cluster.size
                for cluster in (
                    exp_clusters[:cluster_idx] + exp_clusters[cluster_idx + 1 :]
                )[floor(log10(screen_thres)) :]
            )
            + cluster_orbs.size
        ):
            # fit logarithmic mean increment magnitude
            fit = np.polynomial.polynomial.Polynomial(
                np.polynomial.polynomial.polyfit(nclusters, mean, 1, w=weights)
            )

            # get t-statistic for 95% significance
            t_stat = stats.t.ppf(0.975, nclusters.size - 2)

            # get residuals of model
            residuals = mean - fit(nclusters)

            # get standard error of model
            s_err = np.sqrt(np.sum(residuals**2) / (nclusters.size - 2))

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

                # initialize number of tuples for cluster for this order
                ntup_order_cluster = 0

                # loop over number of tuples and predictors
                for ntup, ncluster, ncontrib in n_tuples_predictors(
                    exp_space,
                    exp_clusters if not exp_single_orbs else None,
                    cluster_idx,
                    nocc,
                    ref_nelec,
                    ref_nhole,
                    vanish_exc,
                    order,
                ):
                    # check if any tuples for this cluster at this order contribute
                    if ntup > 0:
                        # add to number of tuples for cluster for this order
                        ntup_order_cluster += ntup

                        # estimate logarithmic mean increment magnitude prediction
                        # interval for given predictors and add to mean increment
                        # magnitude as worst-case behaviour
                        log_mean_abs_inc = fit(ncluster) + (
                            t_stat
                            * s_err
                            * np.sqrt(
                                1
                                + 1 / nclusters.size
                                + (ncluster - np.mean(nclusters)) ** 2
                                / np.sum((nclusters - np.mean(nclusters)) ** 2)
                            )
                        )

                        # estimate mean increment magnitude for given predictors
                        mean_abs_inc = ncontrib * np.exp(log_mean_abs_inc)

                        # get factor due to sign cancellation
                        insert_idx = np.digitize(mean_abs_inc, bins).item()
                        if insert_idx < signs.size and ntot_bins[insert_idx] > 0:
                            if signs[insert_idx] == 0:
                                p = 0.5 / ntot_bins[insert_idx]
                            else:
                                p = signs[insert_idx]
                            # get 95% prediction interval for binomial distribution of
                            # sign factor according to Nelson
                            sign_factor = ntup * p + stats.norm.ppf(0.975) * np.sqrt(
                                (ntup * p * (1 - p) * (ntup + ntot_bins[insert_idx]))
                                / ntot_bins[insert_idx]
                            )
                            sign_factor = min(ntup, sign_factor)
                            if ntup % 2 == 0:
                                # round up to closest even number
                                sign_factor = ceil(sign_factor / 2) * 2
                            else:
                                # round up to closest odd number
                                sign_factor = ceil(sign_factor) // 2 * 2 + 1
                        else:
                            sign_factor = ntup

                        # calculate the error for this order
                        est_error[cluster_idx, order_idx] += sign_factor * mean_abs_inc

                # check if cluster produces increments for this order
                if ntup_order_cluster > 0:
                    # add to number of tuples for cluster for all remaining orders
                    ntup_all_cluster += ntup_order_cluster

                    # estimate the mean absolute increment for all increments at this
                    # order
                    est_mean_inc[cluster_idx, order_idx] = (
                        est_error[cluster_idx, order_idx] / ntup_order_cluster
                    )

                    # add to total error
                    tot_error[cluster_idx] += est_error[cluster_idx, order_idx]

                    # stop if the last few orders contribute less than 1% or if
                    # accumulated error is larger than threshold
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
                    ) or tot_error[cluster_idx] > error_thresh:
                        break

            # calculate difference to allowed error
            error_diff[cluster_idx] = (
                ntup_all_cluster / ntup_all_total
            ) * error_thresh - tot_error[cluster_idx]

    # get index in expansion space for minimum cluster contribution
    min_idx = int(np.argmax(error_diff))

    # screen cluster away if contribution is smaller than threshold
    if error_diff[min_idx] > 0.0:
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
        logger.info2(" " + 47 * "-")
        logger.info2("  Order | Est. mean abs. increment | Est. error")
        logger.info2(" " + 47 * "-")
        for order_idx, (order, mean_inc, error) in enumerate(
            zip(
                range(curr_order + 1, max_order + 1),
                est_mean_inc[min_idx],
                est_error[min_idx],
            )
        ):
            if sum(est_error[min_idx][order_idx:]) == 0.0:
                break
            logger.info2(
                f"  {order:5} |        {mean_inc:>10.4e}        | {error:>10.4e}"
            )
        logger.info2(" " + 47 * "-" + "\n")

        # add screened cluster contribution to error
        mbe_tot_error += tot_error[min_idx]

        return min_idx, mbe_tot_error

    # check if geometric mean absolute increment contribution for minimum error cluster
    # comes close to convergence threshold
    elif np.sum(screen["inc_count"][:, min_idx]) > 0 and (
        0.0
        < np.sum(screen["log_inc_sum"][:, min_idx])
        / np.sum(screen["inc_count"][:, min_idx])
        < np.log(1e1 * conv_tol)
    ):
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

    # cluster with minimum contribution is not screened away
    else:
        return None, mbe_tot_error
