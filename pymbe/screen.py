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
from scipy import optimize
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
    est_rel_factor = np.zeros_like(est_error)
    est_mean_abs_inc = np.zeros_like(est_error)

    # initialize array for cluster contribution errors
    tot_error = np.zeros(len(exp_clusters), dtype=np.float64)

    # initialize array for error difference to threshold
    error_diff = np.zeros_like(tot_error)

    # loop over clusters
    for cluster_idx, cluster_orbs in enumerate(exp_clusters):
        # get mask for predictors with more than single sample
        mask = screen["inc_count"][:, cluster_orbs[0]] > 1

        # get corresponding predictors
        predictors = screen["predictors"][mask].astype(np.float64)
        predictors[:, 1] = np.log(predictors[:, 1])

        # calculate logarithm of mean increment magnitude
        mean = np.log(
            screen["inc_sum"][mask, cluster_orbs[0]]
            / screen["inc_count"][mask, cluster_orbs[0]]
        )

        # calculate variance of the logarithm of mean increment magnitude
        variance = (
            screen["log_inc_sum2"][mask, cluster_orbs[0]]
            - 2 * mean * screen["log_inc_sum"][mask, cluster_orbs[0]]
        ) / screen["inc_count"][mask, cluster_orbs[0]] + mean**2
        variance /= screen["inc_count"][mask, cluster_orbs[0]]

        # require at least 3 points to fit and ensure all two-orbital correlations have
        # been calculated
        if (
            predictors.shape[0] >= 3
            and curr_order
            >= (exp_clusters[:cluster_idx] + exp_clusters[cluster_idx + 1 :])[-1].size
            + cluster_orbs.size
        ):
            # fit parameters
            fit_params: Tuple[int, ...]

            # check if second predictor is unique
            if len(np.unique(predictors[:, 1], return_counts=True)[0]) == 1:
                # define fit function
                def fit(x: np.ndarray, *args: float):
                    return args[0] + args[1] * x[:, 0]

                # fit logarithmic mean increment magnitude
                (intercept, ncluster_slope), cov = optimize.curve_fit(
                    fit,
                    predictors,
                    mean,
                    p0=(-1, -1),
                    bounds=([-np.inf, -np.inf], [np.inf, 0]),
                    maxfev=1000000,
                    sigma=np.sqrt(variance),
                    absolute_sigma=True,
                )
                intercept_err, ncluster_slope_err = np.sqrt(np.diag(cov))
                intercept += 2 * intercept_err
                ncluster_slope += 2 * ncluster_slope_err
                fit_params = (intercept, ncluster_slope)
            else:
                # define fit function
                def fit(x: np.ndarray, *args: float):
                    return args[0] + args[1] * x[:, 0] + args[2] * x[:, 1]

                # fit logarithmic mean increment magnitude
                (intercept, ncluster_slope, ncontrib_slope), cov = optimize.curve_fit(
                    fit,
                    predictors,
                    mean,
                    p0=(-1, -1, 1),
                    bounds=([-np.inf, -np.inf, 0], [np.inf, 0, np.inf]),
                    maxfev=1000000,
                    sigma=np.sqrt(variance),
                    absolute_sigma=True,
                )
                intercept_err, ncluster_slope_err, ncontrib_slope_err = np.sqrt(
                    np.diag(cov)
                )
                intercept += 2 * intercept_err
                ncluster_slope += 2 * ncluster_slope_err
                ncontrib_slope += 2 * ncontrib_slope_err
                fit_params = (intercept, ncluster_slope, ncontrib_slope)

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

                        # estimate mean increment magnitude for given predictors
                        mean_abs_inc = np.exp(
                            fit(np.array([[ncluster, np.log(ncontrib)]]), *fit_params)
                        )

                        # get factor due to sign cancellation
                        insert_idx = np.digitize(mean_abs_inc, bins).item()
                        if insert_idx < signs.size and ntot_bins[insert_idx] > 0:
                            prev_bin_slice = slice(
                                insert_idx, min(insert_idx + 3, signs.size)
                            )
                            sign_factor = np.average(
                                signs[prev_bin_slice], weights=ntot_bins[prev_bin_slice]
                            )
                        else:
                            sign_factor = 1.0

                        # calculate the error for this order
                        est_error[cluster_idx, order_idx] += (
                            sign_factor * ntup * mean_abs_inc
                        )

                # check if cluster produces increments for this order
                if ntup_order_cluster > 0:
                    # add to number of tuples for cluster for all remaining orders
                    ntup_all_cluster += ntup_order_cluster

                    # estimate the mean absolute increment for all increments at this
                    # order
                    est_mean_abs_inc[cluster_idx, order_idx] = (
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
        logger.info2(" " + 70 * "-")
        logger.info2(
            "  Order | Est. relative factor | Est. mean abs. increment | Est. error"
        )
        logger.info2(" " + 70 * "-")
        for order_idx, (order, factor, mean_abs_inc, error) in enumerate(
            zip(
                range(curr_order + 1, max_order + 1),
                est_rel_factor[min_idx],
                est_mean_abs_inc[min_idx],
                est_error[min_idx],
            )
        ):
            if sum(est_error[min_idx][order_idx:]) == 0.0:
                break
            logger.info2(
                f"  {order:5} |      {factor:>10.4e}      |        "
                f"{mean_abs_inc:>10.4e}        | {error:>10.4e}"
            )
        logger.info2(" " + 70 * "-" + "\n")

        # add screened cluster contribution to error
        mbe_tot_error += tot_error[min_idx]

        return min_idx, mbe_tot_error

    # check if geometric mean absolute increment contribution for minimum error cluster
    # comes close to convergence threshold
    elif (
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
