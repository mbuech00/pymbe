#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
driver module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import logging
import numpy as np
from typing import TYPE_CHECKING

from pymbe.mbe import main as mbe_main
from pymbe.output import (
    main_header,
    mbe_header,
    mbe_results,
    mbe_end,
    screen_results,
    purge_header,
    purge_results,
    purge_end,
)
from pymbe.purge import main as purge_main
from pymbe.tools import n_tuples, min_orbs, inc_dim, inc_shape, write_file

if TYPE_CHECKING:

    from pymbe.parallel import MPICls
    from pymbe.expansion import ExpCls


# get logger
logger = logging.getLogger("pymbe_logger")


def master(mpi: MPICls, exp: ExpCls) -> None:
    """
    this function is the main pymbe master function
    """
    # print expansion headers
    logger.info(main_header(mpi=mpi, method=exp.method))

    # print output from restarted calculation
    if exp.restarted:
        for i in range(exp.min_order, exp.start_order):

            # print mbe header
            logger.info(
                mbe_header(
                    i,
                    exp.n_tuples["calc"][i - exp.min_order],
                    1.0 if i < exp.screen_start else exp.screen_perc,
                )
            )

            # print mbe end
            logger.info(mbe_end(i, exp.time["mbe"][i - exp.min_order]))

            # print mbe results
            logger.info(
                mbe_results(
                    exp.target,
                    exp.fci_state_root,
                    exp.min_order,
                    i,
                    exp.mbe_tot_prop,
                    exp.mean_inc[i - exp.min_order],
                    exp.min_inc[i - exp.min_order],
                    exp.max_inc[i - exp.min_order],
                )
            )

            # print screening results
            exp.screen_orbs = np.setdiff1d(
                exp.exp_space[i - exp.min_order], exp.exp_space[i - exp.min_order + 1]
            )
            if 0 < exp.screen_orbs.size:
                logger.info(screen_results(i, exp.screen_orbs, exp.exp_space))

    # begin or resume mbe expansion depending
    for exp.order in range(exp.start_order, exp.max_order + 1):

        # theoretical and actual number of tuples at current order
        if len(exp.n_tuples["inc"]) == exp.order - exp.min_order:
            min_occ, min_virt = min_orbs(exp.occup, exp.ref_space, exp.vanish_exc)
            exp.n_tuples["theo"].append(
                n_tuples(
                    exp.exp_space[0][exp.exp_space[0] < exp.nocc],
                    exp.exp_space[0][exp.nocc <= exp.exp_space[0]],
                    0,
                    0,
                    exp.order,
                )
            )
            exp.n_tuples["calc"].append(
                n_tuples(
                    exp.exp_space[-1][exp.exp_space[-1] < exp.nocc],
                    exp.exp_space[-1][exp.nocc <= exp.exp_space[-1]],
                    min_occ,
                    min_virt,
                    exp.order,
                )
            )
            exp.n_tuples["inc"].append(exp.n_tuples["calc"][-1])
            if exp.rst:
                write_file(
                    exp.order, np.asarray(exp.n_tuples["theo"][-1]), "mbe_n_tuples_theo"
                )
                write_file(
                    exp.order, np.asarray(exp.n_tuples["calc"][-1]), "mbe_n_tuples_calc"
                )
                write_file(
                    exp.order, np.asarray(exp.n_tuples["inc"][-1]), "mbe_n_tuples_inc"
                )

        # print mbe header
        logger.info(
            mbe_header(
                exp.order,
                exp.n_tuples["calc"][-1],
                1.0 if exp.order < exp.screen_start else exp.screen_perc,
            )
        )

        # main mbe function
        hashes_win, inc_win, tot, mean_inc, min_inc, max_inc = mbe_main(mpi, exp)

        # append window to hashes
        if len(exp.hashes) == len(exp.n_tuples["inc"]):
            exp.hashes[-1] = hashes_win
        else:
            exp.hashes.append(hashes_win)

        # append window to increments
        if len(exp.incs) == len(exp.n_tuples["inc"]):
            exp.incs[-1] = inc_win
        else:
            exp.incs.append(inc_win)

        # append total property
        exp.mbe_tot_prop.append(tot)
        if exp.order > exp.min_order:
            exp.mbe_tot_prop[-1] += exp.mbe_tot_prop[-2]

        # append increment statistics
        if len(exp.mean_inc) > exp.order - exp.min_order:
            exp.mean_inc[-1] = mean_inc
            exp.min_inc[-1] = min_inc
            exp.max_inc[-1] = max_inc
        else:
            exp.mean_inc.append(mean_inc)
            exp.min_inc.append(min_inc)
            exp.max_inc.append(max_inc)

        # print mbe end
        logger.info(mbe_end(exp.order, exp.time["mbe"][-1]))

        # print mbe results
        logger.info(
            mbe_results(
                exp.target,
                exp.fci_state_root,
                exp.min_order,
                exp.order,
                exp.mbe_tot_prop,
                exp.mean_inc[-1],
                exp.min_inc[-1],
                exp.max_inc[-1],
            )
        )

        # update screen_orbs
        if exp.order > exp.min_order:
            exp.screen_orbs = np.setdiff1d(exp.exp_space[-2], exp.exp_space[-1])

        # print screening results
        if 0 < exp.screen_orbs.size:
            logger.info(screen_results(exp.order, exp.screen_orbs, exp.exp_space))

        # print header
        logger.info(purge_header(exp.order))

        # main purging function
        exp.incs, exp.hashes, exp.n_tuples = purge_main(mpi, exp)

        # print purging results
        if exp.order + 1 <= exp.exp_space[-1].size:
            logger.info(purge_results(exp.n_tuples, exp.min_order, exp.order))

        # print purge end
        logger.info(purge_end(exp.order, exp.time["purge"][-1]))

        # write restart files
        if exp.rst:
            if exp.screen_orbs.size > 0:
                for k in range(exp.order - exp.min_order + 1):
                    buf = exp.hashes[k].Shared_query(0)[0]
                    hashes = np.ndarray(
                        buffer=buf,  # type: ignore
                        dtype=np.int64,
                        shape=(exp.n_tuples["inc"][k],),
                    )
                    write_file(k + exp.min_order, hashes, "mbe_hashes")
                    buf = exp.incs[k].Shared_query(0)[0]
                    inc = np.ndarray(
                        buffer=buf,  # type: ignore
                        dtype=np.float64,
                        shape=inc_shape(exp.n_tuples["inc"][k], inc_dim(exp.target)),
                    )
                    write_file(k + exp.min_order, inc, "mbe_inc")
                    write_file(
                        k + exp.min_order,
                        np.asarray(exp.n_tuples["inc"][k]),
                        "mbe_n_tuples_inc",
                    )
            else:
                buf = exp.hashes[-1].Shared_query(0)[0]
                hashes = np.ndarray(
                    buffer=buf,  # type: ignore
                    dtype=np.int64,
                    shape=(exp.n_tuples["inc"][-1],),
                )
                write_file(exp.order, hashes, "mbe_hashes")
                buf = exp.incs[-1].Shared_query(0)[0]
                inc = np.ndarray(
                    buffer=buf,  # type: ignore
                    dtype=np.float64,
                    shape=inc_shape(exp.n_tuples["inc"][-1], inc_dim(exp.target)),
                )
                write_file(exp.order, inc, "mbe_inc")
                write_file(
                    exp.order, np.asarray(exp.n_tuples["inc"][-1]), "mbe_n_tuples_inc"
                )
            write_file(exp.order, np.asarray(exp.mbe_tot_prop[-1]), "mbe_tot")
            write_file(exp.order, np.asarray(exp.time["mbe"][-1]), "mbe_time_mbe")
            write_file(exp.order, np.asarray(exp.time["purge"][-1]), "mbe_time_purge")

        # convergence check
        if exp.exp_space[-1].size < exp.order + 1 or exp.order == exp.max_order:

            # final order
            exp.final_order = exp.order

            # total timing
            exp.time["total"] = [
                mbe + purge for mbe, purge in zip(exp.time["mbe"], exp.time["purge"])
            ]

            # final results
            logger.info("\n\n")

            break

    # wake up slaves
    mpi.global_comm.bcast({"task": "exit"}, root=0)


def slave(mpi: MPICls, exp: ExpCls) -> None:
    """
    this function is the main pymbe slave function
    """
    # set loop/waiting logical
    slave = True

    # enter slave state
    while slave:

        # task id
        msg = mpi.global_comm.bcast(None, root=0)

        if msg["task"] == "mbe":

            # receive order
            exp.order = msg["order"]

            # actual number of tuples at current order
            if len(exp.n_tuples["inc"]) == exp.order - exp.min_order:
                min_occ, min_virt = min_orbs(exp.occup, exp.ref_space, exp.vanish_exc)
                exp.n_tuples["inc"].append(
                    n_tuples(
                        exp.exp_space[-1][exp.exp_space[-1] < exp.nocc],
                        exp.exp_space[-1][exp.nocc <= exp.exp_space[-1]],
                        min_occ,
                        min_virt,
                        exp.order,
                    )
                )

            # main mbe function
            hashes_win, inc_win = mbe_main(
                mpi,
                exp,
                rst_read=msg["rst_read"],
                tup_idx=msg["tup_idx"],
                tup=msg["tup"],
            )

            # append window to hashes
            if len(exp.hashes) == len(exp.n_tuples["inc"]):
                exp.hashes[-1] = hashes_win
            else:
                exp.hashes.append(hashes_win)

            # append window to increments
            if len(exp.incs) == len(exp.n_tuples["inc"]):
                exp.incs[-1] = inc_win
            else:
                exp.incs.append(inc_win)

            # update screen_orbs
            if exp.order == exp.min_order:
                exp.screen_orbs = np.array([], dtype=np.int64)
            else:
                exp.screen_orbs = np.setdiff1d(exp.exp_space[-2], exp.exp_space[-1])

        elif msg["task"] == "purge":

            # receive order
            exp.order = msg["order"]

            # main purging function
            exp.incs, exp.hashes, exp.n_tuples = purge_main(mpi, exp)

        elif msg["task"] == "exit":

            slave = False
