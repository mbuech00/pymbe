#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
output module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from datetime import datetime
from pyscf import symm
from typing import TYPE_CHECKING

from pymbe.tools import git_version, time_str, intervals

if TYPE_CHECKING:
    from typing import List, Dict

    from pymbe.parallel import MPICls


# output parameters
HEADER = f"{('-' * 45):^87}"
DIVIDER = " " + "-" * 92
FILL = " " + "|" * 92
BAR_LENGTH = 50


def main_header(mpi: MPICls) -> str:
    """
    this function prints the main pymbe header
    """
    string: str = "\n\n"
    string += "   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo\n"
    string += "   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8\n"
    string += "    888   .d88' oooo    ooo  888b     d'888   888     888  888        \n"
    string += "    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8   \n"
    string += "    888           `88..8'    8  `888'   888   888    `88b  888    \"  \n"
    string += "    888            `888'     8    Y     888   888    .88P  888       o\n"
    string += "   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8\n"
    string += "                .o..P'                                                \n"
    string += "                `Y8P'                                                 \n"
    string += "\n\n"
    # date & time
    string += (
        f"   -- date & time    : {datetime.now().strftime('%Y-%m-%d & %H:%M:%S'):s}\n"
    )

    # git hash
    string += f"   -- git version    : {git_version():s}\n"

    # mpi info
    if mpi is not None:
        string += "   -- local masters  :\n"
        for master_idx in range(mpi.num_masters):
            string += (
                f"   #### rank / node  : {mpi.master_global_ranks[master_idx]:>6d} / "
                f"{mpi.master_global_hosts[master_idx]:s}\n"
            )

    string += "\n\n"

    return string


def mbe_header(
    order: int, n_incs: int, screen_type: str, screen_perc: float, screen_thres: float
) -> str:
    """
    this function prints the mbe header
    """
    # set string
    string: str = "\n\n" + DIVIDER + "\n"
    string += (
        f" STATUS-{order:d}:  order k = {order:d} MBE started  ---  {n_incs:d} "
        f"tuples in total "
    )
    if screen_type == "fixed":
        string += f"(perc: {screen_perc:.2f})"
    elif screen_type in ["adaptive", "adaptive_truncation"]:
        string += f"(thres: {screen_thres:.0e})"
    string += "\n" + DIVIDER

    return string


def mbe_debug(
    symmetry: str,
    orbsym: np.ndarray,
    nelec_tup: np.ndarray,
    order: int,
    cas_idx: np.ndarray,
    tup: np.ndarray,
) -> str:
    """
    this function prints mbe debug information
    """
    # tuple
    tup_lst = [i for i in tup]

    # symmetry
    try:
        tup_sym = [symm.addons.irrep_id2name(symmetry, i) for i in orbsym[tup]]
    except KeyError:
        tup_sym = [symm.addons.irrep_id2name("C1", i) for i in orbsym[tup]]

    string: str = (
        f" INC-{order:d}: order = {order:d}, tup = {tup_lst:}, space = "
        f"({(nelec_tup[0] + nelec_tup[1]):d}e,{cas_idx.size:d}o)\n"
    )
    string += f"      symmetry = {tup_sym}\n"

    return string


def mbe_status(order: int, prog: float) -> str:
    """
    this function prints the status of an mbe phase
    """
    status: int = int(round(BAR_LENGTH * prog))
    remainder: int = BAR_LENGTH - status

    # set string
    string: str = (
        f" STATUS-{order:d}:   [{'#' * status + '-' * remainder}]   ---  "
        f"{(prog * 100.0):>6.2f} %"
    )

    return string


def mbe_end(order: int, time: float) -> str:
    """
    this function prints the end mbe information
    """
    # set string
    string: str = DIVIDER + "\n"
    string += (
        f" STATUS-{order:d}:  order k = {order:d} MBE done in {time_str(time):s}\n"
    )
    string += DIVIDER

    return string


def redundant_results(
    order: int, n_screen: int, n_van: int, n_calc: int, symm: bool
) -> str:
    """
    this function prints the number of redundant increments
    """
    # set string
    string: str
    if not symm:
        string = f" RESULT-{order:d}:  total number of vanishing increments skipped: "
        string += f"{n_screen - n_calc:}\n"
    else:
        string = f" RESULT-{order:d}:  total number of redundant increments skipped: "
        string += f"{n_screen - n_calc:}\n"
        string += f" RESULT-{order:d}:  {n_screen - n_van:} increments are vanishing "
        string += f"due to occupation\n"
        string += f" RESULT-{order:d}:  {n_van - n_calc:} increments are redundant due "
        string += f"to symmetry\n"
    string += DIVIDER

    return string


def screen_results(
    order: int,
    orbs: np.ndarray,
    n_screen_orbs: int,
    screen_type: str,
    error: float,
) -> str:
    """
    this function prints the screened MOs
    """
    # set string
    string: str = FILL + "\n"
    string += DIVIDER + "\n"
    string += f" RESULT-{order:d}:  screened MOs --- "
    # divide orbs into intervals
    orbs_ints = [i for i in intervals(orbs)]
    for idx, i in enumerate(orbs_ints):
        elms = f"{i[0]:}-{i[1]:}" if len(i) > 1 else f"{i[0]:}"
        if 0 < idx:
            string += f" RESULT-{order:d}:{'':19s}"
        string += f"[{elms}]\n"
    string += DIVIDER + "\n"
    string += f" RESULT-{order:d}:  total number of screened MOs: {n_screen_orbs:}\n"
    if screen_type == "adaptive":
        string += DIVIDER + "\n"
        string += f" RESULT-{order:d}:  total error: {error:>13.4e}\n"
    string += DIVIDER

    return string


def purge_header(order: int) -> str:
    """
    this function prints the purging header
    """
    # set string
    string: str = FILL + "\n"
    string += DIVIDER + "\n"
    string += f" STATUS-{order:d}:  order k = {order:d} purging started\n"
    string += DIVIDER

    return string


def purge_results(
    n_tuples: Dict[str, List[int]], n_incs: List[np.ndarray], min_order: int, order: int
) -> str:
    """
    this function prints the updated number of tuples
    """
    # init string
    string: str = FILL + "\n"
    string += DIVIDER + "\n"
    string += f" RESULT-{order:d}:  after purging of tuples --- "
    for k in range(min_order, order + 1):
        if min_order < k:
            string += f" RESULT-{order:d}:{'':30s}"
        red = (
            (1.0 - np.sum(n_incs[k - min_order]) / n_tuples["prev"][k - min_order])
            * 100.0
            if n_tuples["prev"][k - min_order] > 0
            else 0.0
        )
        string += f"no. of tuples at k = {k:2d} has been reduced by: {red:6.2f} %\n"
    sum_n_tuples_prev = sum(n_tuples["prev"])
    sum_n_tuples_inc = sum(np.sum(order_n_incs) for order_n_incs in n_incs)
    total_red_abs = sum_n_tuples_prev - sum_n_tuples_inc
    total_red_rel = (1.0 - sum_n_tuples_inc / sum_n_tuples_prev) * 100.0
    string += DIVIDER + "\n"
    string += (
        f" RESULT-{order:d}:  total number of reduced tuples: {total_red_abs} "
        f"({total_red_rel:.2f} %)\n"
    )
    string += DIVIDER + "\n"
    string += FILL + "\n"
    string += DIVIDER

    return string


def purge_end(order: int, time: float) -> str:
    """
    this function prints the end purging information
    """
    string: str = (
        f" STATUS-{order:d}:  order k = {order:d} purging done in {time_str(time):s}\n"
    )
    string += DIVIDER

    return string


def ref_space_results(orbs: np.ndarray, ref_space: np.ndarray) -> str:
    """
    this function prints the MOs added to the reference space
    """
    # set string
    string: str = FILL + "\n"
    string += DIVIDER + "\n"
    string += (
        f" STATUS:  MOs added to the reference space: "
        + ", ".join(str(orb) for orb in orbs)
        + "\n"
    )
    string += DIVIDER + "\n"
    string += (
        f" STATUS:  New reference space: "
        + "["
        + ", ".join(str(orb) for orb in ref_space)
        + "]"
        + "\n"
    )
    string += DIVIDER + "\n"
    string += f" STATUS:  Restarting calculation...\n"
    string += DIVIDER + "\n\n\n\n"

    return string
