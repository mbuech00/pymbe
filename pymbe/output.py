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

    from typing import List, Tuple, Dict, Union, Any, Optional

    from pymbe.parallel import MPICls


# output parameters
HEADER = f"{('-' * 45):^87}"
DIVIDER = " " + "-" * 92
FILL = " " + "|" * 92
BAR_LENGTH = 50


def main_header(mpi: Optional[MPICls] = None, method: Optional[str] = None) -> str:
    """
    this function prints the main pymbe header
    """
    string: str = (
        "\n\n   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo\n"
    )
    string += "   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8\n"
    string += "    888   .d88' oooo    ooo  888b     d'888   888     888  888\n"
    string += "    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8\n"
    string += "    888           `88..8'    8  `888'   888   888    `88b  888    \"\n"
    string += "    888            `888'     8    Y     888   888    .88P  888       o\n"
    string += "   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8\n"
    string += "                .o..P'\n"
    string += "                `Y8P'\n\n\n"
    # date & time
    string += (
        f"   -- date & time   : {datetime.now().strftime('%Y-%m-%d & %H:%M:%S'):s}\n"
    )

    # git hash
    string += f"   -- git version   : {git_version():s}\n"

    # mpi info
    if mpi is not None:
        string += "   -- local masters :\n"
        for master_idx in range(mpi.num_masters):
            string += (
                f"   #### rank / node : {mpi.master_global_ranks[master_idx]:>6d} / "
                f"{mpi.master_global_hosts[master_idx]:s}\n"
            )

    string += "\n\n"

    # method
    if method is not None:

        string += HEADER + "\n"
        string += f"{method.upper() + ' expansion':^87s}\n"
        string += HEADER

    return string


def mbe_header(order: int, n_tuples: int, thres: float) -> str:
    """
    this function prints the mbe header
    """
    # set string
    string: str = "\n\n" + DIVIDER + "\n"
    string += (
        f" STATUS-{order:d}:  order k = {order:d} MBE started  ---  {n_tuples:d} "
        f"tuples in total (thres: {thres:.2f})\n"
    )
    string += DIVIDER

    return string


def mbe_debug(
    symmetry: str,
    orbsym: np.ndarray,
    root: int,
    nelec_tup: Tuple[int, int],
    inc_tup: Union[float, np.ndarray],
    order: int,
    cas_idx: np.ndarray,
    tup: np.ndarray,
) -> str:
    """
    this function prints mbe debug information
    """
    # symmetry
    tup_lst = [i for i in tup]

    tup_sym = [symm.addons.irrep_id2name(symmetry, i) for i in orbsym[tup]]

    string: str = (
        f" INC-{order:d}: order = {order:d}, tup = {tup_lst:}, space = "
        f"({(nelec_tup[0] + nelec_tup[1]):d}e,{cas_idx.size:d}o)\n"
    )
    string += f"      symmetry = {tup_sym}\n"

    if isinstance(inc_tup, float):
        string += f"      increment for root {root:d} = {inc_tup:.4e}\n"
    else:
        string += (
            f"      increment for root {root:d} = ({inc_tup[0]:.4e}, {inc_tup[1]:.4e}, "
            f"{inc_tup[2]:.4e})\n"
        )

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


def mbe_results(
    target: str,
    root: int,
    min_order: int,
    order: int,
    prop_tot: List[np.ndarray],
    mean_inc: np.ndarray,
    min_inc: np.ndarray,
    max_inc: np.ndarray,
) -> str:
    """
    this function prints mbe results statistics
    """
    # calculate total inc
    if target in ["energy", "excitation"]:
        if order == min_order:
            tot_inc = prop_tot[order - min_order].item()
        else:
            tot_inc = (
                prop_tot[order - min_order] - prop_tot[order - min_order - 1]
            ).item()
    elif target in ["dipole", "trans"]:
        if order == min_order:
            tot_inc = np.linalg.norm(prop_tot[order - min_order])
        else:
            tot_inc = np.linalg.norm(prop_tot[order - min_order]) - np.linalg.norm(
                prop_tot[order - min_order - 1]
            )

    # set header
    if target == "energy":
        header = f"energy for root {root} (total increment = {tot_inc:.4e})"
    elif target == "excitation":
        header = f"excitation energy for root {root} (total increment = {tot_inc:.4e})"
    elif target == "dipole":
        header = f"dipole moment for root {root} (total increment = {tot_inc:.4e})"
    elif target == "trans":
        header = (
            f"transition dipole moment for excitation 0 -> {root} (total increment = "
            f"{tot_inc:.4e})"
        )
    # set string
    string: str = FILL + "\n"
    string += DIVIDER + "\n"
    string += f" RESULT-{order:d}:{header:^81}\n"
    string += DIVIDER + "\n"

    if target in ["energy", "excitation"]:

        # set string
        string += DIVIDER + "\n"
        string += (
            f" RESULT-{order:d}:      mean increment     |      "
            "min. abs. increment     |     max. abs. increment\n"
        )
        string += DIVIDER + "\n"
        string += (
            f" RESULT-{order:d}:     {mean_inc.item():>13.4e}       |        "
            f"{min_inc.item():>13.4e}         |       {max_inc.item():>13.4e}\n"
        )

    elif target in ["dipole", "trans"]:

        # set components
        string += DIVIDER
        comp = ("x-component", "y-component", "z-component")

        # loop over x, y, and z
        for k in range(3):

            # set string
            string += f"\n RESULT-{order:d}:{comp[k]:^81}\n"
            string += DIVIDER + "\n"
            string += (
                f" RESULT-{order:d}:      mean increment     |      "
                "min. abs. increment     |     max. abs. increment\n"
            )
            string += DIVIDER + "\n"
            string += (
                f" RESULT-{order:d}:     {mean_inc[k]:>13.4e}       |        "
                f"{min_inc[k]:>13.4e}         |       {max_inc[k]:>13.4e}\n"
            )
            if k < 2:
                string += DIVIDER

    # set string
    string += DIVIDER + "\n"
    string += FILL + "\n"
    string += DIVIDER

    return string


def screen_results(order: int, orbs: np.ndarray, exp_space: List[np.ndarray]) -> str:
    """
    this function prints the screened MOs
    """
    # init string
    string: str = f" RESULT-{order:d}:  screened MOs --- "
    # divide orbs into intervals
    orbs_ints = [i for i in intervals(orbs)]
    for idx, i in enumerate(orbs_ints):
        elms = f"{i[0]:}-{i[1]:}" if len(i) > 1 else f"{i[0]:}"
        if 0 < idx:
            string += f" RESULT-{order:d}:{'':19s}"
        string += f"[{elms}]\n"
    total_screen = np.setdiff1d(exp_space[0], exp_space[-1])
    string += DIVIDER + "\n"
    string += (
        f" RESULT-{order:d}:  total number of screened MOs: {total_screen.size:}\n"
    )
    string += DIVIDER + "\n"
    string += FILL + "\n"
    string += DIVIDER

    return string


def purge_header(order: int) -> str:
    """
    this function prints the purging header
    """
    # set string
    string: str = f" STATUS-{order:d}:  order k = {order:d} purging started\n"
    string += DIVIDER

    return string


def purge_results(n_tuples: Dict[str, List[int]], min_order: int, order: int) -> str:
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
            1.0 - n_tuples["inc"][k - min_order] / n_tuples["theo"][k - min_order]
        ) * 100.0
        string += f"no. of tuples at k = {k:2d} has been reduced by: {red:6.2f} %\n"
    total_red_abs = sum(n_tuples["theo"]) - sum(n_tuples["inc"])
    total_red_rel = (1.0 - sum(n_tuples["inc"]) / sum(n_tuples["theo"])) * 100.0
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
