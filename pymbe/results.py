#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from typing import TYPE_CHECKING

try:
    import matplotlib

    PLT_FOUND = True
except (ImportError, OSError):
    pass
    PLT_FOUND = False
if PLT_FOUND:
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    try:
        import seaborn as sns

        SNS_FOUND = True
    except (ImportError, OSError):
        pass
        SNS_FOUND = False

from pymbe.tools import time_str

if TYPE_CHECKING:
    from typing import Dict, List

    from pymbe.expansion import ExpCls


# results parameters
DIVIDER = f"{('-' * 137):^143}"


def _time(time: Dict[str, List[float]], comp: str, idx: int) -> str:
    """
    this function returns the final timings in (HHH : MM : SS) format
    """
    # init time
    if comp in ["mbe", "purge"]:
        req_time = time[comp][idx]
    elif comp == "sum":
        req_time = time["mbe"][idx] + time["purge"][idx]
    elif comp in ["tot_mbe", "tot_purge"]:
        req_time = np.sum(time[comp[4:]])
    elif comp == "tot_sum":
        req_time = np.sum(time["mbe"]) + np.sum(time["purge"])
    return time_str(req_time)


def timings_prt(
    method: str,
    min_order: int,
    final_order: int,
    n_tuples: Dict[str, List[int]],
    time: Dict[str, List[float]],
) -> str:
    """
    this function returns the timings table
    """
    string: str = DIVIDER[:106] + "\n"
    string += f"{f'MBE-{method.upper()} timings':^106}\n"

    string += DIVIDER[:106] + "\n"
    string += (
        f"{'':3}{'MBE order':^14}{'|':1}{'MBE':^18}{'|':1}{'purging':^18}{'|':1}"
        f"{'sum':^18}{'|':1}{'calculations':^18}{'|':1}{'in %':^13}\n"
    )

    string += DIVIDER[:106] + "\n"
    for i, j in enumerate(range(min_order, final_order + 1)):
        calc_i = n_tuples["calc"][i]
        rel_i = n_tuples["calc"][i] / n_tuples["theo"][i] * 100.0
        calc_tot = sum(n_tuples["calc"][: i + 1])
        rel_tot = calc_tot / sum(n_tuples["theo"][: i + 1]) * 100.0
        string += (
            f"{'':3}{j:>8d}{'':6}{'|':1}"
            f"{_time(time, 'mbe', i):>16s}{'':2}{'|':1}"
            f"{_time(time, 'purge', i):>16s}{'':2}{'|':1}"
            f"{_time(time, 'sum', i):>16s}{'':2}{'|':1}"
            f"{calc_i:>16d}{'':2}{'|':1}"
            f"{rel_i:>10.2f}\n"
        )

    string += DIVIDER[:106] + "\n"
    string += (
        f"{'':3}{'total':^14s}{'|':1}"
        f"{_time(time, 'tot_mbe', -1):>16s}{'':2}{'|':1}"
        f"{_time(time, 'tot_purge', -1):>16s}{'':2}{'|':1}"
        f"{_time(time, 'tot_sum', -1):>16s}{'':2}{'|':1}"
        f"{calc_tot:>16d}{'':2}{'|':1}"
        f"{rel_tot:>10.2f}\n"
    )

    string += DIVIDER[:106] + "\n"

    return string


def results_plt(
    prop: np.ndarray,
    min_order: int,
    final_order: int,
    marker: str,
    color: str,
    label: str,
    ylabel: str,
) -> matplotlib.figure.Figure:
    """
    this function plots the target property
    """
    # check if matplotlib is available
    if not PLT_FOUND:
        raise ModuleNotFoundError("No module named matplotlib")

    # set seaborn
    if SNS_FOUND:
        sns.set(style="darkgrid", palette="Set2", font="DejaVu Sans")

    # set subplot
    fig, ax = plt.subplots()

    # plot results
    ax.plot(
        np.arange(min_order, final_order + 1),
        prop,
        marker=marker,
        linewidth=2,
        mew=1,
        color=color,
        linestyle="-",
        label=label,
    )

    # set x limits
    ax.set_xlim([0.5, final_order + 1 - 0.5])

    # turn off x-grid
    ax.xaxis.grid(False)

    # set labels
    ax.set_xlabel("Expansion order")
    ax.set_ylabel(ylabel)

    # force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # despine
    if SNS_FOUND:
        sns.despine()

    # set legend
    ax.legend(loc=1, frameon=False)

    return fig
