#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
interface module
"""

from __future__ import annotations

__author__ = "Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import os
import ctypes
import numpy as np
from math import floor, log10
from pyscf import ao2mo
from typing import TYPE_CHECKING

from pymbe.expansion import CONV_TOL

try:
    from pymbe.settings import MBECCLIB

    cclib = ctypes.cdll.LoadLibrary(MBECCLIB)
    CCLIB_AVAILABLE = True
except (ImportError, OSError):
    CCLIB_AVAILABLE = False

if TYPE_CHECKING:
    from typing import Tuple


MAX_MEM = 131071906


def mbecc_interface(
    method: str,
    cc_backend: str,
    orb_type: str,
    point_group: str,
    orbsym: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    nelec: np.ndarray,
    higher_amp_extrap: bool,
    verbose: int,
) -> Tuple[float, int]:
    """
    this function returns the results of a cc calculation using the mbecc interface
    """
    # check for path to MBECC library
    if not CCLIB_AVAILABLE:
        msg = (
            "settings.py not found for module interface. Please create "
            + f"{os.path.join(os.path.dirname(__file__), 'settings.py'):}\n"
        )
        raise ModuleNotFoundError(msg)

    # method keys in cfour
    method_dict = {"ccsd": 10, "ccsd(t)": 22, "ccsdt": 18, "ccsdtq": 46}

    # cc module
    cc_module_dict = {"ecc": 0, "ncc": 1}

    # point group
    point_group_dict = {
        "C1": 1,
        "C2": 2,
        "Ci": 3,
        "Cs": 4,
        "D2": 5,
        "C2v": 6,
        "C2h": 7,
        "D2h": 8,
    }

    # settings
    method_val = ctypes.c_int64(method_dict[method])
    cc_module_val = ctypes.c_int64(cc_module_dict[cc_backend])
    point_group_val = ctypes.c_int64(
        point_group_dict[point_group] if orb_type != "local" else 1
    )
    non_canonical = ctypes.c_int64(0 if orb_type == "can" else 1)
    maxcor = ctypes.c_int64(MAX_MEM)  # max memory in integer words
    conv = ctypes.c_int64(-int(floor(log10(abs(CONV_TOL)))))
    max_cycle = ctypes.c_int64(500)
    t3_extrapol = ctypes.c_int64(1 if higher_amp_extrap else 0)
    t4_extrapol = ctypes.c_int64(1 if higher_amp_extrap else 0)
    verbose_val = ctypes.c_int64(1 if verbose >= 3 else 0)

    n_act = orbsym.size
    h2e = ao2mo.restore(1, h2e, n_act)

    # initialize variables
    n_act_val = ctypes.c_int64(n_act)  # number of orbitals
    cc_energy = ctypes.c_double()  # cc-energy output
    success = ctypes.c_int64()  # success flag

    # perform cc calculation
    cclib.cc_interface(
        ctypes.byref(method_val),
        ctypes.byref(cc_module_val),
        ctypes.byref(non_canonical),
        ctypes.byref(maxcor),
        nelec.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(n_act_val),
        orbsym.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(point_group_val),
        h1e.ctypes.data_as(ctypes.c_void_p),
        h2e.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(conv),
        ctypes.byref(max_cycle),
        ctypes.byref(t3_extrapol),
        ctypes.byref(t4_extrapol),
        ctypes.byref(verbose_val),
        ctypes.byref(cc_energy),
        ctypes.byref(success),
    )

    # convergence check
    if success.value != 1:
        # redo calculation in debug mode if not converged
        verbose_val = ctypes.c_int64(1)

        cclib.cc_interface(
            ctypes.byref(method_val),
            ctypes.byref(cc_module_val),
            ctypes.byref(non_canonical),
            ctypes.byref(maxcor),
            nelec.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(n_act_val),
            orbsym.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(point_group_val),
            h1e.ctypes.data_as(ctypes.c_void_p),
            h2e.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(conv),
            ctypes.byref(max_cycle),
            ctypes.byref(t3_extrapol),
            ctypes.byref(t4_extrapol),
            ctypes.byref(verbose_val),
            ctypes.byref(cc_energy),
            ctypes.byref(success),
        )

    return cc_energy.value, success.value
