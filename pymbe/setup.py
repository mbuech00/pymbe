#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
setup module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import sys
import os
import logging
import numpy as np
from json import load, dump
from pyscf import symm, ao2mo
from typing import TYPE_CHECKING, cast

from pymbe.parallel import MPICls, kw_dist, system_dist
from pymbe.tools import RST, logger_config, assertion, ground_state_sym


if TYPE_CHECKING:

    from typing import Dict, Union, Optional, Tuple

    from pymbe.pymbe import MBE


# get logger
logger = logging.getLogger("pymbe_logger")


def main(mbe: MBE) -> MBE:

    # mpi object
    mbe.mpi = MPICls()

    # input handling
    if mbe.mpi.global_master:

        # check for restart folder
        if not os.path.isdir(RST):

            # copy attributes from mol object
            if mbe.mol:

                # number of orbitals
                if mbe.norb is None:
                    mbe.norb = mbe.mol.nao.item()

                # number of electrons
                if mbe.nelec is None:
                    mbe.nelec = mbe.mol.nelec

                # point group
                if mbe.point_group is None and mbe.orb_type != "local":
                    mbe.point_group = mbe.mol.groupname

            # convert number of electrons to numpy array
            if isinstance(mbe.nelec, int):
                mbe.nelec = (-(mbe.nelec // -2), mbe.nelec // 2)
            if mbe.nelec is not None:
                mbe.nelec = np.asarray(mbe.nelec, dtype=np.int64)

            # set default value for point group
            if mbe.point_group is None:
                mbe.point_group = "C1"
            mbe.point_group = symm.addons.std_symb(mbe.point_group)

            # set default value for orbital symmetry
            if (
                mbe.orbsym is None
                and mbe.point_group == "C1"
                and isinstance(mbe.norb, int)
            ):
                mbe.orbsym = np.zeros(mbe.norb, dtype=np.int64)

            # set default value for fci wavefunction state symmetry
            if mbe.fci_state_sym is None:
                if isinstance(mbe.orbsym, np.ndarray) and isinstance(
                    mbe.nelec, np.ndarray
                ):
                    mbe.fci_state_sym = ground_state_sym(
                        mbe.orbsym, mbe.nelec, cast(str, mbe.point_group)
                    )
                else:
                    mbe.fci_state_sym = 0

            # set default value for fci wavefunction state root
            if mbe.fci_state_root is None:
                if mbe.target in ["energy", "dipole", "rdm12"]:
                    mbe.fci_state_root = 0
                elif mbe.target in ["excitation", "trans"]:
                    mbe.fci_state_root = 1

            # prepare integrals
            if isinstance(mbe.eri, np.ndarray) and isinstance(mbe.norb, int):

                # reorder electron repulsion integrals
                mbe.eri = ao2mo.restore(4, mbe.eri, mbe.norb)

            # set default value for base model property
            if mbe.base_prop is None and mbe.base_method is None:
                if mbe.target in ["energy", "excitation"]:
                    mbe.base_prop = 0.0
                elif mbe.target in ["dipole", "trans"]:
                    mbe.base_prop = np.zeros(3, dtype=np.float64)
                elif mbe.target == "rdm12" and mbe.norb is not None:
                    mbe.base_prop = (
                        np.zeros(2 * (mbe.norb,), dtype=np.float64),
                        np.zeros(4 * (mbe.norb,), dtype=np.float64),
                    )

            # create restart folder
            if mbe.rst:
                os.mkdir(RST)

            # restart logical
            mbe.restarted = False

        else:

            # read keywords
            mbe = restart_read_kw(mbe)

            # read system quantities
            mbe = restart_read_system(mbe)

            # restart logical
            mbe.restarted = True

        # configure logger on global master
        logger_config(mbe.verbose)

        # sanity check
        sanity_check(mbe)

    # bcast keywords
    mbe = kw_dist(mbe)

    # write keywords
    if not mbe.restarted and mbe.mpi.global_master and mbe.rst:
        restart_write_kw(mbe)

    # bcast system quantities
    mbe = system_dist(mbe)

    # write system quantities
    if not mbe.restarted and mbe.mpi.global_master and mbe.rst:
        restart_write_system(mbe)

    # configure logging on slaves
    if not mbe.mpi.global_master:
        logger_config(mbe.verbose)

    return mbe


def sanity_check(mbe: MBE) -> None:
    """
    this function performs sanity checks of all mbe attributes
    """
    # expansion model
    assertion(
        isinstance(mbe.method, str),
        "electronic structure method (method keyword argument) must be a string",
    )
    assertion(
        mbe.method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq", "fci"],
        "valid electronic structure methods (method keyword argument) are: ccsd, "
        "ccsd(t), ccsdt, ccsdtq and fci",
    )
    assertion(
        isinstance(mbe.cc_backend, str),
        "coupled-cluster backend (cc_backend keyword argument) must be a string",
    )
    assertion(
        mbe.cc_backend in ["pyscf", "ecc", "ncc"],
        "valid coupled-cluster backends (cc_backend keyword argument) are: pyscf, ecc "
        "and ncc",
    )
    assertion(
        isinstance(mbe.hf_guess, bool),
        "hf initial guess for fci calculations (hf_guess keyword argument) must be a "
        "bool",
    )
    if mbe.method != "fci":
        assertion(
            mbe.hf_guess,
            "non-hf initial guess (hf_guess keyword argument) only valid for fci calcs "
            "(method keyword argument)",
        )
        if mbe.method == "ccsdt":
            assertion(
                mbe.cc_backend != "pyscf",
                "ccsdt (method keyword argument) is not available with the pyscf "
                "coupled-cluster backend (cc_backend keyword argument)",
            )
        if mbe.method == "ccsdtq":
            assertion(
                mbe.cc_backend == "ncc",
                "ccsdtq (method keyword argument) is not available with the pyscf and "
                "ecc coupled-cluster backends (cc_backend keyword argument)",
            )

    # targets
    assertion(
        isinstance(mbe.target, str),
        "expansion target property (target keyword argument) must be a string",
    )
    assertion(
        mbe.target in ["energy", "excitation", "dipole", "trans", "rdm12"],
        "invalid choice for target property (target keyword argument). valid choices "
        "are: energy, excitation energy (excitation), dipole, transition dipole "
        "(trans) and 1- and 2-particle reduced density matrices (rdm12)",
    )
    if mbe.method != "fci":
        assertion(
            mbe.target in ["energy", "dipole", "rdm12"],
            "excited target states (target keyword argument) not implemented for "
            "chosen expansion model (method keyword argument)",
        )
        if mbe.cc_backend in ["ecc", "ncc"]:
            assertion(
                mbe.target == "energy",
                "calculation of targets (target keyword argument) other than energy "
                "are not possible using the ecc and ncc backends (cc_backend keyword "
                "argument)",
            )

    # system
    assertion(
        isinstance(mbe.ncore, int) and mbe.ncore >= 0,
        "number of core orbitals (ncore keyword argument) must be an int >= 0",
    )
    assertion(
        isinstance(mbe.norb, int) and mbe.norb > 0,
        "number of orbitals (norb keyword argument) must be an int > 0",
    )
    assertion(
        (
            isinstance(mbe.nelec, np.ndarray)
            and mbe.nelec.size == 2
            and isinstance(mbe.nelec[0], np.int64)
            and isinstance(mbe.nelec[1], np.int64)
            and (mbe.nelec[0] > 0 or mbe.nelec[1] > 0)
        ),
        "number of electrons (nelec keyword argument) must be an int > 0 or a tuple of "
        "ints > 0 with dimension 2 or a np.ndarray of ints > 0 with dimension 2",
    )
    if cast(np.ndarray, mbe.nelec)[0] != cast(np.ndarray, mbe.nelec)[1]:
        if mbe.method != "fci" or mbe.base_method is not None:
            assertion(
                mbe.cc_backend == "pyscf",
                "the ecc and ncc backends (cc_backend keyword argument) are designed "
                "for closed-shell systems only",
            )
            logger.warning(
                "Warning: All open-shell CC calculations with the pyscf backend "
                "estimate the unrestricted CC property on the basis of a ROHF "
                "reference function instead of the fully restricted CC property."
            )
        assertion(
            mbe.target != "rdm12",
            "1- and 2-particle reduced density matrix calculations are currently not "
            "implemented for open-shell systems.",
        )
    assertion(
        isinstance(mbe.point_group, str),
        "symmetry (point_group keyword argument) must be a str",
    )
    assertion(
        isinstance(mbe.orbsym, np.ndarray),
        "orbital symmetry (orbsym keyword argument) must be a np.ndarray",
    )
    assertion(
        isinstance(mbe.fci_state_sym, (str, int)),
        "state wavefunction symmetry (fci_state_sym keyword argument) must be a str or "
        "int",
    )
    if isinstance(mbe.fci_state_sym, str):
        try:
            mbe.fci_state_sym = symm.addons.irrep_name2id(
                mbe.point_group, mbe.fci_state_sym
            )
        except Exception as err:
            raise ValueError(
                "illegal choice of state wavefunction symmetry (fci_state_sym keyword "
                f"argument) -- PySCF error: {err}"
            )
    assertion(
        isinstance(mbe.fci_state_root, int) and mbe.fci_state_root >= 0,
        "target state (root keyword argument) must be an int >= 0",
    )
    hf_wfnsym = ground_state_sym(
        cast(np.ndarray, mbe.orbsym),
        cast(np.ndarray, mbe.nelec),
        cast(str, mbe.point_group),
    )
    if mbe.method == "fci" and mbe.hf_guess:
        assertion(
            mbe.fci_state_sym == hf_wfnsym,
            "illegal choice of fci state wavefunction symmetry (fci_state_sym keyword "
            "argument) when enforcing hf initial guess (hf_guess keyword argument) "
            "because fci_state_sym does not equal hf state symmetry",
        )
    if mbe.method != "fci" or mbe.base_method is not None:
        assertion(
            mbe.fci_state_sym == hf_wfnsym,
            "illegal choice of fci state wavefunction symmetry (fci_state_sym keyword "
            "argument) for chosen expansion model (method or base_method keyword "
            "argument) because fci_state_sym does not equal hf state symmetry",
        )
        assertion(
            mbe.fci_state_root == 0,
            "excited target states (root keyword argument) not implemented for chosen "
            "expansion model (method or base_method keyword argument)",
        )
    if mbe.target in ["excitation", "trans"]:
        assertion(
            cast(int, mbe.fci_state_root) > 0,
            "calculation of excitation energies or transition dipole moments (target "
            "keyword argument) requires target state root (state_root keyword "
            "argument) >= 1",
        )

    # orbital representation
    assertion(
        isinstance(mbe.orb_type, str),
        "orbital representation (orb_type keyword argument) must be a string",
    )
    assertion(
        mbe.orb_type in ["can", "local", "ccsd", "ccsd(t)", "casscf"],
        "valid orbital representations (orb_type keyword argument) are currently: "
        "canonical (can), pipek-mezey (local), natural (ccsd or ccsd(t) or casscf "
        "orbs (casscf))",
    )
    if mbe.orb_type == "local":
        assertion(
            mbe.point_group == "C1",
            "the combination of local orbitals (orb_type keyword argument) and point "
            "group symmetry (point_group keyword argument) different from c1 is not "
            "allowed",
        )

    # integrals
    assertion(
        isinstance(mbe.hcore, np.ndarray),
        "core hamiltonian integral (hcore keyword argument) must be a np.ndarray",
    )
    assertion(
        isinstance(mbe.eri, np.ndarray),
        "electron repulsion integral (eri keyword argument) must be a np.ndarray",
    )
    if mbe.target in ["dipole", "trans"]:
        assertion(
            isinstance(mbe.dipole_ints, np.ndarray),
            "dipole integrals (dipole_ints keyword argument) must be a np.ndarray",
        )

    # reference space
    assertion(
        isinstance(mbe.ref_space, np.ndarray),
        "reference space (ref_space keyword argument) must be a np.ndarray of orbital "
        "indices",
    )
    if cast(np.ndarray, mbe.nelec)[0] != cast(np.ndarray, mbe.nelec)[1]:
        assertion(
            np.all(
                np.isin(
                    np.arange(
                        np.amin(cast(np.ndarray, mbe.nelec)),
                        np.amax(cast(np.ndarray, mbe.nelec)),
                    ),
                    mbe.ref_space,
                )
            ),
            "all partially occupied orbitals have to be included in the reference "
            "space (ref_space keyword argument)",
        )

    # base model
    assertion(
        isinstance(mbe.base_method, (str, type(None))),
        "base model electronic structure method (base_method keyword argument) must be "
        "a str or None",
    )
    if mbe.base_method is not None:
        assertion(
            mbe.base_method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"],
            "valid base model electronic structure methods (base_method keyword "
            "argument) are currently: ccsd, ccsd(t), ccsdt and ccsdtq",
        )
        if mbe.base_method == "ccsdt":
            assertion(
                mbe.cc_backend != "pyscf",
                "ccsdt (base_method keyword argument) is not available with pyscf "
                "coupled-cluster backend (cc_backend keyword argument)",
            )
        if mbe.base_method == "ccsdtq":
            assertion(
                mbe.cc_backend == "ncc",
                "ccsdtq (base_method keyword argument) is not available with pyscf and "
                "ecc coupled-cluster backends (cc_backend keyword argument)",
            )
        assertion(
            mbe.target in ["energy", "dipole", "rdm12"],
            "excited target states (target keyword argument) not implemented for base "
            "model calculations (base_method keyword argument)",
        )
        if mbe.cc_backend in ["ecc", "ncc"]:
            assertion(
                mbe.target == "energy",
                "calculation of targets (target keyword argument) other than energy "
                "are not possible using the ecc and ncc coupled-cluster backends "
                "(cc_backend keyword argument)",
            )
        assertion(
            mbe.fci_state_root == 0,
            "excited target states (root keyword argument) not implemented for base "
            "model (base_method keyword argument)",
        )
        if mbe.target == "energy":
            assertion(
                isinstance(mbe.base_prop, float),
                "base model energy (base_prop keyword argument) must be a float",
            )
        elif mbe.target == "dipole":
            assertion(
                isinstance(mbe.base_prop, np.ndarray),
                "base model dipole moment (base_prop keyword argument) must be a "
                "np.ndarray",
            )
        elif mbe.target == "rdm12":
            assertion(
                isinstance(mbe.base_prop, tuple)
                and len(mbe.base_prop) == 2
                and isinstance(mbe.base_prop[0], np.ndarray)
                and isinstance(mbe.base_prop[1], np.ndarray),
                "base model 1- and 2-particle density matrices (base_prop keyword argument) "
                "must be a tuple of np.ndarray with dimension 2",
            )

    # screening
    assertion(
        isinstance(mbe.screen_start, int) and mbe.screen_start >= 2,
        "screening start order (screen_start keyword argument) must be an int >= 2",
    )
    assertion(
        isinstance(mbe.screen_perc, float) and mbe.screen_perc <= 1.0,
        "screening threshold (screen_perc keyword argument) must be a float <= 1.",
    )
    if mbe.max_order is not None:
        assertion(
            isinstance(mbe.max_order, int) and mbe.max_order >= 1,
            "maximum expansion order (max_order keyword argument) must be an int >= 1",
        )

    # restart
    assertion(
        isinstance(mbe.rst, bool),
        "restart logical (rst keyword argument) must be a bool",
    )
    assertion(
        isinstance(mbe.rst_freq, int) and mbe.rst_freq >= 1,
        "restart frequency (rst_freq keyword argument) must be an int >= 1",
    )

    # verbose
    assertion(
        isinstance(mbe.verbose, int) and mbe.verbose >= 0,
        "verbose option (verbose keyword argument) must be an int >= 0",
    )

    # pi pruning
    assertion(
        isinstance(mbe.pi_prune, bool),
        "pruning of pi-orbitals (pi_prune keyword argument) must be a bool",
    )
    if mbe.pi_prune:
        assertion(
            mbe.point_group in ["D2h", "C2v"],
            "pruning of pi-orbitals (pi_prune keyword argument) is only implemented "
            "for linear D2h and C2v symmetries (point_group keyword argument)",
        )
        assertion(
            isinstance(mbe.orbsym_linear, np.ndarray),
            "linear point group orbital symmetry (orbsym_linear keyword argument) must "
            "be a np.ndarray",
        )


def restart_write_kw(mbe: MBE) -> None:
    """
    this function writes the keyword restart file
    """
    # define keywords
    keywords = {
        "method": mbe.method,
        "cc_backend": mbe.cc_backend,
        "hf_guess": mbe.hf_guess,
        "target": mbe.target,
        "point_group": mbe.point_group,
        "fci_state_sym": mbe.fci_state_sym,
        "fci_state_root": mbe.fci_state_root,
        "orb_type": mbe.orb_type,
        "base_method": mbe.base_method,
        "screen_start": mbe.screen_start,
        "screen_perc": mbe.screen_perc,
        "max_order": mbe.max_order,
        "rst": mbe.rst,
        "rst_freq": mbe.rst_freq,
        "verbose": mbe.verbose,
        "pi_prune": mbe.pi_prune,
    }

    # write keywords
    with open(os.path.join(RST, "keywords.rst"), "w") as f:
        dump(keywords, f)


def restart_read_kw(mbe: MBE) -> MBE:
    """
    this function reads the keyword restart file
    """
    # read keywords
    with open(os.path.join(RST, "keywords.rst"), "r") as f:
        keywords = load(f)

    # set keywords as MBE attributes
    for key, val in keywords.items():
        setattr(mbe, key, val)

    return mbe


def restart_write_system(mbe: MBE) -> None:
    """
    this function writes all system quantities restart files
    """
    # define system quantities
    system: Dict[str, Union[Optional[int], Tuple[int, int], np.ndarray, float]] = {
        "ncore": mbe.ncore,
        "norb": mbe.norb,
        "nelec": mbe.nelec,
        "orbsym": mbe.orbsym,
        "hcore": mbe.hcore,
        "eri": mbe.eri,
        "ref_space": mbe.ref_space,
    }

    if isinstance(mbe.base_prop, (float, np.ndarray)):
        system["base_prop"] = mbe.base_prop
    elif isinstance(mbe.base_prop, tuple):
        system["base_prop1"] = mbe.base_prop[0]
        system["base_prop2"] = mbe.base_prop[1]

    if mbe.dipole_ints is not None:
        system["dipole_ints"] = mbe.dipole_ints

    if mbe.orbsym_linear is not None:
        system["orbsym_linear"] = mbe.orbsym_linear

    # write system quantities
    np.savez(os.path.join(RST, "system"), **system)  # type: ignore


def restart_read_system(mbe: MBE) -> MBE:
    """
    this function reads all system quantities restart files
    """
    # read system quantities
    system_npz = np.load(os.path.join(RST, "system.npz"))

    # create system dictionary
    system = {}
    for file in system_npz.files:
        system[file] = system_npz[file]

    # close npz object
    system_npz.close()

    # define scalar values
    scalars = ["ncore", "norb"]

    if mbe.target in ["energy", "excitation"]:
        scalars.append("base_prop")
    elif mbe.target == "rdm12":
        system["base_prop"] = (system.pop("base_prop1"), system.pop("base_prop2"))

    # convert to scalars
    for scalar in scalars:
        system[scalar] = system[scalar].item()

    # set system quantities as MBE attributes
    for key, val in system.items():
        setattr(mbe, key, val)

    return mbe


def settings() -> None:
    """
    this function sets and asserts some general settings
    """
    # only run with python3+
    assertion(3 <= sys.version_info[0], "PyMBE only runs under python3+")

    # PYTHONHASHSEED = 0
    pythonhashseed = os.environ.get("PYTHONHASHSEED", -1)
    assertion(
        int(pythonhashseed) == 0,
        "environment variable PYTHONHASHSEED must be set to zero",
    )
