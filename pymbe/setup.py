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

    from typing import Dict, Any

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
                    if isinstance(mbe.mol.nao, int):
                        mbe.norb = mbe.mol.nao
                    else:
                        mbe.norb = mbe.mol.nao.item()

                # number of electrons
                if mbe.nelec is None:
                    mbe.nelec = mbe.mol.nelec

                # point group
                if mbe.point_group is None:
                    if mbe.orb_type == "local":
                        mbe.point_group = mbe.mol.topgroup
                    else:
                        mbe.point_group = mbe.mol.groupname

            # convert number of electrons to numpy array
            if isinstance(mbe.nelec, int):
                mbe.nelec = np.asarray((mbe.nelec // 2, mbe.nelec // 2), dtype=np.int64)
            elif isinstance(mbe.nelec, tuple):
                mbe.nelec = np.asarray(mbe.nelec, dtype=np.int64)
            elif isinstance(mbe.nelec, list):
                mbe.nelec = [
                    np.asarray(state, dtype=np.int64)
                    if isinstance(state, tuple)
                    else state
                    for state in mbe.nelec
                ]

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
                if isinstance(mbe.orbsym, np.ndarray):
                    if isinstance(mbe.nelec, np.ndarray):
                        mbe.fci_state_sym = ground_state_sym(
                            mbe.orbsym, mbe.nelec, cast(str, mbe.point_group)
                        )
                    elif isinstance(mbe.nelec, list):
                        mbe.fci_state_sym = [
                            ground_state_sym(
                                mbe.orbsym,
                                cast(np.ndarray, state),
                                cast(str, mbe.point_group),
                            )
                            for state in mbe.nelec
                        ]
                else:
                    mbe.fci_state_sym = 0

            # set default value for fci wavefunction state root
            if mbe.fci_state_root is None:
                if mbe.target in ["energy", "dipole", "rdm12", "genfock"]:
                    mbe.fci_state_root = 0
                elif mbe.target in ["excitation", "trans"]:
                    mbe.fci_state_root = 1

            # set default values for state-averaged rdm12 and genfock calculations
            if mbe.target in ["rdm12", "genfock"]:
                if isinstance(mbe.nelec, list):
                    mbe.hf_guess = False
                    if isinstance(mbe.fci_state_sym, int):
                        mbe.fci_state_sym = [
                            mbe.fci_state_sym for _ in range(len(mbe.nelec))
                        ]
                    if isinstance(mbe.fci_state_root, int):
                        mbe.fci_state_root = [
                            mbe.fci_state_root for _ in range(len(mbe.nelec))
                        ]
                    if mbe.fci_state_weights is None:
                        mbe.fci_state_weights = [
                            1 / len(mbe.nelec) for _ in range(len(mbe.nelec))
                        ]
                elif isinstance(mbe.fci_state_sym, list):
                    mbe.hf_guess = False
                    if isinstance(mbe.nelec, np.ndarray):
                        mbe.nelec = [mbe.nelec for _ in range(len(mbe.fci_state_sym))]
                    if isinstance(mbe.fci_state_root, int):
                        mbe.fci_state_root = [
                            mbe.fci_state_root for _ in range(len(mbe.fci_state_sym))
                        ]
                    if mbe.fci_state_weights is None:
                        mbe.fci_state_weights = [
                            1 / len(mbe.fci_state_sym)
                            for _ in range(len(mbe.fci_state_sym))
                        ]
                elif isinstance(mbe.fci_state_root, list):
                    mbe.hf_guess = False
                    if isinstance(mbe.nelec, np.ndarray):
                        mbe.nelec = [mbe.nelec for _ in range(len(mbe.fci_state_root))]
                    if isinstance(mbe.fci_state_sym, int):
                        mbe.fci_state_sym = [
                            mbe.fci_state_sym for _ in range(len(mbe.fci_state_root))
                        ]
                    if mbe.fci_state_weights is None:
                        mbe.fci_state_weights = [
                            1 / len(mbe.fci_state_root)
                            for _ in range(len(mbe.fci_state_root))
                        ]
                elif isinstance(mbe.fci_state_weights, list):
                    mbe.hf_guess = False
                    if isinstance(mbe.nelec, np.ndarray):
                        mbe.nelec = [
                            mbe.nelec for _ in range(len(mbe.fci_state_weights))
                        ]
                    if isinstance(mbe.fci_state_sym, int):
                        mbe.fci_state_sym = [
                            mbe.fci_state_sym for _ in range(len(mbe.fci_state_weights))
                        ]
                    if isinstance(mbe.fci_state_root, int):
                        mbe.fci_state_root = [
                            mbe.fci_state_root
                            for _ in range(len(mbe.fci_state_weights))
                        ]

            # prepare integrals
            if isinstance(mbe.eri, np.ndarray) and isinstance(mbe.norb, int):

                # reorder electron repulsion integrals
                mbe.eri = ao2mo.restore(4, mbe.eri, mbe.norb)

            # set default value for expansion space
            if mbe.exp_space is None and isinstance(mbe.norb, int):
                mbe.exp_space = np.array(
                    [i for i in range(mbe.norb) if i not in mbe.ref_space],
                    dtype=np.int64,
                )

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
                elif (
                    mbe.target == "genfock"
                    and isinstance(mbe.full_nocc, int)
                    and isinstance(mbe.norb, int)
                    and isinstance(mbe.full_norb, int)
                ):
                    mbe.base_prop = (
                        0.0,
                        np.zeros(
                            (mbe.full_nocc + mbe.norb, mbe.full_norb), dtype=np.float64
                        ),
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
        mbe.target in ["energy", "excitation", "dipole", "trans", "rdm12", "genfock"],
        "invalid choice for target property (target keyword argument). valid choices "
        "are: energy, excitation energy (excitation), dipole, transition dipole "
        "(trans), 1- and 2-particle reduced density matrices (rdm12) and generalized "
        "Fock matrix (genfock)",
    )
    if mbe.method != "fci":
        assertion(
            mbe.target in ["energy", "dipole", "rdm12", "genfock"],
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
        isinstance(mbe.norb, int) and mbe.norb > 0,
        "number of orbitals (norb keyword argument) must be an int > 0",
    )
    assertion(
        (
            (
                isinstance(mbe.nelec, np.ndarray)
                and mbe.nelec.size == 2
                and mbe.nelec.dtype == np.int64
                and (mbe.nelec[0] > 0 or mbe.nelec[1] > 0)
            )
            or (
                isinstance(mbe.nelec, list)
                and all(
                    [
                        isinstance(state, np.ndarray)
                        and state.size == 2
                        and state.dtype == np.int64
                        and (state[0] > 0 or state[1] > 0)
                        for state in mbe.nelec
                    ]
                )
            )
        ),
        "number of electrons (nelec keyword argument) must be an int > 0 or a tuple of "
        "ints > 0 with dimension 2 or a np.ndarray of ints > 0 with dimension 2 or a "
        "list of tuples of ints > 0 with dimension 2 or a list of np.ndarray of "
        "ints > 0 with dimension 2",
    )
    if (isinstance(mbe.nelec, np.ndarray) and mbe.nelec[0] != mbe.nelec[1]) or (
        isinstance(mbe.nelec, list)
        and any([state[0] != state[1] for state in mbe.nelec])
    ):
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
        isinstance(mbe.point_group, str),
        "symmetry (point_group keyword argument) must be a str",
    )
    assertion(
        isinstance(mbe.fci_state_sym, (str, int))
        or (
            isinstance(mbe.fci_state_sym, list)
            and all([isinstance(state, (str, int)) for state in mbe.fci_state_sym])
        ),
        "state wavefunction symmetry (fci_state_sym keyword argument) must be a str or "
        "int or a list of str or int",
    )
    try:
        if isinstance(mbe.fci_state_sym, str):
            mbe.fci_state_sym = symm.addons.irrep_name2id(
                mbe.point_group, mbe.fci_state_sym
            )
        elif isinstance(mbe.fci_state_sym, list) and all(
            [isinstance(state, str) for state in mbe.fci_state_sym]
        ):
            mbe.fci_state_sym = [
                symm.addons.irrep_name2id(mbe.point_group, state)
                for state in mbe.fci_state_sym
            ]
    except Exception as err:
        raise ValueError(
            "illegal choice of state wavefunction symmetry (fci_state_sym keyword "
            f"argument) -- PySCF error: {err}"
        )
    assertion(
        (isinstance(mbe.fci_state_root, int) and mbe.fci_state_root >= 0)
        or (
            isinstance(mbe.fci_state_root, list)
            and all(
                [isinstance(state, int) and state >= 0 for state in mbe.fci_state_root]
            )
        ),
        "target state (root keyword argument) must be an int >= 0",
    )
    if (
        isinstance(mbe.nelec, list)
        and isinstance(mbe.fci_state_sym, list)
        and isinstance(mbe.fci_state_root, list)
        and isinstance(mbe.fci_state_weights, list)
    ):
        assertion(
            mbe.target in ["rdm12", "genfock"],
            "only 1- and 2-particle reduced density matrices and generalized Fock "
            "matrices can be determined as state-averaged properties, all other mbe "
            "targets must only have keywords describing a single state and can "
            "therefore not be lists (nelec, fci_state_sym, fci_state_root, "
            "fci_state_weights keyword arguments)",
        )
        assertion(
            mbe.method == "fci",
            "only the fci method (method keyword argument) can be used to calculate "
            "state-averaged 1- and 2-particle reduced density matrices, cc methods "
            "must only have keywords describing a single state and can therefore not "
            "be lists (nelec, fci_state_sym, fci_state_root, fci_state_weights keyword "
            "arguments)",
        )
        assertion(
            len(mbe.nelec)
            == len(mbe.fci_state_sym)
            == len(mbe.fci_state_root)
            == len(mbe.fci_state_weights),
            "keywords describing different states for the calculation of state-averaged "
            "1- and 2-particle reduced density matrices (nelec, fci_state_sym, "
            "fci_state_root, fci_state_weights keyword arguments) must all have the "
            "same length",
        )
        states = [
            state
            for state in zip(
                *[
                    [tuple(nelec_state) for nelec_state in mbe.nelec],
                    mbe.fci_state_sym,
                    mbe.fci_state_root,
                ]
            )
        ]
        assertion(
            len(set(states)) == len(states),
            "keywords describing multiple states for the calculation of state-averaged "
            "1- and 2-particle reduced density matrices (nelec, fci_state_sym, "
            "fci_state_root keyword arguments) must describe different states, every "
            "state must therefore differ from every other state in either "
            "multiplicity, symmetry or root",
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
    if mbe.orb_type in ["can", "ccsd", "ccsd(t)", "casscf"]:
        assertion(
            isinstance(mbe.orbsym, np.ndarray) and mbe.orbsym.shape == (mbe.norb,),
            "orbital symmetry (orbsym keyword argument) must be a np.ndarray with "
            "shape (norb,)",
        )
    elif mbe.orb_type == "local":
        assertion(
            (isinstance(mbe.orbsym, np.ndarray) and mbe.orbsym.shape == (mbe.norb,))
            or (
                isinstance(mbe.orbsym, list)
                and all([isinstance(symm_op, dict) for symm_op in mbe.orbsym])
                and all([len(symm_op) == mbe.norb for symm_op in mbe.orbsym])
                and all(
                    [
                        [isinstance(orb, int) for orb in symm_op.keys()]
                        for symm_op in mbe.orbsym
                    ]
                )
                and all(
                    [
                        [isinstance(tup, tuple) for tup in symm_op.values()]
                        for symm_op in mbe.orbsym
                    ]
                )
                and all(
                    [
                        [
                            [isinstance(orb, int) for orb in tup]
                            for tup in symm_op.values()
                        ]
                        for symm_op in mbe.orbsym
                    ]
                )
            ),
            "orbital symmetry (orbsym keyword argument) must be a np.ndarray with "
            "shape (norb,) or a list of symmetry operation dictionaries with orbital "
            "indices as keys and tuples of orbitals that this orbital transforms into "
            "as values.",
        )
        if (
            mbe.orb_type == "local"
            and mbe.target in ["rdm12", "genfock"]
            and mbe.point_group != "C1"
        ):
            logger.warning(
                "Warning: 1- and 2-particle reduced density matrix and generalized "
                "Fock matrix calculations while exploiting local orbital symmetry are "
                "currently not possible. The symmetry of the local orbitals is not "
                "utilized in the current calculation."
            )

    # integrals
    assertion(
        isinstance(mbe.hcore, np.ndarray) and mbe.hcore.shape == 2 * (mbe.norb,),
        "core hamiltonian integral (hcore keyword argument) must be a np.ndarray with "
        "shape (norb, norb)",
    )
    assertion(
        isinstance(mbe.eri, np.ndarray)
        and (
            mbe.eri.shape == 2 * (cast(int, mbe.norb) * (cast(int, mbe.norb) + 1) / 2,)
            or mbe.eri.shape == 4 * (mbe.norb,)
        ),
        "electron repulsion integral (eri keyword argument) must be a np.ndarray with "
        "shape (mbe.norb * (mbe.norb + 1) / 2, (mbe.norb * (mbe.norb + 1) / 2)) or "
        "(norb, norb, norb, norb)",
    )

    # reference and expansion spaces
    assertion(
        isinstance(mbe.ref_space, np.ndarray),
        "reference space (ref_space keyword argument) must be a np.ndarray of orbital "
        "indices",
    )
    if (isinstance(mbe.nelec, np.ndarray) and mbe.nelec[0] != mbe.nelec[1]) or (
        isinstance(mbe.nelec, list)
        and any([state[0] != state[1] for state in mbe.nelec])
    ):
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
    assertion(
        isinstance(mbe.exp_space, np.ndarray),
        "expansion space (exp_space keyword argument) must be a np.ndarray of orbital "
        "indices",
    )
    assertion(
        np.intersect1d(mbe.ref_space, cast(np.ndarray, mbe.exp_space)).size == 0,
        "reference space (ref_space keyword argument) and expansion space (exp_space "
        "keyword argument) must be mutually exclusive",
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
            mbe.target in ["energy", "dipole", "rdm12", "genfock"],
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
                isinstance(mbe.base_prop, np.ndarray) and mbe.base_prop.shape == (3,),
                "base model dipole moment (base_prop keyword argument) must be a "
                "np.ndarray with shape (3,)",
            )
        elif mbe.target == "rdm12":
            assertion(
                isinstance(mbe.base_prop, tuple)
                and len(mbe.base_prop) == 2
                and isinstance(mbe.base_prop[0], np.ndarray)
                and mbe.base_prop[0].shape == 2 * (mbe.norb,)
                and isinstance(mbe.base_prop[1], np.ndarray)
                and mbe.base_prop[1].shape == 4 * (mbe.norb,),
                "base model 1- and 2-particle density matrices (base_prop keyword "
                "argument) must be a tuple with dimension 2, rdm1 must be a np.ndarray "
                "with shape (norb, norb), rdm2 must be a np.ndarray with shape "
                "(norb, norb, norb, norb)",
            )
        elif mbe.target == "genfock":
            assertion(
                isinstance(mbe.base_prop, tuple)
                and len(mbe.base_prop) == 2
                and isinstance(mbe.base_prop[0], float)
                and isinstance(mbe.base_prop[1], np.ndarray)
                and mbe.base_prop[1].shape
                == (cast(int, mbe.full_nocc) + cast(int, mbe.norb), mbe.full_norb),
                "base model for generalized fock matrix calculation (base_prop keyword "
                "argument) must be a tuple with dimension 2, the first element "
                "describes the energy and must be a float, the second element "
                "describes the generalized fock matrix and must be a np.ndarray with "
                "shape (full_nocc + norb, full_norb)",
            )

    # screening
    assertion(
        isinstance(mbe.screen_type, str),
        "screening type (screen_type keyword argument) must be a string",
    )
    assertion(
        mbe.screen_type in ["fixed", "adaptive"],
        "valid screening types (screen_type keyword argument) are: fixed and adaptive",
    )
    assertion(
        isinstance(mbe.screen_start, int) and mbe.screen_start >= 2,
        "screening start order (screen_start keyword argument) must be an int >= 2",
    )
    assertion(
        isinstance(mbe.screen_perc, float) and mbe.screen_perc <= 1.0,
        "screening percentage (screen_perc keyword argument) must be a float <= 1.",
    )
    assertion(
        isinstance(mbe.screen_thres, float) and mbe.screen_thres > 0.0,
        "screening threshold (screen_thres keyword argument) must be a float > 0.",
    )
    assertion(
        isinstance(mbe.screen_func, str),
        "screening function (screen_func keyword argument) must be an str",
    )
    assertion(
        mbe.screen_func in ["max", "abs_sum", "sum", "rnd"],
        "valid screening functions (screen_func keyword argument) are: max, abs_sum, "
        "sum and rnd",
    )
    if mbe.screen_func == "sum":
        assertion(
            mbe.target in ["energy", "excitation"],
            "screening with the sum screening function (screen_func keyword argument) "
            "only works for scalar targets such as energy or excitation (target "
            "keyword argument)",
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
            isinstance(mbe.orbsym_linear, np.ndarray)
            and mbe.orbsym_linear.shape == (mbe.norb,),
            "linear point group orbital symmetry (orbsym_linear keyword argument) must "
            "be a np.ndarray with shape (norb,)",
        )

    # exclude single excitations
    if mbe.target in ["rdm12", "genfock"]:
        assertion(
            isinstance(mbe.no_singles, bool),
            "excluding single excitations (no_singles keyword argument) must be a bool",
        )

    # optional integrals for (transition) dipole moment
    if mbe.target in ["dipole", "trans"]:
        assertion(
            isinstance(mbe.dipole_ints, np.ndarray)
            and mbe.dipole_ints.shape == (3, mbe.norb, mbe.norb),
            "dipole integrals (dipole_ints keyword argument) must be a np.ndarray with "
            "shape (3, norb, norb)",
        )

    # optional parameters for generalized Fock matrix
    if mbe.target == "genfock":
        assertion(
            isinstance(mbe.full_norb, int) and mbe.full_norb > 0,
            "number of orbitals in the full system (full_norb keyword argument) must "
            "be an int > 0",
        )
        assertion(
            isinstance(mbe.full_nocc, int) and mbe.full_nocc > 0,
            "number of occupied orbitals in the full system (full_nocc keyword "
            "argument) must be an int > 0",
        )
        assertion(
            isinstance(mbe.inact_fock, np.ndarray)
            and mbe.inact_fock.shape
            == (mbe.full_norb, cast(int, mbe.full_nocc) + cast(int, mbe.norb)),
            "inactive Fock matrix (inact_fock keyword argument) must be a np.ndarray "
            "with shape (full_norb, full_nocc + norb)",
        )
        assertion(
            isinstance(mbe.eri_goaa, np.ndarray)
            and mbe.eri_goaa.shape
            == (mbe.full_norb, mbe.full_nocc, mbe.norb, mbe.norb),
            "general-occupied-active-active electron repulsion integral (eri_goaa "
            "keyword argument) must be a np.ndarray with shape "
            "(mbe.full_norb, mbe.full_nocc, mbe.norb, mbe.norb)",
        )
        assertion(
            isinstance(mbe.eri_gaao, np.ndarray)
            and mbe.eri_gaao.shape
            == (mbe.full_norb, mbe.norb, mbe.norb, mbe.full_nocc),
            "general-active-active-occupied electron repulsion integral (eri_gaao "
            "keyword argument) must be a np.ndarray with shape "
            "(mbe.full_norb, mbe.norb, mbe.norb, mbe.full_nocc)",
        )
        assertion(
            isinstance(mbe.eri_gaaa, np.ndarray)
            and mbe.eri_gaaa.shape == (mbe.full_norb, mbe.norb, mbe.norb, mbe.norb),
            "general-active-active-active electron repulsion integral (eri_gaaa "
            "keyword argument) must be a np.ndarraywith shape "
            "(mbe.full_norb, mbe.norb, mbe.norb, mbe.norb)",
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
        "no_singles": mbe.no_singles,
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
    system: Dict[str, Any] = {
        "norb": mbe.norb,
        "nelec": mbe.nelec,
        "orbsym": mbe.orbsym,
        "hcore": mbe.hcore,
        "eri": mbe.eri,
        "ref_space": mbe.ref_space,
        "exp_space": mbe.exp_space,
        "base_prop": mbe.base_prop,
        "orbsym_linear": mbe.orbsym_linear,
        "dipole_ints": mbe.dipole_ints,
        "full_norb": mbe.full_norb,
        "full_nocc": mbe.full_nocc,
        "inact_fock": mbe.inact_fock,
        "eri_goaa": mbe.eri_goaa,
        "eri_gaao": mbe.eri_gaao,
        "eri_gaaa": mbe.eri_gaaa,
    }

    # write system quantities
    np.savez(os.path.join(RST, "system"), **system)


def restart_read_system(mbe: MBE) -> MBE:
    """
    this function reads all system quantities restart files
    """
    # read system quantities
    system_npz = np.load(os.path.join(RST, "system.npz"), allow_pickle=True)

    # create system dictionary
    system = {}
    for file in system_npz.files:
        system[file] = system_npz[file]

    # close npz object
    system_npz.close()

    # revert to scalars
    for key, value in system.items():
        if value.ndim == 0:
            system[key] = value.item()

    # revert to list
    if system["orbsym"].dtype == object:
        system["orbsym"] = list(system["orbsym"])

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