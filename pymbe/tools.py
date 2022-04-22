#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import os
import re
import sys
import logging
import numpy as np
import scipy.special as sc
from mpi4py import MPI
from pyscf import symm, ao2mo
from pyscf.lib.exceptions import PointGroupSymmetryError
from itertools import islice, combinations, groupby
from math import floor, sin, cos, pi, sqrt
from subprocess import Popen, PIPE
from traceback import format_stack
from typing import TYPE_CHECKING

from pymbe.parallel import open_shared_win

if TYPE_CHECKING:

    from typing import Tuple, List, Generator, Union, Optional


# get logger
logger = logging.getLogger("pymbe_logger")

# restart folder
RST = os.getcwd() + "/rst"

# pi-orbitals
PI_SYMM_D2H = np.array(
    [2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27]
)
PI_SYMM_C2V = np.array([2, 3, 10, 11, 12, 13, 20, 21, 22, 23])


class RDMCls:
    """
    this class holds the 1- and 2-particle RDMs and defines all necessary operations
    """

    def __init__(self, rdm1: np.ndarray, rdm2: np.ndarray):
        """
        initializes a RDM object
        """
        self.rdm1 = rdm1
        self.rdm2 = rdm2

    def __getitem__(self, idx: np.ndarray) -> RDMCls:
        """
        this function ensures RDMCls can be retrieved using one-dimensional indexing of
        RDMCls objects
        """
        return RDMCls(
            self.rdm1[idx.reshape(-1, 1), idx],
            self.rdm2[
                idx.reshape(-1, 1, 1, 1),
                idx.reshape(1, -1, 1, 1),
                idx.reshape(1, 1, -1, 1),
                idx,
            ],
        )

    def __setitem__(
        self,
        idx: np.ndarray,
        values: Union[RDMCls, Tuple[np.ndarray, np.ndarray]],
    ) -> RDMCls:
        """
        this function implements setting RDMs through indexing for the RDMCls objects
        this function ensures RDMCls indexed in one dimension can be set using RDMCls
        or tuples of numpy arrays
        """
        if isinstance(values, RDMCls):
            self.rdm1[idx.reshape(-1, 1), idx] = values.rdm1
            self.rdm2[
                idx.reshape(-1, 1, 1, 1),
                idx.reshape(1, -1, 1, 1),
                idx.reshape(1, 1, -1, 1),
                idx,
            ] = values.rdm2
        elif isinstance(values, tuple):
            self.rdm1[idx.reshape(-1, 1), idx] = values[0]
            self.rdm2[
                idx.reshape(-1, 1, 1, 1),
                idx.reshape(1, -1, 1, 1),
                idx.reshape(1, 1, -1, 1),
                idx,
            ] = values[1]
        else:
            return NotImplemented

        return self

    def __add__(self, other: RDMCls) -> RDMCls:
        """
        this function implements addition for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            return RDMCls(self.rdm1 + other.rdm1, self.rdm2 + other.rdm2)
        else:
            return NotImplemented

    def __iadd__(self, other: Union[RDMCls, packedRDMCls]) -> RDMCls:
        """
        this function implements inplace addition for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            self.rdm1 += other.rdm1
            self.rdm2 += other.rdm2
            return self
        else:
            return NotImplemented

    def __sub__(self, other: RDMCls) -> RDMCls:
        """
        this function implements subtraction for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            return RDMCls(self.rdm1 - other.rdm1, self.rdm2 - other.rdm2)
        else:
            return NotImplemented

    def __isub__(self, other: Union[RDMCls, packedRDMCls]) -> RDMCls:
        """
        this function implements inplace subtraction for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            self.rdm1 -= other.rdm1
            self.rdm2 -= other.rdm2
            return self
        else:
            return NotImplemented

    def __mul__(self, other: int) -> RDMCls:
        """
        this function implements multiplication for the RDMCls objects
        """
        if isinstance(other, int):
            return RDMCls(other * self.rdm1, other * self.rdm2)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __imul__(self, other: int) -> RDMCls:
        """
        this function implements inplace multiplication for the RDMCls objects
        """
        if isinstance(other, RDMCls):
            self.rdm1 *= other
            self.rdm2 *= other
            return self
        else:
            return NotImplemented

    def __truediv__(self, other: Union[int, float]) -> RDMCls:
        """
        this function implements division for the RDMCls objects
        """
        if isinstance(other, (int, float)):
            return RDMCls(self.rdm1 / other, self.rdm2 / other)
        else:
            return NotImplemented

    def __itruediv__(self, other: Union[int, float]) -> RDMCls:
        """
        this function implements inplace division for the RDMCls objects
        """
        if isinstance(other, (int, float)):
            self.rdm1 /= other
            self.rdm2 /= other
            return self
        else:
            return NotImplemented

    def fill(self, value: float) -> None:
        """
        this function defines the fill function for RDMCls objects
        """
        self.rdm1.fill(value)
        self.rdm2.fill(value)

    def copy(self) -> RDMCls:
        """
        this function creates a copy of RDMCls objects
        """
        return RDMCls(self.rdm1.copy(), self.rdm2.copy())


class packedRDMCls:
    """
    this class describes packed RDMs, instances of this class can either be created
    normally using __init__() or by opening a shared memory instance with
    open_shared_RDM
    """

    rdm1_size: List[int] = []
    pack_rdm1: List[Tuple[np.ndarray, np.ndarray]] = []
    unpack_rdm1: List[np.ndarray] = []
    rdm2_size: List[int] = []
    pack_rdm2: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    unpack_rdm2: List[np.ndarray] = []

    def __init__(self, rdm1: np.ndarray, rdm2: np.ndarray, idx: int = -1) -> None:
        """
        this function initializes a packedRDMCls object
        """
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.idx = idx

    @classmethod
    def reset(cls):
        """
        this function resets the class to ensure class attributes are not kept from
        previous runs
        """
        cls.rdm1_size = []
        cls.pack_rdm1 = []
        cls.unpack_rdm1 = []
        cls.rdm2_size = []
        cls.pack_rdm2 = []
        cls.unpack_rdm2 = []

    @classmethod
    def open_shared_RDM(
        cls, inc_win: Tuple[MPI.Win, MPI.Win], n_tuples: int, idx: int
    ) -> packedRDMCls:
        """
        this factory function initializes a packedRDMCls object in shared memory
        """
        # open shared windows
        rdm1 = open_shared_win(inc_win[0], np.float64, (n_tuples, cls.rdm1_size[idx]))
        rdm2 = open_shared_win(inc_win[1], np.float64, (n_tuples, cls.rdm2_size[idx]))

        return cls(rdm1, rdm2, idx)

    @classmethod
    def get_pack_idx(cls, norb) -> None:
        """
        this function generates packing and unpacking indices for the 1- and 2-particle
        rdms and should be called at every order before a packedRDMCls object is
        initialized at this order
        """
        pack_rdm1 = np.triu_indices(norb)
        unpack_rdm1 = np.zeros((norb, norb), dtype=np.int64)
        rdm1_size = pack_rdm1[0].size
        indices = np.arange(rdm1_size)
        unpack_rdm1[pack_rdm1] = indices
        unpack_rdm1[pack_rdm1[1], pack_rdm1[0]] = indices

        rdm2_size = (7 * norb**4 + 8 * norb**3 - norb**2 - 2 * norb) // 12
        pack_rdm2 = np.empty((rdm2_size, 4), dtype=np.int64)
        unpack_rdm2 = np.empty(4 * (norb,), dtype=np.int64)

        i = 0
        for s in range(norb):
            for r in range(s + 1):
                for q in range(s + 1):
                    for p in range(norb):
                        pack_rdm2[i, :] = np.array([p, q, r, s], dtype=np.int64)
                        unpack_rdm2[p, q, r, s] = i
                        unpack_rdm2[r, s, p, q] = i
                        unpack_rdm2[q, p, s, r] = i
                        unpack_rdm2[s, r, q, p] = i
                        i += 1
                for q in range(s + 1, norb):
                    for p in range(r + 1):
                        pack_rdm2[i, :] = np.array([p, q, r, s], dtype=np.int64)
                        unpack_rdm2[p, q, r, s] = i
                        unpack_rdm2[r, s, p, q] = i
                        unpack_rdm2[q, p, s, r] = i
                        unpack_rdm2[s, r, q, p] = i
                        i += 1
            for r in range(s + 1, norb):
                for q in range(s + 1):
                    for p in range(q + 1):
                        pack_rdm2[i, :] = np.array([p, q, r, s], dtype=np.int64)
                        unpack_rdm2[p, q, r, s] = i
                        unpack_rdm2[r, s, p, q] = i
                        unpack_rdm2[q, p, s, r] = i
                        unpack_rdm2[s, r, q, p] = i
                        i += 1
                for q in range(s + 1, norb):
                    for p in range(min(r + 1, q + 1)):
                        pack_rdm2[i, :] = np.array([p, q, r, s], dtype=np.int64)
                        unpack_rdm2[p, q, r, s] = i
                        unpack_rdm2[r, s, p, q] = i
                        unpack_rdm2[q, p, s, r] = i
                        unpack_rdm2[s, r, q, p] = i
                        i += 1

        cls.rdm1_size.append(rdm1_size)
        cls.pack_rdm1.append(pack_rdm1)
        cls.unpack_rdm1.append(unpack_rdm1)
        cls.rdm2_size.append(rdm2_size)
        cls.pack_rdm2.append(
            (pack_rdm2[:, 0], pack_rdm2[:, 1], pack_rdm2[:, 2], pack_rdm2[:, 3])
        )
        cls.unpack_rdm2.append(unpack_rdm2)

    def __getitem__(self, idx: Union[int, np.int64, slice, np.ndarray]) -> packedRDMCls:
        """
        this function ensures packedRDMCls can be retrieved through indexing
        packedRDMCls objects
        """
        if isinstance(idx, (int, np.integer, slice, np.ndarray)):
            return packedRDMCls(self.rdm1[idx], self.rdm2[idx], self.idx)
        else:
            return NotImplemented

    def __setitem__(
        self,
        idx: Union[int, np.int64, slice, np.ndarray],
        values: Union[packedRDMCls, RDMCls, np.ndarray, float],
    ) -> packedRDMCls:
        """
        this function ensures indexed packedRDMCls can be set using packedRDMCls or
        RDMCls objects
        """
        if isinstance(idx, (slice, np.ndarray)) and isinstance(values, packedRDMCls):
            self.rdm1[idx] = values.rdm1
            self.rdm2[idx] = values.rdm2
        elif isinstance(idx, (int, np.integer)) and isinstance(values, RDMCls):
            self.rdm1[idx] = values.rdm1[self.pack_rdm1[self.idx]]
            self.rdm2[idx] = values.rdm2[self.pack_rdm2[self.idx]]
        else:
            return NotImplemented

        return self

    def __radd__(self, other: RDMCls) -> RDMCls:
        """
        this function ensures the packedRDMCls object is unpacked when added to a RDMCls
        object
        """
        if isinstance(other, RDMCls) and self.rdm1.ndim == 1 and self.rdm2.ndim == 1:
            return other + RDMCls(
                self.rdm1[self.unpack_rdm1[self.idx]],
                self.rdm2[self.unpack_rdm2[self.idx]],
            )
        else:
            return NotImplemented

    def fill(self, value: float) -> None:
        """
        this function defines the fill function for packedRDMCls objects
        """
        self.rdm1.fill(value)
        self.rdm2.fill(value)


def logger_config(verbose: int) -> None:
    """
    this function configures the pymbe logger
    """
    # corresponding logging level
    verbose_level = {0: 30, 1: 20, 2: 10, 3: 10}

    # set level for logger
    logger.setLevel(verbose_level[verbose])

    # add new handler to log to stdout
    handler = logging.StreamHandler(sys.stdout)

    # create new formatter
    formatter = logging.Formatter("%(message)s")

    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(handler)

    # prevent logger from propagating handlers from parent loggers
    logger.propagate = False


def git_version() -> str:
    """
    this function returns the git revision as a string
    see: https://github.com/numpy/numpy/blob/master/setup.py#L70-L92
    """

    def _minimal_ext_cmd(cmd: List[str]) -> bytes:
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = Popen(cmd, stdout=PIPE, env=env, cwd=get_pymbe_path()).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_pymbe_path() -> str:
    """
    this function returns the path to pymbe
    """
    return os.path.dirname(__file__)


def assertion(cond: Union[bool, np.bool_], reason: str) -> None:
    """
    this function returns an assertion of a given condition
    """
    if not cond:
        # get stack
        stack = "".join(format_stack()[:-1])
        # print stack
        logger.error("\n\n" + stack)
        logger.error("\n\n*** PyMBE assertion error: " + reason + " ***\n\n")
        # abort mpi
        MPI.COMM_WORLD.Abort()


def time_str(time: float) -> str:
    """
    this function returns time as a HH:MM:SS string
    """
    # hours, minutes, and seconds
    hours = time // 3600.0
    minutes = (time - (time // 3600) * 3600.0) // 60.0
    seconds = time - hours * 3600.0 - minutes * 60.0

    # init time string
    string: str = ""

    # write time string
    if hours > 0:
        string += f"{hours:.0f}h "

    if minutes > 0:
        string += f"{minutes:.0f}m "

    string += f"{seconds:.2f}s"

    return string


def hash_2d(a: np.ndarray) -> np.ndarray:
    """
    this function converts a 2d numpy array to a 1d array of hashes
    """
    return np.fromiter(map(hash_1d, a), dtype=np.int64, count=a.shape[0])


def hash_1d(a: np.ndarray) -> int:
    """
    this function converts a 1d numpy array to a hash
    """
    return hash(a.tobytes())


def hash_lookup(a: np.ndarray, b: Union[int, np.ndarray]) -> Optional[np.ndarray]:
    """
    this function finds occurences of b in a through a binary search
    """
    left = a.searchsorted(b, side="left")
    right = a.searchsorted(b, side="right")
    if ((right - left) > 0).all():
        return left
    else:
        return None


def tuples(
    occ_space: np.ndarray,
    virt_space: np.ndarray,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
    order_start: int = 1,
    occ_start: int = 0,
    virt_start: int = 0,
) -> Generator[np.ndarray, None, None]:
    """
    this function is the main generator for tuples
    """
    # combinations of occupied and virtual MOs
    for k in range(order_start, order):
        if _valid_tup(ref_nelec, ref_nhole, k, order - k, vanish_exc):
            for tup_occ in islice(combinations(occ_space, k), occ_start, None):
                for tup_virt in islice(
                    combinations(virt_space, order - k), virt_start, None
                ):
                    yield np.array(tup_occ + tup_virt, dtype=np.int64)
                virt_start = 0
            occ_start = 0

    # only occupied MOs
    if _valid_tup(ref_nelec, ref_nhole, order, 0, vanish_exc) and 0 <= occ_start:
        for tup_occ in islice(combinations(occ_space, order), occ_start, None):
            yield np.array(tup_occ, dtype=np.int64)

    # only virtual MOs
    if _valid_tup(ref_nelec, ref_nhole, 0, order, vanish_exc) and 0 <= virt_start:
        for tup_virt in islice(combinations(virt_space, order), virt_start, None):
            yield np.array(tup_virt, dtype=np.int64)


def start_idx(
    occ_space: np.ndarray,
    virt_space: np.ndarray,
    tup_occ: Optional[np.ndarray],
    tup_virt: Optional[np.ndarray],
) -> Tuple[int, int, int]:
    """
    this function return the start indices for a given occupied and virtual tuple
    """
    if tup_occ is None and tup_virt is None:
        order_start = 1
        occ_start = virt_start = 0
    elif tup_occ is not None and tup_virt is not None:
        order_start = int(tup_occ.size)
        occ_start = int(_comb_idx(occ_space, tup_occ))
        virt_start = int(_comb_idx(virt_space, tup_virt))
    elif tup_occ is not None and tup_virt is None:
        order_start = int(tup_occ.size)
        occ_start = int(_comb_idx(occ_space, tup_occ))
        virt_start = 0
    elif tup_occ is None and tup_virt is not None:
        order_start = int(tup_virt.size)
        occ_start = -1
        virt_start = int(_comb_idx(virt_space, tup_virt))
    return order_start, occ_start, virt_start


def _comb_idx(space: np.ndarray, tup: np.ndarray) -> float:
    """
    this function return the index of a given (ordered) combination returned from
    itertools.combinations
    """
    idx = _idx(space, tup[0], tup.size)
    idx += sum(
        (
            _idx(space[tup[i - 1] < space], tup[i], tup[i:].size)
            for i in range(1, tup.size)
        )
    )
    return idx


def _idx(space: np.ndarray, idx: int, order: int) -> float:
    """
    this function return the start index of element space[idx] in position (order+1)
    from the right in a given combination
    """
    return sum(
        (sc.binom(space[i < space].size, (order - 1)) for i in space[space < idx])
    )


def n_tuples(
    occ_space: np.ndarray,
    virt_space: np.ndarray,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
) -> int:
    """
    this function returns the total number of tuples of a given order
    """
    # init n_tuples
    n = 0.0

    # combinations of occupied and virtual MOs
    for k in range(1, order):
        if _valid_tup(ref_nelec, ref_nhole, k, order - k, vanish_exc):
            n += sc.binom(occ_space.size, k) * sc.binom(virt_space.size, order - k)

    # only occupied MOs
    if _valid_tup(ref_nelec, ref_nhole, order, 0, vanish_exc):
        n += sc.binom(occ_space.size, order)

    # only virtual MOs
    if _valid_tup(ref_nelec, ref_nhole, 0, order, vanish_exc):
        n += sc.binom(virt_space.size, order)

    return int(n)


def cas(ref_space: np.ndarray, tup: np.ndarray) -> np.ndarray:
    """
    this function returns a cas space
    """
    return np.sort(np.append(ref_space, tup))


def core_cas(
    nocc: int, ref_space: np.ndarray, tup: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns a core and a cas space
    """
    cas_idx = cas(ref_space, tup)
    core_idx = np.setdiff1d(np.arange(nocc), cas_idx)
    return core_idx, cas_idx


def _cas_idx_cart(cas_idx: np.ndarray) -> np.ndarray:
    """
    this function returns a cartesian product of (cas_idx, cas_idx)
    """
    return np.array(np.meshgrid(cas_idx, cas_idx)).T.reshape(-1, 2)


def _coor_to_idx(ij: Tuple[int, int]) -> int:
    """
    this function returns the lower triangular index corresponding to (i, j)
    """
    i = ij[0]
    j = ij[1]
    if i >= j:
        return i * (i + 1) // 2 + j
    else:
        return j * (j + 1) // 2 + i


def idx_tril(cas_idx: np.ndarray) -> np.ndarray:
    """
    this function returns lower triangular cas indices
    """
    cas_idx_cart = _cas_idx_cart(cas_idx)
    return np.unique(
        np.fromiter(
            map(_coor_to_idx, cas_idx_cart),
            dtype=cas_idx_cart.dtype,
            count=cas_idx_cart.shape[0],
        )
    )


def pi_space(
    group: str, orbsym: np.ndarray, exp_space: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns pi-orbitals and hashes from total expansion space
    """
    # all pi-orbitals
    if group == "Dooh":
        pi_space_arr = exp_space[np.in1d(orbsym[exp_space], PI_SYMM_D2H)]
    else:
        pi_space_arr = exp_space[np.in1d(orbsym[exp_space], PI_SYMM_C2V)]

    # get all degenerate pi-pairs
    pi_pairs = pi_space_arr.reshape(-1, 2)

    # get hashes of all degenerate pi-pairs
    pi_hashes = hash_2d(pi_pairs)
    pi_pairs = pi_pairs[np.argsort(pi_hashes)]
    pi_hashes.sort()

    return (pi_pairs.reshape(-1), pi_hashes)


def _pi_orbs(pi_space: np.ndarray, tup: np.ndarray) -> np.ndarray:
    """
    this function returns pi-orbitals from tuple of orbitals
    """
    return tup[np.in1d(tup, pi_space)]


def pi_prune(pi_space: np.ndarray, pi_hashes: np.ndarray, tup: np.ndarray) -> bool:
    """
    this function returns True for a tuple of orbitals allowed under pruning wrt
    degenerate pi-orbitals
    """
    # get all pi-orbitals in tup
    tup_pi_orbs = _pi_orbs(pi_space, tup)

    if tup_pi_orbs.size == 0:
        # no pi-orbitals
        return True

    if tup_pi_orbs.size % 2 > 0:
        # always prune tuples with an odd number of pi-orbitals
        return False

    # get hashes of pi-pairs
    tup_pi_hashes = hash_2d(tup_pi_orbs.reshape(-1, 2))
    tup_pi_hashes.sort()

    # get indices of pi-pairs
    idx = hash_lookup(pi_hashes, tup_pi_hashes)

    return idx is not None


def symm_eqv_tup(
    cas_idx: np.ndarray, symm_orbs: np.ndarray, ref_space: np.ndarray
) -> int:
    """
    this function returns the number of stabilizers for a tuple if it is the
    lexicographically greatest with respect to all symmetry operations, otherwise
    returns zero
    """
    # initialize number of stabilizers
    nstab = 0

    # initialize number of valid symmetry operations for this tuple
    nsymm = 0

    # loop over symmetry operations in point group
    for symm_op in symm_orbs:

        # get permuted cas space by applying symmetry operation
        perm_cas = symm_op[cas_idx]

        # check if any orbitals cannot be transformed
        if (perm_cas == -1).any():

            # skip this symmetry operation
            continue

        # increment number of symmetry operations
        nsymm += 1

        # sort permuted cas space
        perm_cas.sort()

        # check if reference space is included in permuted cas space
        if not np.isin(ref_space, perm_cas, assume_unique=True).all():

            # not a valid tuple
            continue

        # loop over orbs in cas space and permuted cas space
        for orb, perm_orb in zip(cas_idx, perm_cas):

            # check if orb in cas space is smaller than orb in permuted cas space
            if orb < perm_orb:

                # tuple is not unique and not lexicographically smaller
                return 0

            # check if orb in cas space is greater than orb in permuted cas space
            elif orb > perm_orb:

                # tuple is lexicographically greater
                break

        # symmetry operation is a stabilizer for this tuple
        else:

            # increment number of stabilizers
            nstab += 1

    # calculate number of equivalent tuples
    neqvtups = nsymm // nstab

    return neqvtups


def get_lex_tup(
    tup: np.ndarray, symm_orbs: np.ndarray, ref_space: np.ndarray
) -> np.ndarray:
    """
    this function returns the symmetrically equivalent but lexicographically greater
    tuple
    """
    # generate full cas space
    cas_idx = cas(ref_space, tup)

    # initialize current lexicographically greatest cas space
    lex_cas = cas_idx.copy()

    # loop over symmetry operations in point group
    for symm_op in symm_orbs:

        # get permuted cas space by applying symmetry operation
        perm_cas = symm_op[cas_idx]

        # check if any orbitals cannot be transformed
        if (perm_cas == -1).any():

            # skip this symmetry operation
            continue

        # sort permuted cas space
        perm_cas.sort()

        # check if reference space is included in permuted cas space
        if not np.isin(ref_space, perm_cas, assume_unique=True).all():

            # not a valid tuple
            continue

        # loop over orbs in lex_cas and perm_cas
        for lex_orb, perm_orb in zip(lex_cas, perm_cas):

            # check if orb in lex_cas is smaller than orb in perm_cas
            if lex_orb < perm_orb:

                # set permuted cas space as lexicographically greatest
                lex_cas = perm_cas

                # perm_cas is lexicographically greater
                break

            # check if orb in lex_cas is greater than orb in perm_cas
            elif lex_orb > perm_orb:

                # perm_cas is lexicographically smaller
                break

    # remove reference space
    lex_tup = np.setdiff1d(lex_cas, ref_space, assume_unique=True)

    return lex_tup


def get_nelec(occup: np.ndarray, tup: np.ndarray) -> np.ndarray:
    """
    this function returns the number of electrons in a given tuple of orbitals
    """
    occup_tup = occup[tup]
    return np.array(
        [np.count_nonzero(occup_tup > 0.0), np.count_nonzero(occup_tup > 1.0)]
    )


def get_nhole(nelec: np.ndarray, tup: np.ndarray) -> np.ndarray:
    """
    this function returns the number of holes in a given tuple of orbitals
    """
    return tup.size - nelec


def get_nexc(nelec: np.ndarray, nhole: np.ndarray) -> int:
    """
    this function returns the number of possible excitations in a given tuple of
    orbitals
    """
    return np.sum(np.minimum(nelec, nhole))


def _valid_tup(
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    tup_nocc: int,
    tup_nvirt: int,
    vanish_exc: int,
) -> bool:
    """
    this function returns true if a tuple kind produces a non-vanishing correlation
    energy
    """
    return (
        get_nexc(
            ref_nelec + np.array([tup_nocc, tup_nocc]),
            ref_nhole + np.array([tup_nvirt, tup_nvirt]),
        )
        > vanish_exc
    )


def mat_idx(site_idx: int, nx: int, ny: int) -> Tuple[int, int]:
    """
    this function returns x and y indices of a matrix
    """
    y = site_idx % nx
    x = int(floor(float(site_idx) / ny))
    return x, y


def near_nbrs(site_xy: Tuple[int, int], nx: int, ny: int) -> List[Tuple[int, int]]:
    """
    this function returns a list of nearest neighbour indices
    """
    up = ((site_xy[0] - 1) % nx, site_xy[1])
    down = ((site_xy[0] + 1) % nx, site_xy[1])
    left = (site_xy[0], (site_xy[1] + 1) % ny)
    right = (site_xy[0], (site_xy[1] - 1) % ny)
    return [up, down, left, right]


def is_file(order: int, string: str) -> bool:
    """
    this function looks to see if a general restart file corresponding to the input
    string exists
    """
    if order is None:
        return os.path.isfile(os.path.join(RST, f"{string}.npy"))
    else:
        return os.path.isfile(os.path.join(RST, f"{string}_{order}.npy"))


def write_file(order: Optional[int], arr: np.ndarray, string: str) -> None:
    """
    this function writes a general restart file corresponding to the input string
    """
    if order is None:
        np.save(os.path.join(RST, f"{string}"), arr)
    else:
        np.save(os.path.join(RST, f"{string}_{order}"), arr)


def read_file(order: int, string: str) -> np.ndarray:
    """
    this function reads a general restart file corresponding to the input string
    """
    if order is None:
        return np.load(os.path.join(RST, f"{string}.npy"))
    else:
        return np.load(os.path.join(RST, f"{string}_{order}.npy"))


def natural_keys(txt: str) -> List[Union[int, str]]:
    """
    this function return keys to sort a string in human order (as
    alist.sort(key=natural_keys))
    see: http://nedbatchelder.com/blog/200712/human_sorting.html
    see: https://stackoverflow.com/questions/5967500
    """
    return [_convert(c) for c in re.split(r"(\d+)", txt)]


def _convert(txt: str) -> Union[int, str]:
    """
    this function converts strings with numbers in them
    """
    return int(txt) if txt.isdigit() else txt


def intervals(a: np.ndarray) -> Generator[List[int], None, None]:
    """
    this generator converts sequential numbers into intervals
    """
    for key, group in groupby(enumerate(a), lambda x: x[1] - x[0]):
        group_lst = list(group)
        if len(group_lst) == 1:
            yield [group_lst[0][1]]
        else:
            yield [group_lst[0][1], group_lst[-1][1]]


def ground_state_sym(orbsym: np.ndarray, nelec: np.ndarray, point_group: str) -> int:
    """
    this function determines the symmetry of the hf ground state
    """
    wfnsym = np.array([0])
    for irrep in orbsym[np.amin(nelec) : np.amax(nelec)]:
        wfnsym = symm.addons.direct_prod(wfnsym, irrep, groupname=point_group)
    return wfnsym.item()


def get_vhf(eri: np.ndarray, nocc: int, norb: int):
    """
    this function determines the Hartree-Fock potential from the electron repulsion
    integrals
    """
    eri = ao2mo.restore(1, eri, norb)

    vhf = np.empty((nocc, norb, norb), dtype=np.float64)
    for i in range(nocc):
        idx = np.asarray([i])
        vhf[i] = np.einsum("pqrs->rs", eri[idx[:, None], idx, :, :]) * 2.0
        vhf[i] -= np.einsum("pqrs->ps", eri[:, idx[:, None], idx, :]) * 2.0 * 0.5

    return vhf


def get_occup(norb: int, nelec: np.ndarray) -> np.ndarray:
    """
    this function generates the Hartree-Fock occupation vector
    """
    occup = np.zeros(norb, dtype=np.int64)
    occup[: np.amin(nelec)] = 2
    occup[np.amin(nelec) : np.amax(nelec)] = 1

    return occup


def get_symm_op_matrices(
    point_group: str, l_max: int
) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    this function generates all cartesian and spherical symmetry operation matrices for
    a given point group
    """
    symm_ops = [ident_matrix(l_max)]

    # 3D rotation group
    if point_group == "SO(3)":

        # same-atom symmetries are currently not exploited
        pass

    # proper cyclic groups Cn
    elif point_group[0] == "C" and point_group[1:].isnumeric():

        tot_main_rot = int(point_group[1:])

        # Cn
        for i in range(1, tot_main_rot):

            symm_ops.append(
                rot_matrix(np.array([0.0, 0.0, 1.0]), i * 2 * pi / tot_main_rot, l_max)
            )

    # improper cyclic group Ci
    elif point_group == "Ci":

        # i
        symm_ops.append(inv_matrix(l_max))

    # improper cyclic group Cs
    elif point_group == "Cs":

        # sigma_h
        symm_ops.append(reflect_matrix(np.array([0.0, 0.0, 1.0]), l_max))

    # improper cyclic group Sn
    elif point_group[0] == "S":

        tot_main_rot = int(point_group[1:])

        # Cn, Sn and i
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            if i % 2 == 0:
                symm_ops.append(rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))
            else:
                if rot_angle == pi:
                    symm_ops.append(inv_matrix(l_max))
                else:
                    symm_ops.append(
                        rot_reflect_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                    )

    # dihedral groups Dn
    elif point_group[0] == "D" and point_group[1:].isnumeric():

        tot_main_rot = int(point_group[1:])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # C2
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * 2 * pi
            symm_ops.append(
                rot_matrix(np.array([cos(theta), sin(theta), 0.0]), pi, l_max)
            )

    # Dnh
    elif point_group[0] == "D" and point_group[-1] == "h":

        # treat Dooh as D2h because same-atom symmetries are currently not exploited
        if point_group[1:-1] == "oo":
            tot_main_rot = 2
        else:
            tot_main_rot = int(point_group[1:-1])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # C2
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * 2 * pi
            symm_ops.append(
                rot_matrix(np.array([cos(theta), sin(theta), 0.0]), pi, l_max)
            )

        # Sn and i
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            if rot_angle == pi:
                symm_ops.append(inv_matrix(l_max))
            else:
                symm_ops.append(
                    rot_reflect_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                )

        # sigma_h
        symm_ops.append(reflect_matrix(np.array([0.0, 0.0, 1.0]), l_max))

        # sigma_v and sigma_d
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * 2 * pi
            symm_ops.append(
                reflect_matrix(np.array([cos(theta), sin(theta), 0.0]), l_max)
            )

    # Dnd
    elif point_group[0] == "D" and point_group[-1] == "d":

        tot_main_rot = int(point_group[1:-1])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # C2
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * 2 * pi
            symm_ops.append(
                rot_matrix(np.array([cos(theta), sin(theta), 0.0]), pi, l_max)
            )

        # S_2n
        for i in range(0, tot_main_rot):
            rot_angle = ((2 * i + 1) / (2 * tot_main_rot)) * 2 * pi
            if rot_angle == pi:
                symm_ops.append(inv_matrix(l_max))
            else:
                symm_ops.append(
                    rot_reflect_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                )

        # sigma_d
        for i in range(0, tot_main_rot):
            theta = ((2 * i + 1) / (2 * tot_main_rot)) * 2 * pi
            symm_ops.append(
                reflect_matrix(np.array([cos(theta), sin(theta), 0.0]), l_max)
            )

    # Cnv
    elif point_group[0] == "C" and point_group[-1] == "v":

        # treat Coov as C2v because same-atom symmetries are currently not exploited
        if point_group[1:-1] == "oo":
            tot_main_rot = 2
        else:
            tot_main_rot = int(point_group[1:-1])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # sigma_v and sigma_d
        for i in range(0, tot_main_rot):
            theta = (i / tot_main_rot) * 2 * pi
            symm_ops.append(
                reflect_matrix(np.array([cos(theta), sin(theta), 0.0]), l_max)
            )

    # Cnh
    elif point_group[0] == "C" and point_group[-1] == "h":

        tot_main_rot = int(point_group[1:-1])

        # Cn
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            symm_ops.append(rot_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max))

        # Sn and i
        for i in range(1, tot_main_rot):
            rot_angle = (i / tot_main_rot) * 2 * pi
            if rot_angle == pi:
                symm_ops.append(inv_matrix(l_max))
            else:
                symm_ops.append(
                    rot_reflect_matrix(np.array([0.0, 0.0, 1.0]), rot_angle, l_max)
                )

        # sigma_h
        symm_ops.append(reflect_matrix(np.array([0.0, 0.0, 1.0]), l_max))

    # cubic group O
    elif point_group == "O":

        corners, edges, surfaces = cubic_coords()

        # C3
        for coord in corners:
            symm_ops.append(rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(rot_matrix(coord, 4 * pi / 3, l_max))

        # C4
        tot_n_rot = 4
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in surfaces:
                symm_ops.append(rot_matrix(coord, rot_angle, l_max))

        # C2
        for coord in edges:
            symm_ops.append(rot_matrix(coord, pi, l_max))

    # cubic group T
    elif point_group == "T":

        corners, edges, surfaces = cubic_coords()

        # C2
        for coord in surfaces:
            symm_ops.append(rot_matrix(coord, pi, l_max))

        # C3
        for coord in corners:
            symm_ops.append(rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(rot_matrix(coord, 4 * pi / 3, l_max))

    # cubic group Oh
    elif point_group == "Oh":

        corners, edges, surfaces = cubic_coords()

        # C3
        for coord in corners:
            symm_ops.append(rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(rot_matrix(coord, 4 * pi / 3, l_max))

        # C4
        tot_n_rot = 4
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in surfaces:
                symm_ops.append(rot_matrix(coord, rot_angle, l_max))

        # C2
        for coord in edges:
            symm_ops.append(rot_matrix(coord, pi, l_max))

        # i
        symm_ops.append(inv_matrix(l_max))

        # sigma
        for coord in surfaces:
            symm_ops.append(reflect_matrix(coord, l_max))

        # S6
        for coord in corners:
            symm_ops.append(rot_reflect_matrix(coord, pi / 6, l_max))
            symm_ops.append(rot_reflect_matrix(coord, 5 * pi / 6, l_max))

        # S4
        tot_n_rot = 4
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            if rot_angle != pi:
                for coord in surfaces:
                    symm_ops.append(rot_reflect_matrix(coord, rot_angle, l_max))

        # sigma_d
        for coord in edges:
            symm_ops.append(reflect_matrix(coord, l_max))

    # cubic group Th
    elif point_group == "Th":

        corners, edges, surfaces = cubic_coords()

        # C2
        for coord in surfaces:
            symm_ops.append(rot_matrix(coord, pi, l_max))

        # C3
        for coord in corners:
            symm_ops.append(rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(rot_matrix(coord, 4 * pi / 3, l_max))

        # i
        symm_ops.append(inv_matrix(l_max))

        # sigma
        for coord in surfaces:
            symm_ops.append(reflect_matrix(coord, l_max))

        # S6
        for coord in corners:
            symm_ops.append(rot_reflect_matrix(coord, pi / 6, l_max))
            symm_ops.append(rot_reflect_matrix(coord, 5 * pi / 6, l_max))

    # cubic group Td
    elif point_group == "Td":

        corners, edges, surfaces = cubic_coords()

        # C2
        for coord in surfaces:
            symm_ops.append(rot_matrix(coord, pi, l_max))

        # C3
        for coord in corners:
            symm_ops.append(rot_matrix(coord, 2 * pi / 3, l_max))
            symm_ops.append(rot_matrix(coord, 4 * pi / 3, l_max))

        # S4
        for coord in surfaces:
            symm_ops.append(rot_reflect_matrix(coord, pi / 2, l_max))
            symm_ops.append(rot_reflect_matrix(coord, 3 * pi / 2, l_max))

        # sigma_d
        for coord in edges:
            symm_ops.append(reflect_matrix(coord, l_max))

    # icosahedral group I
    elif point_group == "I":

        corners, edges, surfaces = icosahedric_coords()

        # C5
        tot_n_rot = 5
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in corners:
                symm_ops.append(rot_matrix(coord, rot_angle, l_max))

        # C3
        tot_n_rot = 3
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in surfaces:
                symm_ops.append(rot_matrix(coord, rot_angle, l_max))

        # C2
        for coord in edges:
            symm_ops.append(rot_matrix(coord, pi, l_max))

    # icosahedral group Ih
    elif point_group == "Ih":

        corners, edges, surfaces = icosahedric_coords()

        # C5
        tot_n_rot = 5
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in corners:
                symm_ops.append(rot_matrix(coord, rot_angle, l_max))

        # C3
        tot_n_rot = 3
        for i in range(1, tot_n_rot):
            rot_angle = (i / tot_n_rot) * 2 * pi
            for coord in surfaces:
                symm_ops.append(rot_matrix(coord, rot_angle, l_max))

        # C2
        for coord in edges:
            symm_ops.append(rot_matrix(coord, pi, l_max))

        # i
        symm_ops.append(inv_matrix(l_max))

        # S10
        tot_main_rot = 10
        for i in range(0, tot_main_rot // 2):
            rot_angle = ((2 * i + 1) / tot_main_rot) * 2 * pi
            if rot_angle != pi:
                for coord in corners:
                    symm_ops.append(rot_reflect_matrix(coord, rot_angle, l_max))

        # S6
        tot_main_rot = 6
        for i in range(0, tot_main_rot // 2):
            rot_angle = ((2 * i + 1) / tot_main_rot) * 2 * pi
            if rot_angle != pi:
                for coord in surfaces:
                    symm_ops.append(rot_reflect_matrix(coord, rot_angle, l_max))

        # sigma
        for coord in edges:
            symm_ops.append(reflect_matrix(coord, l_max))

    else:

        raise PointGroupSymmetryError("Unknown Point Group.")

    return symm_ops


def cubic_coords() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    this function defines the coordinates of specific points within a cube
    """
    sqrt2d2 = sqrt(2) / 2

    corners = [
        np.array([0.0, sqrt2d2, -sqrt2d2]),
        np.array([-sqrt2d2, 0.0, -sqrt2d2]),
        np.array([0.0, sqrt2d2, sqrt2d2]),
        np.array([-sqrt2d2, 0.0, sqrt2d2]),
    ]

    edges = [
        np.array([sqrt2d2, 0.0, sqrt2d2]),
        np.array([sqrt2d2, 0.0, -sqrt2d2]),
        np.array([-sqrt2d2, 0.0, sqrt2d2]),
        np.array([-sqrt2d2, 0.0, -sqrt2d2]),
        np.array([sqrt2d2, sqrt2d2, 0.0]),
        np.array([sqrt2d2, -sqrt2d2, 0.0]),
    ]

    surfaces = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    return corners, edges, surfaces


def icosahedric_coords() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    this function defines the coordinates of specific points within an icosahedron
    """
    gr = (1 + sqrt(5)) / 2

    corners = [
        np.array([gr, 0.0, 1.0]),
        np.array([-gr, 0.0, 1.0]),
        np.array([0.0, 1.0, gr]),
        np.array([0.0, 1.0, -gr]),
        np.array([1.0, gr, 0.0]),
        np.array([1.0, -gr, 0.0]),
    ]

    onepgrd3 = (1 + gr) / 3
    twopgrd3 = (1 + gr) / 3
    edges = [
        np.array([onepgrd3, onepgrd3, onepgrd3]),
        np.array([onepgrd3, onepgrd3, -onepgrd3]),
        np.array([-onepgrd3, onepgrd3, onepgrd3]),
        np.array([onepgrd3, -onepgrd3, onepgrd3]),
        np.array([-twopgrd3, 0.0, 1 / 3]),
        np.array([0.0, 1 / 3, -twopgrd3]),
        np.array([1 / 3, -twopgrd3, 0.0]),
        np.array([twopgrd3, 0.0, 1 / 3]),
        np.array([0.0, 1 / 3, twopgrd3]),
        np.array([1 / 3, twopgrd3, 0.0]),
    ]

    onepgrd2 = (1 + gr) / 2
    onemgrd2 = (1 - gr) / 2
    surfaces = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([onepgrd2, gr / 2, 1 / 2]),
        np.array([-1 / 2, onepgrd2, gr / 2]),
        np.array([gr / 2, 1 / 2, -onepgrd2]),
        np.array([gr / 2, 1 / 2, onepgrd2]),
        np.array([1 / 2, -onepgrd2, gr / 2]),
        np.array([onemgrd2, gr / 2, -1 / 2]),
        np.array([-gr / 2, 1 / 2, onepgrd2]),
        np.array([onepgrd2, -gr / 2, 1 / 2]),
        np.array([1 / 2, onemgrd2, gr / 2]),
        np.array([-onepgrd2, gr / 2, 1 / 2]),
        np.array([gr / 2, -1 / 2, onepgrd2]),
        np.array([1 / 2, onepgrd2, 1 / gr]),
    ]

    return corners, edges, surfaces


def ident_matrix(l_max: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function constructs a cartesian identity matrix and identity matrices for all
    spherical harmonics until l_max
    """
    return np.eye(3), [np.eye(2 * l + 1) for l in range(l_max + 1)]


def inv_matrix(l_max: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function constructs a cartesian inversion matrix and inversion matrices for
    all spherical harmonics until l_max
    """
    return -np.eye(3), [(-1) ** l * np.eye(2 * l + 1) for l in range(l_max + 1)]


def rot_matrix(
    axis: np.ndarray, angle: float, l_max: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function constructs a cartesian rotation matrix and rotation matrices using
    the Wigner D-matrices for all spherical harmonics until l_max
    """
    axis = axis / np.linalg.norm(axis)

    sin_a = sin(angle)
    cos_a = cos(angle)

    rot = np.zeros((3, 3), dtype=np.float64)

    rot[0, 0] = cos_a + axis[0] ** 2 * (1 - cos_a)
    rot[0, 1] = axis[0] * axis[1] * (1 - cos_a) - axis[2] * sin_a
    rot[0, 2] = axis[0] * axis[2] * (1 - cos_a) + axis[1] * sin_a
    rot[1, 0] = axis[1] * axis[0] * (1 - cos_a) + axis[2] * sin_a
    rot[1, 1] = cos_a + axis[1] ** 2 * (1 - cos_a)
    rot[1, 2] = axis[1] * axis[2] * (1 - cos_a) - axis[0] * sin_a
    rot[2, 0] = axis[2] * axis[0] * (1 - cos_a) - axis[1] * sin_a
    rot[2, 1] = axis[2] * axis[1] * (1 - cos_a) + axis[0] * sin_a
    rot[2, 2] = cos_a + axis[2] ** 2 * (1 - cos_a)

    alpha, beta, gamma = symm.Dmatrix.get_euler_angles(np.eye(3), rot)

    D_mats = []

    for l in range(l_max + 1):

        D_mats.append(symm.Dmatrix.Dmatrix(l, alpha, beta, gamma, reorder_p=True))

    return rot, D_mats


def reflect_matrix(
    normal: np.ndarray, l_max: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    this function constructs a reflection matrix and reflection matrices for all
    spherical harmonics until l_max
    """
    cart_rot_mat, sph_rot_mats = rot_matrix(normal, pi, l_max)
    cart_inv_mat, sph_inv_mats = inv_matrix(l_max)

    return cart_rot_mat @ cart_inv_mat, [
        rot @ inv for rot, inv in zip(sph_rot_mats, sph_inv_mats)
    ]


def rot_reflect_matrix(
    axis: np.ndarray, angle: float, l_max: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    constructs a cartesian rotation-reflection matrix and rotation-reflection matrices
    for all spherical harmonics until l_max
    """
    cart_rot_mat, sph_rot_mats = rot_matrix(axis, angle, l_max)
    cart_reflect_mat, sph_reflect_mats = reflect_matrix(axis, l_max)

    return cart_rot_mat @ cart_reflect_mat, [
        rot @ reflect for rot, reflect in zip(sph_rot_mats, sph_reflect_mats)
    ]
