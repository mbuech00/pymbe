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
import operator
import numpy as np
from math import comb
from pyscf import symm, ao2mo, fci
from itertools import islice, combinations, groupby
from bisect import insort
from subprocess import Popen, PIPE
from typing import TYPE_CHECKING, overload

from pymbe.logger import logger

if TYPE_CHECKING:
    from typing import Tuple, List, Generator, Union, Optional, Dict, Set, Callable


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

    def __add__(self, other: Union[RDMCls, packedRDMCls]) -> RDMCls:
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

    def __sub__(self, other: Union[RDMCls, packedRDMCls]) -> RDMCls:
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

    def __mul__(self, other: Union[int, float]) -> RDMCls:
        """
        this function implements multiplication for the RDMCls objects
        """
        if isinstance(other, (int, float)):
            return RDMCls(other * self.rdm1, other * self.rdm2)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __imul__(self, other: Union[int, float]) -> RDMCls:
        """
        this function implements inplace multiplication for the RDMCls objects
        """
        if isinstance(other, (int, float)):
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

    __rtruediv__ = __truediv__

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
    this class describes packed RDMs
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

    @overload
    def __getitem__(self, idx: Union[int, np.int64]) -> RDMCls:
        ...

    @overload
    def __getitem__(self, idx: Union[slice, np.ndarray, List[int]]) -> packedRDMCls:
        ...

    def __getitem__(
        self, idx: Union[int, np.int64, slice, np.ndarray, List[int]]
    ) -> Union[RDMCls, packedRDMCls]:
        """
        this function ensures packedRDMCls can be retrieved through indexing
        packedRDMCls objects
        """
        if isinstance(idx, (int, np.integer)):
            return RDMCls(
                self.rdm1[idx][self.unpack_rdm1[self.idx]],
                self.rdm2[idx][self.unpack_rdm2[self.idx]],
            )
        elif isinstance(idx, (slice, np.ndarray, list)):
            return packedRDMCls(self.rdm1[idx], self.rdm2[idx], self.idx)
        else:
            return NotImplemented

    def __setitem__(
        self,
        idx: Union[int, np.int64, slice, np.ndarray, list],
        values: Union[
            float, np.ndarray, RDMCls, packedRDMCls, GenFockCls, packedGenFockCls
        ],
    ) -> packedRDMCls:
        """
        this function ensures indexed packedRDMCls can be set using packedRDMCls or
        RDMCls objects
        """
        if isinstance(idx, (slice, np.ndarray, list)) and isinstance(
            values, packedRDMCls
        ):
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


class GenFockCls:
    """
    this class holds the energy and generalized Fock matrix elements and defines all
    necessary operations
    """

    def __init__(self, energy: float, rdm1: np.ndarray, gen_fock: np.ndarray):
        """
        initializes a GenFock object
        """
        self.energy = energy
        self.rdm1 = rdm1
        self.gen_fock = gen_fock

    def __getitem__(
        self,
        idx: Tuple[np.ndarray, np.ndarray],
    ) -> GenFockCls:
        """
        this function ensures GenFockCls can be retrieved using one-dimensional
        indexing of GenFockCls objects
        """
        return GenFockCls(
            self.energy, self.rdm1[idx[0].reshape(-1, 1), idx[0]], self.gen_fock[idx[1]]
        )

    def __setitem__(
        self,
        idx: Tuple[np.ndarray, np.ndarray],
        values: Union[GenFockCls, Tuple[float, np.ndarray, np.ndarray]],
    ) -> GenFockCls:
        """
        this function implements setting GenFockCls through indexing for the GenFockCls
        objects and through tuples of a float and two arrays
        """
        if isinstance(values, GenFockCls):
            self.energy = values.energy
            self.rdm1[idx[0].reshape(-1, 1), idx[0]] = values.rdm1
            self.gen_fock[idx[1]] = values.gen_fock
        elif isinstance(values, tuple):
            self.energy = values[0]
            self.rdm1[idx[0].reshape(-1, 1), idx[0]] = values[1]
            self.gen_fock[idx[1]] = values[2]
        else:
            return NotImplemented

        return self

    def __add__(self, other: Union[GenFockCls, packedGenFockCls]) -> GenFockCls:
        """
        this function implements addition for the GenFockCls objects
        """
        if isinstance(other, GenFockCls):
            return GenFockCls(
                self.energy + other.energy,
                self.rdm1 + other.rdm1,
                self.gen_fock + other.gen_fock,
            )
        else:
            return NotImplemented

    def __iadd__(self, other: Union[GenFockCls, packedGenFockCls]) -> GenFockCls:
        """
        this function implements inplace addition for the GenFockCls objects
        """
        if isinstance(other, GenFockCls):
            self.energy += other.energy
            self.rdm1 += other.rdm1
            self.gen_fock += other.gen_fock
            return self
        else:
            return NotImplemented

    def __sub__(self, other: Union[GenFockCls, packedGenFockCls]) -> GenFockCls:
        """
        this function implements subtraction for the GenFockCls objects
        """
        if isinstance(other, GenFockCls):
            return GenFockCls(
                self.energy - other.energy,
                self.rdm1 - other.rdm1,
                self.gen_fock - other.gen_fock,
            )
        else:
            return NotImplemented

    def __isub__(self, other: Union[GenFockCls, packedGenFockCls]) -> GenFockCls:
        """
        this function implements inplace subtraction for the GenFockCls objects
        """
        if isinstance(other, GenFockCls):
            self.energy -= other.energy
            self.rdm1 -= other.rdm1
            self.gen_fock -= other.gen_fock
            return self
        else:
            return NotImplemented

    def __mul__(self, other: Union[int, float]) -> GenFockCls:
        """
        this function implements multiplication for the GenFockCls objects
        """
        if isinstance(other, (int, float)):
            return GenFockCls(
                other * self.energy, other * self.rdm1, other * self.gen_fock
            )
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __imul__(self, other: Union[int, float]) -> GenFockCls:
        """
        this function implements inplace multiplication for the GenFockCls objects
        """
        if isinstance(other, (int, float)):
            self.energy *= other
            self.rdm1 *= other
            self.gen_fock *= other
            return self
        else:
            return NotImplemented

    def __truediv__(self, other: Union[int, float]) -> GenFockCls:
        """
        this function implements division for the GenFockCls objects
        """
        if isinstance(other, (int, float)):
            return GenFockCls(
                self.energy / other, self.rdm1 / other, self.gen_fock / other
            )
        else:
            return NotImplemented

    def __itruediv__(self, other: Union[int, float]) -> GenFockCls:
        """
        this function implements inplace division for the GenFockCls objects
        """
        if isinstance(other, (int, float)):
            self.energy /= other
            self.rdm1 /= other
            self.gen_fock /= other
            return self
        else:
            return NotImplemented

    def fill(self, value: float) -> None:
        """
        this function defines the fill function for GenFockCls objects
        """
        self.energy = value
        self.rdm1.fill(value)
        self.gen_fock.fill(value)

    def copy(self) -> GenFockCls:
        """
        this function creates a copy of GenFockCls objects
        """
        return GenFockCls(self.energy, self.rdm1.copy(), self.gen_fock.copy())


class packedGenFockCls:
    """
    this class describes a packed version of GenFockCls
    """

    rdm1_size: List[int] = []
    pack_rdm1: List[Tuple[np.ndarray, np.ndarray]] = []
    unpack_rdm1: List[np.ndarray] = []

    def __init__(
        self, energy: np.ndarray, rdm1: np.ndarray, gen_fock: np.ndarray, idx: int = -1
    ) -> None:
        """
        this function initializes a packedGenFockCls object
        """
        self.energy = energy
        self.rdm1 = rdm1
        self.gen_fock = gen_fock
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

    @classmethod
    def get_pack_idx(cls, norb) -> None:
        """
        this function generates packing and unpacking indices for the 1-particle rdms
        and should be called at every order before a packedGenFockCls object is
        initialized at this order
        """
        pack_rdm1 = np.triu_indices(norb)
        unpack_rdm1 = np.zeros((norb, norb), dtype=np.int64)
        rdm1_size = pack_rdm1[0].size
        indices = np.arange(rdm1_size)
        unpack_rdm1[pack_rdm1] = indices
        unpack_rdm1[pack_rdm1[1], pack_rdm1[0]] = indices

        cls.rdm1_size.append(rdm1_size)
        cls.pack_rdm1.append(pack_rdm1)
        cls.unpack_rdm1.append(unpack_rdm1)

    @overload
    def __getitem__(self, idx: Union[int, np.int64]) -> GenFockCls:
        ...

    @overload
    def __getitem__(self, idx: Union[slice, np.ndarray, List[int]]) -> packedGenFockCls:
        ...

    def __getitem__(
        self, idx: Union[int, np.int64, slice, np.ndarray, List[int]]
    ) -> Union[GenFockCls, packedGenFockCls]:
        """
        this function ensures packedGenFockCls can be retrieved through indexing
        packedGenFockCls objects
        """
        if isinstance(idx, (int, np.int64, slice, np.ndarray, list)):
            return packedGenFockCls(
                self.energy[idx], self.rdm1[idx], self.gen_fock[idx], self.idx
            )
        else:
            return NotImplemented

    def __setitem__(
        self,
        idx: Union[int, np.int64, slice, np.ndarray, List[int]],
        values: Union[
            float, np.ndarray, RDMCls, packedRDMCls, GenFockCls, packedGenFockCls
        ],
    ) -> packedGenFockCls:
        """
        this function ensures indexed packedGenFockCls can be set using packedGenFockCls
        or GenFockCls objects
        """
        if isinstance(idx, (slice, np.ndarray, list)) and isinstance(
            values, packedGenFockCls
        ):
            self.energy[idx] = values.energy
            self.rdm1[idx] = values.rdm1
            self.gen_fock[idx] = values.gen_fock
        elif isinstance(idx, (int, np.integer)) and isinstance(values, GenFockCls):
            self.energy[idx] = values.energy
            self.rdm1[idx] = values.rdm1[self.pack_rdm1[self.idx]]
            self.gen_fock[idx] = values.gen_fock
        else:
            return NotImplemented

        return self

    def __radd__(self, other: GenFockCls) -> GenFockCls:
        """
        this function ensures the packedGenFockCls object is unpacked when added to a
        GenFockCls object
        """
        if (
            isinstance(other, GenFockCls)
            and self.energy.size == 1
            and self.rdm1.ndim == 1
            and self.gen_fock.ndim == 2
        ):
            return other + GenFockCls(
                self.energy.item(), self.rdm1[self.unpack_rdm1[self.idx]], self.gen_fock
            )
        else:
            return NotImplemented

    def fill(self, value: float) -> None:
        """
        this function defines the fill function for packedGenFockCls objects
        """
        self.energy.fill(value)
        self.rdm1.fill(value)
        self.gen_fock.fill(value)


def logger_config(verbose: int) -> None:
    """
    this function configures the pymbe logger
    """
    # corresponding logging level
    verbose_level = {0: 30, 1: 20, 2: 15, 3: 10, 4: 5}

    # set level for logger
    logger.setLevel(verbose_level[verbose])


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
    order_start: int = 0,
    occ_start: int = 0,
    virt_start: int = 0,
) -> Generator[np.ndarray, None, None]:
    """
    this function is the main generator for tuples
    """
    for k in range(order_start, order + 1):
        if valid_tup(ref_nelec, ref_nhole, k, order - k, vanish_exc):
            if k == 0:
                # only virtual MOs
                for tup_virt in islice(
                    combinations(virt_space, order), virt_start, None
                ):
                    yield np.array(tup_virt, dtype=np.int64)
                virt_start = 0
            elif 0 < k < order:
                # combinations of occupied and virtual MOs
                for tup_occ in islice(combinations(occ_space, k), occ_start, None):
                    for tup_virt in islice(
                        combinations(virt_space, order - k), virt_start, None
                    ):
                        yield np.array(tup_occ + tup_virt, dtype=np.int64)
                    virt_start = 0
                occ_start = 0
            elif k == order:
                # only occupied MOs
                for tup_occ in islice(combinations(occ_space, order), occ_start, None):
                    yield np.array(tup_occ, dtype=np.int64)
                occ_start = 0


def orb_tuples_with_nocc(
    occ_space: np.ndarray,
    virt_space: np.ndarray,
    order: int,
    nocc: int,
    orb: int
) -> Generator[np.ndarray, None, None]:
    """
    this function is the main generator for tuples for a given occupation that include
    a certain orbital
    """
    # orbital is occupied
    if orb in occ_space:
        # remove orbital
        occ_space = np.delete(occ_space, np.where(occ_space == orb)[0][0])

        # only virtual orbitals
        if nocc == 1:
            for tup_virt in (
                list(tup) for tup in combinations(virt_space, order - 1)
            ):
                yield np.array([orb] + tup_virt, dtype=np.int64)

        # combinations of occupied and virtual MOs
        elif 1 < nocc < order:
            for tup_occ in (list(tup) for tup in combinations(occ_space, nocc - 1)):
                insort(tup_occ, orb)
                for tup_virt in (
                    list(tup) for tup in combinations(virt_space, order - nocc)
                ):
                    yield np.array(tup_occ + tup_virt, dtype=np.int64)

        # only occupied MOs
        elif nocc == order:
            for tup_occ in (
                list(tup) for tup in combinations(occ_space, order - 1)
            ):
                insort(tup_occ, orb)
                yield np.array(tup_occ, dtype=np.int64)

    # orbital is virtual
    elif orb in virt_space:
        # remove orbital
        virt_space = np.delete(virt_space, np.where(virt_space == orb)[0][0])

        # only virtual MOs
        if nocc == 0:
            for tup_virt in (
                list(tup) for tup in combinations(virt_space, order - 1)
            ):
                insort(tup_virt, orb)
                yield np.array(tup_virt, dtype=np.int64)

        # combinations of occupied and virtual MOs
        elif 0 < nocc < order - 1:
            for tup_occ in (list(tup) for tup in combinations(occ_space, nocc)):
                for tup_virt in (
                    list(tup) for tup in combinations(virt_space, order - 1 - nocc)
                ):
                    insort(tup_virt, orb)
                    yield np.array(tup_occ + tup_virt, dtype=np.int64)

        # only occupied MOs
        elif nocc == order - 1:
            for tup_occ in (
                list(tup) for tup in combinations(occ_space, order - 1)
            ):
                yield np.array(tup_occ + [orb], dtype=np.int64)


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
        order_start = 0
        occ_start = virt_start = 0
    elif tup_occ is not None and tup_virt is not None:
        order_start = tup_occ.size
        occ_start = _comb_idx(occ_space, tup_occ)
        virt_start = _comb_idx(virt_space, tup_virt)
    elif tup_occ is not None and tup_virt is None:
        order_start = tup_occ.size
        occ_start = _comb_idx(occ_space, tup_occ)
        virt_start = 0
    elif tup_occ is None and tup_virt is not None:
        order_start = 0
        occ_start = 0
        virt_start = _comb_idx(virt_space, tup_virt)
    return order_start, occ_start, virt_start


def tuples_with_nocc(
    occ_space: np.ndarray, virt_space: np.ndarray, order: int, nocc: int
) -> Generator[np.ndarray, None, None]:
    """
    this function is the main generator for tuples for a given number of occupied
    orbitals
    """
    # only virtual MOs
    if nocc == 0:
        for tup_virt in combinations(virt_space, order):
            yield np.array(tup_virt, dtype=np.int64)
    # combinations of occupied and virtual MOs
    elif 0 < nocc < order:
        for tup_occ in combinations(occ_space, nocc):
            for tup_virt in combinations(virt_space, order - nocc):
                yield np.array(tup_occ + tup_virt, dtype=np.int64)
    # only occupied MOs
    elif nocc == order:
        for tup_occ in combinations(occ_space, order):
            yield np.array(tup_occ, dtype=np.int64)


def tuples_and_virt_with_nocc(
    occ_space: np.ndarray, virt_space: np.ndarray, order: int, nocc: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    this function is the main generator for tuples and their corresponding virtual
    subset for a given number of occupied orbitals
    """
    # only virtual MOs
    if nocc == 0:
        for tup_virt in combinations(virt_space, order):
            yield np.array(tup_virt, dtype=np.int64), np.array(tup_virt, dtype=np.int64)
    # combinations of occupied and virtual MOs
    elif 0 < nocc < order:
        for tup_occ in combinations(occ_space, nocc):
            for tup_virt in combinations(virt_space, order - nocc):
                yield np.array(tup_occ + tup_virt, dtype=np.int64), np.array(
                    tup_virt, dtype=np.int64
                )
    # only occupied MOs
    elif nocc == order:
        for tup_occ in combinations(occ_space, order):
            yield np.array(tup_occ, dtype=np.int64), np.array([], dtype=np.int64)


def _comb_idx(space: np.ndarray, tup: np.ndarray) -> int:
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


def _idx(space: np.ndarray, idx: int, order: int) -> int:
    """
    this function return the start index of element space[idx] in position (order+1)
    from the right in a given combination
    """
    return sum((comb(space[i < space].size, (order - 1)) for i in space[space < idx]))


def n_tuples(
    occ_space: np.ndarray,
    virt_space: np.ndarray,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
    tup_nocc: int,
) -> int:
    """
    this function returns the total number of tuples of a given order
    """
    # init n_tuples
    n = 0

    for k in range(order + 1):
        if valid_tup(ref_nelec, ref_nhole, k, order - k, vanish_exc):
            n += comb(occ_space.size, k) * comb(virt_space.size, order - k)

    return n


def n_tuples_with_nocc(
    occ_space: np.ndarray,
    virt_space: np.ndarray,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
    tup_nocc: int,
) -> int:
    """
    this function returns the total number of tuples of a given order and a given
    occupation
    """
    # check if tuple is valid for chosen method
    if valid_tup(ref_nelec, ref_nhole, tup_nocc, order - tup_nocc, vanish_exc):
        return comb(occ_space.size, tup_nocc) * comb(virt_space.size, order - tup_nocc)
    else:
        return 0


def orb_n_tuples(
    occ_space: np.ndarray,
    virt_space: np.ndarray,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    order: int,
    occ_type: str,
) -> int:
    """
    this function returns the both the total number of tuples of a given order that
    include a specific occupied or a specific virtual orbital
    """
    # initialize ntup_occ
    ntup = 0

    if occ_type == "occ" and occ_space.size > 0:
        # combinations of occupied and virtual MOs
        for k in range(1, order):
            if valid_tup(ref_nelec, ref_nhole, k, order - k, vanish_exc):
                ntup += comb(occ_space.size - 1, k - 1) * comb(
                    virt_space.size, order - k
                )

        # only occupied MOs
        if valid_tup(ref_nelec, ref_nhole, order, 0, vanish_exc):
            ntup += comb(occ_space.size - 1, order - 1)

    elif occ_type == "virt" and virt_space.size > 0:
        # combinations of occupied and virtual MOs
        for k in range(1, order):
            if valid_tup(ref_nelec, ref_nhole, k, order - k, vanish_exc):
                ntup += comb(occ_space.size, k) * comb(
                    virt_space.size - 1, order - k - 1
                )

        # only virtual MOs
        if valid_tup(ref_nelec, ref_nhole, 0, order, vanish_exc):
            ntup += comb(virt_space.size - 1, order - 1)

    return ntup


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
    core_idx = np.setdiff1d(np.arange(nocc), cas_idx, assume_unique=True)
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
    cas_idx: np.ndarray,
    symm_orbs: List[Dict[int, Tuple[int, ...]]],
    ref_space: Optional[np.ndarray],
) -> Optional[Set[Tuple[int, ...]]]:
    """
    this function returns a set of symmetry-equivalent cas_idx that will yield the same
    CASCI property if the supplied cas_idx is lexicographically greatest with respect
    to all symmetry operations, otherwise returns None
    """
    # initialize set of symmetry-equivalent tuples
    eqv_set = {tuple(cas_idx)}

    # loop over symmetry operations in point group
    for symm_op in symm_orbs:
        # get permuted cas space by applying symmetry operation
        perm_cas = apply_symm_op(symm_op, cas_idx)

        # skip this symmetry operation
        if perm_cas is None or (
            ref_space is not None
            and np.intersect1d(ref_space, perm_cas, assume_unique=True).size
            < ref_space.size
        ):
            continue

        # sort permuted cas space
        perm_cas.sort()

        # loop over orbs in cas space and permuted cas space
        for orb, perm_orb in zip(cas_idx, perm_cas):
            # check if orb in cas space is smaller than orb in permuted cas space
            if orb < perm_orb:
                # tuple is not unique and not lexicographically smaller
                return None

            # check if orb in cas space is greater than orb in permuted cas space
            elif orb > perm_orb:
                # tuple is lexicographically greater
                break

        eqv_set.add(tuple(perm_cas))

    return eqv_set


def symm_eqv_inc(
    symm_orbs: np.ndarray,
    eqv_tup_set: Set[Tuple[int, ...]],
    ref_space: Optional[np.ndarray],
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    this function returns a list of sets of tuples that yield symmetrically equivalent
    increments and returns the lexicographically greatest increment for every set
    """
    # initialize list of sets of tuples with the same increments
    eqv_inc_sets = []

    # initialize list of lexicographically greates increment for every set
    lex_cas = []

    # check if all tuples in set will produce the same increment
    while len(eqv_tup_set) > 0:
        # start with random tuple in set of equivalent tuples
        curr_cas = eqv_tup_set.pop()

        # add new set of equivalent increments
        eqv_inc_sets.append({curr_cas})

        # add new lexicographically greatest equivalent increment
        lex_cas.append(np.array(curr_cas, dtype=np.int64))

        # get permuted cas spaces by applying symmetry operations
        perm_cas_spaces = symm_orbs[:, curr_cas]

        # sort permuted cas spaces
        perm_cas_spaces.sort()

        # loop over symmetry operations
        for perm_cas in perm_cas_spaces:
            # skip this symmetry operation
            if perm_cas[0] == -1 or (
                ref_space is not None
                and np.intersect1d(ref_space, perm_cas, assume_unique=True).size
                < ref_space.size
            ):
                continue

            # convert to tuple
            tup_perm_cas = tuple(perm_cas)

            # check if tuple has been added to set of tuples with the same increments
            if tup_perm_cas not in eqv_inc_sets[-1]:
                # add permuted tuple to set of tuples with the same increments
                eqv_inc_sets[-1].add(tup_perm_cas)

                # remove permuted tuple from set of symmetrically equivalent tuples
                eqv_tup_set.remove(tup_perm_cas)

                # loop over orbs in lex_cas and perm_cas
                for lex_orb, perm_orb in zip(lex_cas[-1], perm_cas):
                    # check if orb in lex_cas is smaller than orb in perm_cas
                    if lex_orb < perm_orb:
                        # set permuted cas space as lexicographically greatest
                        lex_cas[-1] = perm_cas

                        # perm_cas is lexicographically greater
                        break

                    # check if orb in lex_cas is greater than orb in perm_cas
                    elif lex_orb > perm_orb:
                        # perm_cas is lexicographically smaller
                        break

    # remove reference space
    if ref_space is not None:
        lex_tup = [
            np.setdiff1d(cas_idx, ref_space, assume_unique=True) for cas_idx in lex_cas
        ]
        eqv_inc_tups = [
            [
                np.setdiff1d(cas_idx, ref_space, assume_unique=True)
                for cas_idx in eqv_inc_set
            ]
            for eqv_inc_set in eqv_inc_sets
        ]
    else:
        lex_tup = lex_cas
        eqv_inc_tups = [
            [np.array(cas_idx, dtype=np.int64) for cas_idx in eqv_inc_set]
            for eqv_inc_set in eqv_inc_sets
        ]

    return lex_tup, eqv_inc_tups


def is_lex_tup(
    cas_idx: np.ndarray, symm_orbs: np.ndarray, ref_space: Optional[np.ndarray]
) -> bool:
    """
    this function returns whether the supplied cas_idx is the lexicographically
    greatest cas_idx
    """
    # get permuted cas spaces by applying symmetry operations
    perm_cas_spaces = symm_orbs[:, cas_idx]

    # sort permuted cas spaces
    perm_cas_spaces.sort()

    # loop over symmetry operations in point group
    for perm_cas in perm_cas_spaces:
        # skip this symmetry operation
        if (
            ref_space is not None
            and np.intersect1d(ref_space, perm_cas, assume_unique=True).size
            < ref_space.size
        ):
            continue

        # loop over orbs in cas_idx and perm_cas
        for curr_orb, perm_orb in zip(cas_idx, perm_cas):
            # check if orb in cas_idx is smaller than orb in perm_cas
            if curr_orb < perm_orb:
                # perm_cas is lexicographically greater
                return False

            # check if orb in lex_cas is greater than orb in perm_cas
            elif curr_orb > perm_orb:
                # perm_cas is lexicographically smaller
                break

    return True


def get_lex_cas(
    cas_idx: np.ndarray, symm_orbs: np.ndarray, ref_space: Optional[np.ndarray]
) -> np.ndarray:
    """
    this function returns the symmetrically equivalent but lexicographically greater
    cas_idx
    """
    # initialize current lexicographically greatest cas space
    lex_cas = cas_idx

    # get permuted cas spaces by applying symmetry operations
    perm_cas_spaces = symm_orbs[:, cas_idx]

    # sort permuted cas spaces
    perm_cas_spaces.sort()

    # loop over symmetry operations in point group
    for perm_cas in perm_cas_spaces:
        # skip this symmetry operation
        if (
            ref_space is not None
            and np.intersect1d(ref_space, perm_cas, assume_unique=True).size
            < ref_space.size
        ):
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

    return lex_cas


def apply_symm_op(
    symm_op: Dict[int, Tuple[int, ...]], tup: np.ndarray
) -> Optional[np.ndarray]:
    """
    this function applies a symmetry operation to a tuple of orbitals
    """
    try:
        perm_set: Set[int] = set()
        try:
            perm_set.update(*operator.itemgetter(*tup)(symm_op))
        except TypeError:
            perm_set.update(operator.itemgetter(*tup)(symm_op))
    except KeyError:
        return None
    set_len = len(perm_set)
    if set_len == tup.size:
        return np.fromiter(perm_set, np.int64, count=set_len)
    else:
        return None


def reduce_symm_eqv_orbs(
    symm_eqv_orbs: List[Dict[int, Tuple[int, ...]]], space: np.ndarray
) -> List[Dict[int, Tuple[int, ...]]]:
    """
    this function only returns symmetry-equivalent orbitals included in the supplied
    space
    """
    return [
        {
            orb: tuple(tup_orb for tup_orb in tup)
            for orb, tup in symm_op.items()
            if orb in space and np.isin(tup, space).all()
        }
        for symm_op in symm_eqv_orbs
    ]


def get_eqv_inc_orbs(
    symm_eqv_orbs: List[Dict[int, Tuple[int, ...]]], nsymm: int, norb: int
) -> np.ndarray:
    """
    this function returns the orbital combinations necessary to produce a
    symmetry-equivalent increment
    """
    eqv_inc_orbs = np.empty((nsymm, norb), dtype=np.int64)
    for op, symm_op in enumerate(symm_eqv_orbs):
        for orb1 in range(norb):
            orb2 = symm_op.get(orb1, None)
            if orb2 is not None and len(orb2) == 1:
                eqv_inc_orbs[op, orb1] = orb2[0]
            else:
                eqv_inc_orbs[op, orb1] = -1
    return eqv_inc_orbs


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
    return np.add.reduce(np.minimum(nelec, nhole))


def valid_tup(
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    tup_nocc: int,
    tup_nvirt: int,
    vanish_exc: int,
) -> bool:
    """
    this function returns true if a tuple kind produces a non-vanishing correlation
    energy by calculating the number of excitations (equivalent to get_nexc but faster)
    and comparing to the number of vanishing excitation for the used model
    """
    return (
        min(ref_nelec[0] + tup_nocc, ref_nhole[0] + tup_nvirt)
        + min(ref_nelec[1] + tup_nocc, ref_nhole[1] + tup_nvirt)
    ) > vanish_exc


def is_file(string: str, order: int) -> bool:
    """
    this function looks to see if a general restart file corresponding to the input
    string exists
    """
    if order is None:
        return os.path.isfile(os.path.join(RST, f"{string}.npy"))
    else:
        return os.path.isfile(os.path.join(RST, f"{string}_{order}.npy"))


def write_file(
    arr: np.ndarray,
    string: str,
    order: Optional[int] = None,
    nocc: Optional[int] = None,
) -> None:
    """
    this function writes a general restart file corresponding to the input string
    """
    if order is None and nocc is None:
        np.save(os.path.join(RST, f"{string}"), arr)
    elif nocc is None:
        np.save(os.path.join(RST, f"{string}_{order}"), arr)
    else:
        np.save(os.path.join(RST, f"{string}_{order}_{nocc}"), arr)


def write_file_mult(
    arrs: Dict[str, np.ndarray], string: str, order: Optional[int] = None
) -> None:
    """
    this function writes a general restart file corresponding to the input string for
    multiple arrays
    """
    if order is None:
        np.savez(os.path.join(RST, f"{string}"), **arrs)
    else:
        np.savez(os.path.join(RST, f"{string}_{order}"), **arrs)


def read_file(string: str, order: Optional[int] = None) -> np.ndarray:
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


def get_vhf(eri: np.ndarray, nocc: int, norb: int) -> np.ndarray:
    """
    this function determines the Hartree-Fock potential from the electron repulsion
    integrals
    """
    eri = ao2mo.restore(1, eri, norb)

    vhf = np.empty((nocc, norb, norb), dtype=np.float64)
    for i in range(nocc):
        vhf[i] = 2.0 * eri[i, i, :, :] - eri[:, i, i, :]

    return vhf


def get_occup(norb: int, nelec: np.ndarray) -> np.ndarray:
    """
    this function generates the Hartree-Fock occupation vector
    """
    occup = np.zeros(norb, dtype=np.int64)
    occup[: np.amin(nelec)] = 2
    occup[np.amin(nelec) : np.amax(nelec)] = 1

    return occup


def e_core_h1e(
    hcore: np.ndarray, vhf: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    this function returns core energy and cas space 1e integrals
    """
    # determine effective core fock potential
    if core_idx.size > 0:
        core_vhf = np.sum(vhf[core_idx], axis=0)
    else:
        core_vhf = 0.0

    # calculate core energy
    e_core = np.trace((hcore + 0.5 * core_vhf)[core_idx[:, None], core_idx]) * 2.0

    # extract cas integrals
    h1e_cas = (hcore + core_vhf)[cas_idx[:, None], cas_idx]

    return e_core, h1e_cas


def cf_prefactor(contrib_order: int, order: int, max_order: int) -> int:
    """
    this function calculates the prefactor for the closed form MBE
    """
    return (-1) ** (order - contrib_order) * comb(
        max_order - contrib_order - 1, order - contrib_order
    )


def hop_no_singles(
    solver: Union[fci.direct_spin0_symm.FCI, fci.direct_spin1_symm.FCI],
    norb: int,
    nelec: np.ndarray,
    spin: int,
    h1e: np.ndarray,
    h2e: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    this function generates a special function for the hamiltonian operation where
    singles are omitted
    """
    if spin == 0:
        link_index = fci.cistring.gen_linkstr_index_trilidx(range(norb), nelec[0])
        na = link_index.shape[0]
        t1_addrs = np.array(
            [
                fci.cistring.str2addr(norb, nelec[0], x)
                for x in fci.cistring.tn_strs(norb, nelec[0], 1)
            ]
        )
        h2e_abs = solver.absorb_h1e(h1e, h2e, norb, nelec, 0.5)

        def hop(c):
            hc = solver.contract_2e(h2e_abs, c.reshape(na, na), norb, nelec, link_index)
            hc[t1_addrs, 0] = 0.0
            hc[0, t1_addrs] = 0.0
            return hc.ravel()

    else:
        link_indexa = fci.cistring.gen_linkstr_index_trilidx(range(norb), nelec[0])
        link_indexb = fci.cistring.gen_linkstr_index_trilidx(range(norb), nelec[1])
        t1_addrs_a = np.array(
            [
                fci.cistring.str2addr(norb, nelec[0], x)
                for x in fci.cistring.tn_strs(norb, nelec[0], 1)
            ]
        )
        t1_addrs_b = np.array(
            [
                fci.cistring.str2addr(norb, nelec[1], x)
                for x in fci.cistring.tn_strs(norb, nelec[1], 1)
            ]
        )
        h2e_abs = solver.absorb_h1e(h1e, h2e, norb, nelec, 0.5)

        def hop(c):
            hc = solver.contract_2e(h2e_abs, c, norb, nelec, (link_indexa, link_indexb))
            if t1_addrs_a.size > 0:
                hc[t1_addrs_a] = 0.0
            if t1_addrs_b.size > 0:
                hc[t1_addrs_b * na] = 0.0
            return hc.ravel()

    return hop


def get_subspace_det_addr(
    cas: np.ndarray,
    cas_nelec: np.ndarray,
    subspace: np.ndarray,
    subspace_nelec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function determines the adresses of subspace determinants in a CAS wavefunction
    """
    # get indices of orbitals inside and outside subspace within cas
    inside_idx = np.searchsorted(cas, subspace)
    outside_idx = np.delete(np.arange(cas.size), inside_idx)

    # get indices of occupied orbitals outside subspace (should be the same regardless
    # of spin if subspace includes all singly occupied orbitals)
    occ_outside_idx = outside_idx[outside_idx < cas_nelec[0]]

    # write bitstring for occupied orbitals outside subspace
    outside_str = 0
    for i in occ_outside_idx:
        outside_str = outside_str | (1 << i)

    # make bitstrings for subspace determinants
    strs_a = fci.cistring.make_strings(inside_idx, subspace_nelec[0])
    strs_b = fci.cistring.make_strings(inside_idx, subspace_nelec[1])

    # add occupied orbitals outside subspace
    strs_a = [string | outside_str for string in strs_a]
    strs_b = [string | outside_str for string in strs_b]

    # get adresses
    subspace_addr_a = fci.cistring.strs2addr(cas.size, cas_nelec[0], strs_a)
    subspace_addr_b = fci.cistring.strs2addr(cas.size, cas_nelec[1], strs_b)

    return subspace_addr_a, subspace_addr_b


def init_wfn(norb: int, nelec: np.ndarray, length: int) -> np.ndarray:
    """
    this function initializes a wavefunction
    """
    # determine size of ci vector
    na = fci.cistring.num_strings(norb, nelec[0])
    nb = fci.cistring.num_strings(norb, nelec[1])

    # initialize ci vector
    civec = np.zeros((length, na, nb))

    return civec
