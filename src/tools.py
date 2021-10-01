#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import os
import re
import sys
import numpy as np
import scipy.special as sc
from mpi4py import MPI
from itertools import islice, combinations, groupby
from math import floor, fsum as math_fsum
from subprocess import Popen, PIPE
from traceback import format_stack
from contextlib import contextmanager
from typing import Tuple, Set, List, Dict, Any, Generator, Union

# restart folder
RST = os.getcwd()+'/rst'
# pi-orbitals
PI_SYMM_D2H = np.array([2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27])
PI_SYMM_C2V = np.array([2, 3, 10, 11, 12, 13, 20, 21, 22, 23])


class Logger:
        """
        this class pipes all write statements to both stdout and output_file
        """
        def __init__(self, output_file: str, both: bool = True) -> None:
            """
            init Logger
            """
            self.terminal = sys.stdout
            self.log = open(output_file, 'a')
            self.both = both

        def write(self, message: str) -> None:
            """
            define write
            """
            self.log.write(message)
            if self.both:
                self.terminal.write(message)

        def flush(self):
            """
            define flush
            """
            pass


@contextmanager
def suppress_stdout():
        """
        this function suppresses stdout
        see: https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
        """
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


def git_version() -> str:
        """
        this function returns the git revision as a string
        see: https://github.com/numpy/numpy/blob/master/setup.py#L70-L92
        """
        def _minimal_ext_cmd(cmd: List[str]) -> bytes:
            env = {}
            for k in ['SYSTEMROOT', 'PATH', 'HOME']:
                v = os.environ.get(k)
                if v is not None:
                    env[k] = v
            # LANGUAGE is used on win32
            env['LANGUAGE'] = 'C'
            env['LANG'] = 'C'
            env['LC_ALL'] = 'C'
            out = Popen(cmd, stdout = PIPE, env=env, cwd=get_pymbe_path()).communicate()[0]
            return out

        try:
            out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
            GIT_REVISION = out.strip().decode('ascii')
        except OSError:
            GIT_REVISION = "Unknown"

        return GIT_REVISION


def get_pymbe_path() -> str:
        """
        this function returns the path to pymbe
        """
        return os.path.dirname(__file__)


def assertion(cond: bool, reason: str) -> None:
        """
        this function returns an assertion of a given condition
        """
        if not cond:
            # get stack
            stack = ''.join(format_stack()[:-1])
            # print stack
            print('\n\n'+stack)
            print('\n\n*** PyMBE assertion error: '+reason+' ***\n\n')
            # abort mpi
            MPI.COMM_WORLD.Abort()


def time_str(time: float) -> str:
        """
        this function returns time as a HH:MM:SS string
        """
        # hours, minutes, and seconds
        hours = time // 3600.
        minutes = (time - (time // 3600) * 3600.) // 60.
        seconds = time - hours * 3600. - minutes * 60.

        # init time string
        string: str = ''
        form: Tuple[float, ...] = ()

        # write time string
        if hours > 0:
            string += '{:.0f}h '
            form += (hours,)

        if minutes > 0:
            string += '{:.0f}m '
            form += (minutes,)

        string += '{:.2f}s'
        form += (seconds,)

        return string.format(*form)


def fsum(a: np.ndarray) -> Union[float, np.ndarray]:
        """
        this function uses math.fsum to safely sum 1d array or 2d array (column-wise)
        """
        if a.ndim == 1:
            return math_fsum(a)
        elif a.ndim == 2:
            return np.fromiter(map(math_fsum, a.T), dtype=a.dtype, count=a.shape[1])
        else:
            raise NotImplementedError('tools.py: _fsum()')


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


def hash_lookup(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, None]:
        """
        this function finds occurences of b in a through a binary search
        """
        left = a.searchsorted(b, side='left')
        right = a.searchsorted(b, side='right')
        if ((right - left) > 0).all():
            return left
        else:
            return None


def tuples(occ_space: np.ndarray, virt_space: np.ndarray, ref_occ: bool, ref_virt: bool, order: int, \
           order_start: int = 1, occ_start: int = 0, virt_start: int = 0) -> Generator[np.ndarray, None, None]:
        """
        this function is the main generator for tuples
        """
        # combinations of occupied and virtual MOs
        for k in range(order_start, order):
            for tup_occ in islice(combinations(occ_space, k), occ_start, None):
                for tup_virt in islice(combinations(virt_space, order - k), virt_start, None):
                    yield np.array(tup_occ + tup_virt, dtype=np.int64)
                virt_start = 0
            occ_start = 0

        # only occupied MOs
        if ref_virt and 0 <= occ_start:
            for tup_occ in islice(combinations(occ_space, order), occ_start, None):
                yield np.array(tup_occ, dtype=np.int64)

        # only virtual MOs
        if ref_occ and 0 <= virt_start:
            for tup_virt in islice(combinations(virt_space, order), virt_start, None):
                yield np.array(tup_virt, dtype=np.int64)


def start_idx(occ_space: np.ndarray, virt_space: np.ndarray, \
              tup_occ: np.ndarray, tup_virt: np.ndarray) -> Tuple[int, int, int]:
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
        this function return the index of a given (ordered) combination
        returned from itertools.combinations
        """
        idx = _idx(space, tup[0], tup.size)
        idx += sum((_idx(space[tup[i-1] < space], tup[i], tup[i:].size) for i in range(1, tup.size)))
        return idx


def _idx(space: np.ndarray, idx: int, order: int) -> float:
        """
        this function return the start index of element space[idx] in
        position (order+1) from the right in a given combination
        """
        return sum((sc.binom(space[i < space].size, (order - 1)) for i in space[space < idx]))


def n_tuples(occ_space: np.ndarray, virt_space: np.ndarray, \
                ref_occ: bool, ref_virt: bool, order: int) -> int:
        """
        this function returns the total number of tuples of a given order
        """
        # init n_tuples
        n = 0.

        # combinations of occupied and virtual MOs
        for k in range(1, order):
            n += sc.binom(occ_space.size, k) * sc.binom(virt_space.size, order - k)

        # only occupied MOs
        if ref_virt:
            n += sc.binom(occ_space.size, order)

        # only virtual MOs
        if ref_occ:
            n += sc.binom(virt_space.size, order)

        return int(n)


def cas(ref_space: np.ndarray, tup: np.ndarray) -> np.ndarray:
        """
        this function returns a cas space
        """
        return np.sort(np.append(ref_space, tup))


def core_cas(nocc: int, ref_space: np.ndarray, tup: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        i = ij[0]; j = ij[1]
        if i >= j:
            return i * (i + 1) // 2 + j
        else:
            return j * (j + 1) // 2 + i


def idx_tril(cas_idx: np.ndarray) -> np.ndarray:
        """
        this function returns lower triangular cas indices
        """
        cas_idx_cart = _cas_idx_cart(cas_idx)
        return np.unique(np.fromiter(map(_coor_to_idx, cas_idx_cart), \
                                     dtype=cas_idx_cart.dtype, count=cas_idx_cart.shape[0]))


def pi_space(group: str, orbsym: np.ndarray, exp_space: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns pi-orbitals and hashes from total expansion space
        """
        # all pi-orbitals
        if group == 'Dooh':
            pi_space_arr = exp_space[np.in1d(orbsym[exp_space], PI_SYMM_D2H)]
        else:
            pi_space_arr = exp_space[np.in1d(orbsym[exp_space], PI_SYMM_C2V)]

        # get all degenerate pi-pairs
        pi_pairs = pi_space_arr.reshape(-1, 2)

        # get hashes of all degenerate pi-pairs
        pi_hashes = hash_2d(pi_pairs)
        pi_pairs = pi_pairs[np.argsort(pi_hashes)]
        pi_hashes.sort()

        return pi_pairs.reshape(-1,), pi_hashes


def _pi_orbs(pi_space: np.ndarray, tup: np.ndarray) -> np.ndarray:
        """
        this function returns pi-orbitals from tuple of orbitals
        """
        return tup[np.in1d(tup, pi_space)]


def pi_prune(pi_space: np.ndarray, pi_hashes: np.ndarray, tup: np.ndarray) -> bool:
        """
        this function returns True for a tuple of orbitals allowed under pruning wrt degenerate pi-orbitals
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


def occ_prune(occup: np.ndarray, tup: np.ndarray) -> bool:
        """
        this function returns True for a tuple of orbitals allowed under pruning wrt occupied orbitals
        """
        return np.any(occup[tup] > 0.)


def virt_prune(occup: np.ndarray, tup: np.ndarray) -> bool:
        """
        this function returns True for a tuple of orbitals allowed under pruning wrt virtual orbitals
        """
        return np.any(occup[tup] == 0.)


def nelec(occup: np.ndarray, tup: np.ndarray) -> Tuple[int, int]:
        """
        this function returns the number of electrons in a given tuple of orbitals
        """
        occup_tup = occup[tup]
        return (np.count_nonzero(occup_tup > 0.), np.count_nonzero(occup_tup > 1.))


def ndets(occup: np.ndarray, cas_idx: np.ndarray, \
            ref_space: np.ndarray = None, n_elec: Tuple[int, ...] = None) -> int:
        """
        this function returns the number of determinants in given casci calculation (ignoring point group symmetry)
        """
        if n_elec is None:
            n_elec = nelec(occup, cas_idx)

        n_orbs = cas_idx.size

        if ref_space is not None:
            ref_n_elec = nelec(occup, ref_space)
            n_elec = tuple(map(sum, zip(n_elec, ref_n_elec)))
            n_orbs += ref_space.size

        return int(sc.binom(n_orbs, n_elec[0]) * sc.binom(n_orbs, n_elec[1]))


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
        this function looks to see if a general restart file corresponding to the input string exists
        """
        if order is None:
            return os.path.isfile(os.path.join(RST, '{:}.npy'.format(string)))
        else:
            return os.path.isfile(os.path.join(RST, '{:}_{:}.npy'.format(string, order)))


def write_file(order: Union[None, int], arr: np.ndarray, string: str) -> None:
        """
        this function writes a general restart file corresponding to the input string
        """
        if order is None:
            np.save(os.path.join(RST, '{:}'.format(string)), arr)
        else:
            np.save(os.path.join(RST, '{:}_{:}'.format(string, order)), arr)


def read_file(order: int, string: str) -> np.ndarray:
        """
        this function reads a general restart file corresponding to the input string
        """
        if order is None:
            return np.load(os.path.join(RST, '{:}.npy'.format(string)))
        else:
            return np.load(os.path.join(RST, '{:}_{:}.npy'.format(string, order)))


def natural_keys(txt: str) -> List[Union[int, str]]:
        """
        this function return keys to sort a string in human order (as alist.sort(key=natural_keys))
        see: http://nedbatchelder.com/blog/200712/human_sorting.html
        see: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
        """
        return [_convert(c) for c in re.split('(\d+)', txt)]


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


def inc_dim(target: str) -> int:
        """
        this function returns the dimension of increments
        """
        return 1 if target in ['energy', 'excitation'] else 3


def inc_shape(n: int, dim: int) -> Union[Tuple[int], Tuple[int, int]]:
        """
        this function returns the shape of increments
        """
        return (n,) if dim == 1 else (n, dim)
