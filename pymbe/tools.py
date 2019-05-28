#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools module containing all helper functions used in pymbe
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import os
import sys
import traceback
import subprocess
import numpy as np
import scipy.special
import functools
import itertools
import math

import parallel

PI_THRES = 1.0e-04
NON_DEG_ID = np.array([0, 1, 4, 5])
DEG_ID = np.array([2, 3, 6, 7]) 


class Logger(object):
        """
        this class pipes all write statements to both stdout and output_file
        """
        def __init__(self, output_file, both=True):
            """
            init Logger
            """
            self.terminal = sys.stdout
            self.log = open(output_file, 'a')
            self.both = both

        def write(self, message):
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


def git_version():
        """
        this function returns the git revision as a string
        see: https://github.com/numpy/numpy/blob/master/setup.py#L70-L92

        :return: string
        """
        def _minimal_ext_cmd(cmd):
            env = {}
            for k in ['SYSTEMROOT', 'PATH', 'HOME']:
                v = os.environ.get(k)
                if v is not None:
                    env[k] = v
            # LANGUAGE is used on win32
            env['LANGUAGE'] = 'C'
            env['LANG'] = 'C'
            env['LC_ALL'] = 'C'
            out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, cwd=get_pymbe_path()).communicate()[0]
            return out

        try:
            out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
            GIT_REVISION = out.strip().decode('ascii')
        except OSError:
            GIT_REVISION = "Unknown"

        return GIT_REVISION


def get_pymbe_path():
        """
        this function returns the path to pymbe

        :return: string
        """
        return os.path.dirname(os.path.realpath(sys.argv[0]))


def assertion(cond, reason):
        """
        this function returns an assertion of a given condition

        :param cond: condition. bool
        :param reason: reason for aborting. string
        """
        if not cond:

            # get stack
            stack = ''.join(traceback.format_stack()[:-1])

            # print stack
            print('\n\n'+stack)
            print('\n\n*** PyMBE assertion error: '+reason+' ***\n\n')

            # abort mpi
            parallel.abort()


def enum(*sequential, **named):
        """
        this function returns hardcoded enums
        see: https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python

        :return: enum
        """
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)


def time_str(time):
        """
        this function returns time as a HH:MM:SS string

        :param time: time in seconds. scalar
        :return: string
        """
        # hours, minutes, and seconds
        hours = int(time // 3600)
        minutes = int((time - (time // 3600) * 3600.)//60)
        seconds = time - hours * 3600. - minutes * 60.

        # init time string
        string = ''
        form = ()

        # write time string
        if hours > 0:

            string += '{:}h '
            form += (hours,)

        if minutes > 0:

            string += '{:}m '
            form += (minutes,)

        string += '{:.2f}s'
        form += (seconds,)

        return string.format(*form)


def fsum(a):
        """
        this function uses math.fsum to safely sum 1d array or 2d array (column-wise)

        :param a: quantity to sum. numpy array of shape (n_a,) or (n_a_1, n_a_2)
        :return: scalar or numpy array of shape (n_a_1,) depending on dimensions of a
        """
        if a.ndim == 1:
            return math.fsum(a)
        elif a.ndim == 2:
            return np.fromiter(map(math.fsum, a.T), dtype=a.dtype, count=a.shape[1])
        else:
            raise NotImplementedError('tools.py: _fsum()')


def hash_2d(a):
        """
        this function converts a 2d numpy array to a 1d array of hashes

        :param a: 2d array of integers. numpy array of shape (n_a_1, n_a_2) [*32-bit integers]
        :return: numpy array of shape (n_a_1,) [*64-bit integers]
        """
        return np.fromiter(map(hash_1d, a), dtype=np.int64, count=a.shape[0])


def hash_1d(a):
        """
        this function converts a 1d numpy array to a hash

        :param a: 1d array of integers. numpy array of shape (n_a,) [*32-bit integers]
        :return: 64-bit integer
        """
        return hash(a.tobytes())


def hash_compare(a, b):
        """
        this function finds occurences of b in a through a binary search

        :param a: main array. numpy array of shape (n_a,)
        :param b: test array. numpy array of shape (n_b,)
        :return: numpy array of shape (n_b,) or None
        """
        left = a.searchsorted(b, side='left')
        right = a.searchsorted(b, side='right')
        if ((right - left) > 0).all():
            return left
        else:
            return None


def tasks(n_tasks, n_slaves, task_size):
        """
        this function returns an array of tasks

        :param n_tasks: number of tasks. integer
        :param n_slaves: number of slaves. integer
        :param task_size: mpi task size. integer
        :return: list of numpy arrays of various shapes
        """
        if n_slaves * task_size < n_tasks:
            return np.array_split(np.arange(n_tasks), n_tasks // task_size)
        else:
            return np.array_split(np.arange(n_tasks), n_slaves)


def cas(ref_space, tup):
        """
        this function returns a cas space

        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param tup: current orbital tuple. numpy array of shape (order,)
        :return: numpy array of shape (n_ref_tot + order,)
        """
        return np.sort(np.append(ref_space, tup))


def core_cas(nocc, ref_space, tup):
        """
        this function returns a core and a cas space

        :param nocc: number of occupied orbitals. integer
        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param tup: current orbital tuple. numpy array of shape (order,)
        :return: numpy array of shapes (n_inactive,) and (n_cas,)
        """
        cas_idx = cas(ref_space, tup)
        core_idx = np.setdiff1d(np.arange(nocc), cas_idx)
        return core_idx, cas_idx


def _cas_idx_cart(cas_idx):
        """
        this function returns a cartesian product of (cas_idx, cas_idx)

        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :return: numpy array of shape (n_cas**2, 2)
        """
        return np.array(np.meshgrid(cas_idx, cas_idx)).T.reshape(-1, 2)


def _coor_to_idx(ij):
        """
        this function returns the lower triangular index corresponding to (i, j)

        :param ij: combined i (ij[0]) and j (ij[1]) indices
        :return: integer
        """
        i = ij[0]; j = ij[1]
        if i >= j:
            return i * (i + 1) // 2 + j
        else:
            return j * (j + 1) // 2 + i


def cas_idx_tril(cas_idx):
        """
        this function returns lower triangular cas indices

        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :return: numpy array of shape (n_cas*(n_cas + 1) // 2,)
        """
        cas_idx_cart = _cas_idx_cart(cas_idx)
        return np.unique(np.fromiter(map(_coor_to_idx, cas_idx_cart), \
                                        dtype=cas_idx_cart.dtype, count=cas_idx_cart.shape[0]))


def non_deg_orbs(orbsym, tup):
        """
        this function returns non-degenerate orbitals from tuple of orbitals

        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param tup: tuple of orbitals. numpy array of shape (n_tup,)
        :return: numpy array of shape (n_non_deg_orbs,)
        """
        return tup[np.in1d(orbsym[tup], NON_DEG_ID)]


def _pi_orbs(orbsym, tup):
        """
        this function returns pi-orbitals from tuple of orbitals

        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param tup: tuple of orbitals. numpy array of shape (n_tup,)
        :return: numpy array of shape (n_pi_orbs,)
        """
        return tup[np.in1d(orbsym[tup], DEG_ID)]


def n_pi_orbs(orbsym, tup):
        """ this function returnsnumber of pi-orbitals in tuple of orbitals

        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param tup: tuple of orbitals. numpy array of shape (n_tup,)
        :return: integer
        """
        return np.count_nonzero(_pi_orbs(orbsym, tup))


def _pi_pairs(orbsym, tup):
        """
        this function returns pairs of pi-orbitals from tuple of orbitals

        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param tup: tuple of orbitals. numpy array of shape (n_tup,)
        :return: numpy array of shape (n_pi_pairs, 2)
        """
        return np.array(list(itertools.combinations(_pi_orbs(orbsym, tup), 2)))


def _pi_deg(mo_energy, pair):
        """
        this function returns True for degenerate pairs of pi-orbitals

        :param mo_energy: orbital energies. numpy array of shape (n_orb,)
        :param pair: orbital pair. numpy array of shape (2,)
        :return: bool
        """
        return np.abs(mo_energy[pair[1]] - mo_energy[pair[0]]) < PI_THRES


def pi_pairs_deg(mo_energy, orbsym, tup):
        """
        this function returns pairs of degenerate pi-orbitals from tuple of orbitals

        :param mo_energy: orbital energies. numpy array of shape (n_orb,)
        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param tup: tuple of orbitals. numpy array of shape (n_tup,)
        :return: numpy array of shape (n_pi_deg_pairs, 2)
        """
        return np.array([pair for pair in _pi_pairs(orbsym, tup) if _pi_deg(mo_energy, pair)], dtype=np.int32)


def pi_prune(mo_energy, orbsym, tup):
        """
        this function returns True for a tuple of orbitals allowed under pruning wrt degenerate pi-orbitals

        :param mo_energy: orbital energies. numpy array of shape (n_orb,)
        :param orbsym: orbital symmetries. numpy array of shape (n_orb,)
        :param tup: tuple of orbitals. numpy array of shape (n_tup,)
        :return: bool
        """
        # get all pi-orbitals in tup
        pi_orbs = _pi_orbs(orbsym, tup)

        if pi_orbs.size == 0:

            # no pi-orbitals
            return True

        else:

            if pi_orbs.size % 2 > 0:

                # always prune tuples with an odd number of pi-orbitals
                return False

            else:

                # check if all pi-orbitals are pair-wise degenerate
                return pi_orbs.size == pi_pairs_deg(mo_energy, orbsym, tup).size


def seed_prune(seed, occup, tup):
        """
        this function returns True for a tuple of orbitals allowed under pruning wrt seed orbitals

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param tup: current orbital tuple. numpy array of shape (order,)
        :return: bool
        """
        if seed == 'occ':
            return np.any(occup[tup] > 0.0)
        else:
            return np.any(occup[tup] == 0.0)


def corr_prune(occup, tup):
        """
        this function returns True for a tuple of orbitals allowed under pruning wrt a mix of occupied and virtual orbitals

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param tup: current orbital tuple. numpy array of shape (order,)
        :return: bool
        """
        return np.any(occup[tup] > 0.0) and np.any(occup[tup] == 0.0)


def nelec(occup, tup):
        """
        this function returns the number of electrons in a given tuple of orbitals

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param tup: current orbital tuple. numpy array of shape (order,)
        :return: tuple of integers
        """
        occup_tup = occup[tup]
        return (np.count_nonzero(occup_tup > 0.0), np.count_nonzero(occup_tup > 1.0))


def ndets(occup, cas_idx, ref_space=None, n_elec=None):
        """
        this function returns the number of determinants in given casci calculation (ignoring point group symmetry)

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param n_elec: number of electrons in cas space. tuple of two integers
        :return: scalar
        """
        if n_elec is None:
            n_elec = nelec(occup, cas_idx)
        n_orbs = cas_idx.size
        if ref_space is not None:
            ref_n_elec = nelec(occup, ref_space)
            n_elec = tuple(map(sum, zip(n_elec, ref_n_elec)))
            n_orbs += ref_space.size
        return scipy.special.binom(n_orbs, n_elec[0]) * scipy.special.binom(n_orbs, n_elec[1])


def mat_idx(site_xy, nx, ny):
        """
        this function returns x and y indices of a matrix

        :param site_xy: matrix index. integer
        :param nx: x-dimension of matrix. integer
        :param ny: y-dimension of matrix. integer
        :return: tuple of integers
        """
        x = site_xy % nx
        y = int(math.floor(float(site_xy) / ny))
        return x, y


def near_nbrs(site_xy, nx, ny):
        """
        this function returns a list of nearest neighbour indices

        :param site_xy: matrix index. integer
        :param nx: x-dimension of matrix. integer
        :param ny: y-dimension of matrix. integer
        :return: list of tuples of integers
        """
        left = ((site_xy[0] - 1) % nx, site_xy[1])
        right = ((site_xy[0] + 1) % nx, site_xy[1])
        down = (site_xy[0], (site_xy[1] + 1) % ny)
        up = (site_xy[0], (site_xy[1] - 1) % ny)
        return [left, right, down, up]


