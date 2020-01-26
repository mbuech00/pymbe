#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
mbe module containing all functions related to MBEs in pymbe
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import functools
import sys
import itertools
from pyscf import gto
from typing import Tuple, Set, List, Dict, Union, Any

import kernel
import output
import expansion
import driver
import system
import calculation
import parallel
import tools


def main(mpi: parallel.MPICls, mol: system.MolCls, \
            calc: calculation.CalcCls, exp: expansion.ExpCls, \
            rst_read: bool = False, rst_write: bool = False, tup_start: int = 0) -> Union[Tuple[Any, ...], MPI.Win]:
        """
        this function is the mbe main function
        """
        if mpi.global_master:

            # read restart files
            rst_read = len(exp.prop[calc.target_mbe]['inc']) > len(exp.prop[calc.target_mbe]['tot'])

            # start index
            tup_start = tools.read_file(exp.order, 'mbe_idx') if rst_read else 0

            # wake up slaves
            msg = {'task': 'mbe', 'order': exp.order, 'rst_read': rst_read, 'tup_start': tup_start}
            mpi.global_comm.bcast(msg, root=0)

        # increment dimensions
        if calc.target_mbe in ['energy', 'excitation']:
            dim = 1
        elif calc.target_mbe in ['dipole', 'trans']:
            dim = 3

        # increment shape
        def shape(n, dim):
            return (n,) if dim == 1 else (n, dim)

        # load eri
        buf = mol.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

        # load hcore
        buf = mol.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # load vhf
        buf = mol.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # load increments for previous orders
        inc = []
        for k in range(exp.order-exp.min_order):
            buf = exp.prop[calc.target_mbe]['inc'][k].Shared_query(0)[0] # type: ignore
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=shape(exp.n_tuples[k], dim)))

        # init increments for present order
        if rst_read:
            inc_win = exp.prop[calc.target_mbe]['inc'][-1]
        else:
            inc_win = MPI.Win.Allocate_shared(8 * exp.n_tuples[-1] * dim if mpi.local_master else 0, 8, comm=mpi.local_comm)
        buf = inc_win.Shared_query(0)[0] # type: ignore
        inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=shape(exp.n_tuples[-1], dim)))
        if mpi.local_master and not rst_read:
            inc[-1][:] = np.zeros_like(inc[-1])

        # init time
        if mpi.global_master:
            if not rst_read:
                exp.time['mbe'].append(0.)
            time = MPI.Wtime()

        # init increment statistics
        min_inc = exp.min_inc[-1] if mpi.global_master and rst_read else np.array([1.e12] * dim, dtype=np.float64)
        max_inc = exp.max_inc[-1] if mpi.global_master and rst_read else np.array([0.] * dim, dtype=np.float64)
        sum_inc = exp.mean_inc[-1] if mpi.global_master and rst_read else np.array([0.] * dim, dtype=np.float64)

        # init determinant statistics
        min_ndets = exp.min_ndets[-1] if mpi.global_master and rst_read else np.array([1e12], dtype=np.int64)
        max_ndets = exp.max_ndets[-1] if mpi.global_master and rst_read else np.array([0], dtype=np.int64)
        sum_ndets = exp.mean_ndets[-1] if mpi.global_master and rst_read else np.array([0], dtype=np.int64)

        # mpi barrier
        mpi.global_comm.Barrier()

        # occupied and virtual expansion spaces
        exp_occ = exp.exp_space[-1][exp.exp_space[-1] < mol.nocc]
        exp_virt = exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]]

        # allow for tuples with only virtual or occupied MOs
        ref_occ = tools.occ_prune(calc.occup, calc.ref_space)
        ref_virt = tools.virt_prune(calc.occup, calc.ref_space)

        # set rst_write
        rst_write = calc.misc['rst'] and mpi.global_size < calc.misc['rst_freq'] < exp.n_tuples[-1]

        # loop until no tuples left
        for tup_idx, tup in enumerate(itertools.islice(tools.tuples(exp_occ, exp_virt, ref_occ, ref_virt, exp.order), \
                                        tup_start, None), tup_start):

            # distribute tuples
            if tup_idx % mpi.global_size != mpi.global_rank:
                continue

            # pi-pruning
            if calc.extra['pi_prune']:
                if not tools.pi_prune(exp.pi_orbs, exp.pi_hashes, tup):
                    inc[-1][tup_idx] = np.nan
                    continue

            # get core and cas indices
            core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, tup)

            # get h2e indices
            cas_idx_tril = tools.cas_idx_tril(cas_idx)

            # get h2e_cas
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

            # calculate increment
            inc_tup, ndets_tup, nelec_tup = _inc(calc.model['method'], calc.base['method'], calc.model['solver'], mol.spin, \
                                                 calc.occup, calc.target_mbe, calc.state['wfnsym'], calc.orbsym, \
                                                 calc.model['hf_guess'], calc.state['root'], calc.prop['hf']['energy'], \
                                                 calc.prop['ref'][calc.target_mbe], e_core, h1e_cas, h2e_cas, \
                                                 core_idx, cas_idx, mol.debug, \
                                                 mol.dipole if calc.target_mbe in ['dipole', 'trans'] else None, \
                                                 calc.mo_coeff if calc.target_mbe in ['dipole', 'trans'] else None, \
                                                 calc.prop['hf']['dipole'] if calc.target_mbe in ['dipole', 'trans'] else None)

            # calculate increment
            if exp.order > exp.min_order:
                inc_tup -= _sum(mol, calc.occup, calc.target_mbe, exp.min_order, \
                                exp.order, inc, exp.exp_space, ref_occ, ref_virt, tup)


            # debug print
            if mol.debug >= 1:
                print(output.mbe_debug(mol.atom, mol.symmetry, calc.orbsym, calc.state['root'], \
                                        ndets_tup, nelec_tup, inc_tup, exp.order, cas_idx, tup))

            # add to inc
            inc[-1][tup_idx] = inc_tup

            # update increment statistics
            min_inc = np.minimum(min_inc, np.abs(inc_tup))
            max_inc = np.maximum(max_inc, np.abs(inc_tup))
            sum_inc += inc_tup

            # update determinant statistics
            min_ndets = np.minimum(min_ndets, ndets_tup)
            max_ndets = np.maximum(max_ndets, ndets_tup)
            sum_ndets += ndets_tup

            # write restart files
            if rst_write and (tup_idx % calc.misc['rst_freq']) < mpi.global_size:

                # reduce increments onto global master
                if mpi.num_masters > 1 and mpi.local_master:
                    inc[-1][:] = parallel.reduce(mpi.master_comm, inc[-1], root=0, op=MPI.SUM)
                    if not mpi.global_master:
                        inc[-1][:] = np.zeros_like(inc[-1])

                # reduce increment statistics onto global master
                min_inc = parallel.reduce(mpi.global_comm, min_inc, root=0, op=MPI.MIN)
                max_inc = parallel.reduce(mpi.global_comm, max_inc, root=0, op=MPI.MAX)
                sum_inc = parallel.reduce(mpi.global_comm, sum_inc, root=0, op=MPI.SUM)
                if not mpi.global_master:
                    min_inc = np.array([1.e12] * dim, dtype=np.float64)
                    max_inc = np.array([0.] * dim, dtype=np.float64)
                    sum_inc = np.array([0.] * dim, dtype=np.float64)

                # reduce determinant statistics onto global master
                min_ndets = parallel.reduce(mpi.global_comm, min_ndets, root=0, op=MPI.MIN)
                max_ndets = parallel.reduce(mpi.global_comm, max_ndets, root=0, op=MPI.MAX)
                sum_ndets = parallel.reduce(mpi.global_comm, sum_ndets, root=0, op=MPI.SUM)
                if not mpi.global_master:
                    min_ndets = np.array([1e12], dtype=np.int64)
                    max_ndets = np.array([0], dtype=np.int64)
                    sum_ndets = np.array([0], dtype=np.int64)

                # reduce mbe_idx onto global master
                mbe_idx = mpi.global_comm.allreduce(tup_idx, op=MPI.MAX)

                # reset rst_write
                rst_write &= (mbe_idx + calc.misc['rst_freq']) < exp.n_tuples[-1]

                # write restart files
                if mpi.global_master:

                    # save mbe_idx
                    tools.write_file(exp.order, np.asarray(mbe_idx + 1), 'mbe_idx')
                    # save increments
                    tools.write_file(exp.order, inc[-1], 'mbe_inc')
                    # save increment statistics
                    tools.write_file(exp.order, max_inc, 'mbe_max_inc')
                    tools.write_file(exp.order, min_inc, 'mbe_min_inc')
                    tools.write_file(exp.order, sum_inc, 'mbe_mean_inc')
                    # save determinant statistics
                    tools.write_file(exp.order, max_ndets, 'mbe_max_ndets')
                    tools.write_file(exp.order, min_ndets, 'mbe_min_ndets')
                    tools.write_file(exp.order, sum_ndets, 'mbe_mean_ndets')
                    # save timing
                    exp.time['mbe'][-1] += MPI.Wtime() - time
                    tools.write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')

                    # re-init time
                    time = MPI.Wtime()

                    # print status
                    print(output.mbe_status(mbe_idx / exp.n_tuples[-1]))

                # mpi barrier
                mpi.master_comm.Barrier()

        # mpi barrier
        mpi.global_comm.Barrier()

        # print final status
        if mpi.global_master:
            print(output.mbe_status(1.))

        # increment statistics
        min_inc = parallel.reduce(mpi.global_comm, min_inc, root=0, op=MPI.MIN)
        max_inc = parallel.reduce(mpi.global_comm, max_inc, root=0, op=MPI.MAX)
        sum_inc = parallel.reduce(mpi.global_comm, sum_inc, root=0, op=MPI.SUM)

        # determinant statistics
        min_ndets = parallel.reduce(mpi.global_comm, min_ndets, root=0, op=MPI.MIN)
        max_ndets = parallel.reduce(mpi.global_comm, max_ndets, root=0, op=MPI.MAX)
        sum_ndets = parallel.reduce(mpi.global_comm, sum_ndets, root=0, op=MPI.SUM)

        # mean increment
        if mpi.global_master:
            mean_inc = sum_inc / exp.n_tuples[-1]

        # mean number of determinants
        if mpi.global_master:
            mean_ndets = np.asarray(np.rint(sum_ndets / exp.n_tuples[-1]), dtype=np.int64)

        # allreduce increments among local masters
        if mpi.local_master:
            inc[-1][:] = parallel.allreduce(mpi.master_comm, inc[-1], op=MPI.SUM)

        # mpi barrier
        mpi.global_comm.Barrier()

        # collect results on global master
        if mpi.global_master:

            # write restart files
            if calc.misc['rst']:
                # save idx
                tools.write_file(exp.order, np.asarray(exp.n_tuples[-1]-1), 'mbe_idx')
                # save increments
                tools.write_file(exp.order, inc[-1], 'mbe_inc')

            # total property
            tot = sum_inc

            # save timing
            exp.time['mbe'][-1] += MPI.Wtime() - time

            return inc_win, tot, mean_ndets, min_ndets, max_ndets, mean_inc, min_inc, max_inc

        else:

            return inc_win


def _inc(main_method: str, base_method: Union[str, None], solver: str, spin: int, \
            occup: np.ndarray, target_mbe: str, state_wfnsym: str, orbsym: np.ndarray, hf_guess: bool, \
            state_root: int, e_hf: float, res_ref: Union[float, np.ndarray], e_core: float, \
            h1e_cas: np.ndarray, h2e_cas: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray, \
            debug: int, ao_dipole: Union[np.ndarray, None], mo_coeff: Union[np.ndarray, None], \
            dipole_hf: Union[np.ndarray, None]) -> Tuple[Union[float, np.ndarray], int, Tuple[int, int]]:
        """
        this function calculates the current-order contribution to the increment associated with a given tuple

        example:
        >>> n = 4
        >>> occup = np.array([2.] * (n // 2) + [0.] * (n // 2))
        >>> orbsym = np.zeros(n, dtype=np.int64)
        >>> h1e_cas, h2e_cas = kernel.hubbard_h1e((1, n), False), kernel.hubbard_eri((1, n), 2.)
        >>> core_idx, cas_idx = np.array([]), np.arange(n)
        >>> e, ndets, nelec = _inc('fci', None, 'pyscf_spin0', 0, occup, 'energy', None, orbsym, 0,
        ...                        0, 0., 0., 0., h1e_cas, h2e_cas, core_idx, cas_idx, False, None, None, None)
        >>> np.isclose(e, -2.875942809005048)
        True
        >>> ndets
        36
        >>> nelec
        (2, 2)
        """
        # nelec
        nelec = tools.nelec(occup, cas_idx)

        # perform main calc
        res_full, ndets = kernel.main(main_method, solver, spin, occup, target_mbe, state_wfnsym, orbsym, \
                                        hf_guess, state_root, e_hf, e_core, h1e_cas, h2e_cas, \
                                        core_idx, cas_idx, nelec, debug, ao_dipole, mo_coeff, dipole_hf)

        # perform base calc
        if base_method is not None:
            res_full -= kernel.main(base_method, '', spin, occup, target_mbe, state_wfnsym, orbsym, \
                                      hf_guess, state_root, e_hf, e_core, h1e_cas, h2e_cas, \
                                      core_idx, cas_idx, nelec, debug, ao_dipole, mo_coeff, dipole_hf)[0]

        return res_full - res_ref, ndets, nelec


def _sum(mol: system.MolCls, occup: np.ndarray, target_mbe: str, min_order: int, order: int, \
            inc: List[np.ndarray], exp_space: List[np.ndarray], ref_occ: bool, ref_virt: bool, \
            tup: np.ndarray) -> Union[float, np.ndarray]:
        """
        this function performs a recursive summation and returns the final increment associated with a given tuple

        example:
        >>> occup = np.array([2.] * 2 + [0.] * 2)
        >>> ref_space = {'occ': np.arange(2, dtype=np.int64),
        ...              'virt': np.array([], dtype=np.int64),
        ...              'tot': np.arange(2, dtype=np.int64)}
        >>> exp_space = {'occ': np.array([], dtype=np.int64),
        ...              'virt': np.arange(2, 4, dtype=np.int64),
        ...              'tot': np.arange(2, 4, dtype=np.int64),
        ...              'pi_orbs': np.array([], dtype=np.int64),
        ...              'pi_hashes': np.array([], dtype=np.int64)}
        >>> min_order, order = 1, 2
        ... # [[2], [3]]
        ... # [[2, 3]]
        >>> hashes = [np.sort(np.array([-4760325697709127167, -4199509873246364550])),
        ...           np.array([-5475322122992870313])]
        >>> inc = [np.array([-.1, -.2])]
        >>> tup = np.arange(2, 4, dtype=np.int64)
        >>> np.isclose(_sum(occup, ref_space, exp_space, 'energy', min_order, order, inc, hashes, tup, False), -.3)
        True
        >>> inc = [np.array([[0., 0., .1], [0., 0., .2]])]
        >>> np.allclose(_sum(occup, ref_space, exp_space, 'dipole', min_order, order, inc, hashes, tup, False), np.array([0., 0., .3]))
        True
        >>> ref_space['tot'] = ref_space['occ'] = np.array([], dtype=np.int64)
        >>> exp_space = {'tot': np.arange(4), 'occ': np.arange(2), 'virt': np.arange(2, 4),
        ...              'pi_orbs': np.arange(2, dtype=np.int64), 'pi_hashes': np.array([-3821038970866580488])}
        >>> min_order, order = 2, 4
        ... # [[0, 2], [0, 3], [1, 2], [1, 3]]
        ... # [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        ... # [[0, 1, 2, 3]]
        >>> hashes = [np.sort(np.array([-4882741555304645790, 1455941523185766351, -2163557957507198923, -669804309911520350])),
        ...           np.sort(np.array([-5731810011007442268, 366931854209709639, -7216722148388372205, -3352798558434503475])),
        ...           np.array([-2930228190932741801])]
        >>> inc = [np.array([-.11, -.12, -.11, -.12]), np.array([-.01, -.02, -.03, -.03])]
        >>> tup = np.arange(4, dtype=np.int64)
        >>> np.isclose(_sum(occup, ref_space, exp_space, 'energy', min_order, order, inc, hashes, tup, False), -0.55)
        True
        >>> np.isclose(_sum(occup, ref_space, exp_space, 'energy', min_order, order, inc, hashes, tup, True), -0.05)
        True
        """
        # init res
        if target_mbe in ['energy', 'excitation']:
            res = np.empty(order - min_order, dtype=np.float64)
        else:
            res = np.empty([order - min_order, 3], dtype=np.float64)

        # occupied and virtual subspaces of tuple
        tup_occ = tup[tup < mol.nocc]
        tup_virt = tup[mol.nocc <= tup]

        # compute contributions from lower-order increments
        for k in range(order-1, min_order-1, -1):

            # get indices of subtuples
            idx = np.fromiter((tools.restricted_idx(exp_space[k-min_order][exp_space[k-min_order] < mol.nocc], \
                                                    exp_space[k-min_order][mol.nocc <= exp_space[k-min_order]], \
                                                    tup_sub[tup_sub < mol.nocc], tup_sub[mol.nocc <= tup_sub]) \
                               for tup_sub in tools.tuples(tup_occ, tup_virt, ref_occ, ref_virt, k)), \
                              dtype=np.int64, count=tools.n_tuples(tup_occ, tup_virt, ref_occ, ref_virt, k))

            # sum up order increments
            res[k-min_order] = tools.fsum(inc[k-min_order][idx][~np.isnan(inc[k-min_order][idx])])

        return tools.fsum(res)


if __name__ == "__main__":
    import doctest
    doctest.testmod()



