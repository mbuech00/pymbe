#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
mbe module
"""

from __future__ import annotations

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import logging
import numpy as np
from mpi4py import MPI
from typing import TYPE_CHECKING

from pymbe.kernel import e_core_h1e, main as kernel_main
from pymbe.output import mbe_status, mbe_debug
from pymbe.parallel import mpi_reduce, mpi_allreduce
from pymbe.tools import is_file, read_file, write_file, inc_dim, inc_shape, \
                        occ_prune, virt_prune, pi_prune, tuples, start_idx, \
                        core_cas, idx_tril, nelec, hash_1d, hash_lookup, fsum

if TYPE_CHECKING:

    from typing import Tuple, List, Union, Any, Optional

    from pymbe.parallel import MPICls
    from pymbe.expansion import ExpCls


# get logger
logger = logging.getLogger('pymbe_logger')

SCREEN = 1000. # random, non-sensical number


def main(mpi: MPICls, exp: ExpCls, rst_read: bool = False, tup_idx: int = 0, \
         tup: Optional[np.ndarray] = None) -> Tuple[Any, ...]:
        """
        this function is the mbe main function
        """
        if mpi.global_master:
            # read restart files
            rst_read = is_file(exp.order, 'mbe_idx') and is_file(exp.order, 'mbe_tup')
            # start indices
            tup_idx = read_file(exp.order, 'mbe_idx').item() if rst_read else 0
            # start tuples
            tup = read_file(exp.order, 'mbe_tup') if rst_read else None
            # wake up slaves
            msg = {'task': 'mbe', 'order': exp.order, \
                   'rst_read': rst_read, 'tup_idx': tup_idx, 'tup': tup}
            mpi.global_comm.bcast(msg, root=0)

        # increment dimensions
        dim = inc_dim(exp.target)

        # load eri
        buf = exp.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.norb * (exp.norb + 1) // 2,) * 2)

        # load hcore
        buf = exp.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.norb,) * 2)

        # load vhf
        buf = exp.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.nocc, exp.norb, exp.norb))

        # load hashes for previous orders
        hashes = []
        for k in range(exp.order-exp.min_order):
            buf = exp.prop[exp.target]['hashes'][k].Shared_query(0)[0] # type: ignore
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['inc'][k],)))

        # init hashes for present order
        if rst_read:
            hashes_win = exp.prop[exp.target]['hashes'][-1]
        else:
            hashes_win = MPI.Win.Allocate_shared(8 * exp.n_tuples['inc'][-1] if mpi.local_master else 0, 8, comm=mpi.local_comm)
        buf = hashes_win.Shared_query(0)[0] # type: ignore
        hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples['inc'][-1],)))
        if mpi.local_master and not mpi.global_master:
            hashes[-1][:].fill(0)

        # load increments for previous orders
        inc = []
        for k in range(exp.order-exp.min_order):
            buf = exp.prop[exp.target]['inc'][k].Shared_query(0)[0] # type: ignore
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape = inc_shape(exp.n_tuples['inc'][k], dim)))

        # init increments for present order
        if rst_read:
            inc_win = exp.prop[exp.target]['inc'][-1]
        else:
            inc_win = MPI.Win.Allocate_shared(8 * exp.n_tuples['inc'][-1] * dim if mpi.local_master else 0, 8, comm=mpi.local_comm)
        buf = inc_win.Shared_query(0)[0] # type: ignore
        inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape = inc_shape(exp.n_tuples['inc'][-1], dim)))
        if mpi.local_master and not mpi.global_master:
            inc[-1][:].fill(0.)

        # init time
        if mpi.global_master:
            if not rst_read:
                exp.time['mbe'].append(0.)
            time = MPI.Wtime()

        # init increment statistics
        min_inc = exp.min_inc[-1] if mpi.global_master and rst_read else np.array([1.e12] * dim, dtype=np.float64)
        max_inc = exp.max_inc[-1] if mpi.global_master and rst_read else np.array([0.] * dim, dtype=np.float64)
        mean_inc = exp.mean_inc[-1] if mpi.global_master and rst_read else np.array([0.] * dim, dtype=np.float64)

        # init pair_corr statistics
        if exp.ref_space.size == 0 and exp.order == exp.min_order and exp.base_method is None:
            pair_corr = [np.zeros(exp.n_tuples['inc'][0], dtype=np.float64), \
                         np.zeros([exp.n_tuples['inc'][0], 2], dtype=np.int32)]
        else:
            pair_corr = None

        # mpi barrier
        mpi.global_comm.Barrier()

        # occupied and virtual expansion spaces
        exp_occ = exp.exp_space[-1][exp.exp_space[-1] < exp.nocc]
        exp_virt = exp.exp_space[-1][exp.nocc <= exp.exp_space[-1]]

        # allow for tuples with only virtual or occupied MOs
        ref_occ = occ_prune(exp.occup, exp.ref_space)
        ref_virt = virt_prune(exp.occup, exp.ref_space)

        # init screen array
        screen = np.zeros(exp.norb, dtype=np.float64)
        if rst_read:
            if mpi.global_master:
                screen = exp.screen
        if exp.order == exp.min_order:
            if ref_occ and not ref_virt:
                screen[exp_occ] = SCREEN
            if not ref_occ and ref_virt:
                screen[exp_virt] = SCREEN

        # set rst_write
        rst_write = exp.rst and mpi.global_size < exp.rst_freq < exp.n_tuples['inc'][-1]

        # start tuples
        if tup is not None:
            tup_occ = tup[tup < exp.nocc]
            tup_virt = tup[exp.nocc <= tup]
            if tup_occ.size == 0:
                tup_occ = None
            if tup_virt.size == 0:
                tup_virt = None
        else:
            tup_occ = tup_virt = None
        order_start, occ_start, virt_start = start_idx(exp_occ, exp_virt, tup_occ, tup_virt)

        # loop until no tuples left
        for tup_idx, tup in enumerate(tuples(exp_occ, exp_virt, ref_occ, ref_virt, exp.order, \
                                             order_start, occ_start, virt_start), tup_idx):

            # distribute tuples
            if tup_idx % mpi.global_size != mpi.global_rank:
                continue

            # write restart files and re-init time
            if rst_write and tup_idx % exp.rst_freq < mpi.global_size:

                # mpi barrier
                mpi.local_comm.Barrier()

                # reduce hashes & increments onto global master
                if mpi.num_masters > 1 and mpi.local_master:
                    hashes[-1][:] = mpi_reduce(mpi.master_comm, hashes[-1], root=0, op=MPI.SUM)
                    if not mpi.global_master:
                        hashes[-1][:].fill(0)
                    inc[-1][:] = mpi_reduce(mpi.master_comm, inc[-1], root=0, op=MPI.SUM)
                    if not mpi.global_master:
                        inc[-1][:].fill(0.)

                # reduce increment statistics onto global master
                min_inc = mpi_reduce(mpi.global_comm, min_inc, root=0, op=MPI.MIN)
                max_inc = mpi_reduce(mpi.global_comm, max_inc, root=0, op=MPI.MAX)
                mean_inc = mpi_reduce(mpi.global_comm, mean_inc, root=0, op=MPI.SUM)
                if not mpi.global_master:
                    min_inc = np.array([1.e12] * dim, dtype=np.float64)
                    max_inc = np.array([0.] * dim, dtype=np.float64)
                    mean_inc = np.array([0.] * dim, dtype=np.float64)

                # reduce screen onto global master
                screen = mpi_reduce(mpi.global_comm, screen, root=0, op=MPI.MAX)

                # reduce mbe_idx onto global master
                mbe_idx = mpi.global_comm.allreduce(tup_idx, op=MPI.MIN)
                # send tup corresponding to mbe_idx to master
                if mpi.global_master:
                    if tup_idx == mbe_idx:
                        mbe_tup = tup
                    else:
                        mbe_tup = np.empty(exp.order, dtype=np.int64)
                        mpi.global_comm.Recv(mbe_tup, source=MPI.ANY_SOURCE, tag=101)
                elif tup_idx == mbe_idx:
                    mpi.global_comm.Send(tup, dest=0, tag=101)
                # update rst_write
                rst_write = mbe_idx + exp.rst_freq < exp.n_tuples['inc'][-1] - mpi.global_size

                if mpi.global_master:
                    # write restart files
                    write_file(exp.order, max_inc, 'mbe_max_inc')
                    write_file(exp.order, min_inc, 'mbe_min_inc')
                    write_file(exp.order, mean_inc, 'mbe_mean_inc')
                    write_file(exp.order, screen, 'mbe_screen')
                    write_file(exp.order, np.asarray(mbe_idx), 'mbe_idx')
                    write_file(exp.order, mbe_tup, 'mbe_tup')
                    write_file(exp.order, hashes[-1], 'mbe_hashes')
                    write_file(exp.order, inc[-1], 'mbe_inc')
                    exp.time['mbe'][-1] += MPI.Wtime() - time
                    write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')
                    # re-init time
                    time = MPI.Wtime()
                    # print status
                    logger.info(mbe_status(exp.order, \
                                           mbe_idx / exp.n_tuples['inc'][-1]))

            # pi-pruning
            if exp.pi_prune:
                if not pi_prune(exp.pi_orbs, exp.pi_hashes, tup):
                    screen[tup] = SCREEN
                    continue

            # get core and cas indices
            core_idx, cas_idx = core_cas(exp.nocc, exp.ref_space, tup)

            # get h2e indices
            cas_idx_tril = idx_tril(cas_idx)

            # get h2e_cas
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = e_core_h1e(exp.nuc_energy, hcore, vhf, core_idx, cas_idx)

            # calculate increment
            inc_tup, n_elec_tup = _inc(exp.method, exp.base_method, \
                                       exp.cc_backend, exp.fci_solver, \
                                       exp.orb_type, exp.spin, exp.occup, \
                                       exp.target, exp.fci_state_sym, \
                                       exp.point_group, exp.orbsym, \
                                       exp.hf_guess, exp.fci_state_root, \
                                       exp.hf_prop, e_core, h1e_cas, h2e_cas, \
                                       core_idx, cas_idx, exp.verbose, \
                                       exp.dipole_ints, exp.ref_prop)

            # calculate increment
            if exp.order > exp.min_order:
                inc_tup -= _sum(exp.nocc, exp.target, exp.min_order, \
                                exp.order, inc, hashes, ref_occ, ref_virt, tup)

            # add hash and increment
            hashes[-1][tup_idx] = hash_1d(tup)
            inc[-1][tup_idx] = inc_tup

            # screening procedure
            if exp.target in ['energy', 'excitation']:
                screen[tup] = np.maximum(screen[tup], np.abs(inc_tup))
            else:
                screen[tup] = np.maximum(screen[tup], np.max(np.abs(inc_tup)))

            # debug print
            logger.debug(mbe_debug(exp.point_group, exp.orbsym, \
                                   exp.fci_state_root, n_elec_tup, inc_tup, \
                                   exp.order, cas_idx, tup))

            # update increment statistics
            min_inc, max_inc, mean_inc = _update(min_inc, max_inc, mean_inc, inc_tup)
            # update pair_corr statistics
            if pair_corr is not None:
                if exp.target in ['energy', 'excitation']:
                    pair_corr[0][tup_idx] = inc_tup # type: ignore
                else:
                    pair_corr[0][tup_idx] = inc_tup[np.argmax(np.abs(inc_tup))] # type: ignore
                pair_corr[1][tup_idx] = tup

        # mpi barrier
        mpi.global_comm.Barrier()

        # print final status
        if mpi.global_master:
            logger.info(mbe_status(exp.order, 1.))

        # allreduce hashes & increments among local masters
        if mpi.local_master:
            hashes[-1][:] = mpi_allreduce(mpi.master_comm, hashes[-1], op=MPI.SUM)
            inc[-1][:] = mpi_allreduce(mpi.master_comm, inc[-1], op=MPI.SUM)

        # sort hashes and increments
        if mpi.local_master:
            inc[-1][:] = inc[-1][np.argsort(hashes[-1])]
            hashes[-1][:].sort()

        # increment statistics
        min_inc = mpi_reduce(mpi.global_comm, min_inc, root=0, op=MPI.MIN)
        max_inc = mpi_reduce(mpi.global_comm, max_inc, root=0, op=MPI.MAX)
        mean_inc = mpi_reduce(mpi.global_comm, mean_inc, root=0, op=MPI.SUM)

        # pair_corr statistics
        if pair_corr is not None:
            pair_corr = [mpi_reduce(mpi.global_comm, pair_corr[0], root=0, op=MPI.SUM), \
                         mpi_reduce(mpi.global_comm, pair_corr[1], root=0, op=MPI.SUM)]

        # mean increment
        if mpi.global_master:
            mean_inc /= exp.n_tuples['inc'][-1]

        # write restart files & save timings
        if mpi.global_master:
            if exp.rst:
                write_file(exp.order, max_inc, 'mbe_max_inc')
                write_file(exp.order, min_inc, 'mbe_min_inc')
                write_file(exp.order, mean_inc, 'mbe_mean_inc')
                write_file(exp.order, np.asarray(exp.n_tuples['inc'][-1]), 'mbe_idx')
                write_file(exp.order, hashes[-1], 'mbe_hashes')
                write_file(exp.order, inc[-1], 'mbe_inc')
            exp.time['mbe'][-1] += MPI.Wtime() - time

        # allreduce screen
        tot_screen = mpi_allreduce(mpi.global_comm, screen, op=MPI.MAX)

        # update expansion space wrt screened orbitals
        nonzero_screen = tot_screen[np.nonzero(tot_screen)[0]]
        thres = 1. if exp.order < exp.screen_start else exp.screen_perc
        screen_idx = int(thres * exp.exp_space[-1].size)
        exp.exp_space.append(exp.exp_space[-1][np.sort(np.argsort(nonzero_screen)[::-1][:screen_idx])])

        # write restart files
        if mpi.global_master:
            if exp.rst:
                write_file(exp.order, tot_screen, 'mbe_screen')
                write_file(exp.order+1, exp.exp_space[-1], 'exp_space')

        # total property
        tot = mean_inc * exp.n_tuples['inc'][-1]

        # mpi barrier
        mpi.local_comm.Barrier()

        if mpi.global_master and pair_corr is not None:
            pair_corr[1] = pair_corr[1][np.argsort(np.abs(pair_corr[0]))[::-1]]
            pair_corr[0] = pair_corr[0][np.argsort(np.abs(pair_corr[0]))[::-1]]
            logger.debug('\n --------------------------------------------------------------------------')
            logger.debug(f'{"pair correlation information":^75s}')
            logger.debug(' --------------------------------------------------------------------------')
            logger.debug(' orbital tuple  |  absolute corr.  |  relative corr.  |  cumulative corr.')
            logger.debug(' --------------------------------------------------------------------------')
            for i in range(10):
                logger.debug(f'   [{pair_corr[1][i][0]:3d},{pair_corr[1][i][1]:3d}]    |' + \
                      f'    {pair_corr[0][i]:.3e}    |' + \
                      f'        {pair_corr[0][i] / pair_corr[0][0]:.2f}      |' + \
                      f'        {np.sum(pair_corr[0][:i+1]) / np.sum(pair_corr[0]):.2f}')
            logger.debug(' --------------------------------------------------------------------------\n')

        if mpi.global_master:
            return hashes_win, inc_win, tot, mean_inc, min_inc, max_inc
        else:
            return hashes_win, inc_win


def _inc(method: str, base: Optional[str], cc_backend: str, fci_solver: str, \
         orb_type: str, spin: int, occup: np.ndarray, target_mbe: str, \
         fci_state_sym: int, point_group: str, orbsym: np.ndarray, \
         hf_guess: bool, fci_state_root: int, \
         hf_prop: Union[float, np.ndarray], \
         e_core: float, h1e_cas: np.ndarray, h2e_cas: np.ndarray, \
         core_idx: np.ndarray, cas_idx: np.ndarray, verbose: int, \
         dipole_ints: Optional[np.ndarray], \
         ref_prop: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Tuple[int, int]]:
        """
        this function calculates the current-order contribution to the increment associated with a given tuple
        """
        # n_elec
        n_elec = nelec(occup, cas_idx)

        # perform main calc
        res_full = kernel_main(method, cc_backend, fci_solver, orb_type, spin, \
                               occup, target_mbe, fci_state_sym, point_group, \
                               orbsym, hf_guess, fci_state_root, hf_prop, \
                               e_core, h1e_cas, h2e_cas, core_idx, cas_idx, \
                               n_elec, verbose, dipole_ints)

        # perform base calc
        if base is not None:
            res_full -= kernel_main(base, cc_backend, '', orb_type, spin, \
                                    occup, target_mbe, fci_state_sym, \
                                    point_group, orbsym, hf_guess, \
                                    fci_state_root, hf_prop, e_core, h1e_cas, \
                                    h2e_cas, core_idx, cas_idx, n_elec, \
                                    verbose, dipole_ints)

        return res_full - ref_prop, n_elec


def _sum(nocc: int, target_mbe: str, min_order: int, order: int, \
         inc: List[np.ndarray], hashes: List[np.ndarray], ref_occ: bool, \
         ref_virt: bool, tup: np.ndarray) -> Union[float, np.ndarray]:
        """
        this function performs a recursive summation and returns the final increment associated with a given tuple
        """
        # init res
        if target_mbe in ['energy', 'excitation']:
            res = np.zeros(order - min_order, dtype=np.float64)
        else:
            res = np.zeros([order - min_order, 3], dtype=np.float64)

        # occupied and virtual subspaces of tuple
        tup_occ = tup[tup < nocc]
        tup_virt = tup[nocc <= tup]

        # compute contributions from lower-order increments
        for k in range(order-1, min_order-1, -1):

            # loop over subtuples
            for tup_sub in tuples(tup_occ, tup_virt, ref_occ, ref_virt, k):

                # compute index
                idx = hash_lookup(hashes[k-min_order], hash_1d(tup_sub))

                # sum up order increments
                if idx is not None:
                    res[k-min_order] += inc[k-min_order][idx]

        return fsum(res)


def _update(min_prop: Union[float, int], max_prop: Union[float, int], \
            sum_prop: Union[float, int], tup_prop: Union[float, int]) -> Tuple[Union[float, int], ...]:
        """
        this function returns updated statistics
        """
        return np.minimum(min_prop, np.abs(tup_prop)), np.maximum(max_prop, np.abs(tup_prop)), sum_prop + tup_prop
