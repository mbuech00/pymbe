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
from typing import Tuple, List, Dict, Union, Any

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
            rst_mbe: bool = False, tup_start: int = 0) -> Tuple[Any, ...]:
        """
        this function is the mbe main function
        """
        if mpi.global_master:

            # restart run
            rst_mbe = len(exp.prop[calc.target_mbe]['inc']) > len(exp.prop[calc.target_mbe]['tot'])

            # start index
            tup_start = tools.read_file(exp.order, 'mbe_idx') if rst_mbe else 0

            # wake up slaves
            msg = {'task': 'mbe', 'order': exp.order, 'rst_mbe': rst_mbe, 'tup_start': tup_start}
            mpi.global_comm.bcast(msg, root=0)

        # load eri
        buf = mol.eri.Shared_query(0)[0]
        eri = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb * (mol.norb + 1) // 2,) * 2)

        # load hcore
        buf = mol.hcore.Shared_query(0)[0]
        hcore = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.norb,) * 2)

        # load vhf
        buf = mol.vhf.Shared_query(0)[0]
        vhf = np.ndarray(buffer=buf, dtype=np.float64, shape=(mol.nocc, mol.norb, mol.norb))

        # load hashes for previous orders
        hashes = []
        for k in range(exp.order-exp.min_order):
            buf = exp.hashes[k].Shared_query(0)[0] # type: ignore
            hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples[k],)))

        # init hashes for present order
        if rst_mbe:
            hash_win = exp.hashes[-1]
            buf = hash_win.Shared_query(0)[0] # type: ignore
        else:
            if mpi.local_master:
                hash_win = MPI.Win.Allocate_shared(8 * exp.n_tuples[-1], 8, comm=mpi.local_comm)
            else:
                hash_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
            buf = hash_win.Shared_query(0)[0] # type: ignore
        hashes.append(np.ndarray(buffer=buf, dtype=np.int64, shape=(exp.n_tuples[-1],)))
        if mpi.local_master:
            hashes[-1][:] = np.zeros_like(hashes[-1])

        # load increments for previous orders
        inc = []
        for k in range(exp.order-exp.min_order):
            buf = exp.prop[calc.target_mbe]['inc'][k].Shared_query(0)[0] # type: ignore
            if calc.target_mbe in ['energy', 'excitation']:
                inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[k],)))
            elif calc.target_mbe in ['dipole', 'trans']:
                inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[k], 3)))

        # init increments for present order
        if rst_mbe:
            inc_win = exp.prop[calc.target_mbe]['inc'][-1]
            buf = inc_win.Shared_query(0)[0] # type: ignore
        else:
            if mpi.local_master:
                if calc.target_mbe in ['energy', 'excitation']:
                    inc_win = MPI.Win.Allocate_shared(8 * exp.n_tuples[-1], 8, comm=mpi.local_comm)
                elif calc.target_mbe in ['dipole', 'trans']:
                    inc_win = MPI.Win.Allocate_shared(8 * exp.n_tuples[-1] * 3, 8, comm=mpi.local_comm)
            else:
                inc_win = MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm)
            buf = inc_win.Shared_query(0)[0] # type: ignore
        if calc.target_mbe in ['energy', 'excitation']:
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[-1],)))
        elif calc.target_mbe in ['dipole', 'trans']:
            inc.append(np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.n_tuples[-1], 3)))
        if mpi.local_master:
            inc[-1][:] = np.zeros_like(inc[-1])

        # init time
        if mpi.global_master:
            if not rst_mbe:
                exp.time['mbe'].append(0.)
            time = MPI.Wtime()

        # init determinant statistics
        if mpi.global_master and rst_mbe:
            min_ndets: int = exp.min_ndets[-1]
            max_ndets: int = exp.max_ndets[-1]
            sum_ndets: int = exp.mean_ndets[-1]
        else:
            min_ndets = int(1e12)
            max_ndets = 0
            sum_ndets = 0

        # mpi barrier
        mpi.global_comm.Barrier()

        # occupied and virtual expansion spaces
        occ_space = calc.exp_space[calc.exp_space < mol.nocc]
        virt_space = calc.exp_space[mol.nocc <= calc.exp_space]

        # allow for tuples with only occupied or virtual MOs
        occ_only = tools.virt_prune(calc.occup, calc.ref_space)
        virt_only = tools.occ_prune(calc.occup, calc.ref_space)

        # loop until no tuples left
        for idx, tup in enumerate(itertools.islice(tools.tuples(occ_space, virt_space, occ_only, virt_only, exp.order), \
                                                    tup_start, None)):

            # distribute tuples
            if idx % mpi.global_size != mpi.global_rank:
                continue

            # recast tup as numpy array
            tup = np.asarray(tup, dtype=np.int64)

            # get core and cas indices
            core_idx, cas_idx = tools.core_cas(mol.nocc, calc.ref_space, tup)

            # get h2e indices
            cas_idx_tril = tools.cas_idx_tril(cas_idx)

            # get h2e_cas
            h2e_cas = eri[cas_idx_tril[:, None], cas_idx_tril]

            # compute e_core and h1e_cas
            e_core, h1e_cas = kernel.e_core_h1e(mol.e_nuc, hcore, vhf, core_idx, cas_idx)

            # calculate increment
            inc_tup, ndets, nelec = _inc(calc.model['method'], calc.base['method'], calc.model['solver'], mol.spin, \
                                           calc.occup, calc.target_mbe, calc.state['wfnsym'], calc.orbsym, \
                                            calc.model['hf_guess'], calc.state['root'], calc.prop['hf']['energy'], \
                                            calc.prop['ref'][calc.target_mbe], e_core, h1e_cas, h2e_cas, \
                                            core_idx, cas_idx, mol.debug, \
                                            mol.dipole if calc.target_mbe in ['dipole', 'trans'] else None, \
                                            calc.mo_coeff if calc.target_mbe in ['dipole', 'trans'] else None, \
                                            calc.prop['hf']['dipole'] if calc.target_mbe in ['dipole', 'trans'] else None)

            # calculate increment
            if exp.order > exp.min_order:
                if np.any(inc_tup != 0.):
                    inc_tup -= _sum(calc.occup, calc.target_mbe, exp.min_order, exp.order, \
                                    inc, hashes, occ_only, virt_only, tup, pi_prune=calc.extra['pi_prune'])


            # debug print
            if mol.debug >= 1:
                print(output.mbe_debug(mol.atom, mol.symmetry, calc.orbsym, calc.state['root'], \
                                        ndets, nelec, inc_tup, exp.order, cas_idx, tup))

            # add to hashes
            hashes[-1][idx] = tools.hash_1d(tup)

            # add to inc
            inc[-1][idx] = inc_tup

            # update determinant statistics
            min_ndets = min(min_ndets, ndets)
            max_ndets = max(max_ndets, ndets)
            sum_ndets += ndets

            # write restart files
            if calc.misc['rst'] and tup_start + idx > mpi.global_size and tup_start + idx % calc.misc['rst_freq'] == mpi.global_rank:

                # reduce hashes & increments onto global master
                if mpi.num_masters > 1:

                    hashes[-1][:] = parallel.reduce(mpi.master_comm, hashes[-1], root=0)
                    inc[-1][:] = parallel.reduce(mpi.master_comm, inc[-1], root=0)
                    min_ndets = mpi.global_comm.reduce(min_ndets, root=0, op=MPI.MIN)
                    max_ndets = mpi.global_comm.reduce(max_ndets, root=0, op=MPI.MAX)
                    sum_ndets = mpi.global_comm.reduce(sum_ndets, root=0, op=MPI.SUM)

                    if not mpi.global_master:

                        hashes[-1][:] = np.zeros_like(hashes[-1])
                        inc[-1][:] = np.zeros_like(inc[-1])
                        sum_ndets = 0

                if mpi.global_master:

                    # save idx
                    tools.write_file(exp.order, np.asarray(tup_start + idx), 'mbe_idx')
                    # save hashes
                    tools.write_file(exp.order, hashes[-1], 'mbe_hashes')
                    # save increments
                    tools.write_file(exp.order, inc[-1], 'mbe_inc')
                    # save determinant statistics
                    tools.write_file(exp.order, np.asarray(max_ndets, dtype=np.int64), 'mbe_max_ndets')
                    tools.write_file(exp.order, np.asarray(min_ndets, dtype=np.int64), 'mbe_min_ndets')
                    tools.write_file(exp.order, np.asarray(sum_ndets, dtype=np.int64), 'mbe_mean_ndets')
                    # save timing
                    exp.time['mbe'][-1] += MPI.Wtime() - time
                    tools.write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')

                    # re-init time
                    time = MPI.Wtime()

                    # print status
                    print(output.mbe_status((tup_start + idx) / exp.n_tuples[-1]))

        # mpi barrier
        mpi.global_comm.Barrier()

        # print final status
        if mpi.global_master:
            print(output.mbe_status(1.))

        # determinant statistics
        min_ndets = mpi.global_comm.reduce(min_ndets, root=0, op=MPI.MIN)
        max_ndets = mpi.global_comm.reduce(max_ndets, root=0, op=MPI.MAX)
        sum_ndets = mpi.global_comm.reduce(sum_ndets, root=0, op=MPI.SUM)

        # mean number of determinants
        if mpi.global_master:
            mean_ndets = round(sum_ndets / exp.n_tuples[-1])

        if mpi.local_master:

            # allreduce hashes & increments among local masters
            hashes[-1][:] = parallel.allreduce(mpi.master_comm, hashes[-1])
            inc[-1][:] = parallel.allreduce(mpi.master_comm, inc[-1])

            # sort increments wrt hashes
            inc[-1][:] = inc[-1][np.argsort(hashes[-1])]
            hashes[-1].sort()

        # mpi barrier
        mpi.global_comm.Barrier()

        # collect results on global master
        if mpi.global_master:

            if calc.misc['rst']:

                # save idx
                tools.write_file(exp.order, np.asarray(exp.n_tuples[-1]-1), 'mbe_idx')
                # save hashes
                tools.write_file(exp.order, hashes[-1], 'mbe_hashes')
                # save increments
                tools.write_file(exp.order, inc[-1], 'mbe_inc')

            # total property
            tot = tools.fsum(inc[-1])

            # statistics
            if calc.target_mbe in ['energy', 'excitation']:

                # increments
                if inc[-1].any():
                    mean_inc = np.mean(inc[-1][np.nonzero(inc[-1])])
                    min_inc = np.min(np.abs(inc[-1][np.nonzero(inc[-1])]))
                    max_inc = np.max(np.abs(inc[-1][np.nonzero(inc[-1])]))
                else:
                    mean_inc = min_inc = max_inc = 0.

            elif calc.target_mbe in ['dipole', 'trans']:

                # init result arrays
                mean_inc = np.empty(3, dtype=np.float64)
                min_inc = np.empty(3, dtype=np.float64)
                max_inc = np.empty(3, dtype=np.float64)

                # loop over x, y, and z
                for k in range(3):

                    # increments
                    if inc[-1][:, k].any():
                        mean_inc[k] = np.mean(inc[-1][:, k][np.nonzero(inc[-1][:, k])])
                        min_inc[k] = np.min(np.abs(inc[-1][:, k][np.nonzero(inc[-1][:, k])]))
                        max_inc[k] = np.max(np.abs(inc[-1][:, k][np.nonzero(inc[-1][:, k])]))
                    else:
                        mean_inc[k] = min_inc[k] = max_inc[k] = 0.

            # save timing
            exp.time['mbe'][-1] += MPI.Wtime() - time

            return inc_win, hash_win, tot, mean_ndets, min_ndets, max_ndets, mean_inc, min_inc, max_inc

        else:

            return inc_win, hash_win


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


def _sum(occup: np.ndarray, target_mbe: str, min_order: int, order: int, \
            inc: List[np.ndarray], hashes: List[np.ndarray], occ_only: bool, virt_only: bool, \
            tup: np.ndarray, pi_prune: bool = False) -> Union[float, np.ndarray]:
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
            res = 0.
        else:
            res = np.zeros(3, dtype=np.float64)

        # compute contributions from lower-order increments
        for k in range(order-1, min_order-1, -1):

            # generate array with all subsets of particular tuple
            combs = np.array([comb for comb in itertools.combinations(tup, k)], dtype=np.int64)

            # prune combinations without occupied orbitals
            if not occ_only:
                combs = combs[np.fromiter(map(functools.partial(tools.occ_prune, occup), combs), \
                                              dtype=bool, count=combs.shape[0])]

            # prune combinations without virtual orbitals
            if not virt_only:
                combs = combs[np.fromiter(map(functools.partial(tools.virt_prune, occup), combs), \
                                                  dtype=bool, count=combs.shape[0])]

            # prune combinations with non-degenerate pairs of pi-orbitals
#            if pi_prune:
#                combs = combs[np.fromiter(map(functools.partial(tools.pi_prune, \
#                                              exp_space['pi_orbs'], exp_space['pi_hashes']), combs), \
#                                              dtype=bool, count=combs.shape[0])]

            if combs.size == 0:
                continue

            # convert to sorted hashes
            combs_hash = tools.hash_2d(combs)
            combs_hash.sort()

            # get indices of combinations
            idx = tools.hash_compare(hashes[k-min_order], combs_hash)

            # assertion
            tools.assertion(idx is not None, 'error in recursive increment calculation:\n'
                                             'k = {:}\ntup:\n{:}\ncombs:\n{:}'. \
                                             format(k, tup, combs))

            # add up lower-order increments
            res += tools.fsum(inc[k-min_order][idx])

        return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()



