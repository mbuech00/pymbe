#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
driver module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import numpy as np
from mpi4py import MPI

import mbe
import kernel
import output
import purge
import system
import calculation
import expansion
import parallel
import tools


def master(mpi: parallel.MPICls, mol: system.MolCls, \
            calc:calculation.CalcCls, exp: expansion.ExpCls) -> None:
        """
        this function is the main pymbe master function
        """
        # print expansion headers
        print(output.main_header(mpi=mpi, method=calc.model['method']))

#        # print output from restarted calculation
#        if calc.restart:
#            for i in range(exp.min_order, exp.start_order):
#
#                # print mbe header
#                print(output.mbe_header(i, exp.n_tuples['prop'][i-exp.min_order]))
#
#                # print mbe end
#                print(output.mbe_end(i, exp.time['mbe'][i-exp.min_order], \
#                                     exp.n_tuples['inc'][i-exp.min_order]))
#
#                # print mbe results
#                print(output.mbe_results(calc.occup, calc.target_mbe, calc.state['root'], \
#                                            exp.min_order, i, exp.prop[calc.target_mbe]['tot'], \
#                                            exp.mean_inc[i-exp.min_order], exp.min_inc[i-exp.min_order], \
#                                            exp.max_inc[i-exp.min_order], exp.mean_ndets[i-exp.min_order], \
#                                            exp.min_ndets[i-exp.min_order], exp.max_ndets[i-exp.min_order]))
#
#                # print screening results
#                if 0 < i:
#                    exp.screen_orbs = np.setdiff1d(exp.exp_space[i-exp.min_order-1], exp.exp_space[i-exp.min_order])
#                    if exp.screen_orbs.size > 0:
#                        print(output.screen_results(exp.screen_orbs, exp.exp_space[:i-exp.min_order+1]))

        # begin or resume mbe expansion depending
        for exp.order in range(exp.start_order, exp.max_order+1):

            # theoretical and actual number of tuples at current order
            exp.n_tuples['theo'].append(tools.n_tuples(exp.exp_space[0][exp.exp_space[0] < mol.nocc], \
                                                       exp.exp_space[0][mol.nocc <= exp.exp_space[0]], \
                                                       tools.occ_prune(calc.occup, calc.ref_space), \
                                                       tools.virt_prune(calc.occup, calc.ref_space), exp.order))
            exp.n_tuples['prop'].append(tools.n_tuples(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc], \
                                                       exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]], \
                                                       tools.occ_prune(calc.occup, calc.ref_space), \
                                                       tools.virt_prune(calc.occup, calc.ref_space), exp.order))

            # print mbe header
            print(output.mbe_header(exp.order, exp.n_tuples['prop'][-1]))

            # main mbe function
            hashes_win, n_tuples, inc_win, tot, mean_ndets, min_ndets, max_ndets, \
                mean_inc, min_inc, max_inc, exp.screen_orbs = mbe.main(mpi, mol, calc, exp)

            # append window to hashes
            if len(exp.prop[calc.target_mbe]['hashes']) == len(exp.n_tuples['prop']):
                exp.prop[calc.target_mbe]['hashes'][-1] = hashes_win
            else:
                exp.prop[calc.target_mbe]['hashes'].append(hashes_win)

            # append n_tuples
            if len(exp.n_tuples['inc']) < len(exp.n_tuples['prop']):
                exp.n_tuples['inc'].append(n_tuples)

            # append window to increments
            if len(exp.prop[calc.target_mbe]['inc']) == len(exp.n_tuples['prop']):
                exp.prop[calc.target_mbe]['inc'][-1] = inc_win
            else:
                exp.prop[calc.target_mbe]['inc'].append(inc_win)

            # append total property
            exp.prop[calc.target_mbe]['tot'].append(tot)
            if exp.order > exp.min_order:
                exp.prop[calc.target_mbe]['tot'][-1] += exp.prop[calc.target_mbe]['tot'][-2]

            # append determinant statistics
            if exp.order == exp.start_order and exp.min_order < exp.start_order:
                exp.mean_ndets[-1] = mean_ndets
                exp.min_ndets[-1] = min_ndets
                exp.max_ndets[-1] = max_ndets
            else:
                exp.mean_ndets.append(mean_ndets)
                exp.min_ndets.append(min_ndets)
                exp.max_ndets.append(max_ndets)

            # append increment statistics
            if exp.order == exp.start_order and exp.min_order < exp.start_order:
                exp.mean_inc[-1] = mean_inc
                exp.min_inc[-1] = min_inc
                exp.max_inc[-1] = max_inc
            else:
                exp.mean_inc.append(mean_inc)
                exp.min_inc.append(min_inc)
                exp.max_inc.append(max_inc)

            # print mbe end
            print(output.mbe_end(exp.order, exp.time['mbe'][-1], \
                                 exp.n_tuples['inc'][-1]))

            # print mbe results
            print(output.mbe_results(calc.occup, calc.target_mbe, calc.state['root'], \
                                     exp.min_order, exp.order, exp.prop[calc.target_mbe]['tot'], \
                                     exp.mean_inc[-1], exp.min_inc[-1], exp.max_inc[-1], \
                                     exp.mean_ndets[-1], exp.min_ndets[-1], exp.max_ndets[-1]))

            # print screening results
            if exp.screen_orbs.size > 0:
                print(output.screen_results(exp.screen_orbs, exp.exp_space))

            # print header
            print(output.purge_header(exp.order))

            # main purging function
            exp.prop[calc.target_mbe], exp.n_tuples = purge.main(mpi, mol, calc, exp)

            # print purging results
            if exp.order + 1 <= exp.exp_space[-1].size and exp.n_tuples['inc'][-1] < exp.n_tuples['prop'][-1]:
                print(output.purge_results(exp.n_tuples, exp.min_order, exp.order))

            # print purge end
            print(output.purge_end(exp.order, exp.time['purge'][-1]))

            # write restart files
            if calc.misc['rst']:
                if exp.screen_orbs.size > 0:
                    for k in range(exp.order-exp.min_order+1):
                        buf = exp.prop[calc.target_mbe]['hashes'][k].Shared_query(0)[0] # type: ignore
                        hashes = np.ndarray(buffer=buf, dtype=np.int64, \
                                            shape=(exp.n_tuples['inc'][k],))
                        tools.write_file(k + exp.min_order, hashes, 'mbe_hashes')
                        buf = exp.prop[calc.target_mbe]['inc'][k].Shared_query(0)[0] # type: ignore
                        inc = np.ndarray(buffer=buf, dtype=np.float64, \
                                         shape=tools.inc_shape(exp.n_tuples['inc'][k], tools.inc_dim(calc.target_mbe)))
                        tools.write_file(k + exp.min_order, inc, 'mbe_inc')
                        tools.write_file(k + exp.min_order, np.asarray(exp.n_tuples['inc'][k]), 'mbe_n_tuples_inc')
                else:
                    buf = exp.prop[calc.target_mbe]['hashes'][-1].Shared_query(0)[0] # type: ignore
                    hashes = np.ndarray(buffer=buf, dtype=np.int64, \
                                        shape=(exp.n_tuples['inc'][-1],))
                    tools.write_file(exp.order, hashes, 'mbe_hashes')
                    buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0] # type: ignore
                    inc = np.ndarray(buffer=buf, dtype=np.float64, \
                                     shape=tools.inc_shape(exp.n_tuples['inc'][-1], tools.inc_dim(calc.target_mbe)))
                    tools.write_file(exp.order, inc, 'mbe_inc')
                    tools.write_file(exp.order, np.asarray(exp.n_tuples['inc'][-1]), 'mbe_n_tuples_inc')
                tools.write_file(exp.order, np.asarray(exp.n_tuples['theo'][-1]), 'mbe_n_tuples_theo')
                tools.write_file(exp.order, np.asarray(exp.n_tuples['prop'][-1]), 'mbe_n_tuples_prop')
                tools.write_file(exp.order, np.asarray(exp.prop[calc.target_mbe]['tot'][-1]), 'mbe_tot')
                tools.write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')
                tools.write_file(exp.order, np.asarray(exp.time['purge'][-1]), 'mbe_time_purge')
                tools.write_file(exp.order+1, exp.exp_space[-1], 'exp_space')

            # convergence check
            if exp.exp_space[-1].size < exp.order + 1 or exp.order == exp.max_order:

                # final order
                exp.final_order = exp.order

                # timings
                exp.time['mbe'] = np.asarray(exp.time['mbe'])
                exp.time['purge'] = np.asarray(exp.time['purge'])
                exp.time['total'] = exp.time['mbe'] + exp.time['purge']

                # increments
                exp.mean_inc = np.asarray(exp.mean_inc)
                exp.min_inc = np.asarray(exp.min_inc)
                exp.max_inc = np.asarray(exp.max_inc)

                # ndets
                exp.mean_ndets = np.asarray(exp.mean_ndets)
                exp.min_ndets = np.asarray(exp.min_ndets)
                exp.max_ndets = np.asarray(exp.max_ndets)

                # final results
                exp.prop[calc.target_mbe]['tot'] = np.asarray(exp.prop[calc.target_mbe]['tot'])
                print('\n\n')

                break


def slave(mpi: parallel.MPICls, mol: system.MolCls, \
            calc: calculation.CalcCls, exp: expansion.ExpCls) -> None:
        """
        this function is the main pymbe slave function
        """
        # set loop/waiting logical
        slave = True

        # enter slave state
        while slave:

            # task id
            msg = mpi.global_comm.bcast(None, root=0)

            if msg['task'] == 'mbe':

                # receive order
                exp.order = msg['order']

                # actual number of tuples at current order
                exp.n_tuples['prop'].append(tools.n_tuples(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc], \
                                                           exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]], \
                                                           tools.occ_prune(calc.occup, calc.ref_space), \
                                                           tools.virt_prune(calc.occup, calc.ref_space), exp.order))

                # main mbe function
                hashes_win, n_tuples, inc_win, exp.screen_orbs = mbe.main(mpi, mol, calc, exp, \
                                                                          rst_read_a=msg['rst_read_a'], \
                                                                          rst_read_b=msg['rst_read_b'], \
                                                                          tup_start_a=msg['tup_start_a'], \
                                                                          tup_start_b=msg['tup_start_b'])

                # append window to hashes
                if len(exp.prop[calc.target_mbe]['hashes']) == len(exp.n_tuples['prop']):
                    exp.prop[calc.target_mbe]['hashes'][-1] = hashes_win
                else:
                    exp.prop[calc.target_mbe]['hashes'].append(hashes_win)

                # append n_tuples
                if len(exp.n_tuples['inc']) < len(exp.n_tuples['prop']):
                    exp.n_tuples['inc'].append(n_tuples)

                # append window to increments
                if len(exp.prop[calc.target_mbe]['inc']) == len(exp.n_tuples['prop']):
                    exp.prop[calc.target_mbe]['inc'][-1] = inc_win
                else:
                    exp.prop[calc.target_mbe]['inc'].append(inc_win)

            elif msg['task'] == 'purge':

                # receive order
                exp.order = msg['order']

                # main purging function
                exp.prop[calc.target_mbe], exp.n_tuples = purge.main(mpi, mol, calc, exp)

            elif msg['task'] == 'exit':

                slave = False

        # finalize mpi
        parallel.finalize(mpi)


