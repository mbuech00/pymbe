#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
driver module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import numpy as np
from mpi4py import MPI

import mbe
import kernel
import output
import screen
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

        # print output from restarted calculation
        if calc.restart:
            for i in range(exp.min_order, exp.start_order):

                # print mbe header
                print(output.mbe_header(i, exp.n_tuples['theo'][i-exp.min_order]))

                # print mbe end
                print(output.mbe_end(i, exp.time['mbe'][i-exp.min_order], exp.n_tuples['actual'][i-exp.min_order]))

                # print mbe results
                print(output.mbe_results(calc.occup, calc.target_mbe, calc.state['root'], exp.min_order, \
                                            exp.max_order, i, exp.prop[calc.target_mbe]['tot'], \
                                            exp.mean_inc[i-exp.min_order], exp.min_inc[i-exp.min_order], \
                                            exp.max_inc[i-exp.min_order], exp.mean_ndets[i-exp.min_order], \
                                            exp.min_ndets[i-exp.min_order], exp.max_ndets[i-exp.min_order]))

                # print header
                print(output.screen_header(i))

                # print screening results
                if 0 < i:
                    exp.screen_orbs = np.setdiff1d(exp.exp_space[i-exp.min_order-1], exp.exp_space[i-exp.min_order])
                    if exp.screen_orbs.size > 0:
                        print(output.screen_results(exp.screen_orbs, exp.exp_space[:i-exp.min_order+1]))

                # print screen end
                print(output.screen_end(i, exp.time['screen'][i-exp.min_order], False))

        # begin or resume mbe expansion depending
        for exp.order in range(exp.start_order, exp.max_order+1):

            # theoretical number of tuples at current order
            exp.n_tuples['theo'].append(tools.n_tuples(exp.exp_space[-1][exp.exp_space[-1] < mol.nocc], \
                                                       exp.exp_space[-1][mol.nocc <= exp.exp_space[-1]], \
                                                       tools.occ_prune(calc.occup, calc.ref_space), \
                                                       tools.virt_prune(calc.occup, calc.ref_space), exp.order))

            # print mbe header
            print(output.mbe_header(exp.order, exp.n_tuples['theo'][-1]))

            # main mbe function
            hashes_win, n_tuples, inc_win, tot, \
                    mean_ndets, min_ndets, max_ndets, mean_inc, min_inc, max_inc = mbe.main(mpi, mol, calc, exp)

            # append window to hashes
            if len(exp.prop[calc.target_mbe]['hashes']) > len(exp.prop[calc.target_mbe]['tot']):
                exp.prop[calc.target_mbe]['hashes'][-1] = hashes_win
            else:
                exp.prop[calc.target_mbe]['hashes'].append(hashes_win)

            # append n_tuples
            exp.n_tuples['actual'].append(n_tuples)

            # append window to increments
            if len(exp.prop[calc.target_mbe]['inc']) > len(exp.prop[calc.target_mbe]['tot']):
                exp.prop[calc.target_mbe]['inc'][-1] = inc_win
            else:
                exp.prop[calc.target_mbe]['inc'].append(inc_win)

            # append determinant statistics
            if len(exp.mean_ndets) == len(exp.prop[calc.target_mbe]['inc']):
                exp.mean_ndets[-1] = mean_ndets
                exp.min_ndets[-1] = min_ndets
                exp.max_ndets[-1] = max_ndets
            else:
                exp.mean_ndets.append(mean_ndets)
                exp.min_ndets.append(min_ndets)
                exp.max_ndets.append(max_ndets)

            # append increment statistics
            if len(exp.mean_inc) == len(exp.prop[calc.target_mbe]['inc']):
                exp.mean_inc[-1] = mean_inc
                exp.min_inc[-1] = min_inc
                exp.max_inc[-1] = max_inc
            else:
                exp.mean_inc.append(mean_inc)
                exp.min_inc.append(min_inc)
                exp.max_inc.append(max_inc)

            # append total property
            exp.prop[calc.target_mbe]['tot'].append(tot)
            if exp.order > exp.min_order:
                exp.prop[calc.target_mbe]['tot'][-1] += exp.prop[calc.target_mbe]['tot'][-2]

            # print mbe end
            print(output.mbe_end(exp.order, exp.time['mbe'][-1], exp.n_tuples['actual'][-1]))

            # print mbe results
            print(output.mbe_results(calc.occup, calc.target_mbe, calc.state['root'], exp.min_order, \
                                     exp.max_order, exp.order, exp.prop[calc.target_mbe]['tot'], \
                                     exp.mean_inc[-1], exp.min_inc[-1], exp.max_inc[-1], \
                                     exp.mean_ndets[-1], exp.min_ndets[-1], exp.max_ndets[-1]))

            # init screening time
            exp.time['screen'].append(0.)

            if exp.order < exp.max_order:

                # print header
                print(output.screen_header(exp.order))

                # start time
                time = MPI.Wtime()

                # main screening function
                exp.screen_orbs = screen.main(mpi, mol, calc, exp)

                # update expansion space wrt screened orbitals
                exp.exp_space.append(np.copy(exp.exp_space[-1]))
                for mo in exp.screen_orbs:
                    exp.exp_space[-1] = exp.exp_space[-1][exp.exp_space[-1] != mo]

                # print screening results
                if exp.screen_orbs.size > 0:
                    print(output.screen_results(exp.screen_orbs, exp.exp_space))

                # collect time
                exp.time['screen'][-1] = MPI.Wtime() - time

                # purging logical
                purging = calc.misc['purge'] and 0 < exp.screen_orbs.size and exp.order + 1 <= exp.exp_space[-1].size

                # print screen end
                print(output.screen_end(exp.order, exp.time['screen'][-1], \
                                        purging, exp.exp_space[-1].size < exp.order + 1))

            # init screening time
            exp.time['purge'].append(0.)

            if purging:

                # print header
                print(output.purge_header(exp.order))

                # start time
                time = MPI.Wtime()

                # main purging function
                exp.prop[calc.target_mbe] = purge.main(mpi, mol, calc, exp)

                # print purging results
                print(output.purge_results(exp.n_tuples, exp.min_order, exp.order))

                # collect time
                exp.time['purge'][-1] = MPI.Wtime() - time

                # print screen end
                print(output.purge_end(exp.order, exp.time['purge'][-1]))

            # write restart files
            if calc.misc['rst']:
                tools.write_file(exp.order, np.asarray(exp.n_tuples['theo'][-1]), 'mbe_n_tuples_theo')
                tools.write_file(exp.order, np.asarray(exp.n_tuples['actual'][-1]), 'mbe_n_tuples_actual')
                tools.write_file(exp.order, np.asarray(exp.prop[calc.target_mbe]['tot'][-1]), 'mbe_tot')
                tools.write_file(exp.order, exp.mean_ndets[-1], 'mbe_mean_ndets')
                tools.write_file(exp.order, exp.max_ndets[-1], 'mbe_max_ndets')
                tools.write_file(exp.order, exp.min_ndets[-1], 'mbe_min_ndets')
                tools.write_file(exp.order, exp.mean_inc[-1], 'mbe_mean_inc')
                tools.write_file(exp.order, exp.max_inc[-1], 'mbe_max_inc')
                tools.write_file(exp.order, exp.min_inc[-1], 'mbe_min_inc')
                tools.write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')
                tools.write_file(exp.order, np.asarray(exp.time['screen'][-1]), 'mbe_time_screen')
                tools.write_file(exp.order, np.asarray(exp.time['purge'][-1]), 'mbe_time_purge')
                tools.write_file(exp.order+1, exp.exp_space[-1], 'exp_space')

            # convergence check
            if exp.exp_space[-1].size < exp.order + 1 or exp.order == exp.max_order:

                # final order
                exp.final_order = exp.order

                # timings
                exp.time['mbe'] = np.asarray(exp.time['mbe'])
                exp.time['screen'] = np.asarray(exp.time['screen'])
                exp.time['total'] = exp.time['mbe'] + exp.time['screen']

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

                # receive theoretical number of tuples at current order
                exp.n_tuples['theo'].append(msg['n_tuples_theo'])

                # main mbe function
                hashes_win, n_tuples, inc_win = mbe.main(mpi, mol, calc, exp, \
                                                         rst_read=msg['rst_read'], \
                                                         tup_start=msg['tup_start'])

                # append window to hashes
                if msg['rst_read']:
                    exp.prop[calc.target_mbe]['hashes'][-1] = hashes_win # type: ignore
                else:
                    exp.prop[calc.target_mbe]['hashes'].append(hashes_win) # type: ignore

                # append n_tuples
                exp.n_tuples['actual'].append(n_tuples)

                # append window to increments
                if msg['rst_read']:
                    exp.prop[calc.target_mbe]['inc'][-1] = inc_win # type: ignore
                else:
                    exp.prop[calc.target_mbe]['inc'].append(inc_win) # type: ignore

            elif msg['task'] == 'screen':

                # receive order
                exp.order = msg['order']

                # main screening function
                exp.screen_orbs = screen.main(mpi, mol, calc, exp)

                # update expansion space wrt screened orbitals
                exp.exp_space.append(np.copy(exp.exp_space[-1]))
                for mo in exp.screen_orbs:
                    exp.exp_space[-1] = exp.exp_space[-1][exp.exp_space[-1] != mo]

            elif msg['task'] == 'purge':

                # receive order
                exp.order = msg['order']

                # main purging function
                exp.prop[calc.target_mbe] = purge.main(mpi, mol, calc, exp)

                # update expansion space wrt screened orbitals
                exp.exp_space.append(np.copy(exp.exp_space[-1]))
                for mo in exp.screen_orbs:
                    exp.exp_space[-1] = exp.exp_space[-1][exp.exp_space[-1] != mo]

            elif msg['task'] == 'exit':

                slave = False

        # finalize mpi
        parallel.finalize(mpi)


