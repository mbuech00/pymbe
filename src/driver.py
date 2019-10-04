#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
driver module containing main master and slave pymbe functions
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
            for i in range(exp.start_order - exp.min_order):

                # print mbe header
                print(output.mbe_header(exp.n_tasks[i], i + exp.min_order))

                # print mbe end
                print(output.mbe_end(i + exp.min_order, exp.time['mbe'][i]))

                # print mbe results
                print(output.mbe_results(calc.occup, calc.ref_space, calc.target_mbe, calc.state['root'], exp.min_order, \
                                            exp.max_order, i + exp.min_order, exp.prop[calc.target_mbe]['tot'], \
                                            exp.mean_inc[i], exp.min_inc[i], exp.max_inc[i], \
                                            exp.mean_ndets[i], exp.min_ndets[i], exp.max_ndets[i]))

                # print header
                print(output.screen_header(i + exp.min_order))

                # print screen end
                print(output.screen_end(i + exp.min_order, exp.time['screen'][i]))

        # begin or resume mbe expansion depending
        for exp.order in range(exp.start_order, exp.max_order+1):

            if len(exp.hashes) > len(exp.prop[calc.target_mbe]['tot']):

                # print mbe header
                print(output.mbe_header(exp.n_tasks[-1], exp.order))

                # main mbe function
                inc_win, tot, mean_ndets, min_ndets, max_ndets, \
                    mean_inc, min_inc, max_inc = mbe.master(mpi, mol, calc, exp)

                # append window to increments
                if len(exp.prop[calc.target_mbe]['inc']) < len(exp.hashes):
                    exp.prop[calc.target_mbe]['inc'].append(inc_win)
                else:
                    exp.prop[calc.target_mbe]['inc'][-1] = inc_win

                # append determinant statistics
                if len(exp.max_ndets) == len(exp.hashes):
                    exp.min_ndets[-1] = min_ndets
                    exp.max_ndets[-1] = max_ndets
                    exp.mean_ndets[-1] = mean_ndets
                else:
                    exp.min_ndets.append(min_ndets)
                    exp.max_ndets.append(max_ndets)
                    exp.mean_ndets.append(mean_ndets)

                # append increment statistics
                exp.mean_inc.append(mean_inc)
                exp.min_inc.append(min_inc)
                exp.max_inc.append(max_inc)

                # append total property
                exp.prop[calc.target_mbe]['tot'].append(tot)
                if exp.order > exp.min_order:
                    exp.prop[calc.target_mbe]['tot'][-1] += exp.prop[calc.target_mbe]['tot'][-2]

                # write restart files
                if calc.misc['rst']:
                    tools.write_file(exp.order, exp.prop[calc.target_mbe]['tot'][-1], 'mbe_tot')
                    tools.write_file(exp.order, np.asarray(exp.mean_ndets[-1]), 'mbe_mean_ndets')
                    tools.write_file(exp.order, np.asarray(exp.max_ndets[-1]), 'mbe_max_ndets')
                    tools.write_file(exp.order, np.asarray(exp.min_ndets[-1]), 'mbe_min_ndets')
                    tools.write_file(exp.order, exp.mean_inc[-1], 'mbe_mean_inc')
                    tools.write_file(exp.order, exp.max_inc[-1], 'mbe_max_inc')
                    tools.write_file(exp.order, exp.min_inc[-1], 'mbe_min_inc')
                    tools.write_file(exp.order, np.asarray(exp.time['mbe'][-1]), 'mbe_time_mbe')

                # print mbe end
                print(output.mbe_end(exp.order, exp.time['mbe'][-1]))

            # print mbe results
            print(output.mbe_results(calc.occup, calc.ref_space, calc.target_mbe, calc.state['root'], exp.min_order, \
                                     exp.max_order, exp.order, exp.prop[calc.target_mbe]['tot'], \
                                     exp.mean_inc[-1], exp.min_inc[-1], exp.max_inc[-1], \
                                     exp.mean_ndets[-1], exp.min_ndets[-1], exp.max_ndets[-1]))

            # init screening time
            exp.time['screen'].append(0.0)

            if exp.order < exp.max_order:

                # print header
                print(output.screen_header(exp.order))

                # start time
                time = MPI.Wtime()

                # main screening function
                hashes_win, tuples_win, n_tasks = screen.master(mpi, calc, exp)

                # append n_tasks
                exp.n_tasks.append(n_tasks)

                # collect time
                exp.time['screen'][-1] = MPI.Wtime() - time

            # convergence check
            if exp.n_tasks[-1] == 0 or exp.order == exp.max_order:

                # print screen end
                if exp.n_tasks[-1] == 0:
                    print(output.screen_end(exp.order, exp.time['screen'][-1], conv=True))

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

            else:

                # overwrite window to tuples
                exp.tuples = tuples_win

                # append window to hashes
                exp.hashes.append(hashes_win)

                # write restart files
                if calc.misc['rst']:
                    tools.write_file(exp.order+1, np.asarray(exp.n_tasks[-1]), 'mbe_n_tasks')
                    tools.write_file(exp.order, np.asarray(exp.time['screen'][-1]), 'mbe_time_screen')

                # print screen end
                print(output.screen_end(exp.order, exp.time['screen'][-1]))


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

                # main mbe function
                inc_win = mbe.slave(mpi, mol, calc, exp)

                # append window to increments
                if len(exp.prop[calc.target_mbe]['inc']) < len(exp.hashes):
                    exp.prop[calc.target_mbe]['inc'].append(inc_win)
                else:
                    exp.prop[calc.target_mbe]['inc'][-1] = inc_win

            elif msg['task'] == 'screen':

                # receive order
                exp.order = msg['order']

                # main screening function
                hashes_win, tuples_win, n_tasks = screen.slave(mpi, calc, exp, msg['slaves_needed'])

                # overwrite window to tuples
                exp.tuples = tuples_win

                # append window to hashes
                exp.hashes.append(hashes_win)

                # append n_tasks
                exp.n_tasks.append(n_tasks)

            elif msg['task'] == 'exit':

                slave = False

        # finalize mpi
        parallel.finalize(mpi)

