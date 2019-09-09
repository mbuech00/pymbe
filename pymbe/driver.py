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

import restart
import mbe
import kernel
import output
import screen
import expansion
import tools
import parallel


def master(mpi, mol, calc, exp):
        """
        this function is the main master function

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        """
        # print expansion headers
        print(output.main_header(method=calc.model['method']))

        # print output from restarted calculation
        if calc.restart:
            for i in range(exp.start_order - exp.min_order):

                # print mbe header
                print(output.mbe_header(exp.hashes[i].size, i + exp.min_order))

                # print mbe end
                print(output.mbe_end(exp.prop[calc.target]['inc'][i], i + exp.min_order, exp.time['mbe'][i]))

                # print mbe results
                print(output.mbe_results(calc.occup, calc.ref_space, calc.target, calc.state['root'], exp.min_order, \
                                            exp.max_order, i + exp.min_order, exp.tuples, exp.prop[calc.target]['inc'][i], \
                                            exp.prop[calc.target]['tot'], exp.mean_ndets[i], \
                                            exp.min_ndets[i], exp.max_ndets[i]))

                # print header
                print(output.screen_header(i + exp.min_order))

                # print screen end
                print(output.screen_end(exp.tuples.shape[0], i + exp.min_order, exp.time['screen'][i]))

        # begin or resume mbe expansion depending
        for exp.order in range(exp.start_order, exp.max_order+1):

            if len(exp.hashes) > len(exp.prop[calc.target]['tot']):

                # init mbe time
                exp.time['mbe'].append(0.0)

                # print mbe header
                print(output.mbe_header(exp.tuples.shape[0], exp.order))

                # start time
                time = MPI.Wtime()

                # main mbe function
                inc, mean_ndets, min_ndets, max_ndets = mbe.master(mpi, mol, calc, exp)

                # append number of determinants and increments
                if len(exp.prop[calc.target]['inc']) > len(exp.prop[calc.target]['tot']):
                    exp.prop[calc.target]['inc'][-1] = inc
                else:
                    exp.prop[calc.target]['inc'].append(inc)
                exp.mean_ndets.append(mean_ndets)
                exp.min_ndets.append(min_ndets)
                exp.max_ndets.append(max_ndets)

                # calculate and append total property
                buf = exp.prop[calc.target]['inc'][-1].Shared_query(0)[0]
                inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(exp.hashes[-1].size,))
                exp.prop[calc.target]['tot'].append(tools.fsum(inc))
                if exp.order > exp.min_order:
                    exp.prop[calc.target]['tot'][-1] += exp.prop[calc.target]['tot'][-2]

                # collect time
                exp.time['mbe'][-1] = MPI.Wtime() - time

                # write restart files
                if calc.misc['rst']:
                    restart.mbe_write(exp.order, \
                                      exp.prop[calc.target]['inc'][-1], exp.prop[calc.target]['tot'][-1], \
                                      exp.mean_ndets[-1], exp.max_ndets[-1], exp.min_ndets[-1], \
                                      np.asarray(exp.time['mbe'][-1]))

                # print mbe end
                print(output.mbe_end(inc, exp.order, exp.time['mbe'][-1]))

            # print mbe results
            print(output.mbe_results(calc.occup, calc.ref_space, calc.target, calc.state['root'], exp.min_order, \
                                     exp.max_order, exp.order, exp.tuples, inc, \
                                     exp.prop[calc.target]['tot'], exp.mean_ndets[-1], \
                                     exp.min_ndets[-1], exp.max_ndets[-1]))

            # init screening time
            exp.time['screen'].append(0.0)

            if exp.order < exp.max_order:

                # print header
                print(output.screen_header(exp.order))

                # start time
                time = MPI.Wtime()

                # main screening function
                hashes, tuples = screen.master(mpi, calc, exp)

                # overwrite tuples and append hashes
                exp.tuples = tuples
                exp.hashes.append(hashes)

                # collect time
                exp.time['screen'][-1] = MPI.Wtime() - time

                # write restart files
                if calc.misc['rst'] and exp.tuples.shape[0] > 0:
                    restart.screen_write(exp.order, exp.tuples, exp.hashes[-1], \
                                         np.asarray(exp.time['screen'][-1]))

                # print screen end
                print(output.screen_end(exp.tuples.shape[0], exp.order, exp.time['screen'][-1]))

            # convergence check
            if exp.tuples.shape[0] == 0 or exp.order == exp.max_order:

                # final order
                exp.final_order = exp.order

                # timings
                exp.time['mbe'] = np.asarray(exp.time['mbe'])
                exp.time['screen'] = np.asarray(exp.time['screen'])
                exp.time['total'] = exp.time['mbe'] + exp.time['screen']

                # ndets
                exp.mean_ndets = np.asarray(exp.mean_ndets)
                exp.min_ndets = np.asarray(exp.min_ndets)
                exp.max_ndets = np.asarray(exp.max_ndets)

                # final results
                exp.prop[calc.target]['tot'] = np.asarray(exp.prop[calc.target]['tot'])

                break


def slave(mpi, mol, calc, exp):
        """
        this function is the main slave function

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        """
        # set loop/waiting logical
        slave = True

        # enter slave state
        while slave:

            # task id
            msg = mpi.comm.bcast(None, root=0)

            if msg['task'] == 'mbe':

                # receive order
                exp.order = msg['order']

                # main mbe function
                inc = mbe.slave(mpi, mol, calc, exp)

                # append increments
                exp.prop[calc.target]['inc'].append(inc)

            elif msg['task'] == 'screen':

                # receive order
                exp.order = msg['order']

                # main screening function
                hashes = screen.slave(mpi, calc, exp, msg['slaves_needed'])

                # append hashes
                exp.hashes.append(hashes)

            elif msg['task'] == 'exit':

                slave = False

        # finalize mpi
        parallel.finalize(mpi)


