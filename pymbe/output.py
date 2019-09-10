#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
output module containing all print functions in pymbe
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import os
from datetime import datetime
import numpy as np
from pyscf import symm

import kernel
import tools


# output folder and files
OUT = os.getcwd()+'/output'
OUT_FILE = OUT+'/output.out'
# output parameters
HEADER = '{0:^87}'.format('-'*45)
DIVIDER = ' '+'-'*92
FILL = ' '+'|'*92
BAR_LENGTH = 50


def main_header(method=None):
        """
        this function prints the main pymbe header

        :param method: main method. string
        :return: formatted string
        """
        string = "\n\n   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo\n"
        string += "   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8\n"
        string += "    888   .d88' oooo    ooo  888b     d'888   888     888  888\n"
        string += "    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8\n"
        string += "    888           `88..8'    8  `888'   888   888    `88b  888    \"\n"
        string += "    888            `888'     8    Y     888   888    .88P  888       o\n"
        string += "   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8\n"
        string += "                .o..P'\n"
        string += "                `Y8P'\n\n\n"
        string += "   -- date & time: {:s}\n"
        string += "   -- git version: {:s}\n\n\n"

        # date & time
        form = (datetime.now().strftime('%Y-%m-%d & %H:%M:%S'),)

        # git hash
        form += (tools.git_version(),)

        # method
        if method is not None:

            string += HEADER+'\n'
            string += '{:^87s}\n'
            string += HEADER+'\n\n'

            form += (method.upper()+' expansion',)

        return string.format(*form)


def mbe_header(n_tuples, order):
        """
        this function prints the mbe header

        :param n_tuples: number of tuples at a given order. integer
        :param order. expansion order. integer
        :return: formatted string
        """
        # set string
        string = DIVIDER+'\n'
        string += ' STATUS:  order k = {:d} MBE started  ---  {:d} tuples in total\n'
        string += DIVIDER

        form = (order, n_tuples)

        return string.format(*form)


def mbe_debug(atom, symmetry, orbsym, root, ndets_tup, nelec_tup, inc_tup, order, cas_idx, tup):
        """
        this function prints mbe debug information

        :param atom: molecule information (evaluated as False for model Hamiltonian). string
        :param symmetry: molecular point group. string
        :param orbsym: orbital symmetries. numpy array of shape (n_orbs,)
        :param root: state root. integer
        :param ndets_tup: number of determinants in casci calculation on tuple. scalar
        :param nelec_tup: number of alpha- and beta-electrons. tuples of integers
        :param inc_tup: property increment from tuple. scalar
        :param order: expansion order. integer
        :param cas_idx: cas space indices. numpy array of shape (n_cas,)
        :param tup: tuple of orbitals. numpy array of shape (order,)
        :return: formatted string
        """
        # symmetry
        tup_lst = [i for i in tup]

        if atom:
            tup_sym = [symm.addons.irrep_id2name(symmetry, i) for i in orbsym[tup]]
        else:
            tup_sym = ['A'] * tup.size

        string = ' INC: order = {:d} , tup = {:} , space = ({:d}e,{:d}o) , n_dets = {:.2e}\n'
        string += '      symmetry = {:}\n'
        form = (order, tup_lst, nelec_tup[0] + nelec_tup[1], cas_idx.size, ndets_tup, tup_sym)

        if np.isscalar(inc_tup):
            string += '      increment for root {:d} = {:.4e}\n'
            form += (root, inc_tup,)
        else:
            string += '      increment for root {:d} = ({:.4e}, {:.4e}, {:.4e})\n'
            form += (root, *inc_tup,)

        return string.format(*form)


def mbe_status(prog):
        """
        this function prints the status of an mbe phase

        :param prog: progress. scalar (0. <= prog <= 1.)
        :return: formatted string
        """
        status = int(round(BAR_LENGTH * prog))
        remainder = (BAR_LENGTH - status)

        return ' STATUS:   [{:}]   ---  {:>6.2f} %'.\
            format('#' * status + '-' * remainder, prog * 100.)


def mbe_end(order, time):
        """
        this function prints the end mbe information

        :param order. expansion order. integer
        :param time. time in seconds. scalar
        :return: formatted string
        """
        # set string
        string = DIVIDER+'\n'
        string += ' STATUS:  order k = {:d} MBE done in {:s}\n'
        string += DIVIDER

        form = (order, tools.time_str(time),)

        return string.format(*form)


def mbe_results(occup, ref_space, target, root, min_order, max_order, order, tuples, \
                prop_tot, mean_inc, min_inc, max_inc, mean_ndets, min_ndets, max_ndets):
        """
        this function prints mbe results statistics

        :param occup: orbital occupation. numpy array of shape (n_orbs,)
        :param ref_space: reference space. numpy array of shape (n_ref_tot,)
        :param target: calculation target. string
        :param root: state root. integer
        :param min_order: minimum (start) order. integer
        :param max_order: maximum (final) order. integer
        :param order: current order. integer
        :param tuples: current order tuples. numpy array of shape (n_tuples, order)
        :param prop_tot: total mbe energy. list of scalars or numpy arrays of shape (3,) depending on target
        :param mean_inc: mean increment. float
        :param min_inc: min increment. float
        :param max_inc: max increment. float
        :param mean_ndets: mean number of determinants. float
        :param min_ndets: min number of determinants. float
        :param max_ndets: max number of determinants. float
        :return: formatted string
        """
        # calculate total inc
        if target in ['energy', 'excitation']:

            if order == min_order:
                tot_inc = prop_tot[order-min_order]
            else:
                tot_inc = prop_tot[order-min_order] - prop_tot[order-min_order-1]

        elif target in ['dipole', 'trans']:

            if order == min_order:
                tot_inc = np.linalg.norm(prop_tot[order-min_order])
            else:
                tot_inc = np.linalg.norm(prop_tot[order-min_order]) - np.linalg.norm(prop_tot[order-min_order-1])

        # set header
        if target == 'energy':
            header = 'energy for root {:} (total increment = {:.4e})'. \
                        format(root, tot_inc)
        elif target == 'excitation':
            header = 'excitation energy for root {:} (total increment = {:.4e})'. \
                        format(root, tot_inc)
        elif target == 'dipole':
            header = 'dipole moment for root {:} (total increment = {:.4e})'. \
                        format(root, tot_inc)
        elif target == 'trans':
            header = 'transition dipole moment for excitation 0 -> {:} (total increment = {:.4e})'. \
                        format(root, tot_inc)
        # set string
        string = FILL+'\n'
        string += DIVIDER+'\n'
        string += ' RESULT:{:^81}\n'
        string += DIVIDER+'\n'

        if target in ['energy', 'excitation']:

            # set string
            string += DIVIDER+'\n'
            string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
            string += DIVIDER+'\n'
            string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'

            form = (header, mean_inc, min_inc, max_inc)

        elif target in ['dipole', 'trans']:

            # set components
            string += DIVIDER
            form = (header,)
            comp = ('x-component', 'y-component', 'z-component')

            # loop over x, y, and z
            for k in range(3):

                # set string
                string += '\n RESULT:{:^81}\n'
                string += DIVIDER+'\n'
                string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
                string += DIVIDER+'\n'
                string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}'
                if k < 2:
                    string += '\n'+DIVIDER
                form += (comp[k], mean_inc[k], min_inc[k], max_inc[k],)

        # set string
        string += DIVIDER+'\n'
        string += DIVIDER+'\n'
        string += ' RESULT:   mean # determinants   |      min. # determinants     |     max. # determinants\n'
        string += DIVIDER+'\n'
        string += ' RESULT:        {:>9.3e}        |           {:>9.3e}          |          {:>9.3e}\n'
        string += DIVIDER+'\n'
        form += (mean_ndets, min_ndets, max_ndets)

        if order < max_order:
            string += FILL
        else:
            string += '\n\n'

        return string.format(*form)


def screen_header(order):
        """
        this function prints the screening header

        :param order. expansion order. integer
        :return: formatted string
        """
        # set string
        string = DIVIDER+'\n'
        string += ' STATUS:  order k = {:d} screening started\n'
        string += DIVIDER

        form = (order,)

        return string.format(*form)


def screen_end(order, time, conv=False):
        """
        this function prints the end screening information

        :param order. expansion order. integer
        :param time. time in seconds. scalar
        :return: formatted string
        """
        string = DIVIDER+'\n'
        string += ' STATUS:  order k = {:d} screening done in {:s}\n'

        if conv:
            string += ' STATUS:                  *** convergence has been reached ***                         \n'

        string += DIVIDER+'\n\n'

        form = (order, tools.time_str(time),)

        return string.format(*form)


