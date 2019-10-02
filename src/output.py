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
from typing import List, Tuple, Union, Any

import parallel
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


def main_header(mpi: Union[None, parallel.MPICls] = None, method: Union[None, str] = None) -> str:
        """
        this function prints the main pymbe header
        """
        string : str = "\n\n   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo\n"
        string += "   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8\n"
        string += "    888   .d88' oooo    ooo  888b     d'888   888     888  888\n"
        string += "    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8\n"
        string += "    888           `88..8'    8  `888'   888   888    `88b  888    \"\n"
        string += "    888            `888'     8    Y     888   888    .88P  888       o\n"
        string += "   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8\n"
        string += "                .o..P'\n"
        string += "                `Y8P'\n\n\n"
        # date & time
        string += "   -- date & time   : {:s}\n"
        form: Tuple[Any, ...] = (datetime.now().strftime('%Y-%m-%d & %H:%M:%S'),)
        # git hash
        string += "   -- git version   : {:s}\n"
        form += (tools.git_version(),)
        if mpi is not None:
            string += "   -- local masters :\n"
            for master_idx in range(mpi.num_masters):
                string += "   #### rank / node : {:>6d} / {:s}\n"
                form += (mpi.master_global_ranks[master_idx], mpi.master_global_hosts[master_idx])
        string += "\n\n"

        # method
        if method is not None:

            string += HEADER+'\n'
            string += '{:^87s}\n'
            string += HEADER+'\n\n'

            form += (method.upper()+' expansion',)

        return string.format(*form)


def mbe_header(n_tuples: int, order: int) -> str:
        """
        this function prints the mbe header
        """
        # set string
        string: str = DIVIDER+'\n'
        string += ' STATUS:  order k = {:d} MBE started  ---  {:d} tuples in total\n'
        string += DIVIDER

        form: Tuple[int, int] = (order, n_tuples)

        return string.format(*form)


def mbe_debug(atom: Union[List[str], str], symmetry: str, orbsym: np.ndarray, root: int, \
              ndets_tup: int, nelec_tup: Tuple[int, int], inc_tup: Union[float, np.ndarray], \
              order: int, cas_idx: np.ndarray, tup: np.ndarray) -> str:
        """
        this function prints mbe debug information
        """
        # symmetry
        tup_lst = [i for i in tup]

        if atom:
            tup_sym = [symm.addons.irrep_id2name(symmetry, i) for i in orbsym[tup]]
        else:
            tup_sym = ['A'] * tup.size

        string: str = ' INC: order = {:d} , tup = {:} , space = ({:d}e,{:d}o) , n_dets = {:.2e}\n'
        string += '      symmetry = {:}\n'
        form: Tuple[Any, ...] = (order, tup_lst, nelec_tup[0] + nelec_tup[1], cas_idx.size, ndets_tup, tup_sym)

        if np.isscalar(inc_tup):
            string += '      increment for root {:d} = {:.4e}\n'
            form += (root, inc_tup,)
        else:
            string += '      increment for root {:d} = ({:.4e}, {:.4e}, {:.4e})\n'
            form += (root, *inc_tup,) # type: ignore

        return string.format(*form)


def mbe_status(prog: float) -> str:
        """
        this function prints the status of an mbe phase
        """
        status: int = int(round(BAR_LENGTH * prog))
        remainder: int = (BAR_LENGTH - status)

        return ' STATUS:   [{:}]   ---  {:>6.2f} %'.\
            format('#' * status + '-' * remainder, prog * 100.)


def mbe_end(order: int, time: float) -> str:
        """
        this function prints the end mbe information
        """
        # set string
        string: str = DIVIDER+'\n'
        string += ' STATUS:  order k = {:d} MBE done in {:s}\n'
        string += DIVIDER

        form: Tuple[int, str] = (order, tools.time_str(time),)

        return string.format(*form)


def mbe_results(occup: np.ndarray, ref_space: np.ndarray, target: str, root: int, \
                min_order: int, max_order: int, order: int, prop_tot: List[Union[float, np.ndarray]], \
                mean_inc: Union[float, np.ndarray], min_inc: Union[float, np.ndarray], \
                max_inc: Union[float, np.ndarray], mean_ndets: float, min_ndets: float, max_ndets: float) -> str:
        """
        this function prints mbe results statistics
        """
        # calculate total inc
        tot_inc: float = 0.
        if target in ['energy', 'excitation']:

            if order == min_order:
                tot_inc += prop_tot[order-min_order]
            else:
                tot_inc += prop_tot[order-min_order] - prop_tot[order-min_order-1]

        elif target in ['dipole', 'trans']:

            if order == min_order:
                tot_inc += np.linalg.norm(prop_tot[order-min_order])
            else:
                tot_inc += np.linalg.norm(prop_tot[order-min_order]) - np.linalg.norm(prop_tot[order-min_order-1])

        # set header
        header: str = ''
        if target == 'energy':
            header += 'energy for root {:} (total increment = {:.4e})'. \
                        format(root, tot_inc)
        elif target == 'excitation':
            header += 'excitation energy for root {:} (total increment = {:.4e})'. \
                        format(root, tot_inc)
        elif target == 'dipole':
            header += 'dipole moment for root {:} (total increment = {:.4e})'. \
                        format(root, tot_inc)
        elif target == 'trans':
            header += 'transition dipole moment for excitation 0 -> {:} (total increment = {:.4e})'. \
                        format(root, tot_inc)
        # set string
        string: str = FILL+'\n'
        string += DIVIDER+'\n'
        string += ' RESULT:{:^81}\n'
        string += DIVIDER+'\n'

        if target in ['energy', 'excitation']:

            # set string
            string += DIVIDER+'\n'
            string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
            string += DIVIDER+'\n'
            string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'

            form: Tuple[Any, ...] = (header, mean_inc, min_inc, max_inc)

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
                string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'
                if k < 2:
                    string += DIVIDER
                form += (comp[k], mean_inc[k], min_inc[k], max_inc[k],) # type: ignore

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


def screen_header(order: int) -> str:
        """
        this function prints the screening header
        """
        # set string
        string: str = DIVIDER+'\n'
        string += ' STATUS:  order k = {:d} screening started\n'
        string += DIVIDER

        form: Tuple[int] = (order,)

        return string.format(*form)


def screen_end(order: int, time: float, conv: bool = False) -> str:
        """
        this function prints the end screening information
        """
        string: str = DIVIDER+'\n'
        string += ' STATUS:  order k = {:d} screening done in {:s}\n'

        if conv:
            string += ' STATUS:                  *** convergence has been reached ***                         \n'

        string += DIVIDER+'\n\n'

        form: Tuple[int, str] = (order, tools.time_str(time),)

        return string.format(*form)


