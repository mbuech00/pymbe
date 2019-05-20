#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
output module containing all print functions in pymbe
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.6'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import symm

import kernel
import tools


# output parameters
HEADER = '{0:^87}'.format('-'*45)
DIVIDER = ' '+'-'*92
FILL = ' '+'|'*92


def main_header():
		"""
		this function prints the main pymbe header

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
		string += "   -- git version: {:s}\n\n\n"

		form = (tools.git_version(),)

		return string.format(*form)


def exp_header(method):
		"""
		this function prints the expansion header

		:param method: main method. string
		:return: formatted string
		"""
		# set string
		string = HEADER+'\n'
		string += '{:^87s}\n'
		string += HEADER+'\n\n'

		form = (method.upper()+' expansion',)

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

		if inc_tup.ndim == 1:
			string += '      increment for root {:d} = {:.4e}\n'
			form += (root, inc_tup,)
		else:
			string += '      increment for root {:d} = ({:.4e}, {:.4e}, {:.4e})\n'
			form += (root, *inc_tup,)

		return string.format(*form)


def mbe_end(n_tuples, order, time):
		"""
		this function prints the end mbe information

		:param n_tuples: number of tuples at a given order. integer
		:param order. expansion order. integer
		:param time. time in seconds. scalar
		:return: formatted string
		"""
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {:d} MBE done  ---  {:d} tuples in total in {:s}\n'
		string += DIVIDER

		form = (order, n_tuples, tools.time_str(time),)

		return string.format(*form)


def mbe_results(target, prop_inc, prop_tot, ):
		""" print mbe result statistics """
		if calc.target in ['energy', 'excitation']:
			string = FILL+'\n'
			prop_tot = exp.prop[calc.target]['tot']
			prop_inc = exp.prop[calc.target]['inc'][-1]
			# statistics
			if prop_inc.any():
				mean_val = np.mean(prop_inc[np.nonzero(prop_inc)])
				min_val = np.min(np.abs(prop_inc[np.nonzero(prop_inc)]))
				max_val = np.max(np.abs(prop_inc[np.nonzero(prop_inc)]))
			else:
				mean_val = min_val = max_val = 0.0
			# calculate total inc
			if exp.order == exp.min_order:
				tot_inc = prop_tot[-1]
			else:
				tot_inc = prop_tot[-1] - prop_tot[-2]
			# set header
			if calc.target == 'energy':
				header = 'energy for root {:} (total increment = {:.4e})'. \
							format(calc.state['root'], tot_inc)
			else:
				header = 'excitation energy for root {:} (total increment = {:.4e})'. \
							format(calc.state['root'], tot_inc)
			# set string
			string += DIVIDER+'\n'
			string += ' RESULT:{:^81}\n'
			string += DIVIDER+'\n'
			string += DIVIDER+'\n'
			string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
			string += DIVIDER+'\n'
			string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'
			form = (header, mean_val, min_val, max_val)
			# statistics
			ndets = exp.ndets[-1]
			if ndets.any():
				mean_ndets = np.mean(ndets[np.nonzero(ndets)])
				min_ndets = np.min(ndets[np.nonzero(ndets)])
				max_ndets = np.max(ndets[np.nonzero(ndets)])
			else:
				mean_ndets = min_ndets = max_ndets = 0.0
			string += DIVIDER+'\n'
			string += DIVIDER+'\n'
			string += ' RESULT:   mean # determinants   |      min. # determinants     |     max. # determinants\n'
			string += DIVIDER+'\n'
			string += ' RESULT:        {:>9.3e}        |           {:>9.3e}          |          {:>9.3e}\n'
			cas_idx_max = tools.core_cas(mol.nocc, calc.ref_space, exp.tuples[-1][np.argmax(ndets)])[1]
			nelec_max = np.asarray((np.count_nonzero(calc.occup[cas_idx_max] > 0.), \
									np.count_nonzero(calc.occup[cas_idx_max] > 1.)), dtype=np.int32)
			string += ' RESULT:        ---------        |           ---------          |      {:>2.0f} el. in {:>2.0f} orb.\n'
			string += DIVIDER+'\n'
			form += (mean_ndets, min_ndets, max_ndets, \
						nelec_max[0] + nelec_max[1], cas_idx_max.size)
		else:
			string = FILL+'\n'
			prop_tot = exp.prop[calc.target]['tot']
			# calculate total inc
			if exp.order == exp.min_order:
				tot_inc = np.linalg.norm(prop_tot[-1])
			else:
				tot_inc = np.linalg.norm(prop_tot[-1]) - np.linalg.norm(prop_tot[-2])
			# set header
			if calc.target == 'dipole':
				header = 'dipole moment for root {:} (total increment = {:.4e})'. \
							format(calc.state['root'], tot_inc)
			else:
				header = 'transition dipole moment for excitation 0 -> {:} (total increment = {:.4e})'. \
							format(calc.state['root'], tot_inc)
			# set string/form
			string += DIVIDER+'\n'
			string += ' RESULT:{:^81}\n'
			string += DIVIDER+'\n'
			string += DIVIDER
			form = (header,)
			# set components
			comp = ('x-component', 'y-component', 'z-component')
			# init result arrays
			mean_val = np.empty(3, dtype=np.float64)
			min_val = np.empty(3, dtype=np.float64)
			max_val = np.empty(3, dtype=np.float64)
			# loop over x, y, and z
			for k in range(3):
				prop_inc = exp.prop[calc.target]['inc'][-1][:, k]
				# statistics
				if prop_inc.any():
					mean_val[k] = np.mean(prop_inc[np.nonzero(prop_inc)])
					min_val[k] = np.min(np.abs(prop_inc[np.nonzero(prop_inc)]))
					max_val[k] = np.max(np.abs(prop_inc[np.nonzero(prop_inc)]))
				else:
					mean_val[k] = min_val[k] = max_val[k] = 0.0
				string += '\n RESULT:{:^81}\n'
				string += DIVIDER+'\n'
				string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
				string += DIVIDER+'\n'
				string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}'
				if k < 2:
					string += '\n'+DIVIDER
				form += (comp[k], mean_val[k], min_val[k], max_val[k],)
			# statistics
			ndets = exp.ndets[-1]
			if ndets.any():
				mean_ndets = np.mean(ndets[np.nonzero(ndets)])
				min_ndets = np.min(ndets[np.nonzero(ndets)])
				max_ndets = np.max(ndets[np.nonzero(ndets)])
			else:
				mean_ndets = min_ndets = max_ndets = 0.0
			string += '\n'+DIVIDER+'\n'
			string += DIVIDER+'\n'
			string += ' RESULT:   mean # determinants   |      min. # determinants     |     max. # determinants\n'
			string += DIVIDER+'\n'
			string += ' RESULT:        {:>9.3e}        |           {:>9.3e}          |          {:>9.3e}\n'
			cas_idx_max = tools.core_cas(mol.nocc, calc.ref_space, exp.tuples[-1][np.argmax(ndets)])[1]
			nelec_max = np.asarray((np.count_nonzero(calc.occup[cas_idx_max] > 0.), \
									np.count_nonzero(calc.occup[cas_idx_max] > 1.)), dtype=np.int32)
			string += ' RESULT:        ---------        |           ---------          |      {:>2.0f} el. in {:>2.0f} orb.\n'
			string += DIVIDER+'\n'
			form += (mean_ndets, min_ndets, max_ndets, \
						nelec_max[0] + nelec_max[1], cas_idx_max.size)
		if exp.order < exp.max_order:
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


def screen_end(n_tuples, order, time):
		"""
		this function prints the end screening information

		:param n_tuples: number of tuples at a given order. integer
		:param order. expansion order. integer
		:param time. time in seconds. scalar
		:return: formatted string
		"""
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {:d} screening done in {:s}\n'

		if n_tuples == 0:
			string += ' STATUS:                  *** convergence has been reached ***                         \n'

		string += DIVIDER+'\n\n'

		form = (order, tools.time_str(time),)

		return string.format(*form)
		
	
