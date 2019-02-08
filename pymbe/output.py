#!/usr/bin/env python
# -*- coding: utf-8 -*

""" output.py: print module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.20'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
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
		""" print main header """
		string = "\n\n   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo\n"
		string += "   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8\n"
		string += "    888   .d88' oooo    ooo  888b     d'888   888     888  888\n"
		string += "    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8\n"
		string += "    888           `88..8'    8  `888'   888   888    `88b  888    \"\n"
		string += "    888            `888'     8    Y     888   888    .88P  888       o\n"
		string += "   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8\n"
		string += "                .o..P'\n"
		string += "                `Y8P'\n\n\n"
		string += "   -- git version: {:}\n\n\n"
		form = (tools.git_version(),)
		return string.format(*form)


def exp_header(calc, exp):
		""" print expansion header """
		# set string
		string = HEADER+'\n'
		string += '{0:^87}\n'
		string += HEADER+'\n\n'
		form = (calc.model['method'].upper()+' expansion',)
		return string.format(*form)


def mbe_header(exp):
		""" print mbe header """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} MBE started  ---  {1:d} tuples in total\n'
		string += DIVIDER
		form = (exp.order, exp.tuples[exp.order-1].shape[0])
		return string.format(*form)


def mbe_debug(mol, calc, exp, tup, inc_tup):
		""" print mbe debug information """
		tup_lst = [i for i in tup]
		tup_sym = [symm.addons.irrep_id2name(mol.symmetry, i) for i in calc.orbsym[tup]]
		string = ' INC: order = {:} , tup = {:}\n'
		string += '      symmetry = {:}\n'
		form = (exp.order, tup_lst, tup_sym)
		if calc.target in ['energy', 'excitation']:
			string += '      increment for root {:} = {:.4e}\n'
			form += (calc.state['root'], inc_tup,)
		else:
			string += '      increment for root {:} = ({:.4e}, {:.4e}, {:.4e})\n'
			form += (calc.state['root'], *inc_tup,)
		return string.format(*form)


def mbe_end(calc, exp):
		""" print end of mbe """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} MBE done  ---  {1:d} tuples in total\n'
		string += DIVIDER
		form = (exp.order, exp.count[exp.order-1])
		return string.format(*form)


def mbe_results(mol, calc, exp):
		""" print mbe result statistics """
		if calc.target in ['energy', 'excitation']:
			string = FILL+'\n'
			prop_tot = exp.prop[calc.target]['tot']
			prop_inc = exp.prop[calc.target]['inc'][exp.order-1]
			# statistics
			if prop_inc.any():
				mean_val = np.mean(prop_inc[np.nonzero(prop_inc)])
				min_val = np.min(np.abs(prop_inc[np.nonzero(prop_inc)]))
				max_val = np.max(np.abs(prop_inc[np.nonzero(prop_inc)]))
			else:
				mean_val = min_val = max_val = 0.0
			# calculate total inc
			if exp.order == 1:
				tot_inc = prop_tot[exp.order-1]
			else:
				tot_inc = prop_tot[exp.order-1] - prop_tot[exp.order-2]
			# set header
			if calc.target == 'energy':
				header = 'energy for root {:} (total increment = {:.4e})'.format(calc.state['root'], tot_inc)
			else:
				header = 'excitation energy for root {:} (total increment = {:.4e})'.format(calc.state['root'], tot_inc)
			# set string
			string += DIVIDER+'\n'
			string += ' RESULT:{:^81}\n'
			string += DIVIDER+'\n'
			string += DIVIDER+'\n'
			string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
			string += DIVIDER+'\n'
			string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'
			string += DIVIDER+'\n'
			form = (header, mean_val, min_val, max_val)
		else:
			string = FILL+'\n'
			prop_tot = exp.prop[calc.target]['tot']
			# calculate total inc
			if exp.order == 1:
				tot_inc = np.linalg.norm(prop_tot[exp.order-1])
			else:
				tot_inc = np.linalg.norm(prop_tot[exp.order-1]) - np.linalg.norm(prop_tot[exp.order-2])
			# set header
			if calc.target == 'dipole':
				header = 'dipole moment for root {:} (total increment = {:.4e})'.format(calc.state['root'], tot_inc)
			else:
				header = 'transition dipole for excitation 0 -> {:} (total increment = {:.4e})'.format(calc.state['root'], tot_inc)
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
				prop_inc = exp.prop[calc.target]['inc'][exp.order-1][:, k]
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
				string += ' RESULT:     {:>13.4e}       |        {:>13.4e}         |       {:>13.4e}\n'
				string += DIVIDER+'\n'
				form += (comp[k], mean_val[k], min_val[k], max_val[k],)
		if exp.order < exp.max_order:
			string += FILL
		else:
			string += '\n\n'
		return string.format(*form)


def screen_header(exp, thres):
		""" print screening header """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} screening started (thres. = {1:5.2e})\n'
		string += DIVIDER
		form = (exp.order, thres)
		return string.format(*form)


def screen_end(exp):
		""" print end of screening """
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} screening done\n'
		if exp.tuples[-1].shape[0] == 0:
			string += ' STATUS:                  *** convergence has been reached ***                         \n'
		string += DIVIDER+'\n\n'
		form = (exp.order,)
		return string.format(*form)
		
	
