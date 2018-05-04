#!/usr/bin/env python
# -*- coding: utf-8 -*

""" output.py: print module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import os
import os.path
import shutil
import numpy as np
import contextlib

import kernel


# output parameters
OUT = os.getcwd()+'/output'
HEADER = '{0:^87}'.format('-'*45)
DIVIDER = ' '+'-'*92


def main_header():
		""" print main header """
		# rm out if present
		if os.path.isdir(OUT): shutil.rmtree(OUT, ignore_errors=True)
		# mkdir out
		os.mkdir(OUT)
		# print headers
		for i in [OUT+'/output.out',OUT+'/results.out']:
			with open(i,'a') as f:
				with contextlib.redirect_stdout(f):
					print("\n\n   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo")
					print("   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8")
					print("    888   .d88' oooo    ooo  888b     d'888   888     888  888")
					print("    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8")
					print("    888           `88..8'    8  `888'   888   888    `88b  888    \"")
					print("    888            `888'     8    Y     888   888    .88P  888       o")
					print("   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8")
					print("                .o..P'")
					print("                `Y8P'\n\n")


def exp_header(calc, exp):
		""" print expansion header """
		# set string
		string = HEADER+'\n'
		string += '{0:^87}\n'
		string += HEADER+'\n\n'
		form = (calc.typ+' expansion')
		# now print
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(string.format(form))
		# write also to stdout
		print(string.format(form))


def mbe_header(exp):
		""" print mbe header """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} MBE started  ---  {1:d} tuples in total\n'
		string += DIVIDER
		form = (exp.order, len(exp.tuples[exp.order-exp.start_order]))
		# now print
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(string.format(*form))
		# write also to stdout
		print(string.format(*form))


def mbe_status(exp, prog):
		""" print status bar """
		# write only to stdout
		bar_length = 50
		status = ""
		block = int(round(bar_length * prog))
		print(' STATUS:   [{0}]   ---  {1:>6.2f} % {2}'.\
				format('#' * block + '-' * (bar_length - block), prog * 100, status))


def mbe_end(exp):
		""" print end of mbe """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} MBE done (E = {1:.6e})\n'
		string += DIVIDER
		form = (exp.order, np.sum(exp.property['energy']['inc'][exp.order-exp.start_order]))
		# now print
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(string.format(*form))
		# write also to stdout
		print(string.format(*form))


def mbe_results(mol, calc, exp):
		""" print mbe result statistics """
		# statistics
		mean_val = np.mean(exp.property['energy']['inc'][exp.order-exp.start_order])
		min_idx = np.argmin(np.abs(exp.property['energy']['inc'][exp.order-exp.start_order]))
		min_val = exp.property['energy']['inc'][exp.order-exp.start_order][min_idx]
		max_idx = np.argmax(np.abs(exp.property['energy']['inc'][exp.order-exp.start_order]))
		max_val = exp.property['energy']['inc'][exp.order-exp.start_order][max_idx]
		# core and cas regions
		core, cas = kernel.core_cas(mol, exp, exp.tuples[exp.order-exp.start_order][max_idx])
		cas_ref = '{0:}'.format(sorted(list(set(calc.ref_space.tolist()) - set(core))))
		if calc.ref['METHOD'] == 'HF':
			cas_exp = '{0:}'.format(sorted(list(set(cas) - set(calc.ref_space.tolist()))))
		else:
			cas_exp = '{0:}'.format(sorted(exp.tuples[0][0].tolist()))
			cas_exp += ' + {0:}'.format(sorted(list(set(cas) - set(exp.tuples[0][0].tolist()) - set(calc.ref_space.tolist()))))
		# set string
		string = DIVIDER+'\n'
		string += ' RESULT:      mean increment     |      min. abs. increment     |     max. abs. increment\n'
		string += DIVIDER+'\n'
		string += ' RESULT:     {0:>13.4e}       |        {1:>13.4e}         |       {2:>13.4e}\n'
		if mol.verbose:
			string += DIVIDER+'\n'
			string += ' RESULT:                   info on max. abs. increment:\n'
			string += ' RESULT:  core = {3:}\n'
			string += ' RESULT:  cas  = '+cas_ref+' + '+cas_exp+'\n'
		string += DIVIDER
		form = (mean_val, min_val, max_val)
		if mol.verbose:
			form += (core,)
		# now print
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(string.format(*form))
		# write also to stdout
		print(string.format(*form))


def screen_header(exp, thres):
		""" print screening header """
		# set string
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} screening started (thres. = {1:5.2e})\n'
		string += DIVIDER
		form = (exp.order, thres)
		# now print
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(string.format(*form))
		# write also to stdout
		print(string.format(*form))


def screen_end(exp):
		""" print end of screening """
		string = DIVIDER+'\n'
		string += ' STATUS:  order k = {0:>d} screening done\n'
		if exp.conv_orb[-1]:
			string += ' STATUS:                  *** convergence has been reached ***                         \n'
		string += DIVIDER+'\n\n'
		form = (exp.order)
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(string.format(form))
		# write also to stdout
		print(string.format(form))
		
		
