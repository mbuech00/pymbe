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
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(HEADER)
				print('{0:^87}'.format(calc.typ+' expansion'))
				print(HEADER+'\n\n')
		# write also to stdout
		print('\n\n'+HEADER)
		print('{0:^87}'.format(calc.typ+' expansion'))
		print(HEADER+'\n\n')


def mbe_header(exp):
		""" print mbe header """
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS:  order k = {0:>d} MBE started  ---  {1:d} tuples in total'.\
						format(exp.order,len(exp.tuples[exp.order-exp.start_order])))
				print(' --------------------------------------------------------------------------------------------')
		# write also to stdout
		print(' --------------------------------------------------------------------------------------------')
		print(' STATUS:  order k = {0:>d} MBE started  ---  {1:d} tuples in total'.\
				format(exp.order,len(exp.tuples[exp.order-exp.start_order])))
		print(' --------------------------------------------------------------------------------------------')


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
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS:  order k = {0:>d} MBE done (E = {1:.6e})'.\
						format(exp.order,np.sum(exp.energy['inc'][exp.order-exp.start_order])))
				print(' --------------------------------------------------------------------------------------------')
		# write also to stdout
		print(' --------------------------------------------------------------------------------------------')
		print(' STATUS:  order k = {0:>d} MBE done (E = {1:.6e})'.\
				format(exp.order,np.sum(exp.energy['inc'][exp.order-exp.start_order])))
		print(' --------------------------------------------------------------------------------------------')


def mbe_results(mol, calc, exp):
		""" print mbe result statistics """
		# statistics
		mean_val = np.mean(exp.energy['inc'][exp.order-exp.start_order])
		min_idx = np.argmin(np.abs(exp.energy['inc'][exp.order-exp.start_order]))
		min_val = exp.energy['inc'][exp.order-exp.start_order][min_idx]
		max_idx = np.argmax(np.abs(exp.energy['inc'][exp.order-exp.start_order]))
		max_val = exp.energy['inc'][exp.order-exp.start_order][max_idx]
		# core and cas regions
		core, cas = kernel.core_cas(mol, exp, exp.tuples[exp.order-exp.start_order][max_idx])
		cas_ref = '{0:}'.format(sorted(list(set(calc.ref_space.tolist()) - set(core))))
		if calc.ref['METHOD'] == 'HF':
			cas_exp = '{0:}'.format(sorted(list(set(cas) - set(calc.ref_space.tolist()))))
		else:
			cas_exp = '{0:}'.format(sorted(exp.tuples[0][0].tolist()))
			cas_exp += ' + {0:}'.format(sorted(list(set(cas) - set(exp.tuples[0][0].tolist()) - set(calc.ref_space.tolist()))))
		# now print
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(' --------------------------------------------------------------------------------------------')
				print(' RESULT:      mean increment     |    min. abs. increment   |    max. abs. increment')
				print(' --------------------------------------------------------------------------------------------')
				print(' RESULT:     {0:>13.4e}       |      {1:>13.4e}       |      {2:>13.4e}'.\
						format(mean_val, min_val, max_val))
				# debug print
				if mol.verbose:
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT:                   info on max. abs. increment:')
					print(' RESULT:  core = {0:}'.format(core))
					print(' RESULT:  cas  = '+cas_ref+' + '+cas_exp)
				print(' --------------------------------------------------------------------------------------------')
		# write also to stdout
		print(' --------------------------------------------------------------------------------------------')
		print(' RESULT:      mean increment     |    min. abs. increment   |    max. abs. increment')
		print(' --------------------------------------------------------------------------------------------')
		print(' RESULT:     {0:>13.4e}       |      {1:>13.4e}       |      {2:>13.4e}'.\
				format(mean_val, min_val, max_val))
		# debug print
		if mol.verbose:
			print(' --------------------------------------------------------------------------------------------')
			print(' RESULT:                   info on max. abs. increment:')
			print(' RESULT:  core = {0:}'.format(core))
			print(' RESULT:  cas  = '+cas_ref+' + '+cas_exp)
		print(' --------------------------------------------------------------------------------------------')


def screen_header(exp, thres):
		""" print screening header """
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS:  order k = {0:>d} screening started (thres. = {1:5.2e})'.format(exp.order, thres))
				print(' --------------------------------------------------------------------------------------------')
		# write also to stdout
		print(' --------------------------------------------------------------------------------------------')
		print(' STATUS:  order k = {0:>d} screening started (thres. = {1:5.2e})'.format(exp.order, thres))
		print(' --------------------------------------------------------------------------------------------')


def screen_end(exp):
		""" print end of screening """
		with open(OUT+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				if exp.conv_orb[-1]:
					print(' --------------------------------------------------------------------------------------------')
					print(' STATUS:  order k = {0:>d} screening done'.format(exp.order))
					print(' STATUS:                  *** convergence has been reached ***                         ')
					print(' --------------------------------------------------------------------------------------------\n\n')
				else:
					print(' --------------------------------------------------------------------------------------------')
					print(' STATUS:  order k = {0:>d} screening done'.format(exp.order))
					print(' --------------------------------------------------------------------------------------------\n\n')
		# write also to stdout
		if exp.conv_orb[-1]:
			print(' --------------------------------------------------------------------------------------------')
			print(' STATUS:  order k = {0:>d} screening done'.format(exp.order))
			print(' STATUS:                  *** convergence has been reached ***                         ')
			print(' --------------------------------------------------------------------------------------------\n\n')
		else:
			print(' --------------------------------------------------------------------------------------------')
			print(' STATUS:  order k = {0:>d} screening done'.format(exp.order))
			print(' --------------------------------------------------------------------------------------------\n\n')
		
		