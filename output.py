#!/usr/bin/env python
# -*- coding: utf-8 -*

""" output.py: print module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
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
_out = os.getcwd()+'/output'
_header_str = '{0:^93}'.format('-'*45)


def main_header():
		""" print main header """
		# rm out if present
		if (os.path.isdir(_out)): shutil.rmtree(_out, ignore_errors=True)
		# mkdir out
		os.mkdir(_out)
		# print headers
		for i in [_out+'/output.out',_out+'/results.out']:
			with open(i,'a') as f:
				with contextlib.redirect_stdout(f):
					print('')
					print('')
					print("   ooooooooo.               ooo        ooooo oooooooooo.  oooooooooooo")
					print("   `888   `Y88.             `88.       .888' `888'   `Y8b `888'     `8")
					print("    888   .d88' oooo    ooo  888b     d'888   888     888  888")
					print("    888ooo88P'   `88.  .8'   8 Y88. .P  888   888oooo888'  888oooo8")
					print("    888           `88..8'    8  `888'   888   888    `88b  888    \"")
					print("    888            `888'     8    Y     888   888    .88P  888       o")
					print("   o888o            .8'     o8o        o888o o888bood8P'  o888ooooood8")
					print("                .o..P'")
					print("                `Y8P'")
					print('')
					print('')
					print('   --- an incremental Python-based electronic structure correlation program written by:')
					print('')
					print('           Janus Juul Eriksen')
					print('')
					print('       with contributions from:')
					print('')
					print('            Filippo Lipparini')
					print('              & Juergen Gauss')
					print('')
					print('                                            *****')
					print('                                       ***************')
					print('                                            *****')


def exp_header(calc, exp):
		""" print expansion header """
		with open(_out+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print('\n\n'+_header_str)
				print('{0:^93}'.format(calc.exp_type+' expansion'))
				print(_header_str+'\n\n')
		# write also to stdout
		print('\n\n'+_header_str)
		print('{0:^93}'.format(calc.exp_type+' expansion'))
		print(_header_str+'\n\n')


def mbe_header(calc, exp):
		""" print mbe header """
		with open(_out+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} MBE started  ---  {1:d} tuples in total'.\
						format(exp.order,len(exp.tuples[exp.order-exp.start_order])))
				print(' --------------------------------------------------------------------------------------------')
		# write also to stdout
		print(' --------------------------------------------------------------------------------------------')
		print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} MBE started  ---  {1:d} tuples in total'.\
				format(exp.order,len(exp.tuples[exp.order-exp.start_order])))
		print(' --------------------------------------------------------------------------------------------')


def mbe_status(calc, exp, prog):
		""" print status bar """
		# write only to stdout
		bar_length = 50
		status = ""
		block = int(round(bar_length * prog))
		print(' STATUS-'+exp.level.upper()+':   [{0}]   ---  {1:>6.2f} % {2}'.\
				format('#' * block + '-' * (bar_length - block), prog * 100, status))


def mbe_end(calc, exp):
		""" print end of mbe """
		with open(_out+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} MBE done (E = {1:.6e}, thres. = {2:<5.2e})'.\
						format(exp.order,np.sum(exp.energy['inc'][exp.order-exp.start_order]),exp.thres))
				print(' --------------------------------------------------------------------------------------------')
		# write also to stdout
		print(' --------------------------------------------------------------------------------------------')
		print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} MBE done (E = {1:.6e}, thres. = {2:<5.2e})'.\
				format(exp.order,np.sum(exp.energy['inc'][exp.order-exp.start_order]),exp.thres))
		print(' --------------------------------------------------------------------------------------------')


def mbe_microresults(calc, exp):	
		""" print micro result statistics """
		if ((calc.exp_type == 'combined') and (exp.level == 'macro')):
			# statistics
			mean_val = np.mean(exp.micro_conv[exp.order-exp.start_order])
			min_val = exp.micro_conv[exp.order-exp.start_order][np.argmin(exp.micro_conv[exp.order-exp.start_order])]
			max_val = exp.micro_conv[exp.order-exp.start_order][np.argmax(exp.micro_conv[exp.order-exp.start_order])]
			if (len(exp.micro_conv[exp.order-exp.start_order]) > 1):
				std_val = np.std(exp.micro_conv[exp.order-exp.start_order], ddof=1)
			else:
				std_val = 0.0
			# now print
			with open(_out+'/output.out','a') as f:
				with contextlib.redirect_stdout(f):
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT-MICRO:     mean order    |      min. order     |      max. order     |    std.dev.   ')
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT-MICRO:   {0:>8.1f}        |    {1:>8d}         |    {2:>8d}         |   {3:<13.4e}'.\
							format(mean_val, min_val, max_val, std_val))
					print(' --------------------------------------------------------------------------------------------')
			# write also to stdout
			print(' --------------------------------------------------------------------------------------------')
			print(' --------------------------------------------------------------------------------------------')
			print(' RESULT-MICRO:     mean order    |      min. order     |      max. order     |    std.dev.   ')
			print(' --------------------------------------------------------------------------------------------')
			print(' RESULT-MICRO:   {0:>8.1f}        |    {1:>8d}         |    {2:>8d}         |   {3:<13.4e}'.\
					format(mean_val, min_val, max_val, std_val))
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
		if (calc.exp_ref['METHOD'] == 'HF'):
			casexp = '{0:}'.format(sorted(list(set(cas) - set(calc.ref_space.tolist()))))
		else:
			casexp = '{0:}'.format(sorted(exp.tuples[0][0].tolist()))
			casexp += ' + {0:}'.format(sorted(list(set(cas) - set(exp.tuples[0][0].tolist()) - set(calc.ref_space.tolist()))))
		# now print
		with open(_out+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(' --------------------------------------------------------------------------------------------')
				print(' RESULT-'+exp.level.upper()+':      mean increment     |    min. abs. increment   |    max. abs. increment')
				print(' --------------------------------------------------------------------------------------------')
				print(' RESULT-'+exp.level.upper()+':     {0:>13.4e}       |      {1:>13.4e}       |      {2:>13.4e}'.\
						format(mean_val, min_val, max_val))
				# debug print
				if (mol.verbose):
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT-'+exp.level.upper()+':                   info on max. abs. increment:')
					print(' RESULT-'+exp.level.upper()+':  core = {0:}'.format(core))
					print(' RESULT-'+exp.level.upper()+':  cas  = '+cas_ref+' + '+casexp)
				print(' --------------------------------------------------------------------------------------------')
		# write also to stdout
		print(' --------------------------------------------------------------------------------------------')
		print(' RESULT-'+exp.level.upper()+':      mean increment     |    min. abs. increment   |    max. abs. increment')
		print(' --------------------------------------------------------------------------------------------')
		print(' RESULT-'+exp.level.upper()+':     {0:>13.4e}       |      {1:>13.4e}       |      {2:>13.4e}'.\
				format(mean_val, min_val, max_val))
		# debug print
		if (mol.verbose):
			print(' --------------------------------------------------------------------------------------------')
			print(' RESULT-'+exp.level.upper()+':                   info on max. abs. increment:')
			print(' RESULT-'+exp.level.upper()+':  core = {0:}'.format(core))
			print(' RESULT-'+exp.level.upper()+':  cas  = '+cas_ref+' + '+casexp)
		print(' --------------------------------------------------------------------------------------------')


def screen_header(calc, exp):
		""" print screening header """
		with open(_out+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} screening started'.format(exp.order))
				print(' --------------------------------------------------------------------------------------------')
		# write also to stdout
		print(' --------------------------------------------------------------------------------------------')
		print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} screening started'.format(exp.order))
		print(' --------------------------------------------------------------------------------------------')


def screen_end(calc, exp):
		""" print end of screening """
		with open(_out+'/output.out','a') as f:
			with contextlib.redirect_stdout(f):
				if (exp.conv_orb[-1]):
					print(' --------------------------------------------------------------------------------------------')
					print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} screening done'.format(exp.order))
					print(' STATUS-'+exp.level.upper()+':                  *** convergence has been reached ***                         ')
					print(' --------------------------------------------------------------------------------------------\n\n')
				else:
					print(' --------------------------------------------------------------------------------------------')
					print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} screening done'.format(exp.order))
					print(' --------------------------------------------------------------------------------------------\n\n')
		# write also to stdout
		if (exp.conv_orb[-1]):
			print(' --------------------------------------------------------------------------------------------')
			print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} screening done'.format(exp.order))
			print(' STATUS-'+exp.level.upper()+':                  *** convergence has been reached ***                         ')
			print(' --------------------------------------------------------------------------------------------\n\n')
		else:
			print(' --------------------------------------------------------------------------------------------')
			print(' STATUS-'+exp.level.upper()+':  order k = {0:>d} screening done'.format(exp.order))
			print(' --------------------------------------------------------------------------------------------\n\n')
		
		
