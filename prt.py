#!/usr/bin/env python
# -*- coding: utf-8 -*

""" prt.py: print class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
import numpy as np
from contextlib import redirect_stdout


class PrintCls():
		""" print functions """
		def __init__(self, _out):
				""" init parameters """
				self.out = _out.out_dir+'/output.out'
				self.res = _out.out_dir+'/results.out'
				# summary constants
				self.header_str = '{0:^93}'.format('-'*45)
				# print main header
				self.main_header()
				#
				return


		def main_header(self):
				""" print main header """
				for i in [self.out,self.res]:
					with open(i,'a') as f:
						with redirect_stdout(f):
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
				#
				return


		def exp_header(self, _calc, _exp):
				""" print expansion header """
				with open(self.out,'a') as f:
					with redirect_stdout(f):
						print('\n\n'+self.header_str)
						print('{0:^93}'.format(_calc.exp_type+' expansion'))
						print(self.header_str+'\n\n')
				# write also to stdout
				print('\n\n'+self.header_str)
				print('{0:^93}'.format(_calc.exp_type+' expansion'))
				print(self.header_str+'\n\n')
				#
				return
		
		
		def mbe_header(self, _calc, _exp):
				""" print mbe header """
				with open(self.out,'a') as f:
					with redirect_stdout(f):
						print(' --------------------------------------------------------------------------------------------')
						print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} MBE started  ---  {1:d} tuples in total'.\
								format(_exp.order,len(_exp.tuples[_exp.order-_exp.start_order])))
						print(' --------------------------------------------------------------------------------------------')
				# write also to stdout
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} MBE started  ---  {1:d} tuples in total'.\
						format(_exp.order,len(_exp.tuples[_exp.order-_exp.start_order])))
				print(' --------------------------------------------------------------------------------------------')
				#
				return

		
		def mbe_status(self, _calc, _exp, _prog):
				""" print status bar """
				bar_length = 50
				status = ""
				block = int(round(bar_length * _prog))
				print(' STATUS-'+_exp.level.upper()+':   [{0}]   ---  {1:>6.2f} % {2}'.\
						format('#' * block + '-' * (bar_length - block), _prog * 100, status))
				#
				return
	
	
		def mbe_end(self, _calc, _exp):
				""" print end of mbe """
				with open(self.out,'a') as f:
					with redirect_stdout(f):
						print(' --------------------------------------------------------------------------------------------')
						print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} MBE done (E = {1:.6e}, thres. = {2:<5.2e})'.\
								format(_exp.order,np.sum(_exp.energy['inc'][_exp.order-_exp.start_order]),_exp.thres))
						print(' --------------------------------------------------------------------------------------------')
				# write also to stdout
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} MBE done (E = {1:.6e}, thres. = {2:<5.2e})'.\
						format(_exp.order,np.sum(_exp.energy['inc'][_exp.order-_exp.start_order]),_exp.thres))
				print(' --------------------------------------------------------------------------------------------')
				#
				return


		def mbe_micro_results(self, _calc, _exp):	
				""" print micro result statistics """
				if ((_calc.exp_type == 'combined') and (_exp.level == 'macro')):
					# statistics
					mean_val = np.mean(_exp.micro_conv[_exp.order-_exp.start_order])
					min_val = _exp.micro_conv[_exp.order-_exp.start_order][np.argmin(_exp.micro_conv[_exp.order-_exp.start_order])]
					max_val = _exp.micro_conv[_exp.order-_exp.start_order][np.argmax(_exp.micro_conv[_exp.order-_exp.start_order])]
					if (len(_exp.micro_conv[_exp.order-_exp.start_order]) > 1):
						std_val = np.std(_exp.micro_conv[_exp.order-_exp.start_order], ddof=1)
					else:
						std_val = 0.0
					# now print
					with open(self.out,'a') as f:
						with redirect_stdout(f):
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
				#
				return

	
		def mbe_results(self, _mol, _calc, _exp, _kernel):
				""" print mbe result statistics """
				# statistics
				mean_val = np.mean(_exp.energy['inc'][_exp.order-_exp.start_order])
				min_idx = np.argmin(np.abs(_exp.energy['inc'][_exp.order-_exp.start_order]))
				min_val = _exp.energy['inc'][_exp.order-_exp.start_order][min_idx]
				max_idx = np.argmax(np.abs(_exp.energy['inc'][_exp.order-_exp.start_order]))
				max_val = _exp.energy['inc'][_exp.order-_exp.start_order][max_idx]
				# core and cas regions
				core, cas = _kernel.core_cas(_mol, _exp, _exp.tuples[_exp.order-_exp.start_order][max_idx])
				cas_ref = '{0:}'.format(sorted(list(set(_calc.ref_space.tolist()) - set(core))))
				if (_calc.exp_ref['METHOD'] == 'HF'):
					cas_exp = '{0:}'.format(sorted(list(set(cas) - set(_calc.ref_space.tolist()))))
				else:
					cas_exp = '{0:}'.format(sorted(_exp.tuples[0][0].tolist()))
					cas_exp += ' + {0:}'.format(sorted(list(set(cas) - set(_exp.tuples[0][0].tolist()) - set(_calc.ref_space.tolist()))))
				# now print
				with open(self.out,'a') as f:
					with redirect_stdout(f):
						print(' --------------------------------------------------------------------------------------------')
						print(' RESULT-'+_exp.level.upper()+':      mean increment     |    min. abs. increment   |    max. abs. increment')
						print(' --------------------------------------------------------------------------------------------')
						print(' RESULT-'+_exp.level.upper()+':     {0:>13.4e}       |      {1:>13.4e}       |      {2:>13.4e}'.\
								format(mean_val, min_val, max_val))
						# debug print
						if (_mol.verbose_prt):
							print(' --------------------------------------------------------------------------------------------')
							print(' RESULT-'+_exp.level.upper()+':                   info on max. abs. increment:')
							print(' RESULT-'+_exp.level.upper()+':  core = {0:}'.format(core))
							print(' RESULT-'+_exp.level.upper()+':  cas  = '+cas_ref+' + '+cas_exp)
						print(' --------------------------------------------------------------------------------------------')
				# write also to stdout
				print(' --------------------------------------------------------------------------------------------')
				print(' RESULT-'+_exp.level.upper()+':      mean increment     |    min. abs. increment   |    max. abs. increment')
				print(' --------------------------------------------------------------------------------------------')
				print(' RESULT-'+_exp.level.upper()+':     {0:>13.4e}       |      {1:>13.4e}       |      {2:>13.4e}'.\
						format(mean_val, min_val, max_val))
				# debug print
				if (_mol.verbose_prt):
					print(' --------------------------------------------------------------------------------------------')
					print(' RESULT-'+_exp.level.upper()+':                   info on max. abs. increment:')
					print(' RESULT-'+_exp.level.upper()+':  core = {0:}'.format(core))
					print(' RESULT-'+_exp.level.upper()+':  cas  = '+cas_ref+' + '+cas_exp)
				print(' --------------------------------------------------------------------------------------------')
				#
				return
		
		
		def screen_header(self, _calc, _exp):
				""" print screening header """
				with open(self.out,'a') as f:
					with redirect_stdout(f):
						print(' --------------------------------------------------------------------------------------------')
						print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} screening started'.format(_exp.order))
						print(' --------------------------------------------------------------------------------------------')
				# write also to stdout
				print(' --------------------------------------------------------------------------------------------')
				print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} screening started'.format(_exp.order))
				print(' --------------------------------------------------------------------------------------------')
				#
				return
		
		
		def screen_end(self, _calc, _exp):
				""" print end of screening """
				with open(self.out,'a') as f:
					with redirect_stdout(f):
						if (_exp.conv_orb[-1]):
							print(' --------------------------------------------------------------------------------------------')
							print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} screening done'.format(_exp.order))
							print(' STATUS-'+_exp.level.upper()+':                  *** convergence has been reached ***                         ')
							print(' --------------------------------------------------------------------------------------------\n\n')
						else:
							print(' --------------------------------------------------------------------------------------------')
							print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} screening done'.format(_exp.order))
							print(' --------------------------------------------------------------------------------------------\n\n')
				# write also to stdout
				if (_exp.conv_orb[-1]):
					print(' --------------------------------------------------------------------------------------------')
					print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} screening done'.format(_exp.order))
					print(' STATUS-'+_exp.level.upper()+':                  *** convergence has been reached ***                         ')
					print(' --------------------------------------------------------------------------------------------\n\n')
				else:
					print(' --------------------------------------------------------------------------------------------')
					print(' STATUS-'+_exp.level.upper()+':  order k = {0:>d} screening done'.format(_exp.order))
					print(' --------------------------------------------------------------------------------------------\n\n')
				#
				return
		
		
