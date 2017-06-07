#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_summary.py: summary print utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from contextlib import redirect_stdout

from bg_print import print_main_header


divider_str = '{0:^143}'.format('-'*137)
header_str = '{0:^143}'.format('-'*45)

def summary_overall_res(molecule):
		""" print overall results """
		# set local variables
		if (molecule['occ_orbs'] == 'LOCAL'):
			occ_orbs = 'local'
		else:
			occ_orbs = 'canonical'
		if (molecule['virt_orbs'] == 'MP2'):
			virt_orbs = 'MP2 NOs'
		else:
			virt_orbs = 'canonical'
		if (molecule['mpi_parallel']):
			mpi_size = molecule['mpi_size']
		else:
			mpi_size = 1
		nocc = molecule['nocc']
		if (molecule['frozen']): nocc -= molecule['ncore']
		# write summary to bg_results.out
		with open(molecule['out_dir']+'/bg_results.out','a') as f:
			with redirect_stdout(f):
				print('')
				print('')
				print(header_str)
				print('{0:^143}'.format('overall results'))
				print(header_str)
				print('')
				print(divider_str)
				print('{0:14}{1:21}{2:11}{3:1}{4:12}{5:21}{6:11}{7:1}{8:13}{9:}'.\
						format('','molecular information','','|','',\
							'expansion information','','|','','calculation information'))
				print(divider_str)
				print(('{0:12}{1:9}{2:7}{3:1}{4:2}{5:<12s}{6:3}{7:1}{8:9}{9:15}{10:4}{11:1}'
					'{12:2}{13:<6s}{14:7}{15:1}{16:7}{17:16}{18:7}{19:1}{20:2}{21:}').\
						format('','basis set','','=','',molecule['basis'],\
							'','|','','expansion model','','=','',molecule['model'],\
							'','|','','mpi parallel run','','=','',molecule['mpi_parallel']))
				print(('{0:12}{1:11}{2:5}{3:1}{4:2}{5:<5}{6:10}{7:1}{8:9}{9:14}{10:5}{11:1}'
					'{12:2}{13:<8s}{14:5}{15:1}{16:7}{17:21}{18:2}{19:1}{20:2}{21:}').\
						format('','frozen core','','=','',str(molecule['frozen']),\
							'','|','','expansion type','','=','',molecule['exp'],\
							'','|','','number of mpi masters','','=','',1))
				print(('{0:12}{1:14}{2:2}{3:1}{4:2}{5:<2d}{6:^3}{7:<4d}{8:6}{9:1}{10:9}{11:14}{12:5}'
					'{13:1}{14:2}{15:<5.2e}{16:5}{17:1}{18:7}{19:20}{20:3}{21:1}{22:2}{23:}').\
						format('','# occ. / virt.','','=','',nocc,'/',molecule['nvirt'],\
							'','|','','exp. threshold','','=','',molecule['prim_exp_thres_init'],\
							'','|','','number of mpi slaves','','=','',mpi_size-1))
				print(('{0:12}{1:13}{2:3}{3:1}{4:2}{5:<9s}{6:6}{7:1}{8:9}{9:12}{10:7}{11:1}{12:2}'
					'{13:<6.2f}{14:7}{15:1}{16:7}{17:18}{18:5}{19:1}{20:1}{21:>13.6e}').\
						format('','occ. orbitals','','=','',occ_orbs,\
							'','|','','damp. factor','','=','',molecule['prim_exp_scaling'],\
							'','|','','final corr. energy','','=','',molecule['prim_energy'][-1]))
				print(('{0:12}{1:14}{2:2}{3:1}{4:2}{5:<9s}{6:6}{7:1}{8:9}{9:16}{10:3}{11:1}{12:2}'
					'{13:<5.2e}{14:5}{15:1}{16:7}{17:17}{18:6}{19:1}{20:1}{21:>13.6e}').\
						format('','virt. orbitals','','=','',virt_orbs,\
							'','|','','energy threshold','','=','',molecule['prim_energy_thres'],\
							'','|','','final convergence','','=','',\
							molecule['prim_energy'][-1]-molecule['prim_energy'][-2]))
				print(divider_str)
		#
		return


def summary_detail_res(molecule):
		""" print detailed results """
		# init total number of tuples
		total_tup = 0
		# write summary to bg_results.out
		with open(molecule['out_dir']+'/bg_results.out','a') as f:
			with redirect_stdout(f):
				print('')
				print('')
				print(header_str)
				print('{0:^143}'.format('detailed results'))
				print(header_str)
				print('')
				print(divider_str)
				print(('{0:6}{1:8}{2:3}{3:1}{4:7}{5:18}{6:7}{7:1}'
					'{8:7}{9:26}{10:6}{11:1}{12:6}{13:}').\
						format('','BG order','','|','','total corr. energy',\
							'','|','','total time (HHH : MM : SS)',\
							'','|','','number of calcs. (abs. / %  --  total)'))
				print(divider_str)
				# loop over orders
				for i in range(0,len(molecule['prim_energy'])):
					# sum up total time and number of tuples
					total_time = np.sum(molecule['time_kernel'][:i+1])\
									+np.sum(molecule['time_summation'][:i+1])\
									+np.sum(molecule['time_screen'][:i+1])
					total_tup += len(molecule['prim_tuple'][i])
					print(('{0:7}{1:>4d}{2:6}{3:1}{4:9}{5:>13.6e}{6:10}{7:1}{8:14}{9:03d}{10:^3}{11:02d}'
						'{12:^3}{13:02d}{14:12}{15:1}{16:7}{17:>9d}{18:^3}{19:>6.2f}{20:^8}{21:>9d}').\
							format('',i+1,'','|','',molecule['prim_energy'][i],\
								'','|','',int(total_time//3600),':',\
								int((total_time-(total_time//3600)*3600.)//60),':',\
								int(total_time-(total_time//3600)*3600.\
								-((total_time-(total_time//3600)*3600.)//60)*60.),\
								'','|','',len(molecule['prim_tuple'][i]),'/',\
								(float(len(molecule['prim_tuple'][i]))\
								/float(molecule['theo_work'][i]))*100.00,'--',total_tup))
				print(divider_str)
				# reference calc
				if (molecule['ref']):
					print(divider_str)
					print('{0:5}{1:9}{2:3}{3:1}{4:9}{5:>13.6e}'.\
						format('','reference','','|','',molecule['e_ref']))
					print('{0:5}{1:10}{2:2}{3:1}{4:9}{5:>13.6e}'.\
						format('','difference','','|','',molecule['e_ref']-molecule['prim_energy'][-1]))
					print(divider_str)
		#
		return


def summary_phase_time(molecule):
		""" print phase timings """
		# write summary to bg_results.out
		with open(molecule['out_dir']+'/bg_results.out','a') as f:
			with redirect_stdout(f):
				print('')
				print('')
				print(header_str)
				print('{0:^143}'.format('phase timings'))
				print(header_str)
				print('')
				print(divider_str)
				print(('{0:6}{1:8}{2:3}{3:1}{4:5}{5:32}{6:3}{7:1}'
					'{8:3}{9:35}{10:3}{11:1}{12:3}{13:}').\
						format('','BG order','','|','','time: kernel (HHH : MM : SS / %)',\
							'','|','','time: summation (HHH : MM : SS / %)',\
							'','|','','time: screen (HHH : MM : SS / %)'))
				print(divider_str)
				for i in range(0,len(molecule['prim_energy'])):
					# set shorthand notation
					time_k = molecule['time_kernel'][i]
					time_f = molecule['time_summation'][i]
					time_s = molecule['time_screen'][i]
					time_t = molecule['time_tot'][i]
					print(('{0:7}{1:>4d}{2:6}{3:1}{4:11}{5:03d}{6:^3}{7:02d}{8:^3}'
						'{9:02d}{10:^3}{11:>6.2f}{12:7}{13:1}{14:10}{15:03d}{16:^3}'
						'{17:02d}{18:^3}{19:02d}{20:^3}{21:>6.2f}{22:9}{23:1}{24:9}'
						'{25:03d}{26:^3}{27:02d}{28:^3}{29:02d}{30:^3}{31:>6.2f}').\
							format('',i+1,'','|','',int(time_k//3600),':',\
								int((time_k-(time_k//3600)*3600.)//60),':',\
								int(time_k-(time_k//3600)*3600.\
								-((time_k-(time_k//3600)*3600.)//60)*60.),'/',(time_k/time_t)*100.0,\
								'','|','',int(time_f//3600),':',\
								int((time_f-(time_f//3600)*3600.)//60),':',\
								int(time_f-(time_f//3600)*3600.\
								-((time_f-(time_f//3600)*3600.)//60)*60.),'/',(time_f/time_t)*100.0,\
								'','|','',int(time_s//3600),':',int((time_s-(time_s//3600)*3600.)//60),':',\
								int(time_s-(time_s//3600)*3600.\
								-((time_s-(time_s//3600)*3600.)//60)*60.),'/',(time_s/time_t)*100.0))
				print(divider_str)
				print(divider_str)
				# set shorthand notation
				time_k = molecule['time_kernel'][-1]
				time_f = molecule['time_summation'][-1]
				time_s = molecule['time_screen'][-1]
				time_t = molecule['time_tot'][-1]
				print(('{0:8}{1:5}{2:4}{3:1}{4:11}{5:03d}{6:^3}{7:02d}{8:^3}'
					'{9:02d}{10:^3}{11:>6.2f}{12:7}{13:1}{14:10}{15:03d}{16:^3}'
					'{17:02d}{18:^3}{19:02d}{20:^3}{21:>6.2f}{22:9}{23:1}{24:9}'
					'{25:03d}{26:^3}{27:02d}{28:^3}{29:02d}{30:^3}{31:>6.2f}').\
						format('','total','','|','',int(time_k//3600),':',\
							int((time_k-(time_k//3600)*3600.)//60),':',int(time_k-(time_k//3600)*3600.\
							-((time_k-(time_k//3600)*3600.)//60)*60.),'/',(time_k/time_t)*100.0,\
							'','|','',int(time_f//3600),':',int((time_f-(time_f//3600)*3600.)//60),':',\
							int(time_f-(time_f//3600)*3600.\
							-((time_f-(time_f//3600)*3600.)//60)*60.),'/',(time_f/time_t)*100.0,\
							'','|','',int(time_s//3600),':',int((time_s-(time_s//3600)*3600.)//60),':',\
							int(time_s-(time_s//3600)*3600.\
							-((time_s-(time_s//3600)*3600.)//60)*60.),'/',(time_s/time_t)*100.0))
				print(divider_str)
		#
		return


def summary_mpi_time(molecule):
		""" print mpi timings """
		with open(molecule['out_dir']+'/bg_results.out','a') as f:
			with redirect_stdout(f):
				print('')
				print('')
				print(header_str)
				print('{0:^143}'.format('mpi timings'))
				print(header_str)
				print('')
				print(divider_str)
				print(('{0:6}{1:13}{2:3}{3:1}{4:1}{5:35}{6:1}{7:1}'
						'{8:1}{9:38}{10:1}{11:1}{12:1}{13:}').\
						format('','mpi processor','','|','','time: kernel (work/comm/idle, in %)',\
							'','|','','time: summation (work/comm/idle, in %)',\
							'','|','','time: screen (work/comm/idle, in %)'))
				print(divider_str)
				print(('{0:4}{1:6}{2:^4}{3:<8d}{4:1}{5:6}{6:>6.2f}{7:^3}'
						'{8:>6.2f}{9:^3}{10:>6.2f}{11:7}{12:1}{13:7}{14:>6.2f}'
						'{15:^3}{16:>6.2f}{17:^3}{18:>6.2f}{19:9}{20:1}{21:6}'
						'{22:>6.2f}{23:^3}{24:>6.2f}{25:^3}{26:>6.2f}').\
							format('','master','--',0,'|','',molecule['dist_kernel'][0][0],'/',\
								molecule['dist_kernel'][1][0],'/',molecule['dist_kernel'][2][0],\
								'','|','',molecule['dist_summation'][0][0],'/',\
								molecule['dist_summation'][1][0],'/',molecule['dist_summation'][2][0],\
								'','|','',molecule['dist_screen'][0][0],'/',\
								molecule['dist_screen'][1][0],'/',molecule['dist_screen'][2][0]))
				print(divider_str)
				for i in range(1,molecule['mpi_size']):
					print(('{0:4}{1:6}{2:^4}{3:<8d}{4:1}{5:6}{6:>6.2f}{7:^3}'
							'{8:>6.2f}{9:^3}{10:>6.2f}{11:7}{12:1}{13:7}{14:>6.2f}'
							'{15:^3}{16:>6.2f}{17:^3}{18:>6.2f}{19:9}{20:1}{21:6}'
							'{22:>6.2f}{23:^3}{24:>6.2f}{25:^3}{26:>6.2f}').\
								format('','slave ','--',i,'|','',molecule['dist_kernel'][0][i],'/',\
									molecule['dist_kernel'][1][i],'/',molecule['dist_kernel'][2][i],\
									'','|','',molecule['dist_summation'][0][i],'/',\
									molecule['dist_summation'][1][i],'/',molecule['dist_summation'][2][i],\
									'','|','',molecule['dist_screen'][0][i],'/',\
									molecule['dist_screen'][1][i],'/',molecule['dist_screen'][2][i]))
				#
				print(divider_str)
				print(divider_str)
				#
				print(('{0:4}{1:14}{2:4}{3:1}{4:6}{5:>6.2f}{6:^3}{7:>6.2f}{8:^3}'
						'{9:>6.2f}{10:7}{11:1}{12:7}{13:>6.2f}{14:^3}{15:>6.2f}{16:^3}{17:>6.2f}'
						'{18:9}{19:1}{20:6}{21:>6.2f}{22:^3}{23:>6.2f}{24:^3}{25:>6.2f}').\
							format('','mean  : slaves','','|','',\
								np.mean(molecule['dist_kernel'][0][1:]),'/',\
								np.mean(molecule['dist_kernel'][1][1:]),'/',\
								np.mean(molecule['dist_kernel'][2][1:]),\
								'','|','',np.mean(molecule['dist_summation'][0][1:]),'/',\
								np.mean(molecule['dist_summation'][1][1:]),'/',\
								np.mean(molecule['dist_summation'][2][1:]),\
								'','|','',np.mean(molecule['dist_screen'][0][1:]),'/',\
								np.mean(molecule['dist_screen'][1][1:]),'/',\
								np.mean(molecule['dist_screen'][2][1:])))
				#
				print(('{0:4}{1:14}{2:4}{3:1}{4:6}{5:>6.2f}{6:^3}{7:>6.2f}{8:^3}'
						'{9:>6.2f}{10:7}{11:1}{12:7}{13:>6.2f}{14:^3}{15:>6.2f}{16:^3}{17:>6.2f}'
						'{18:9}{19:1}{20:6}{21:>6.2f}{22:^3}{23:>6.2f}{24:^3}{25:>6.2f}').\
							format('','stdev : slaves','','|','',\
								np.std(molecule['dist_kernel'][0][1:],ddof=1),'/',\
								np.std(molecule['dist_kernel'][1][1:],ddof=1),'/',\
								np.std(molecule['dist_kernel'][2][1:],ddof=1),\
								'','|','',np.std(molecule['dist_summation'][0][1:],ddof=1),'/',\
								np.std(molecule['dist_summation'][1][1:],ddof=1),'/',\
								np.std(molecule['dist_summation'][2][1:],ddof=1),\
								'','|','',np.std(molecule['dist_screen'][0][1:],ddof=1),'/',\
								np.std(molecule['dist_screen'][1][1:],ddof=1),'/',\
								np.std(molecule['dist_screen'][2][1:],ddof=1)))
				#
				print(divider_str)
		#
		return


def summary_end(molecule):
		""" print end of summary """
		#
		with open(molecule['out_dir']+'/bg_results.out','a') as f:
			with redirect_stdout(f):
				print('\n')
		#
		return


def summary(molecule):
		""" driver function for summary printing """
		# print main header
		print_main_header(molecule,'bg_results.out')
		# print overall results
		summary_overall_res(molecule)
		# print detailed results
		summary_detail_res(molecule)
		# print phase timings
		summary_phase_time(molecule)
		# print mpi timings
		if (molecule['mpi_parallel']): summary_mpi_time(molecule)
		# print summary end
		summary_end(molecule)
		#
		return


