#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_results.py: summary print and plotting utilities for Bethe-Goldstone correlation calculations."""

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
import numpy as np
from itertools import cycle
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns


class ResCls():
		""" result class """
		def __init__(self, _out_dir):
				""" init parameters """
				self.output = _out_dir+'/bg_results.out'
				# summary constants
				self.divider_str = '{0:^143}'.format('-'*137)
				self.header_str = '{0:^143}'.format('-'*45)
				#
				return self


		def main(self, _mpi, _mol, _calc, _exp, _time):
				""" main driver for summary printing and plotting """
				#
				#** timings **#
				#
				# calculate timings
				_time.calc_time(_mpi, _exp)
				#
				#** summary **#
				#
				# overall results
				self.overall_res(_mpi, _mol, _calc, _exp)
				# detailed results
				self.detail_res(_exp, _time)
				# phase timings
				self.phase_time(_exp, _time)
				# print mpi timings
				if (_mpi.parallel): self.mpi_time(_mpi, _time)
				#
				#** plotting **#
				#
				# total energies
				self.abs_energy(_mol, _calc, _exp)
				# number of calculations
				self.n_tuples(_mol, _calc, _exp)
				# orbital entanglement matrices
				self.orb_ent_all(_mol, _calc, _exp)
				self.orb_ent(_mol, _calc, _exp)
				# individual orbital contributions by order
				self.orb_con_order(_mol, _calc, _exp)
				# orbital/energy distributions
				self.orb_dist(_mol, _calc, _exp)
				# total orbital contributions
				self.orb_con_tot(_mol, _calc, _exp)
				# plot timings
				self.time(_mol, _calc, _exp, _time)
				#
				return


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
						print(divider_str+'\n')
				#
				return


		def abs_energy_plot(molecule):
				""" plot absolute energy """
				# set seaborn
				sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
				# set 1 plot
				fig, ax = plt.subplots()
				# set title
				ax.set_title('Total '+molecule['model']+' energy')
				# set upper limit
				u_limit = molecule['u_limit']
				if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): u_limit -= molecule['ncore']
				# plot results
				ax.plot(list(range(1,len(molecule['prim_energy'])+1)),molecule['prim_energy'],marker='x',linewidth=2,linestyle='-',label='BG('+molecule['model']+')')
				# set x limits
				ax.set_xlim([0.5,u_limit+0.5])
				# turn off x-grid
				ax.xaxis.grid(False)
				# set labels
				ax.set_xlabel('Expansion order')
				ax.set_ylabel('Energy (in Hartree)')
				# force integer ticks on x-axis
				ax.xaxis.set_major_locator(MaxNLocator(integer=True))
				# despine
				sns.despine()
				# make insert
				with sns.axes_style("whitegrid"):
					# define frame
					insert = plt.axes([.35, .50, .50, .30],frameon=True)
					# plot results
					insert.plot(list(range(2,len(molecule['prim_energy'])+1)),molecule['prim_energy'][1:],marker='x',linewidth=2,linestyle='-')
					# set x limits
					plt.setp(insert,xticks=list(range(3,len(molecule['prim_energy'])+1)))
					insert.set_xlim([2.5,len(molecule['prim_energy'])+0.5])
					# set number of y ticks
					insert.locator_params(axis='y',nbins=6)
					# set y limits
					insert.set_ylim([molecule['prim_energy'][-1]-0.01,molecule['prim_energy'][-1]+0.01])
					# turn off x-grid
					insert.xaxis.grid(False)
				# set legends
				ax.legend(loc=1)
				# save plot
				plt.savefig(molecule['wrk_dir']+'/output/abs_energy_plot.pdf', bbox_inches = 'tight', dpi=1000)
				#
				return


		def n_tuples_plot(molecule):
				""" plot number of tuples """
				# set seaborn
				sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
				# set 1 plot
				fig, ax = plt.subplots()
				# set title
				ax.set_title('Total number of '+molecule['model']+' tuples')
				# set upper limit
				u_limit = molecule['u_limit']
				if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): u_limit -= molecule['ncore']
				# init prim list
				prim = []
				# set prim list
				for i in range(0,u_limit):
					if (i < len(molecule['prim_tuple'])):
						prim.append(len(molecule['prim_tuple'][i]))
					else:
						prim.append(0)
				# plot results
				sns.barplot(list(range(1,u_limit+1)),molecule['theo_work'],palette='Greens',label='Theoretical number',log=True)
				sns.barplot(list(range(1,u_limit+1)),prim,palette='Blues_r',label='BG('+molecule['model']+') expansion',log=True)
				# turn off x-grid
				ax.xaxis.grid(False)
				# set x- and y-limits
				ax.set_xlim([-0.5,u_limit-0.5])
				ax.set_ylim(bottom=0.7)
				# set x-ticks
				if (u_limit < 8):
					ax.set_xticks(list(range(0,u_limit)))
					ax.set_xticklabels(list(range(1,u_limit+1)))
				else:
					ax.set_xticks(list(range(0,u_limit,u_limit//8)))
					ax.set_xticklabels(list(range(1,u_limit+1,u_limit//8)))
				# set x- and y-labels
				ax.set_xlabel('Expansion order')
				ax.set_ylabel('Number of correlated tuples')
				# set legend
				plt.legend(loc=1)
				leg = ax.get_legend()
				leg.legendHandles[0].set_color(sns.color_palette('Greens')[-1])
				leg.legendHandles[1].set_color(sns.color_palette('Blues_r')[0])
				# despind
				sns.despine()
				# tight layout
				fig.tight_layout()
				# save plot
				plt.savefig(molecule['wrk_dir']+'/output/n_tuples_plot.pdf', bbox_inches = 'tight', dpi=1000)
				# del prim list
				del prim
				#
				return


		def orb_ent_all_plot(molecule):
				""" plot orbital entanglement (all plots) """
				# set seaborn
				sns.set(style='white',font='DejaVu Sans')
				# set colormap
				cmap = sns.cubehelix_palette(as_cmap=True)
				# set number of subplots
				h_length = len(molecule['prim_orb_ent_rel'])//2
				if (len(molecule['prim_orb_ent_rel']) % 2 != 0): h_length += 1
				ratio = 0.98/float(h_length)
				fig, ax = plt.subplots(h_length+1, 2, sharex='col', sharey='row', gridspec_kw = {'height_ratios': [ratio]*h_length+[0.02]})
				# set figure size
				fig.set_size_inches([8.268,11.693])
				# set location for colorbar
				cbar_ax = fig.add_axes([0.06,0.02,0.88,0.03])
				# init mask array
				mask_arr = np.zeros_like(molecule['prim_orb_ent_rel'][0],dtype=np.bool)
				# set title
				fig.suptitle('Entanglement matrices')
				# plot results
				for i in range(0,len(molecule['prim_orb_ent_rel'])):
					mask_arr = (molecule['prim_orb_ent_rel'][i] == 0.0)
					sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][i]),ax=ax.flat[i],mask=np.transpose(mask_arr),cmap=cmap,\
								xticklabels=False,yticklabels=False,cbar=True,cbar_ax=cbar_ax,cbar_kws={'format':'%.0f', 'orientation': 'horizontal'},\
								annot=False,vmin=0.0,vmax=100.0)
					ax.flat[i].set_title('BG order = '+str(i+2))
				# remove ticks
				ax[-1,0].set_yticklabels([]); ax[-1,1].set_yticklabels([])
				# despine
				sns.despine(left=True,right=True,top=True,bottom=True)
				# tight layout
				fig.tight_layout()
				# adjust subplots (to make room for title)
				plt.subplots_adjust(top=0.95)
				# save plot
				plt.savefig(molecule['wrk_dir']+'/output/orb_ent_all_plot.pdf', bbox_inches = 'tight', dpi=1000)
				# del mask array
				del mask_arr
				#
				return


		def orb_ent_plot(molecule):
				""" plot orbital entanglement (first and last plot) """
				# set seaborn
				sns.set(style='white',font='DejaVu Sans')
				# set colormap
				cmap = sns.cubehelix_palette(as_cmap=True)
				# make 2 plots + 1 colorbar
				fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1.0,1.0,0.06]})
				# set figure size
				fig.set_size_inches([11.0,5.0])
				# fix for colorbar
				ax1.get_shared_y_axes().join(ax2)
				# set mask array
				mask_arr = (molecule['prim_orb_ent_rel'][0] == 0.0)
				# plot results
				sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][0]),ax=ax1,mask=np.transpose(mask_arr),cmap=cmap,\
							xticklabels=False,yticklabels=False,cbar=False,\
							annot=False,vmin=0.0,vmax=100.0)
				# set title
				ax1.set_title('Entanglement matrix, order = 2')
				# set mask array
				mask_arr = (molecule['prim_orb_ent_rel'][-1] == 0.0)
				# plot results
				sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][-1]),ax=ax2,mask=np.transpose(mask_arr),cmap=cmap,\
							xticklabels=False,yticklabels=False,cbar=True,cbar_ax=cbar_ax,cbar_kws={'format':'%.0f'},\
							annot=False,vmin=0.0,vmax=100.0)
				# set title
				ax2.set_title('Entanglement matrix, order = '+str(len(molecule['prim_energy'])))
				# despine
				sns.despine(left=True,right=True,top=True,bottom=True)
				# tight layout
				fig.tight_layout()
				# save plot
				plt.savefig(molecule['wrk_dir']+'/output/orb_ent_plot.pdf', bbox_inches = 'tight', dpi=1000)
				# del mask array
				del mask_arr
				#
				return


		def orb_con_order_plot(molecule):
				""" plot orbital contributions (individually, order by order) """
				# set seaborn
				sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
				# set 1 plot
				fig, ax = plt.subplots()
				# transpose orb_con_abs array
				orb_con_arr = np.transpose(np.asarray(molecule['prim_orb_con_abs']))
				# plot results
				for i in range(0,len(orb_con_arr)):
					# determine x-range
					end = len(molecule['prim_energy'])
					for j in range(1,len(orb_con_arr[i])):
						if ((orb_con_arr[i,j]-orb_con_arr[i,j-1]) == 0.0):
							end = j-1
							break
					ax.plot(list(range(1,end+1)),orb_con_arr[i,:end],linewidth=2)
				# set x-limits
				ax.set_xlim([0.5,len(molecule['prim_energy'])+0.5])
				# turn off x-grid
				ax.xaxis.grid(False)
				# set x- and y-labels
				ax.set_xlabel('Expansion order')
				ax.set_ylabel('Accumulated orbital contribution (in Hartree)')
				# for integer ticks on x-axis
				ax.xaxis.set_major_locator(MaxNLocator(integer=True))
				# despine
				sns.despine()
				# tight layout
				fig.tight_layout()
				# save plot
				plt.savefig(molecule['wrk_dir']+'/output/orb_con_order_plot.pdf', bbox_inches = 'tight', dpi=1000)
				# del orb_con array
				del orb_con_arr
				#
				return


		def orb_dist_plot(molecule):
				""" plot orbital distribution """
				# set seaborn
				sns.set(style='white', palette='Set2')
				# define color palette
				palette = cycle(sns.color_palette())
				# set number of subplots
				w_length = len(molecule['prim_energy_inc'])//2
				if ((len(molecule['prim_energy_inc']) % 2 != 0) and (len(molecule['prim_energy_inc'][-1]) != 1)): w_length += 1
				fig, axes = plt.subplots(2, w_length, figsize=(12, 8), sharex=False, sharey=False)
				# set title
				fig.suptitle('Distribution of energy contributions')
				# save threshold
				thres = molecule['prim_exp_thres_init']
				# set lists and plot results
				for i in range(0,len(molecule['prim_energy_inc'])):
					if (len(molecule['prim_energy_inc'][i]) != 1):
						# sort energy increments
						e_inc_sort = np.sort(molecule['prim_energy_inc'][i])
						# init counting list
						e_inc_count = np.zeros(len(e_inc_sort),dtype=np.float64)
						# count
						for j in range(0,len(e_inc_count)): e_inc_count[j] = ((j+1)/len(e_inc_count))*100.0
						# init contribution list
						e_inc_contrib = np.zeros(len(e_inc_sort),dtype=np.float64)
						# calc contributions
						for j in range(0,len(e_inc_contrib)): e_inc_contrib[j] = np.sum(e_inc_sort[:j+1])
						# plot contributions
						l1 = axes.flat[i].step(e_inc_sort,e_inc_count,where='post',linewidth=2,linestyle='-',color=sns.xkcd_rgb['salmon'],label='Contributions')
						# plot x = 0.0
						l2 = axes.flat[i].axvline(x=0.0,ymin=0.0,ymax=100.0,linewidth=2,linestyle='--',color=sns.xkcd_rgb['royal blue'])
						# plot threshold span
						axes.flat[i].axvspan(0.0-thres,0.0+thres,color=sns.xkcd_rgb['amber'],alpha=0.5)
						# update thres
						thres = molecule['prim_exp_scaling']**(i+1) * molecule['prim_exp_thres_init']
						# change to second y-axis
						ax2 = axes.flat[i].twinx()
						# plot counts
						l3 = ax2.step(e_inc_sort,e_inc_contrib,where='post',linewidth=2,linestyle='-',color=sns.xkcd_rgb['kelly green'],label='Energy')
						# set title
						axes.flat[i].set_title('E-{0:} = {1:4.2e}'.format(i+1,np.sum(e_inc_sort)))
						# set nice axis formatting
						delta = (np.abs(np.max(e_inc_sort)-np.min(e_inc_sort)))*0.05
						axes.flat[i].set_xlim([np.min(e_inc_sort)-delta,np.max(e_inc_sort)+delta])
						axes.flat[i].set_xticks([np.min(e_inc_sort),np.max(e_inc_sort)])
						axes.flat[i].xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
						ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
						axes.flat[i].set_yticks([0.0,25.0,50.0,75.0,100.0])
						# make ticks coloured and remove y2-axis ticks for most of the subplots
						axes.flat[i].tick_params('y',colors=sns.xkcd_rgb['salmon'])
						ax2.tick_params('y',colors=sns.xkcd_rgb['kelly green'])
						if (not ((i == 0) or (i == w_length))): axes.flat[i].set_yticks([])
						# set legend
						if (i == 0):
							lns = l1+l3
							labs = [l.get_label() for l in lns]
							plt.legend(lns,labs,loc=2,fancybox=True,frameon=True)
				# tight layout
				plt.tight_layout()
				# remove ticks for most of the subplots and despine
				if ((len(molecule['prim_energy_inc']) % 2 != 0) and (len(molecule['prim_energy_inc'][-1]) != 1)):
					axes.flat[-1].set_xticks([])
					axes.flat[-1].set_yticks([])
					axes.flat[-1].set_xticklabels([])
					axes.flat[-1].set_yticklabels([])
					sns.despine(left=True,bottom=True,ax=axes.flat[-1])
				# adjust subplots (to make room for title)
				plt.subplots_adjust(top=0.925)
				# save plot
				plt.savefig(molecule['wrk_dir']+'/output/orb_dist_plot.pdf', bbox_inches = 'tight', dpi=1000)
				# del lists
				del e_inc_sort; del e_inc_count; del e_inc_contrib
				#
				return


		def orb_con_tot_plot(molecule):
				""" plot total orbital contributions """
				# set seaborn
				sns.set(style='whitegrid',font='DejaVu Sans')
				# set 1 plot
				fig, ax = plt.subplots()
				# set orb_con and mask arrays
				orb_con_arr = 100.0*np.array(molecule['prim_orb_con_rel'])
				mask_arr = np.zeros_like(orb_con_arr,dtype=np.bool)
				mask_arr = (orb_con_arr == 0.0)
				# plot results
				sns.heatmap(orb_con_arr,ax=ax,mask=mask_arr,cmap='coolwarm',cbar_kws={'format':'%.0f'},\
							xticklabels=False,yticklabels=range(1,len(molecule['prim_orb_con_rel'])+1),cbar=True,\
							annot=False,vmin=0.0,vmax=np.amax(orb_con_arr))
				# set title
				ax.set_title('Total orbital contributions (in %)')
				# set y-label and y-ticks
				ax.set_ylabel('Expansion order')
				ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
				# despine
				sns.despine(left=True,bottom=True)
				# tight layout
				fig.tight_layout()
				# save plot
				plt.savefig(molecule['wrk_dir']+'/output/orb_con_tot_plot.pdf', bbox_inches = 'tight', dpi=1000)
				# del lists
				del orb_con_arr; del mask_arr
				#
				return


		def time_plot(molecule):
				""" plot total and mpi timings """
				# set seaborn
				sns.set(style='whitegrid',palette='Set2',font='DejaVu Sans')
				# set number of subplots - 2 with mpi, 1 without
				if (molecule['mpi_parallel']):
					fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
				else:
					fig, ax1 = plt.subplots()
				# set color palette
				sns.set_color_codes('pastel')
				# set upper limit
				u_limit = molecule['u_limit']
				if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): u_limit -= molecule['ncore']
				# set x-range
				order = list(range(1,len(molecule['prim_energy'])+2))
				# define y-ticks
				y_labels = list(range(1,len(molecule['prim_energy'])+1))
				y_labels.append('total')
				# set title
				ax1.set_title('Phase timings')
				# set result arrays and plot results
				kernel_dat = (molecule['time_kernel']/molecule['time_tot'])*100.0
				sum_dat = kernel_dat + (molecule['time_summation']/molecule['time_tot'])*100.0
				screen_dat = sum_dat + (molecule['time_screen']/molecule['time_tot'])*100.0
				screen = sns.barplot(screen_dat,order,ax=ax1,orient='h',label='screen',color=sns.xkcd_rgb['salmon'])
				summation = sns.barplot(sum_dat,order,ax=ax1,orient='h',label='summation',color=sns.xkcd_rgb['windows blue'])
				kernel = sns.barplot(kernel_dat,order,ax=ax1,orient='h',label='kernel',color=sns.xkcd_rgb['amber'])
				# set x- and y-limits
				ax1.set_ylim([-0.5,(len(molecule['prim_energy'])+1)-0.5])
				ax1.set_xlim([0.0,100.0])
				# set y-ticks
				ax1.set_yticklabels(y_labels)
				# set legend
				handles,labels = ax1.get_legend_handles_labels()
				handles = [handles[2],handles[1],handles[0]]
				labels = [labels[2],labels[1],labels[0]]
				ax1.legend(handles,labels,ncol=3,loc=9,fancybox=True,frameon=True)
				# invert plot
				ax1.invert_yaxis()
				# if not mpi, set labels. if mpi, plot mpi timings
				if (not molecule['mpi_parallel']):
					ax1.set_xlabel('Distribution (in %)')
					ax1.set_ylabel('Expansion order')
				else:
					# set title
					ax2.set_title('MPI timings')
					# set result arrays and plot results
					work_dat = molecule['dist_order'][0]
					comm_dat = work_dat + molecule['dist_order'][1]
					idle_dat = comm_dat + molecule['dist_order'][2]
					idle = sns.barplot(idle_dat,order,ax=ax2,orient='h',label='idle',color=sns.xkcd_rgb['sage'])
					comm = sns.barplot(comm_dat,order,ax=ax2,orient='h',label='comm',color=sns.xkcd_rgb['baby blue'])
					work = sns.barplot(work_dat,order,ax=ax2,orient='h',label='work',color=sns.xkcd_rgb['wine'])
					# set x- and y-limits
					ax2.set_ylim([-0.5,(len(molecule['prim_energy'])+1)-0.5])
					ax2.set_xlim([0.0,100.0])
					# set y-ticks
					ax2.set_yticklabels(y_labels)
					# set legend
					handles,labels = ax2.get_legend_handles_labels()
					handles = [handles[2],handles[1],handles[0]]
					labels = [labels[2],labels[1],labels[0]]
					ax2.legend(handles,labels,ncol=3,loc=9,fancybox=True,frameon=True)
					# set x- and y-labels
					fig.text(0.52,0.0,'Distribution (in %)',ha='center',va='center')
					fig.text(0.0,0.5,'Expansion order',ha='center',va='center',rotation='vertical')
					# invert plot
					ax2.invert_yaxis()
				# despine
				sns.despine(left=True,bottom=True)
				# tight layout
				fig.tight_layout()
				# save plot
				plt.savefig(molecule['wrk_dir']+'/output/time_plot.pdf', bbox_inches = 'tight', dpi=1000)
				# del lists
				del screen_dat; del kernel_dat; del sum_dat
				if (molecule['mpi_parallel']): del work_dat; del comm_dat; del idle_dat
				#
				return


