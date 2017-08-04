#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_results.py: summary print and plotting utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
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
try:
	import seaborn as sns
except ImportError:
	sys.stderr.write('\nImportError : seaborn module not found\n\n')


class ResCls():
		""" result class """
		def __init__(self, _mpi, _mol, _calc, _out):
				""" init parameters """
				self.out_dir = _out.out_dir
				self.output = self.out_dir+'/bg_results.out'
				# summary constants
				self.divider_str = '{0:^143}'.format('-'*137)
				self.fill_str = '{0:^143}'.format('|'*137)
				self.header_str = '{0:^143}'.format('-'*45)
				# upper limit
				if (_calc.exp_type in ['occupied','combined']):
					self.u_limit = _mol.nocc - _mol.ncore
				else:
					self.u_limit = _mol.nvirt 
				#
				return


		def main(self, _mpi, _mol, _calc, _exp):
				""" main driver for summary printing and plotting """
				# results
				self.results(_mpi, _mol, _calc, _exp)
				# plot of total energy
				self.abs_energy(_mol, _calc, _exp)
				# plot of number of calculations
				self.n_tuples(_mol, _calc, _exp)
				#
				return


		def results(self, _mpi, _mol, _calc, _exp):
				""" print results """
				# write summary to bg_results.out
				with open(self.output,'a') as f:
					with redirect_stdout(f):
						print('\n\n'+self.header_str)
						print('{0:^143}'.format('results'))
						print(self.header_str+'\n')
						print(self.divider_str)
						print('{0:14}{1:21}{2:11}{3:1}{4:12}{5:21}{6:11}{7:1}{8:13}{9:}'.\
								format('','molecular information','','|','',\
									'expansion information','','|','','calculation information'))
						print(self.divider_str)
						print(('{0:12}{1:9}{2:7}{3:1}{4:2}{5:<12s}{6:3}{7:1}{8:9}{9:17}{10:2}{11:1}'
							'{12:2}{13:<8s}{14:5}{15:1}{16:7}{17:16}{18:7}{19:1}{20:2}{21:}').\
								format('','basis set','','=','',_mol.basis,\
									'','|','','exp. model','','=','',_calc.exp_model,\
									'','|','','mpi parallel run','','=','',_mpi.parallel))
						print(('{0:12}{1:11}{2:5}{3:1}{4:2}{5:<5}{6:10}{7:1}{8:9}{9:9}{10:10}{11:1}'
							'{12:2}{13:<8s}{14:5}{15:1}{16:7}{17:21}{18:2}{19:1}{20:2}{21:}').\
								format('','frozen core','','=','',str(_mol.frozen),\
									'','|','','exp. base','','=','',_calc.exp_base,\
									'','|','','number of mpi masters','','=','',_mpi.num_local_masters + 1))
						print(('{0:12}{1:14}{2:2}{3:1}{4:2}{5:<2d}{6:^3}{7:<4d}{8:6}{9:1}{10:9}{11:14}{12:5}'
							'{13:1}{14:2}{15:<8s}{16:5}{17:1}{18:7}{19:20}{20:3}{21:1}{22:2}{23:}').\
								format('','# occ. / virt.','','=','',_mol.nocc-_mol.ncore,'/',_mol.nvirt,\
									'','|','','exp. type','','=','',_calc.exp_type,\
									'','|','','number of mpi slaves','','=','',_mpi.global_size - (_mpi.num_local_masters + 1)))
						print(('{0:12}{1:13}{2:3}{3:1}{4:2}{5:<9s}{6:6}{7:1}{8:9}{9:14}{10:5}{11:1}{12:2}'
							'{13:<6.2f}{14:7}{15:1}{16:7}{17:18}{18:5}{19:1}{20:1}{21:>13.6e}').\
								format('','occ. orbitals','','=','',_calc.exp_occ,\
									'','|','','exp. threshold','','=','',_calc.exp_thres,\
									'','|','','final corr. energy','','=','',_exp.energy_tot[-1]))
						print(('{0:12}{1:14}{2:2}{3:1}{4:2}{5:<9s}{6:6}{7:1}{8:9}{9:16}{10:3}{11:1}{12:2}'
							'{13:<5.2e}{14:5}{15:1}{16:7}{17:17}{18:6}{19:1}{20:1}{21:>13.6e}').\
								format('','virt. orbitals','','=','',_calc.exp_virt,\
									'','|','','energy threshold','','=','',_calc.energy_thres,\
									'','|','','final convergence','','=','',\
									_exp.energy_tot[-1] - _exp.energy_tot[-2]))
						print(self.divider_str)
						print(self.fill_str)
						print(self.divider_str)
						print(('{0:6}{1:8}{2:3}{3:1}{4:7}{5:18}{6:7}{7:1}'
							'{8:7}{9:26}{10:6}{11:1}{12:6}{13:}').\
								format('','BG order','','|','','total corr. energy',\
									'','|','','total time (HHH : MM : SS)',\
									'','|','','number of calcs. (abs. / %  --  total)'))
						print(self.divider_str)
						# loop over orders
						total_tup = 0
						for i in range(len(_exp.energy_tot)):
							# sum up total time and number of tuples
							total_time = np.sum(_exp.time_kernel[:i+1])\
											+np.sum(_exp.time_screen[:i+1])
							total_tup += len(_exp.tuples[i])
							print(('{0:7}{1:>4d}{2:6}{3:1}{4:9}{5:>13.6e}{6:10}{7:1}{8:14}{9:03d}{10:^3}{11:02d}'
								'{12:^3}{13:02d}{14:12}{15:1}{16:7}{17:>9d}{18:^3}{19:>6.2f}{20:^8}{21:>9d}').\
									format('',i+1,'','|','',_exp.energy_tot[i],\
										'','|','',int(total_time//3600),':',\
										int((total_time-(total_time//3600)*3600.)//60),':',\
										int(total_time-(total_time//3600)*3600.\
										-((total_time-(total_time//3600)*3600.)//60)*60.),\
										'','|','',len(_exp.tuples[i]),'/',\
										(float(len(_exp.tuples[i])) / \
										float(_exp.theo_work[i]))*100.00,'--',total_tup))
						print(self.divider_str+'\n\n')
				#
				return
	
	
		def abs_energy(self, _mol, _calc, _exp):
				""" plot absolute energy """
				# set seaborn
				sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
				# set 1 plot
				fig, ax = plt.subplots()
				# set title
				ax.set_title('Total '+_calc.exp_model+' correlation energy')
				# plot results
				ax.plot(list(range(1,len(_exp.energy_tot)+1)),
						np.asarray(_exp.energy_tot), marker='x', linewidth=2,
						linestyle='-', label='BG('+_calc.exp_model+')')
				# set x limits
				ax.set_xlim([0.5, self.u_limit + 0.5])
				# turn off x-grid
				ax.xaxis.grid(False)
				# set labels
				ax.set_xlabel('Expansion order')
				ax.set_ylabel('Correlation energy (in Hartree)')
				# force integer ticks on x-axis
				ax.xaxis.set_major_locator(MaxNLocator(integer=True))
				# despine
				sns.despine()
				# make insert
				with sns.axes_style("whitegrid"):
					# define frame
					insert = plt.axes([.35, .50, .50, .30], frameon=True)
					# plot results
					insert.plot(list(range(2,len(_exp.energy_tot)+1)),
								np.asarray(_exp.energy_tot[1:]), marker='x',
								linewidth=2, linestyle='-')
					# set x limits
					plt.setp(insert, xticks=list(range(3,len(_exp.energy_tot)+1)))
					insert.set_xlim([2.5, len(_exp.energy_tot)+0.5])
					# set number of y ticks
					insert.locator_params(axis='y', nbins=6)
					# set y limits
					insert.set_ylim([_exp.energy_tot[-1] - 0.01,
										_exp.energy_tot[-1] + 0.01])
					# turn off x-grid
					insert.xaxis.grid(False)
				# set legends
				ax.legend(loc=1)
				# save plot
				plt.savefig(self.out_dir+'/abs_energy_plot.pdf',
							bbox_inches = 'tight', dpi=1000)
				#
				return


		def n_tuples(self, _mol, _calc, _exp):
				""" plot number of tuples """
				# set seaborn
				sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
				# set 1 plot
				fig, ax = plt.subplots()
				# set title
				ax.set_title('Total number of '+_calc.exp_model+' tuples')
				# init prim list
				prim = []
				# set prim list
				for i in range(self.u_limit):
					if (i < len(_exp.tuples)):
						prim.append(len(_exp.tuples[i]))
					else:
						prim.append(0)
				# plot results
				sns.barplot(list(range(1, self.u_limit+1)),
							_exp.theo_work,palette='Greens',
							label='Theoretical number', log=True)
				sns.barplot(list(range(1, self.u_limit+1)),
							prim,palette='Blues_r',
							label='BG('+_calc.exp_model+') expansion', log=True)
				# turn off x-grid
				ax.xaxis.grid(False)
				# set x- and y-limits
				if (_calc.exp_type == 'occupied'):
					ax.set_xlim([-0.5,(_mol.nocc - _mol.ncore) - 0.5])
				else:
					ax.set_xlim([-0.5,_mol.nvirt - 0.5])
				ax.set_ylim(bottom=0.7)
				# set x-ticks
				if (self.u_limit < 8):
					ax.set_xticks(list(range(self.u_limit)))
					ax.set_xticklabels(list(range(1, self.u_limit+1)))
				else:
					ax.set_xticks(list(range(0, self.u_limit, self.u_limit // 8)))
					ax.set_xticklabels(list(range(1, self.u_limit+1, self.u_limit // 8)))
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
				plt.savefig(self.out_dir+'/n_tuples_plot.pdf',
							bbox_inches = 'tight', dpi=1000)
				#
				return


