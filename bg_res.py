#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_res.py: summary print and plotting utilities for Bethe-Goldstone correlation calculations."""

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
				self.header_str = '{0:^139}'.format('-'*44)
				# upper limit
				if (_calc.exp_type in ['occupied','combined']):
					self.u_limit = _mol.nocc - _mol.ncore
				else:
					self.u_limit = _mol.nvirt 
				# modify reference print out
				if (_calc.exp_ref['METHOD'] == 'CASCI'):
					self.exp_ref = 'CASCI('+str(_calc.ne_act)+'e,'+str(_calc.no_act)+'o)'
				else:
					self.exp_ref = 'HF'
				# modify base print out
				self.exp_base = _calc.exp_base['METHOD']
				# modify orbital print out
				if (_calc.exp_occ == 'CAN'):
					self.exp_occ = 'canonical'
				elif (_calc.exp_occ == 'NO'):
					self.exp_occ = 'natural'
				elif (_calc.exp_occ == 'PM'):
					self.exp_occ = 'pipek-mezey'
				elif (_calc.exp_occ == 'FB'):
					self.exp_occ = 'foster-boys'
				elif (_calc.exp_occ == 'IBO-1'):
					self.exp_occ = 'intrin. bond'
				elif (_calc.exp_occ == 'IBO-2'):
					self.exp_occ = 'intrin. bond'
				if (_calc.exp_virt == 'CAN'):
					self.exp_virt = 'canonical'
				elif (_calc.exp_virt == 'NO'):
					self.exp_virt = 'natural'
				elif (_calc.exp_virt == 'PM'):
					self.exp_virt = 'pipek-mezey'
				elif (_calc.exp_virt == 'FB'):
					self.exp_virt = 'foster-boys'
				elif (_calc.exp_virt == 'DNO'):
					self.exp_virt = 'dist. natural'
				# modify FC print out
				if (_mol.frozen):
					self.frozen = 'true'
				else:
					self.frozen = 'false'
				# modify symmetry print out
				self.hf_symmetry = _mol.hf_symmetry.lower()
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
						print('{0:^138}'.format('results'))
						print(self.header_str+'\n')
						print(self.divider_str)
						print('{0:14}{1:21}{2:11}{3:1}{4:12}{5:21}{6:11}{7:1}{8:13}{9:}'.\
								format('','molecular information','','|','',\
									'expansion information','','|','','calculation information'))
						print(self.divider_str)
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<12s}{6:3}{7:1}{8:8}{9:10}{10:10}{11:1}'
							'{12:2}{13:<11s}{14:2}{15:1}{16:7}{17:21}{18:2}{19:1}{20:2}{21:<2d}{22:^3}{23:<d}').\
								format('','basis set','','=','',_mol.basis,\
									'','|','','exp. model','','=','',_calc.exp_model['METHOD'],\
									'','|','','# mpi masters / slaves','','=','',\
									_mpi.num_local_masters + 1,'/',_mpi.global_size - (_mpi.num_local_masters + 1)))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<5}{6:10}{7:1}{8:8}{9:14}{10:6}{11:1}'
							'{12:2}{13:<11s}{14:2}{15:1}{16:7}{17:10}{18:14}{19:1}{20:1}{21:.6f}').\
								format('','frozen core','','=','',self.frozen,\
									'','|','','exp. reference','','=','',self.exp_ref,\
									'','|','','HF energy','','=','',_calc.hf_e_tot))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<2d}{6:^3}{7:<4d}{8:6}{9:1}{10:8}{11:10}{12:10}'
							'{13:1}{14:2}{15:<11s}{16:2}{17:1}{18:7}{19:18}{20:6}{21:1}{22:1}{23:.6f}').\
								format('','# occ. / virt.','','=','',_mol.nocc-_mol.ncore,'/',_mol.nvirt,\
									'','|','','exp. base','','=','',self.exp_base,\
									'','|','','reference energy','','=','',_calc.ref_e_tot))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:10}{10:10}'
							'{11:1}{12:2}{13:<11s}{14:2}{15:1}{16:7}{17:18}{18:6}{19:1}{20:1}{21:.6f}').\
								format('','orbs. (occ.)','','=','',self.exp_occ,\
									'','|','','exp. type','','=','',_calc.exp_type,\
									'','|','','base model energy','','=','',\
									_calc.hf_e_tot + _calc.e_zero))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:12}{10:8}{11:1}{12:2}'
							'{13:<6.2f}{14:2}{15:5}{16:1}{17:7}{18:18}{19:6}{20:1}{21:1}{22:.6f}').\
								format('','orbs. (virt.)','','=','',self.exp_virt,\
									'','|','','exp. thres.','','=','',_calc.exp_thres,' %',\
									'','|','','final total energy','','=','',_calc.hf_e_tot + _exp.energy_tot[-1]))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<9s}{6:6}{7:1}{8:8}{9:14}{10:6}{11:1}{12:2}'
							'{13:<5.2e}{14:5}{15:1}{16:7}{17:16}{18:8}{19:1}{20:2}{21:.2e}').\
								format('','HF symmetry','','=','',self.hf_symmetry,\
									'','|','','energy thres.','','=','',_calc.energy_thres,\
									'','|','','final abs. conv.','','=','',\
									np.abs(_exp.energy_tot[-1] - _exp.energy_tot[-2])))
						print(self.divider_str)
						print(self.fill_str)
						print(self.divider_str)
						print(('{0:6}{1:9}{2:2}{3:1}{4:7}{5:18}{6:7}{7:1}'
							'{8:7}{9:26}{10:6}{11:1}{12:6}{13:}').\
								format('','MBE order','','|','','total corr. energy',\
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
							print(('{0:7}{1:>4d}{2:6}{3:1}{4:9}{5:>13.5e}{6:10}{7:1}{8:14}{9:03d}{10:^3}{11:02d}'
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
				ax.set_title('Total '+_calc.exp_model['METHOD']+' correlation energy')
				# plot results
				ax.plot(list(range(1,len(_exp.energy_tot)+1)),
						np.asarray(_exp.energy_tot), marker='x', linewidth=2,
						linestyle='-', label='MBE-'+_calc.exp_model['METHOD'])
				# set x limits
				ax.set_xlim([0.5, self.u_limit + 0.5])
				# turn off x-grid
				ax.xaxis.grid(False)
				# set labels
				ax.set_xlabel('Expansion order')
				ax.set_ylabel('Correlation energy (in Hartree)')
				# force integer ticks on x-axis
				ax.xaxis.set_major_locator(MaxNLocator(integer=True))
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
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
				ax.set_title('Total number of '+_calc.exp_model['METHOD']+' tuples')
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
							label='MBE-'+_calc.exp_model['METHOD']+' expansion', log=True)
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


