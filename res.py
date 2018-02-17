#!/usr/bin/env python
# -*- coding: utf-8 -*

""" res.py: summary and plotting class """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import sys
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
				# modify occ-virt print out
				self.occ_virt = '{0:} / {1:}'.format(_mol.nocc-_mol.ncore, _mol.nvirt)
				# modify reference print out
				if (_calc.exp_ref['METHOD'] == 'HF'):
					if (_mol.spin == 0):
						self.exp_ref = 'RHF'
					else:
						self.exp_ref = 'ROHF'
				else:
					self.exp_ref = _calc.exp_ref['METHOD']
				# modify base print out
				if (_calc.exp_base['METHOD'] is None):
					self.exp_base = 'none'
				else:
					self.exp_base = _calc.exp_base['METHOD']
				# modify active space print out
				if (_calc.exp_ref['METHOD'] == 'HF'):
					self.active = 'none'
				else:
					self.active = '{0:} e / {1:} o'.format(_calc.exp_ref['NELEC'][0]+_calc.exp_ref['NELEC'][1], \
															len(_calc.exp_ref['ACTIVE']))
				# modify orbital print out
				if (_calc.exp_occ == 'REF'):
					self.exp_occ = 'reference'
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
				if (_calc.exp_virt == 'REF'):
					self.exp_virt = 'reference'
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
				# modify mpi print out
				self.mpi = '{0:} / {1:}'.format(_mpi.num_local_masters+1, _mpi.global_size-(_mpi.num_local_masters+1))
				# modify threshold print out
				self.thres = '{0:.0e} / {1:.0f}'.format(_calc.exp_thres, _calc.exp_relax)
				#
				return


		def main(self, _mpi, _mol, _calc, _exp):
				""" main driver for summary printing and plotting """
				# results
				self.results(_mpi, _mol, _calc, _exp)
				# plot total energy
				self.abs_energy(_calc, _exp)
				# plot distributions of energy increments
#				self.dist_energy(_calc, _exp)
				#
				return


		def results(self, _mpi, _mol, _calc, _exp):
				""" print results """
				# modify final convergence print out
				if (len(_exp.energy['tot']) == 1):
					self.final_conv = 0.0
				else:
					self.final_conv = np.abs(_exp.energy['tot'][-1] - _exp.energy['tot'][-2])
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
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<12s}{6:3}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
							'{13:<13s}{14:2}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}').\
								format('','basis set','','=','',_mol.basis,\
									'','|','','expansion model','','=','',_calc.exp_model['METHOD'],\
									'','|','','mpi masters / slaves','','=','',self.mpi))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<5}{6:10}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
							'{13:<13s}{14:2}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}').\
								format('','frozen core','','=','',self.frozen,\
									'','|','','reference funct.','','=','',self.exp_ref,\
									'','|','','Hartree-Fock energy','','=','',_calc.energy['hf']))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
							'{13:<13s}{14:2}{15:1}{16:7}{17:18}{18:6}{19:1}{20:1}{21:.6f}').\
								format('','occ. / virt.','','=','',self.occ_virt,\
									'','|','','active space','','=','',self.active,\
									'','|','','base model energy','','=','',_calc.energy['hf']+_calc.energy['base']))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
							'{13:<13s}{14:2}{15:1}{16:7}{17:18}{18:6}{19:1}{20:1}{21:.6f}').\
								format('','orbs. (occ.)','','=','',self.exp_occ,\
									'','|','','expansion base','','=','',self.exp_base,\
									'','|','','final total energy','','=','',\
									_exp.energy['tot'][-1]+_calc.energy['hf']+_calc.energy['base']))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
							'{13:<13s}{14:2}{15:1}{16:7}{17:18}{18:6}{19:1}{20:1}{21:.6f}').\
								format('','orbs. (virt.)','','=','',self.exp_virt,\
									'','|','','expansion type','','=','',_calc.exp_type,\
									'','|','','final corr. energy','','=','',_exp.energy['tot'][-1]))
						print(('{0:11}{1:14}{2:3}{3:1}{4:2}{5:<9s}{6:6}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
							'{13:<13s}{14:2}{15:1}{16:7}{17:16}{18:8}{19:1}{20:2}{21:.2e}').\
								format('','comp. symmetry','','=','',_mol.comp_symmetry,\
									'','|','','thres. / relax.','','=','',self.thres,\
									'','|','','final abs. conv.','','=','',self.final_conv))
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
						for i in range(len(_exp.energy['tot'])):
							# sum up total time and number of tuples
							total_time = np.sum(_exp.time_mbe[:i+1])\
											+np.sum(_exp.time_screen[:i+1])
							total_tup += len(_exp.tuples[i])
							print(('{0:7}{1:>4d}{2:6}{3:1}{4:9}{5:>13.5e}{6:10}{7:1}{8:14}{9:03d}{10:^3}{11:02d}'
								'{12:^3}{13:02d}{14:12}{15:1}{16:7}{17:>9d}{18:^3}{19:>6.2f}{20:^8}{21:>9d}').\
									format('',i+_exp.start_order,'','|','',\
										_exp.energy['tot'][i]+_calc.energy['base'],\
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
	
	
		def abs_energy(self, _calc, _exp):
				""" plot absolute energy """
				# set seaborn
				sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
				# set 1 plot
				fig, ax = plt.subplots()
				# plot results
				ax.plot(list(range(_exp.start_order, len(_exp.energy['tot'])+_exp.start_order)), \
						_exp.energy['tot']+_calc.energy['base'], marker='x', linewidth=2, \
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
				# set legends
				ax.legend(loc=1)
				# tight layout
				plt.tight_layout()
				# save plot
				plt.savefig(self.out_dir+'/abs_energy_plot.pdf',
							bbox_inches = 'tight', dpi=1000)
				#
				return


		def dist_energy(self, _calc, _exp):
				""" plot distribution of energy increments """
				# set seaborn
				sns.set(style='white', palette='Set2', font='DejaVu Sans')
				# set end index
				end = len(_exp.energy['inc'])
				if (len(_exp.energy['inc'][-1]) == 1): end -= 1
				# set number of subplots
				h_length = end // 2
				if (end % 2 != 0): h_length += 1
				fig, ax = plt.subplots(h_length, 2)
				# set figure size
				fig.set_size_inches([8.268,11.693])
				# plot results
				for i in range(end):
					# plot threshold interval
					if (_calc.exp_base['METHOD'] != 'HF'):
						if (i >= 1):
							thres = _calc.exp_thres * _calc.exp_relax ** (i)
							ax.flat[i].axvspan(0.0-thres, 0.0+thres, facecolor='yellow', alpha=0.4)
					else:
						if (i >= 2):
							thres = _calc.exp_thres * _calc.exp_relax ** (i-1)
							ax.flat[i].axvspan(0.0-thres, 0.0+thres, facecolor='yellow', alpha=0.4)
					# plot data
					sns.distplot(_exp.energy['inc'][i], hist=False, color='red', \
									kde_kws={'shade': True}, ax=ax.flat[i])
					# set title
					ax.flat[i].set_title('k = {0:} | N = {1:} | E = {2:.1e}'.format(i+1, len(_exp.energy['inc'][i]), \
																							np.sum(_exp.energy['inc'][i])), size=10)
					# val_max
					val_max = _exp.energy['inc'][i][np.argmax(np.abs(_exp.energy['inc'][i]))]
					# format x-axis
					ax.flat[i].set_xticks([0.0,val_max])
					ax.flat[i].set_xticklabels(['0.0', '{0:.1e}'.format(val_max)])
					# remove y-axis
					plt.setp(ax.flat[i], yticks=[])
					# mark zero and max
					ax.flat[i].axvline(x=0.0, color='black', ymax=0.25)
					ax.flat[i].axvline(x=val_max, color='blue', ymax=0.25)
				# despine
				sns.despine(left=True)
				if (end % 2 != 0):
					sns.despine(ax=ax.flat[-1], left=True, bottom=True)
					plt.setp(ax.flat[-1], xticks=[], yticks=[])
				# tight layout
				plt.tight_layout()
				# save plot
				plt.savefig(self.out_dir+'/dist_energy_plot.pdf',
							bbox_inches = 'tight', dpi=1000)
				#
				return

