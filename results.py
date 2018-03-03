#!/usr/bin/env python
# -*- coding: utf-8 -*

""" results.py: summary and plotting module """

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
import contextlib
import numpy as np
from pyscf import symm
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
try:
	import seaborn as sns
except ImportError:
	sys.stderr.write('\nImportError : seaborn module not found\n\n')


# summary constants
out = os.getcwd()+'/output'
divider_str = '{0:^143}'.format('-'*137)
fill_str = '{0:^143}'.format('|'*137)
header_str = '{0:^139}'.format('-'*44)


def main(mpi, mol, calc, exp):
		""" summary printing and plotting """
		# setup
		info = {}
		info['basis'], info['mult'], info['ref'], info['base'], info['system'], \
			info['frozen'], info['active'], info['occ'], info['virt'], \
			info['mpi'], info['thres'], info['symm'], info['conv'] = _setup(mpi, mol, calc, exp)
		# results
		_table(info, mol, calc, exp)
		# plot
		_plot(calc, exp)
		#
		return


def _setup(mpi, mol, calc, exp):
		""" init parameters """
		# modify basis print out
		basis = _basis(mol)
		# modify spin multiplicity print out
		mult = _mult(mol)
		# modify reference print out
		ref = _ref(mol, calc)
		# modify base print out
		base = _base(calc)
		# modify system size print out
		system = _system(mol, calc)
		# modify frozen core print out
		frozen = _frozen(mol)
		# modify active space print out
		active = _active(calc)
		# modify orbital print out
		occ, virt = _orbs(calc)
		# modify mpi print out
		mpi = _mpi(mpi)
		# modify threshold print out
		thres = _thres(calc)
		# modify symmetry print out
		symm = _symm(mol, calc)
		# modify convergence print out
		conv = _conv(exp)
		#
		return basis, mult, ref, base, system, frozen, active, \
				occ, virt, mpi, thres, symm, conv


def _table(info, mol, calc, exp):
		""" print results """
		# write summary to results.out
		with open(out+'/results.out','a') as f:
			with contextlib.redirect_stdout(f):
				print('\n\n'+header_str)
				print('{0:^138}'.format('results'))
				print(header_str+'\n'); print(divider_str)
				print(_header_1())
				print(divider_str)
				print(_first_row(info, calc))
				print(_second_row(info, calc))
				print(_third_row(info, calc))
				print(_fourth_row(info, calc, exp))
				print(_fifth_row(info, mol, calc))
				print(_sixth_row(info))
				print(divider_str); print(fill_str); print(divider_str)
				print(_header_2())
				print(divider_str)
				for i in _orders(calc, exp): print(i)
				print(divider_str+'\n\n')
		#
		return
	
	
def _plot(calc, exp):
		""" plot correlation energy """
		# set seaborn
		sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 1 plot
		fig, ax = plt.subplots()
		# plot results
		ax.plot(list(range(exp.start_order, len(exp.energy['tot'])+exp.start_order)), \
				exp.energy['tot']+calc.energy['base'], marker='x', linewidth=2, \
				linestyle='-', label='MBE-'+calc.exp_model['METHOD'])
		# set x limits
		ax.set_xlim([0.5, len(calc.exp_space) + 0.5])
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
		plt.savefig(out+'/energy.pdf', bbox_inches = 'tight', dpi=1000)
		#
		return


def _basis(mol):
		""" modify basis print """
		if (isinstance(mol.basis, str)):
			return mol.basis
		elif (isinstance(mol.basis, dict)):
			for i, val in enumerate(mol.basis.items()):
				if (i == 0):
					basis = val[1]
				else:
					basis += '/'+val[1]
			return basis


def _mult(mol):
		""" modify mult print """
		if (mol.spin == 0):
			return 'singlet'
		elif (mol.spin == 1):
			return 'doublet'
		elif (mol.spin == 2):
			return 'triplet'
		elif (mol.spin == 3):
			return 'quartet'
		elif (mol.spin == 4):
			return 'quintet'
		else:
			return '{0:}'.format(mol.spin+1)


def _ref(mol, calc):
		""" modify ref print """
		if (calc.exp_ref['METHOD'] == 'HF'):
			if (mol.spin == 0):
				return 'RHF'
			else:
				return 'ROHF'
		else:
			return calc.exp_ref['METHOD']


def _base(calc):
		""" modify base print """
		if (calc.exp_base['METHOD'] is None):
			return 'none'
		else:
			return calc.exp_base['METHOD']


def _system(mol, calc):
		""" modify system size print """
		system = '{0:} e / {1:} o'.format(mol.nelectron - 2*mol.ncore, len(calc.ref_space) + len(calc.exp_space))
		#
		return system


def _frozen(mol):
		""" modify frozen core print """
		if (mol.frozen):
			return 'true'
		else:
			return 'false'


def _active(calc):
		""" modify active space print """
		if (calc.exp_ref['METHOD'] == 'HF'):
			return 'none'
		else:
			return '{0:} e / {1:} o'.format(calc.exp_ref['NELEC'][0]+calc.exp_ref['NELEC'][1], \
													len(calc.exp_ref['ACTIVE']))


def _orbs(calc):
		""" modify orbital print """
		if (calc.exp_occ == 'REF'):
			occ = 'canonical'
		elif (calc.exp_occ == 'NO'):
			occ = 'natural'
		elif (calc.exp_occ == 'PM'):
			occ = 'pipek-mezey'
		elif (calc.exp_occ == 'FB'):
			occ = 'foster-boys'
		elif (calc.exp_occ == 'IBO-1'):
			occ = 'intrin. bond'
		elif (calc.exp_occ == 'IBO-2'):
			occ = 'intrin. bond'
		if (calc.exp_virt == 'REF'):
			virt = 'canonical'
		elif (calc.exp_virt == 'NO'):
			virt = 'natural'
		elif (calc.exp_virt == 'PM'):
			virt = 'pipek-mezey'
		elif (calc.exp_virt == 'FB'):
			virt = 'foster-boys'
		elif (calc.exp_virt == 'DNO'):
			virt = 'dist. natural'
		#
		return occ, virt


def _mpi(mpi):
		""" modify mpi print """
		return '{0:} / {1:}'.format(mpi.num_local_masters+1, mpi.global_size-(mpi.num_local_masters+1))


def _thres(calc):
		""" modify threshold print """
		return '{0:.0e} / {1:<.1f}'.format(calc.exp_thres, calc.exp_relax)


def _symm(mol, calc):
		""" modify symmetry print """
		return symm.addons.irrep_id2name(mol.symmetry, calc.wfnsym)+' ('+mol.symmetry+')'


def _conv(exp):
		""" modify convergence print """
		if (len(exp.energy['tot']) == 1):
			return 0.0
		else:
			return np.abs(exp.energy['tot'][-1] - exp.energy['tot'][-2])


def _header_1():
		""" table header 1 """
		return '{0:14}{1:21}{2:12}{3:1}{4:12}{5:21}{6:11}{7:1}{8:13}{9:}'.\
				format('','molecular information','','|','',\
					'expansion information','','|','','calculation information')


def _first_row(info, calc):
		""" first row in table """
		return ('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
			'{13:<13s}{14:2}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}').\
				format('','basis set','','=','',info['basis'],\
					'','|','','expansion model','','=','',calc.exp_model['METHOD'],\
					'','|','','mpi masters / slaves','','=','',info['mpi'])


def _second_row(info, calc):
		""" second row in table """
		return ('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
			'{13:<13s}{14:2}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}').\
				format('','spin multiplicity','','=','',info['mult'],\
					'','|','','reference funct.','','=','',info['ref'],\
					'','|','','Hartree-Fock energy','','=','',calc.energy['hf'])


def _third_row(info, calc):
		""" third row in table """
		return ('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
			'{13:<13s}{14:2}{15:1}{16:7}{17:18}{18:6}{19:1}{20:1}{21:.6f}').\
				format('','system size','','=','',info['system'],\
					'','|','','cas size','','=','',info['active'],\
					'','|','','base model energy','','=','',calc.energy['hf']+calc.energy['base'])


def _fourth_row(info, calc, exp):
		""" fourth row in table """
		return ('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
			'{13:<13s}{14:2}{15:1}{16:7}{17:18}{18:6}{19:1}{20:1}{21:.6f}').\
				format('','frozen core','','=','',info['frozen'],\
					'','|','','expansion base','','=','',info['base'],\
					'','|','','final MBE energy','','=','',\
					exp.energy['tot'][-1]+calc.energy['hf']+calc.energy['base'])


def _fifth_row(info, mol, calc):
		""" fifth row in table """
		return ('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
			'{13:<13s}{14:2}{15:1}{16:7}{17:20}{18:4}{19:1}{20:2}{21:<s}').\
				format('','occupied orbitals','','=','',info['occ'],\
					'','|','','expansion type','','=','',calc.exp_type,\
					'','|','','wave funct. symmetry','','=','',info['symm'])


def _sixth_row(info):
		""" sixth row in table """
		return ('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<9s}{6:6}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
				'{13:<13s}{14:2}{15:1}{16:7}{17:16}{18:8}{19:1}{20:2}{21:.3e}').\
					format('','virtual orbitals','','=','',info['virt'],\
						'','|','','thres. / relax.','','=','',info['thres'],\
						'','|','','final abs. conv.','','=','',info['conv'])


def _header_2():
		""" table header 2 """
		return ('{0:6}{1:9}{2:2}{3:1}{4:7}{5:18}{6:7}{7:1}'
			'{8:7}{9:26}{10:6}{11:1}{12:5}{13:15}{14:5}{15:1}{16:5}{17:}').\
				format('','MBE order','','|','','correlation energy',\
					'','|','','total time (HHH : MM : SS)',\
					'','|','','number of calcs.',\
					'','|','','total number')


def _orders(calc, exp):
		""" order table """
		orders = []
		# loop over orders
		total_tup = 0
		for i in range(len(exp.energy['tot'])):
			# sum up total time and number of tuples
			total_time = np.sum(exp.time_mbe[:i+1])\
							+np.sum(exp.time_screen[:i+1])
			total_tup += len(exp.tuples[i])
			orders.append(('{0:7}{1:>4d}{2:6}{3:1}{4:9}{5:>13.5e}{6:10}{7:1}{8:14}{9:03d}{10:^3}{11:02d}'
				'{12:^3}{13:02d}{14:12}{15:1}{16:9}{17:>9d}{18:8}{19:1}{20:6}{21:>9d}').\
					format('',i+exp.start_order,'','|','',\
						exp.energy['tot'][i]+calc.energy['base'],\
						'','|','',int(total_time//3600),':',\
						int((total_time-(total_time//3600)*3600.)//60),':',\
						int(total_time-(total_time//3600)*3600.\
						-((total_time-(total_time//3600)*3600.)//60)*60.),\
						'','|','',len(exp.tuples[i]),'','|','',total_tup))
		#
		return orders

