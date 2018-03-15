#!/usr/bin/env python
# -*- coding: utf-8 -*

""" results.py: summary and plotting module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
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
_out = os.getcwd()+'/output'
_divider_str = '{0:^143}'.format('-'*137)
_fill_str = '{0:^143}'.format('|'*137)


def main(mpi, mol, calc, exp):
		""" summary printing and plotting """
		# setup
		info = {}
		info['basis'], info['mult'], info['ref'], info['base'], info['typ_prot'], \
			info['system'], info['frozen'], info['active'], info['occ'], info['virt'], \
			info['mpi'], info['thres'], info['symm'], info['conv'] = _setup(mpi, mol, calc, exp)
		# results
		_table(info, mol, calc, exp)
		# plot
		_plot(calc, exp)


def _setup(mpi, mol, calc, exp):
		""" init parameters """
		basis = _basis(mol)
		mult = _mult(mol)
		ref = _ref(mol, calc)
		base = _base(calc)
		typ_prot = _typ_prot(calc)
		system = _system(mol, calc)
		frozen = _frozen(mol)
		active = _active(calc)
		occ, virt = _orbs(calc)
		mpi = _mpi(mpi)
		thres = _thres(calc)
		symm = _symm(mol, calc)
		conv = _conv(exp)
		return basis, mult, ref, base, typ_prot, system, frozen, \
				active, occ, virt, mpi, thres, symm, conv


def _table(info, mol, calc, exp):
		""" print results """
		# write summary to results.out
		with open(_out+'/results.out','a') as f:
			with contextlib.redirect_stdout(f):
				print(_divider_str)
				print(_header_1())
				print(_divider_str)
				print(_first_row(info, calc))
				print(_second_row(info, calc))
				print(_third_row(info, calc))
				print(_fourth_row(info, calc, exp))
				print(_fifth_row(info, mol, calc))
				print(_sixth_row(info))
				print(_divider_str); print(_fill_str); print(_divider_str)
				print(_header_2())
				print(_divider_str)
				for i in _orders(calc, exp): print(i)
				print(_divider_str+'\n\n')
	
	
def _plot(calc, exp):
		""" plot results """
		# plot correlation energy
		_energy(calc, exp)
		# plot maximal increments
		_increments(calc, exp)


def _basis(mol):
		""" modify basis print """
		if isinstance(mol.basis, str):
			return mol.basis
		elif isinstance(mol.basis, dict):
			for i, val in enumerate(mol.basis.items()):
				if i == 0:
					basis = val[1]
				else:
					basis += '/'+val[1]
			return basis


def _mult(mol):
		""" modify mult print """
		if mol.spin == 0:
			return 'singlet'
		elif mol.spin == 1:
			return 'doublet'
		elif mol.spin == 2:
			return 'triplet'
		elif mol.spin == 3:
			return 'quartet'
		elif mol.spin == 4:
			return 'quintet'
		else:
			return '{0:}'.format(mol.spin+1)


def _ref(mol, calc):
		""" modify ref print """
		if calc.ref['METHOD'] == 'HF':
			if mol.spin == 0:
				return 'RHF'
			else:
				return 'ROHF'
		else:
			return calc.ref['METHOD']


def _base(calc):
		""" modify base print """
		if calc.base['METHOD'] is None:
			return 'none'
		else:
			return calc.base['METHOD']


def _typ_prot(calc):
		""" modify type / protocol print """
		typ_prot = '{0:} / {1:}'.format(calc.typ, calc.protocol)
		return typ_prot


def _system(mol, calc):
		""" modify system size print """
		system = '{0:} e / {1:} o'.format(mol.nelectron - 2*mol.ncore, len(calc.ref_space) + len(calc.exp_space))
		return system


def _frozen(mol):
		""" modify frozen core print """
		if mol.frozen:
			return 'true'
		else:
			return 'false'


def _active(calc):
		""" modify active space print """
		if calc.ref['METHOD'] == 'HF':
			return 'none'
		else:
			return '{0:} e / {1:} o'.format(calc.ne_act[0] + calc.ne_act[1], calc.no_act)


def _orbs(calc):
		""" modify orbital print """
		if calc.occ == 'REF':
			occ = 'canonical'
		elif calc.occ == 'NO':
			occ = 'natural'
		elif calc.occ == 'PM':
			occ = 'pipek-mezey'
		elif calc.occ == 'FB':
			occ = 'foster-boys'
		elif calc.occ == 'IBO-1':
			occ = 'intrin. bond'
		elif calc.occ == 'IBO-2':
			occ = 'intrin. bond'
		if calc.virt == 'REF':
			virt = 'canonical'
		elif calc.virt == 'NO':
			virt = 'natural'
		elif calc.virt == 'PM':
			virt = 'pipek-mezey'
		elif calc.virt == 'FB':
			virt = 'foster-boys'
		elif calc.virt == 'DNO':
			virt = 'dist. natural'
		return occ, virt


def _mpi(mpi):
		""" modify mpi print """
		return '{0:} / {1:}'.format(mpi.num_local_masters+1, mpi.global_size-(mpi.num_local_masters+1))


def _thres(calc):
		""" modify threshold print """
		return '{0:.0e} / {1:<.1f}'.format(calc.thres, calc.relax)


def _symm(mol, calc):
		""" modify symmetry print """
		if calc.model['METHOD'] in ['SCI','FCI']:
			return symm.addons.irrep_id2name(mol.symmetry, calc.wfnsym)+' ('+mol.symmetry+')'
		else:
			return 'unknown'


def _conv(exp):
		""" modify convergence print """
		if len(exp.energy['tot']) == 1:
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
					'','|','','expansion model','','=','',calc.model['METHOD'],\
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
					'','|','','base model','','=','',info['base'],\
					'','|','','final MBE energy','','=','',\
					exp.energy['tot'][-1] + calc.energy['ref'] + \
					(calc.energy['base'] + (calc.energy['hf'] - calc.energy['ref_base'])))


def _fifth_row(info, mol, calc):
		""" fifth row in table """
		return ('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:8}{9:16}{10:2}{11:1}{12:2}'
			'{13:<13s}{14:2}{15:1}{16:7}{17:20}{18:4}{19:1}{20:2}{21:<s}').\
				format('','occupied orbitals','','=','',info['occ'],\
					'','|','','type / protocol','','=','',info['typ_prot'],\
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
			total_time = np.sum(exp.time['mbe'][:i+1])\
							+np.sum(exp.time['screen'][:i+1])
			total_tup += len(exp.tuples[i])
			orders.append(('{0:7}{1:>4d}{2:6}{3:1}{4:9}{5:>13.5e}{6:10}{7:1}{8:14}{9:03d}{10:^3}{11:02d}'
				'{12:^3}{13:02d}{14:12}{15:1}{16:9}{17:>9d}{18:8}{19:1}{20:6}{21:>9d}').\
					format('',i+exp.start_order,'','|','',\
						exp.energy['tot'][i] + \
						(calc.energy['ref'] - calc.energy['hf']) - \
						(calc.energy['ref_base'] - calc.energy['hf']) + \
						calc.energy['base'],\
						'','|','',int(total_time//3600),':',\
						int((total_time-(total_time//3600)*3600.)//60),':',\
						int(total_time-(total_time//3600)*3600.\
						-((total_time-(total_time//3600)*3600.)//60)*60.),\
						'','|','',len(exp.tuples[i]),'','|','',total_tup))
		return orders


def _energy(calc, exp):
		""" plot correlation energy """
		# set seaborn
		sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 1 plot
		fig, ax = plt.subplots()
		# array of total correlation energy
		corr = exp.energy['tot'] + (calc.energy['ref'] - calc.energy['hf']) - \
				(calc.energy['ref_base'] - calc.energy['hf']) + calc.energy['base']
		# plot results
		ax.plot(np.asarray(list(range(exp.start_order, len(exp.energy['tot'])+exp.start_order))), \
				corr, marker='x', linewidth=2, color='green', \
				linestyle='-', label='MBE-'+calc.model['METHOD'])
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
		plt.savefig(_out+'/energy.pdf', bbox_inches = 'tight', dpi=1000)


def _increments(calc, exp):
		""" plot maximal increments """
		# set seaborn
		sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 1 plot
		fig, ax = plt.subplots()
		# array of maximal increments
		max_val = np.empty_like(exp.energy['inc'])
		for i in range(len(max_val)):
			max_idx = np.argmax(np.abs(exp.energy['inc'][i]))
			max_val[i] = np.abs(exp.energy['inc'][i][max_idx])
		# plot results
		ax.semilogy(np.asarray(list(range(exp.start_order, len(exp.energy['tot'])+exp.start_order))), \
				max_val, marker='x', linewidth=2, color='red', \
				linestyle='-', label='MBE-'+calc.model['METHOD'])
		# set x limits
		ax.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax.xaxis.grid(False)
		# set labels
		ax.set_xlabel('Expansion order')
		ax.set_ylabel('Maximal increment (in Hartree)')
		# force integer ticks on x-axis
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
		# despine
		sns.despine()
		# set legends
		ax.legend(loc=1)
		# tight layout
		plt.tight_layout()
		# save plot
		plt.savefig(_out+'/increments.pdf', bbox_inches = 'tight', dpi=1000)


