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
	SNS_FOUND = True
except (ImportError, OSError):
	pass
	SNS_FOUND = False

# results parameters
OUT = os.getcwd()+'/output'
DIVIDER = '{0:^143}'.format('-'*137)
FILL = '{0:^143}'.format('|'*137)


def main(mpi, mol, calc, exp):
		""" printing and plotting of results """
		# convert final results to numpy arrays
		if calc.target['energy']:
			exp.prop['energy']['tot'] = np.asarray(exp.prop['energy']['tot'])
		if calc.target['excitation']:
			exp.prop['excitation']['tot'] = np.asarray(exp.prop['excitation']['tot'])
		if calc.target['dipole']:
			exp.prop['dipole']['tot'] = np.asarray(exp.prop['dipole']['tot'])
		if calc.target['trans']:
			exp.prop['trans']['tot'] = np.asarray(exp.prop['trans']['tot'])
		# setup
		info = {}
		info['model_type'], info['basis'], info['state'], info['ref'], info['base'], info['prot'], \
			info['system'], info['frozen'], info['hubbard'], info['active'], \
			info['occ'], info['virt'], info['mpi'], info['thres'], info['symm'], \
			info['energy'], info['excitation'], \
			info['dipole'], info['nuc_dipole'], info['trans'] = _setup(mpi, mol, calc, exp)
		info['final_order'] = exp.time['mbe'].size
		# results
		_table(info, mpi, mol, calc, exp)
		# plot
		_plot(info, calc, exp)


def _setup(mpi, mol, calc, exp):
		""" init parameters """
		model_type = _model_type(calc)
		basis = _basis(mol)
		state = _state(mol, calc)
		ref = _ref(mol, calc)
		base = _base(calc)
		prot = _prot(calc)
		system = _system(mol, calc)
		frozen = _frozen(mol)
		if mol.atom:
			hubbard = None
		else:
			hubbard = _hubbard(mol)
		active = _active(calc)
		occ, virt = _orbs(calc)
		mpi = _mpi(mpi, calc)
		thres = _thres(calc)
		symm = _symm(mol, calc)
		if calc.target['energy']:
			energy = _energy(calc, exp)
		else:
			energy = None
		if calc.target['excitation']:
			excitation = _excitation(calc, exp)
		else:
			excitation = None
		if calc.target['dipole']:
			dipole, nuc_dipole = _dipole(mol, calc, exp)
		else:
			dipole = nuc_dipole = None
		if calc.target['trans']:
			trans = _trans(mol, calc, exp)
		else:
			trans = None
		return model_type, basis, state, ref, base, prot, system, frozen, \
				hubbard, active, occ, virt, mpi, thres, symm, \
				energy, excitation, dipole, nuc_dipole, trans


def _table(info, mpi, mol, calc, exp):
		""" print results """
		# write results to results.out
		with open(OUT+'/results.out','a') as f:
			with contextlib.redirect_stdout(f):
				_summary_prt(info, mol, calc, exp)
				_timings_prt(info, calc, exp)
				if calc.target['energy']:
					_energy_prt(info, calc, exp)
				if calc.target['excitation']:
					_excitation_prt(info, calc, exp)
				if calc.target['dipole']:
					_dipole_prt(info, calc, exp)
				if calc.target['trans']:
					_trans_prt(info, calc, exp)
	

def _plot(info, calc, exp):
		""" plot results """
		# plot MBE quantitites
		if calc.target['energy']:
			_energies_plot(info, calc, exp)
		if calc.target['excitation']:
			_excitation_plot(info, calc, exp)
		if calc.target['dipole']:
			_dipole_plot(info, calc, exp)
		if calc.target['trans']:
			_trans_plot(info, calc, exp)
			_osc_strength_plot(info, calc, exp)


def _model_type(calc):
		""" model / type print """
		return '{0:} / {1:}'.format(calc.model['method'].upper(), calc.model['type'])


def _basis(mol):
		""" basis print """
		if isinstance(mol.basis, str):
			return mol.basis
		elif isinstance(mol.basis, dict):
			for i, val in enumerate(mol.basis.items()):
				if i == 0:
					basis = val[1]
				else:
					basis += '/'+val[1]
			return basis


def _state(mol, calc):
		""" state print """
		string = '{:}'.format(calc.state['root'])
		if mol.spin == 0:
			string += ' / singlet'
		elif mol.spin == 1:
			string += ' / doublet'
		elif mol.spin == 2:
			string += ' / triplet'
		elif mol.spin == 3:
			string += ' / quartet'
		elif mol.spin == 4:
			string += ' / quintet'
		else:
			string += ' / {:}'.format(mol.spin+1)
		return string


def _ref(mol, calc):
		""" ref print """
		if calc.ref['method'] == 'hf':
			if mol.spin == 0:
				return 'RHF'
			else:
				return 'ROHF'
		elif calc.ref['method'] == 'casci':
			return 'CASCI'
		elif calc.ref['method'] == 'casscf':
			if calc.ref['root'] == 0:
				return 'CASSCF'
			else:
				for i in range(calc.ref['root']+1):
					if calc.ref['weights'][i] > 0.0:
						weight = '{:.2f}'.format(calc.ref['weights'][i])[-3:]
					else:
						weight = '-'
					if i == 0:
						weights = weight
					else:
						weights += '/'+weight
				return 'CASSCF('+weights+')'


def _base(calc):
		""" base print """
		if calc.base['method'] is None:
			return 'none'
		else:
			return calc.base['method'].upper()


def _prot(calc):
		""" protocol print """
		if calc.extra['filter'] is not None:
			return calc.prot['scheme']+' ('+calc.extra['filter']+')'
		else:
			return calc.prot['scheme']


def _system(mol, calc):
		""" system size print """
		return '{0:} e / {1:} o'.format(mol.nelectron - 2*mol.ncore, len(calc.ref_space) + len(calc.exp_space))


def _hubbard(mol):
		""" hubbard print """
		if mol.dim == 1:
			hubbard = ['1d / {0:}x{1:}'.format(mol.nsites, 1)]
		else:
			hubbard = ['2d / {0:}x{1:}'.format(int(np.sqrt(mol.nsites)), int(np.sqrt(mol.nsites)))]
		hubbard.append('{0:} / {1:}'.format(mol.u, mol.t))
		return hubbard


def _frozen(mol):
		""" frozen core print """
		if mol.frozen:
			return 'true'
		else:
			return 'false'


def _active(calc):
		""" active space print """
		if calc.ref['method'] == 'hf':
			return 'none'
		else:
			return '{0:} e / {1:} o'.format(calc.ne_act[0] + calc.ne_act[1], calc.no_act)


def _orbs(calc):
		""" orbital print """
		if calc.orbs['occ'] == 'can':
			occ = 'canonical'
		elif calc.orbs['occ'] == 'cisd':
			occ = 'cisd natural'
		elif calc.orbs['occ'] == 'ccsd':
			occ = 'ccsd natural'
		elif calc.orbs['occ'] == 'pm':
			occ = 'pipek-mezey'
		elif calc.orbs['occ'] == 'fb':
			occ = 'foster-boys'
		elif calc.orbs['occ'] == 'ibo-1':
			occ = 'intrin. bond'
		elif calc.orbs['occ'] == 'ibo-2':
			occ = 'intrin. bond'
		if calc.orbs['virt'] == 'can':
			virt = 'canonical'
		elif calc.orbs['virt'] == 'cisd':
			virt = 'cisd natural'
		elif calc.orbs['virt'] == 'ccsd':
			virt = 'ccsd natural'
		elif calc.orbs['virt'] == 'pm':
			virt = 'pipek-mezey'
		elif calc.orbs['virt'] == 'fb':
			virt = 'foster-boys'
		return occ, virt


def _mpi(mpi, calc):
		""" mpi print """
		return '{0:} / {1:}'.format(calc.mpi['masters'], mpi.global_size - calc.mpi['masters'])


def _thres(calc):
		""" threshold print """
		return '{0:.0e} / {1:<.1f}'.format(calc.thres['init'], calc.thres['relax'])


def _symm(mol, calc):
		""" symmetry print """
		if calc.model['method'] == 'fci':
			if mol.atom:
				string = symm.addons.irrep_id2name(mol.symmetry, calc.state['wfnsym'])+' ('+mol.symmetry+')'
				if calc.extra['lz_sym']:
					string += ' Lz'
				return string
			else:
				return 'C1 (A)'
		else:
			return 'unknown'


def _energy(calc, exp):
		""" final energies """
		return exp.prop['energy']['tot'] \
				+ calc.prop['hf']['energy'] + calc.base['energy'] \
				+ (calc.prop['ref']['energy'] - calc.base['ref'])


def _excitation(calc, exp):
		""" final energies """
		return exp.prop['excitation']['tot'] + calc.prop['ref']['excitation']


def _dipole(mol, calc, exp):
		""" final molecular dipole moments """
		# nuclear dipole moment
		charges = mol.atom_charges()
		coords  = mol.atom_coords()
		nuc_dipole = np.einsum('i,ix->x', charges, coords)
		dipole = exp.prop['dipole']['tot'] \
						+ calc.prop['hf']['dipole'] \
						+ calc.prop['ref']['dipole']
		return dipole, nuc_dipole


def _trans(mol, calc, exp):
		""" final molecular transition dipole moments """
		return exp.prop['trans']['tot'] + calc.prop['ref']['trans']


def _time(exp, comp, idx):
		""" convert time to (HHH : MM : SS) format """
		# init time
		if comp in ['mbe', 'screen']:
			time = exp.time[comp][idx]
		elif comp == 'sum':
			time = exp.time['mbe'][idx] + exp.time['screen'][idx]
		elif comp in ['tot_mbe', 'tot_screen']:
			time = np.sum(exp.time[comp[4:]])
		elif comp == 'tot_sum':
			time = np.sum(exp.time['mbe']) + np.sum(exp.time['screen'])
		hours = int(time // 3600)
		minutes = int((time - (time // 3600) * 3600.)//60)
		seconds = time - hours * 3600. - minutes * 60.
		# init time string
		time_str = ''
		if hours > 0:
			time_str += '{:}h '.format(hours)
		if minutes > 0:
			time_str += '{:}m '.format(minutes)
		time_str += '{:.2f}s'.format(seconds)
		return time_str


def _summary_prt(info, mol, calc, exp):
		""" summary table """
		print(DIVIDER)
		print('{0:14}{1:21}{2:12}{3:1}{4:12}{5:21}{6:11}{7:1}{8:13}{9:}'. \
				format('','molecular information','','|','', \
					'expansion information','','|','','calculation information'))
		print(DIVIDER)
		if mol.atom:
			print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
					'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}'. \
						format('','basis set','','=','',info['basis'], \
							'','|','','model / type','','=','',info['model_type'], \
							'','|','','mpi masters / slaves','','=','',info['mpi']))
			print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
					'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}'. \
						format('','frozen core','','=','',info['frozen'], \
							'','|','','ref. function','','=','',info['ref'], \
							'','|','','Hartree-Fock energy','','=','',calc.prop['hf']['energy']))
		else:
			print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
					'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}'. \
						format('','hubbard lattice','','=','',info['hubbard'][0], \
							'','|','','model / type','','=','',info['model_type'], \
							'','|','','mpi masters / slaves','','=','',info['mpi']))
			print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
					'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}'. \
						format('','hubbard U / t','','=','',info['hubbard'][1], \
							'','|','','ref. function','','=','',info['ref'], \
							'','|','','Hartree-Fock energy','','=','',calc.prop['hf']['energy']))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}'. \
					format('','system size','','=','',info['system'], \
						'','|','','cas size','','=','',info['active'], \
						'','|','','base model energy','','=','', \
						calc.prop['hf']['energy']+calc.base['energy']))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}'. \
					format('','state / mult.','','=','',info['state'], \
						'','|','','base model','','=','',info['base'], \
						'','|','','MBE total energy','','=','', \
						calc.prop['hf']['energy'] if info['energy'] is None else info['energy'][-1]))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}'.\
					format('','occupied orbs','','=','',info['occ'], \
						'','|','','screen. prot.','','=','',info['prot'], \
						'','|','','total time','','=','',_time(exp, 'tot_sum', -1)))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}'. \
					format('','virtual orbs','','=','',info['virt'], \
						'','|','','screen. thres.','','=','',info['thres'], \
						'','|','','wave funct. symmetry','','=','',info['symm']))
		print(DIVIDER)
		print(FILL)
		print(DIVIDER+'\n')


def _timings_prt(info, calc, exp):
		""" timings """
		print(DIVIDER[:98])
		print('{0:^98}'.format('MBE timings'))
		print(DIVIDER[:98])
		print('{0:6}{1:9}{2:2}{3:1}{4:8}{5:3}{6:8}{7:1}{8:5}{9:9}{10:5}'
				'{11:1}{12:8}{13:3}{14:8}{15:1}{16:5}{17:}'. \
				format('','MBE order','','|','','MBE','','|','','screening', \
						'','|','','sum','','|','','calculations'))
		print(DIVIDER[:98])
		calcs = 0
		for i in range(info['final_order']):
			calc_i = exp.count[i]
			calcs += calc_i
			print('{0:7}{1:>4d}{2:6}{3:1}{4:2}{5:>13s}{6:4}{7:1}{8:2}{9:>13s}{10:4}{11:1}'
					'{12:2}{13:>13s}{14:4}{15:1}{16:5}{17:>9d}'. \
					format('',i+exp.start_order, \
						'','|','',_time(exp, 'mbe', i), \
						'','|','',_time(exp, 'screen', i), \
						'','|','',_time(exp, 'sum', i), \
						'','|','',calc_i))
		print(DIVIDER[:98])
		print('{0:8}{1:5s}{2:4}{3:1}{4:2}{5:>13s}{6:4}{7:1}{8:2}{9:>13s}{10:4}{11:1}'
				'{12:2}{13:>13s}{14:4}{15:1}{16:5}{17:>9d}'. \
				format('','total', \
					'','|','',_time(exp, 'tot_mbe', -1), \
					'','|','',_time(exp, 'tot_screen', -1), \
					'','|','',_time(exp, 'tot_sum', -1), \
					'','|','',calcs))
		print(DIVIDER[:98]+'\n')


def _energy_prt(info, calc, exp):
		""" energies """
		print(DIVIDER[:66])
		string = 'MBE energy (root = {:})'.format(calc.state['root'])
		print('{0:^66}'.format(string))
		print(DIVIDER[:66])
		print('{0:6}{1:9}{2:2}{3:1}{4:5}{5:12}{6:5}{7:1}{8:4}{9:}'. \
				format('','MBE order','','|','','total energy','','|','','correlation energy'))
		print(DIVIDER[:66])
		print('{0:7}{1:>4d}{2:6}{3:1}{4:5}{5:>11.6f}{6:6}{7:1}{8:7}{9:}'. \
				format('',0,'','|','',calc.prop['hf']['energy'],'','|','','-----------'))
		print(DIVIDER[:66])
		for i in range(info['final_order']):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:5}{5:>11.6f}{6:6}{7:1}{8:7}{9:9.4e}'. \
					format('',i+exp.start_order, \
						'','|','',info['energy'][i], \
						'','|','',info['energy'][i] - calc.prop['hf']['energy']))
		print(DIVIDER[:66]+'\n')


def _energies_plot(info, calc, exp):
		""" plot MBE energy """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of MBE total energy
		energy = info['energy'].copy()
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				energy, marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
				linestyle='-', label='state {:}'.format(calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Energy (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total energy increments
		mbe = exp.prop['energy']['tot'].copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				np.abs(mbe), marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
				linestyle='-', label='state {:}'.format(calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/energy_state_{:}.pdf'.format(calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _excitation_prt(info, calc, exp):
		""" excitation energies """
		print(DIVIDER[:43])
		string = 'MBE excitation energy (root = {:})'.format(calc.state['root'])
		print('{0:^46}'.format(string))
		print(DIVIDER[:43])
		print('{0:6}{1:9}{2:2}{3:1}{4:5}{5:}'. \
				format('','MBE order','','|','','excitation energy'))
		print(DIVIDER[:43])
		for i in range(info['final_order']):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:8}{5:9.4e}'. \
					format('',i+exp.start_order, \
						'','|','',info['excitation'][i]))
		print(DIVIDER[:43]+'\n')


def _excitation_plot(info, calc, exp):
		""" plot MBE excitation energy """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of MBE total energy
		excitation = info['excitation'].copy()
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				excitation, marker='x', linewidth=2, mew=1, color='xkcd:dull blue', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Excitation energy (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total energy increments
		mbe = exp.prop['excitation']['tot'].copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				np.abs(mbe), marker='x', linewidth=2, mew=1, color='xkcd:dull blue', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/excitation_states_{:}_{:}.pdf'.format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _dipole_prt(info, calc, exp):
		""" dipole moments """
		print(DIVIDER[:82])
		string = 'MBE dipole moment (root = {:})'.format(calc.state['root'])
		print('{0:^82}'.format(string))
		print(DIVIDER[:82])
		print('{0:6}{1:9}{2:2}{3:1}{4:8}{5:25}{6:9}{7:1}{8:5}{9:}'. \
				format('','MBE order','','|','','dipole components (x,y,z)','','|','','dipole moment'))
		print(DIVIDER[:82])
		print('{0:7}{1:>4d}{2:6}{3:1}{4:4}{5:9.6f}{6:^3}{7:9.6f}{8:^3}{9:9.6f}'
			'{10:5}{11:1}{12:6}{13:9.6f}'. \
				format('',0, \
					'','|','',info['nuc_dipole'][0] - calc.prop['hf']['dipole'][0], \
					'',info['nuc_dipole'][1] - calc.prop['hf']['dipole'][1], \
					'',info['nuc_dipole'][2] - calc.prop['hf']['dipole'][2], \
					'','|','',np.linalg.norm(info['nuc_dipole'] - calc.prop['hf']['dipole'])))
		print(DIVIDER[:82])
		for i in range(info['final_order']):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:4}{5:9.6f}{6:^3}{7:9.6f}{8:^3}{9:9.6f}'
				'{10:5}{11:1}{12:6}{13:9.6f}'. \
					format('',i+exp.start_order, \
						'','|','',info['nuc_dipole'][0] - info['dipole'][i, 0], \
						'',info['nuc_dipole'][1] - info['dipole'][i, 1], \
						'',info['nuc_dipole'][2] - info['dipole'][i, 2], \
						'','|','',np.linalg.norm(info['nuc_dipole'] - info['dipole'][i, :])))
		print(DIVIDER[:82]+'\n')


def _dipole_plot(info, calc, exp):
		""" plot MBE dipole moment """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE dipole moment
		dipole = np.empty(info['final_order'], dtype=np.float64)
		for i in range(info['final_order']):
			dipole[i] = np.linalg.norm(info['nuc_dipole'] - info['dipole'][i, :])
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				dipole, marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
				linestyle='-', label='state {:}'.format(calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Dipole moment (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total dipole increments
		mbe = np.empty_like(dipole)
		for i in range(mbe.size):
			mbe[i] = np.linalg.norm(exp.prop['dipole']['tot'][i, :])
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				np.abs(mbe), marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
				linestyle='-', label='state {:}'.format(calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/dipole_state_{:}.pdf'.format(calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _trans_prt(info, calc, exp):
		""" transition dipole moments """
		print(DIVIDER[:109])
		string = 'MBE transition dipole moment (excitation {:} > {:})'.format(0, calc.state['root'])
		print('{0:^109}'.format(string))
		print(DIVIDER[:109])
		print('{0:6}{1:9}{2:2}{3:1}{4:8}{5:25}{6:9}{7:1}{8:5}{9:13}{10:3}{11:1}{12:4}{13:}'. \
				format('','MBE order','','|','','dipole components (x,y,z)', \
						'','|','','dipole moment','','|','','oscillator strength'))
		print(DIVIDER[:109])
		for i in range(info['final_order']):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:4}{5:9.6f}{6:^3}{7:9.6f}{8:^3}{9:9.6f}'
				'{10:5}{11:1}{12:6}{13:9.6f}{14:6}{15:1}{16:8}{17:9.6f}'. \
					format('',i+exp.start_order, \
						'','|','',info['trans'][i, 0], \
						'',info['trans'][i, 1], \
						'',info['trans'][i, 2], \
						'','|','',np.linalg.norm(info['trans'][i, :]), \
						'','|','',(2./3.) * info['excitation'][i] * np.linalg.norm(info['trans'][i, :])**2))
		print(DIVIDER[:109]+'\n')


def _trans_plot(info, calc, exp):
		""" plot MBE transition dipole moments """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE transition dipole moment
		trans = np.empty(info['final_order'], dtype=np.float64)
		for i in range(info['final_order']):
			trans[i] = np.linalg.norm(info['trans'][i, :])
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				trans, marker='s', linewidth=2, mew=1, color='xkcd:dark magenta', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Transition dipole (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total transition dipole increments
		mbe = trans.copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				np.abs(mbe), marker='s', linewidth=2, mew=1, color='xkcd:dark magenta', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/trans_dipole_states_{:}_{:}.pdf'.format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _osc_strength_plot(info, calc, exp):
		""" plot MBE oscillator strength """
		# set seaborn
		if SNS_FOUND:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE oscillator strength
		osc_strength = np.empty(info['final_order'], dtype=np.float64)
		for i in range(info['final_order']):
			osc_strength[i] = (2./3.) * info['excitation'][i] * np.linalg.norm(info['trans'][i, :])**2
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				osc_strength, marker='+', linewidth=2, mew=1, color='xkcd:royal blue', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax1.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set labels
		ax1.set_ylabel('Oscillator strength (in au)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%8.3f'))
		# array of MBE total oscillator strength increments
		mbe = osc_strength.copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				np.abs(mbe), marker='+', linewidth=2, mew=1, color='xkcd:royal blue', \
				linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))
		# set x limits
		ax2.set_xlim([0.5, len(calc.exp_space) + 0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set labels
		ax2.set_xlabel('Expansion order')
		ax2.set_ylabel('Increments (in au)')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%7.1e'))
		# set upper limit on y-axis
		ax2.set_ylim(top=2.0e-01)
		# no spacing
		plt.subplots_adjust(hspace=0.05)
		# despine
		if SNS_FOUND:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/osc_strength_states_{:}_{:}.pdf'.format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


