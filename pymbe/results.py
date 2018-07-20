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
	has_sns = True
except ImportError:
	pass
	has_sns = False

# results parameters
OUT = os.getcwd()+'/output'
DIVIDER = '{0:^143}'.format('-'*137)
FILL = '{0:^143}'.format('|'*137)


def main(mpi, mol, calc, exp):
		""" printing and plotting of results """
		# convert final results to numpy arrays
		for i in range(calc.nroots):
			exp.prop['energy'][i]['tot'] = np.asarray(exp.prop['energy'][i]['tot'])
			if calc.target['dipole']:
				exp.prop['dipole'][i]['tot'] = np.asarray(exp.prop['dipole'][i]['tot'])
			if calc.target['trans']:
				if i < calc.nroots - 1:
					exp.prop['trans'][i]['tot'] = np.asarray(exp.prop['trans'][i]['tot'])
		# setup
		info = {}
		info['model_type'], info['basis'], info['mult'], info['ref'], info['base'], info['prot'], \
			info['system'], info['frozen'], info['hubbard'], info['active'], \
			info['occ'], info['virt'], info['mpi'], info['thres'], info['symm'], \
			info['energy'], info['dipole'], info['nuc_dipole'], info['trans'] = _setup(mpi, mol, calc, exp)
		info['final_order'] = info['energy'][0].size
		# results
		_table(info, mol, calc, exp)
		# plot
		_plot(info, calc, exp)


def _setup(mpi, mol, calc, exp):
		""" init parameters """
		model_type = _model_type(calc)
		basis = _basis(mol)
		mult = _mult(mol)
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
		energy = _energy(calc, exp)
		if calc.target['dipole']:
			dipole, nuc_dipole = _dipole(mol, calc, exp)
		else:
			dipole = nuc_dipole = None
		if calc.target['trans']:
			trans = _trans(mol, calc, exp)
		else:
			trans = None
		return model_type, basis, mult, ref, base, prot, system, frozen, \
				hubbard, active, occ, virt, mpi, thres, symm, \
				energy, dipole, nuc_dipole, trans


def _table(info, mol, calc, exp):
		""" print results """
		# write results to results.out
		with open(OUT+'/results.out','a') as f:
			with contextlib.redirect_stdout(f):
				_summary_prt(info, mol, calc, exp)
				_timings_prt(info, exp)
				for i in range(calc.nroots):
					_energy_prt(info, calc, exp, i)
				if calc.target['dipole']:
					for i in range(calc.nroots):
						_dipole_prt(info, calc, exp, i)
				if calc.target['trans']:
					for i in range(1, calc.nroots):
						_trans_prt(info, calc, exp, i)
	

def _plot(info, calc, exp):
		""" plot results """
		# plot MBE energies
		for i in range(calc.nroots):
			_energies_plot(info, calc, exp, i)
			# plot MBE dipole moment
			if calc.target['dipole']:
				_dipole_plot(info, calc, exp, i)
			if calc.target['trans']:
				if i > 0:
					_trans_plot(info, calc, exp, i)
					_osc_strength_plot(info, calc, exp, i)


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


def _mult(mol):
		""" mult print """
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
		""" ref print """
		if calc.ref['method'] == 'hf':
			if mol.spin == 0:
				return 'RHF'
			else:
				return 'ROHF'
		else:
			return calc.ref['method'].upper()


def _base(calc):
		""" base print """
		if calc.base['method'] is None:
			return 'none'
		else:
			return calc.base['method'].upper()


def _prot(calc):
		""" protocol print """
		prot = calc.prot['scheme']
		if calc.prot['specific']:
			prot += ' (state {:})'.format(calc.state['root'])
		else:
			prot += ' (all states)'
		return prot


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
				return symm.addons.irrep_id2name(mol.symmetry, calc.state['wfnsym'])+' ('+mol.symmetry+')'
			else:
				return 'C1 (A)'
		else:
			return 'unknown'


def _energy(calc, exp):
		""" final energies """
		# ground state
		energy = [exp.prop['energy'][0]['tot'] \
				+ calc.prop['hf']['energy'] + calc.base['energy'][0] \
				+ (calc.prop['ref']['energy'][0] - calc.base['ref'][0])]
		# excited states
		for i in range(1, calc.nroots):
			energy.append(exp.prop['energy'][i]['tot'] + calc.prop['ref']['energy'][i])
		return energy


def _dipole(mol, calc, exp):
		""" final molecular dipole moments """
		# nuclear dipole moment
		charges = mol.atom_charges()
		coords  = mol.atom_coords()
		nuc_dipole = np.einsum('i,ix->x', charges, coords)
		# ground state
		dipole = [exp.prop['dipole'][0]['tot'] \
						+ calc.prop['hf']['dipole'] \
						+ calc.prop['ref']['dipole'][0]]
		# excited states
		for i in range(1, calc.nroots):
			dipole.append(exp.prop['dipole'][i]['tot'] + calc.prop['ref']['dipole'][i])
		return dipole, nuc_dipole


def _trans(mol, calc, exp):
		""" final molecular transition dipole moments """
		return [exp.prop['trans'][i]['tot'] + calc.prop['ref']['trans'][i] for i in range(calc.nroots-1)]


def _time(exp, comp, idx):
		""" convert time to (HHH : MM : SS) format """
		if comp != 'total':
			hours = int(exp.time[comp][idx]//3600)
			minutes = int((exp.time[comp][idx]-(exp.time[comp][idx]//3600)*3600.)//60)
			seconds = int(exp.time[comp][idx]-(exp.time[comp][idx]//3600)*3600. \
							- ((exp.time[comp][idx]-(exp.time[comp][idx]//3600)*3600.)//60)*60.)
		else:
			hours = int(np.sum(exp.time[comp][:idx+1])//3600)
			minutes = int((np.sum(exp.time[comp][:idx+1])-(np.sum(exp.time[comp][:idx+1])//3600)*3600.)//60)
			seconds = int(np.sum(exp.time[comp][:idx+1])-(np.sum(exp.time[comp][:idx+1])//3600)*3600. \
							- ((np.sum(exp.time[comp][:idx+1])-(np.sum(exp.time[comp][:idx+1])//3600)*3600.)//60)*60.)
		# init time string
		time = ''
		if hours > 0:
			time += '{:}h '.format(hours)
		if minutes > 0:
			time += '{:}m '.format(minutes)
		time += '{:}s'.format(seconds)
		return time


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
						calc.prop['hf']['energy']+calc.base['energy'][0]))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:1}{21:.6f}'. \
					format('','spin multiplicity','','=','',info['mult'], \
						'','|','','base model','','=','',info['base'], \
						'','|','','MBE total energy','','=','',info['energy'][0][-1]))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}'.\
					format('','occupied orbs','','=','',info['occ'], \
						'','|','','screen. prot.','','=','',info['prot'], \
						'','|','','total time','','=','',_time(exp, 'total', exp.order-1)))
		print('{0:9}{1:18}{2:2}{3:1}{4:2}{5:<13s}{6:2}{7:1}{8:7}{9:15}{10:2}{11:1}{12:2}'
				'{13:<16s}{14:1}{15:1}{16:7}{17:21}{18:3}{19:1}{20:2}{21:<s}'. \
					format('','virtual orbs','','=','',info['virt'], \
						'','|','','screen. thres.','','=','',info['thres'], \
						'','|','','wave funct. symmetry','','=','',info['symm']))
		print(DIVIDER)
		print(FILL)
		print(DIVIDER+'\n')


def _timings_prt(info, exp):
		""" timings """
		print(DIVIDER[:98])
		print('{0:^98}'.format('MBE timings'))
		print(DIVIDER[:98])
		print('{0:6}{1:9}{2:2}{3:1}{4:8}{5:3}{6:8}{7:1}{8:5}{9:9}{10:5}'
				'{11:1}{12:7}{13:5}{14:7}{15:1}{16:4}{17:}'. \
				format('','MBE order','','|','','MBE','','|','','screening', \
						'','|','','total','','|','','no. of calcs.'))
		print(DIVIDER[:98])
		for i in range(info['final_order']):
			print('{0:7}{1:>4d}{2:6}{3:1}{4:2}{5:>13s}{6:4}{7:1}{8:2}{9:>13s}{10:4}{11:1}'
					'{12:2}{13:>13s}{14:4}{15:1}{16:5}{17:>9d}'. \
					format('',i+exp.start_order, \
						'','|','',_time(exp, 'mbe', i), \
						'','|','',_time(exp, 'screen', i), \
						'','|','',_time(exp, 'total', i), \
						'','|','',exp.tuples[i].shape[0]))
		print(DIVIDER[:98]+'\n')


def _energy_prt(info, calc, exp, root):
		""" energies """
		if root == 0:
			# ground state
			print(DIVIDER[:66])
			print('{0:^66}'.format('MBE ground state energy'))
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
							'','|','',info['energy'][0][i], \
							'','|','',info['energy'][0][i] - calc.prop['hf']['energy']))
			print(DIVIDER[:66]+'\n')
		else:
			# excited states
			if calc.prot['specific']:
				root_idx = calc.state['root']
			else:
				root_idx = root
			print(DIVIDER[:66])
			string = 'MBE excited state energy (root = {:})'.format(root_idx)
			print('{0:^66}'.format(string))
			print(DIVIDER[:66])
			print('{0:6}{1:9}{2:2}{3:1}{4:5}{5:12}{6:5}{7:1}{8:5}{9:}'. \
					format('','MBE order','','|','','total energy','','|','','excitation energy'))
			print(DIVIDER[:66])
			for i in range(info['final_order']):
				print('{0:7}{1:>4d}{2:6}{3:1}{4:5}{5:>11.6f}{6:6}{7:1}{8:8}{9:9.4e}'. \
						format('',i+exp.start_order, \
							'','|','',info['energy'][0][i] + info['energy'][root][i], \
							'','|','',info['energy'][root][i]))
			print(DIVIDER[:66]+'\n')


def _energies_plot(info, calc, exp, root):
		""" plot MBE energy for state 'root' """
		if calc.prot['specific']:
			root_idx = calc.state['root']
		else:
			root_idx = root
		# set seaborn
		if has_sns:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of MBE total energy
		energy = info['energy'][root].copy()
		if root >= 1: energy += info['energy'][0]
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				energy, marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
				linestyle='-', label='state {:}'.format(root_idx))
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
		mbe = exp.prop['energy'][root]['tot'].copy()
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				np.abs(mbe), marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
				linestyle='-', label='state {:}'.format(root_idx))
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
		if has_sns:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/energy_state_{:}.pdf'.format(root_idx), bbox_inches = 'tight', dpi=1000)


def _dipole_prt(info, calc, exp, root):
		""" dipole moments """
		if root == 0:
			# ground state
			print(DIVIDER[:82])
			print('{0:^82}'.format('MBE ground state dipole moment'))
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
							'','|','',info['nuc_dipole'][0] - info['dipole'][0][i, 0], \
							'',info['nuc_dipole'][1] - info['dipole'][0][i, 1], \
							'',info['nuc_dipole'][2] - info['dipole'][0][i, 2], \
							'','|','',np.linalg.norm(info['nuc_dipole'] - info['dipole'][0][i, :])))
			print(DIVIDER[:82]+'\n')
		else:
			# excited states
			if calc.prot['specific']:
				root_idx = calc.state['root']
			else:
				root_idx = root
			print(DIVIDER[:82])
			string = 'MBE excited state dipole moment (root = {:})'.format(root_idx)
			print('{0:^82}'.format(string))
			print(DIVIDER[:82])
			print('{0:6}{1:9}{2:2}{3:1}{4:8}{5:25}{6:9}{7:1}{8:5}{9:}'. \
					format('','MBE order','','|','','dipole components (x,y,z)','','|','','dipole moment'))
			print(DIVIDER[:82])
			for i in range(info['final_order']):
				print('{0:7}{1:>4d}{2:6}{3:1}{4:4}{5:9.6f}{6:^3}{7:9.6f}{8:^3}{9:9.6f}'
					'{10:5}{11:1}{12:6}{13:9.6f}'. \
						format('',i+exp.start_order, \
							'','|','',info['nuc_dipole'][0] - (info['dipole'][root][i, 0] + info['dipole'][0][i, 0]), \
							'',info['nuc_dipole'][1] - (info['dipole'][root][i, 1] + info['dipole'][0][i, 1]), \
							'',info['nuc_dipole'][2] - (info['dipole'][root][i, 2] + info['dipole'][0][i, 2]), \
							'','|','',np.linalg.norm(info['nuc_dipole'] - (info['dipole'][root][i, :] + info['dipole'][0][i, :]))))
			print(DIVIDER[:82]+'\n')


def _dipole_plot(info, calc, exp, root):
		""" plot MBE dipole moment for state 'root' """
		if calc.prot['specific']:
			root_idx = calc.state['root']
		else:
			root_idx = root
		# set seaborn
		if has_sns:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE dipole moment
		dipole = np.empty(info['final_order'], dtype=np.float64)
		for i in range(info['final_order']):
			if root == 0:
				dipole[i] = np.linalg.norm(info['nuc_dipole'] - info['dipole'][0][i, :])
			else:
				dipole[i] = np.linalg.norm(info['nuc_dipole'] - (info['dipole'][0][i, :] + info['dipole'][root][i, :]))
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				dipole, marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
				linestyle='-', label='state {:}'.format(root_idx))
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
			mbe[i] = np.linalg.norm(exp.prop['dipole'][root]['tot'][i, :])
		mbe[1:] = np.diff(mbe)
		# plot results
		ax2.semilogy(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				np.abs(mbe), marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
				linestyle='-', label='state {:}'.format(root_idx))
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
		if has_sns:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/dipole_state_{:}.pdf'.format(root_idx), bbox_inches = 'tight', dpi=1000)


def _trans_prt(info, calc, exp, root):
		""" transition dipole moments """
		if calc.prot['specific']:
			root_idx = calc.state['root']
		else:
			root_idx = root
		print(DIVIDER[:109])
		string = 'MBE transition dipole moment (excitation {:} > {:})'.format(0, root_idx)
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
						'','|','',info['trans'][root-1][i, 0], \
						'',info['trans'][root-1][i, 1], \
						'',info['trans'][root-1][i, 2], \
						'','|','',np.linalg.norm(info['trans'][root-1][i, :]), \
						'','|','',(2./3.) * info['energy'][root][i] * np.linalg.norm(info['trans'][root-1][i, :])**2))
		print(DIVIDER[:109]+'\n')


def _trans_plot(info, calc, exp, root):
		""" plot MBE transition dipole moment for excitation between states 0 and 'root' """
		if calc.prot['specific']:
			root_idx = calc.state['root']
		else:
			root_idx = root
		# set seaborn
		if has_sns:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE transition dipole moment
		trans = np.empty(info['final_order'], dtype=np.float64)
		for i in range(info['final_order']):
			trans[i] = np.linalg.norm(info['trans'][root-1][i, :])
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				trans, marker='s', linewidth=2, mew=1, color='xkcd:dark magenta', \
				linestyle='-', label='excitation {:} > {:}'.format(0, root_idx))
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
				linestyle='-', label='excitation {:} > {:}'.format(0, root_idx))
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
		if has_sns:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/trans_dipole_states_{:}_{:}.pdf'.format(0, root_idx), bbox_inches = 'tight', dpi=1000)


def _osc_strength_plot(info, calc, exp, root):
		""" plot MBE oscillator strength for excitation between states 0 and 'root' """
		if calc.prot['specific']:
			root_idx = calc.state['root']
		else:
			root_idx = root
		# set seaborn
		if has_sns:
			sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
		# set 2 subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		# array of total MBE oscillator strength
		osc_strength = np.empty(info['final_order'], dtype=np.float64)
		for i in range(info['final_order']):
			osc_strength[i] = (2./3.) * info['energy'][root][i] * np.linalg.norm(info['trans'][root-1][i, :])**2
		# plot results
		ax1.plot(np.asarray(list(range(exp.start_order, info['final_order']+exp.start_order))), \
				osc_strength, marker='+', linewidth=2, mew=1, color='xkcd:royal blue', \
				linestyle='-', label='excitation {:} > {:}'.format(0, root_idx))
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
				linestyle='-', label='excitation {:} > {:}'.format(0, root_idx))
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
		if has_sns:
			sns.despine()
		# set legend
		ax1.legend(loc=1)
		# save plot
		plt.savefig(OUT+'/osc_strength_states_{:}_{:}.pdf'.format(0, root_idx), bbox_inches = 'tight', dpi=1000)



