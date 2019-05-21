#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module containing all summary and plotting functions
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
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

import output
import tools


# results file
RES_FILE = output.OUT+'/results.out'
# results parameters
DIVIDER = '{:^143}'.format('-'*137)
FILL = '{:^143}'.format('|'*137)


def main(mpi, mol, calc, exp):
        """
        this function handles all printing and plotting of results

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        """
        # print header
        print(output.main_header())

        # print atom info
        if mol.atom:
            print(_atom(mol))

        # print summary
        print(_summary_prt(mpi, mol, calc, exp))

        # print timings
        print(_timings_prt(calc, exp))

        # print and plot results
        if calc.target == 'energy' :
            print(_energy_prt(calc, exp))
            _energies_plot(calc, exp)
        if calc.target == 'excitation':
            print(_excitation_prt(calc, exp))
            _excitation_plot(calc, exp)
        if calc.target == 'dipole' :
            print(_dipole_prt(calc, exp))
            _dipole_plot(calc, exp)
        if calc.target == 'trans':
            print(_trans_prt(calc, exp))
            _trans_plot(calc, exp)
            _osc_strength_plot(calc, exp)

        # plot number of determinants
        _ndets_plot(exp)


def _atom(mol):
        """
        this function returns the molecular geometry

        :param mol: pymbe mol object
        :return: formatted string
        """
        # print atom
        string = DIVIDER[:39]+'\n'
        string += '{:^43}\n'
        form = ('geometry',)
        string += DIVIDER[:39]+'\n'
        molecule = mol.atom.split('\n')
        for i in range(len(molecule)-1):
            atom = molecule[i].split()
            for j in range(1, 4):
                atom[j] = float(atom[j])
            string += '   {:<3s} {:>10.5f} {:>10.5f} {:>10.5f}\n'
            form += (*atom,)
        string += DIVIDER[:39]+'\n'
        return string.format(*form)


def _model(calc):
        """
        this function returns the expansion model

        :param calc: pymbe calc object
        :return: formatted string
        """
        return '{:}'.format(calc.model['method'].upper())


def _basis(mol):
        """
        this function returns the basis

        :param mol: pymbe mol object
        :return: formatted string
        """
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
        """
        this function returns the state of interest

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: formatted string
        """
        string = '{:}'.format(calc.state['root'])
        if mol.spin == 0:
            string += ' (singlet)'
        elif mol.spin == 1:
            string += ' (doublet)'
        elif mol.spin == 2:
            string += ' (triplet)'
        elif mol.spin == 3:
            string += ' (quartet)'
        elif mol.spin == 4:
            string += ' (quintet)'
        else:
            string += ' ({:})'.format(mol.spin+1)
        return string


def _ref(mol, calc):
        """
        this function returns the reference function

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: formatted string
        """
        if calc.ref['method'] == 'casci':
            return 'CASCI'
        elif calc.ref['method'] == 'casscf':
            if len(calc.ref['wfnsym']) == 1:
                return 'CASSCF'
            else:
                for i in range(len(set(calc.ref['wfnsym']))):
                    sym = symm.addons.irrep_id2name(mol.symmetry, list(set(calc.ref['wfnsym']))[i])
                    num = np.count_nonzero(np.asarray(calc.ref['wfnsym']) == list(set(calc.ref['wfnsym']))[i])
                    if i == 0:
                        syms = str(num)+'*'+sym
                    else:
                        syms += '/'+sym
                return 'CASSCF('+syms+')'


def _base(calc):
        """
        this function returns the base model

        :param calc: pymbe calc object
        :return: formatted string
        """
        if calc.base['method'] is None:
            return 'none'
        else:
            return calc.base['method'].upper()


def _prot(calc):
        """
        this function returns the screening protocol

        :param calc: pymbe calc object
        :return: formatted string
        """
        if calc.prot['scheme'] == 1:
            return '1st generation'
        elif calc.prot['scheme'] == 2:
            return '2nd generation'
        elif calc.prot['scheme'] == 3:
            return '3rd generation'


def _system(mol):
        """
        this function returns the system size

        :param mol: pymbe mol object
        :return: formatted string
        """
        return '{:} e in {:} o'.format(mol.nelectron - 2 * mol.ncore, mol.norb)


def _hubbard(mol):
        """
        this function returns the hubbard model

        :param mol: pymbe mol object
        :return: formatted string
        """
        hubbard = ['{:} x {:}'.format(mol.matrix[0], mol.matrix[1])]
        hubbard.append('{:} & {:}'.format(mol.u, mol.n))
        return hubbard


def _solver(calc):
        """
        this function returns the chosen fci solver

        :param calc: pymbe calc object
        :return: formatted string
        """
        if calc.model['method'] != 'fci':
            return 'none'
        else:
            if calc.model['solver'] == 'pyscf_spin0':
                return 'PySCF (spin0)'
            elif calc.model['solver'] == 'pyscf_spin1':
                return 'PySCF (spin1)'


def _frozen(mol):
        """
        this function returns the choice of frozen core

        :param mol: pymbe mol object
        :return: formatted string
        """
        if mol.frozen:
            return 'true'
        else:
            return 'false'


def _active(calc):
        """
        this function returns the active space

        :param calc: pymbe calc object
        :return: formatted string
        """
        return '{:} e in {:} o'.format(calc.nelec[0] + calc.nelec[1], calc.ref_space.size)


def _orbs(calc):
        """
        this function returns the choice of orbitals

        :param calc: pymbe calc object
        :return: formatted string
        """
        if calc.orbs['type'] == 'can':
            return 'canonical'
        elif calc.orbs['type'] == 'ccsd':
            return 'CCSD NOs'
        elif calc.orbs['type'] == 'ccsd(t)':
            return 'CCSD(T) NOs'
        elif calc.orbs['type'] == 'local':
            return 'pipek-mezey'


def _mpi(mpi):
        """
        this function returns the mpi information

        :param mpi: pymbe mpi object
        :return: formatted string
        """
        return '{:} & {:}'.format(1, mpi.size - 1)


def _thres(calc):
        """
        this function returns the expansion threshold

        :param calc: pymbe calc object
        :return: formatted string
        """
        return '{:.0e} ({:<.1f})'.format(calc.thres['init'], calc.thres['relax'])


def _symm(mol, calc):
        """
        this function returns the molecular point group symmetry

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: formatted string
        """
        if calc.model['method'] == 'fci':
            if mol.atom:
                string = symm.addons.irrep_id2name(mol.symmetry, calc.state['wfnsym'])+'('+mol.symmetry+')'
                return string
            else:
                return 'C1(A)'
        else:
            return 'unknown'


def _energy(calc, exp):
        """
        this function returns the final total energy

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        return exp.prop['energy']['tot'] \
                + calc.prop['hf']['energy'] \
                + calc.prop['base']['energy'] \
                + calc.prop['ref']['energy']


def _excitation(calc, exp):
        """
        this function returns the final excitation energy

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        return exp.prop['excitation']['tot'] \
                + calc.prop['ref']['excitation']


def _dipole(mol, calc, exp):
        """
        this function returns the final molecular dipole moment

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        # nuclear dipole moment
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nuc_dipole = np.einsum('i,ix->x', charges, coords)
        dipole = exp.prop['dipole']['tot'] \
                        + calc.prop['hf']['dipole'] \
                        + calc.prop['ref']['dipole']
        return dipole, nuc_dipole


def _trans(mol, calc, exp):
        """
        this function returns the final molecular transition dipole moment

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        return exp.prop['trans']['tot'] \
                + calc.prop['ref']['trans']


def _time(exp, comp, idx):
        """
        this function returns the final timings in (HHH : MM : SS) format

        :param exp: pymbe exp object
        :param comp: computation part (mbe, screen, sum, or tot_sum). string
        :param idx: order index. integer
        :return: formatted string
        """
        # init time
        if comp in ['mbe', 'screen']:
            time = exp.time[comp][idx]
        elif comp == 'sum':
            time = exp.time['mbe'][idx] + exp.time['screen'][idx]
        elif comp in ['tot_mbe', 'tot_screen']:
            time = np.sum(exp.time[comp[4:]])
        elif comp == 'tot_sum':
            time = np.sum(exp.time['mbe']) + np.sum(exp.time['screen'])
        return tools.time_str(time)


def _summary_prt(mpi, mol, calc, exp):
        """
        this function returns the summary table

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        string = DIVIDER+'\n'
        string += '{:14}{:21}{:12}{:1}{:12}{:21}{:11}{:1}{:13}{:}\n'
        form = ('','molecular information','','|','', \
                    'expansion information','','|','','calculation information',)
        string += DIVIDER+'\n'

        if mol.atom:

            string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                        '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
            form += ('','basis set','','=','',_basis(mol), \
                        '','|','','exp. model','','=','',_model(calc), \
                        '','|','','mpi masters & slaves','','=','',_mpi(mpi),)

            string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                    '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:1}{:.6f}\n'
            form += ('','frozen core','','=','',_frozen(mol), \
                        '','|','','ref. function','','=','',_ref(mol, calc), \
                        '','|','','Hartree-Fock energy','','=','',calc.prop['hf']['energy'],)

        else:

            string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                    '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
            form += ('','hubbard matrix','','=','',_hubbard(mol)[0], \
                        '','|','','exp. model','','=','',_model(calc), \
                        '','|','','mpi masters & slaves','','=','',_mpi(mpi),)

            string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                    '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:1}{:.6f}\n'
            form += ('','hubbard U/t & n','','=','',_hubbard(mol)[1], \
                        '','|','','ref. function','','=','',_ref(mol, calc), \
                        '','|','','Hartree-Fock energy','','=','',calc.prop['hf']['energy'],)

        string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:1}{:.6f}\n'
        form += ('','system size','','=','',_system(mol), \
                    '','|','','exp. reference','','=','',_active(calc), \
                    '','|','','base model energy','','=','', \
                    calc.prop['hf']['energy']+calc.prop['base']['energy'],)

        string += '{:9}{:18}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:1}{:.6f}\n'
        form += ('','state (mult.)','','=','',_state(mol, calc), \
                    '','|','','base model','','=','',_base(calc), \
                    '','|','','MBE total energy','','=','', \
                    calc.prop['hf']['energy'] if calc.target != 'energy' \
                        else _energy(calc, exp)[-1],)

        string += '{:9}{:17}{:3}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','orbitals','','=','',_orbs(calc), \
                    '','|','','screen. prot.','','=','',_prot(calc), \
                    '','|','','total time','','=','',_time(exp, 'tot_sum', -1),)

        string += '{:9}{:17}{:3}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','FCI solver','','=','',_solver(calc), \
                    '','|','','screen. thres.','','=','',_thres(calc), \
                    '','|','','wave funct. symmetry','','=','',_symm(mol, calc),)

        string += DIVIDER+'\n'+FILL+'\n'+DIVIDER+'\n'

        return string.format(*form)


def _timings_prt(calc, exp):
        """
        this function returns the timings table

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        string = DIVIDER[:98]+'\n'
        string += '{:^98}\n'
        form = ('MBE timings',)

        string += DIVIDER[:98]+'\n'
        string += '{:6}{:9}{:2}{:1}{:8}{:3}{:8}{:1}{:5}{:9}{:5}' \
                '{:1}{:8}{:3}{:8}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','MBE','','|','','screening', \
                    '','|','','sum','','|','','calculations',)

        string += DIVIDER[:98]+'\n'
        calcs = 0

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            calc_i = exp.prop[calc.target]['inc'][i].shape[0]
            calcs += calc_i
            string += '{:7}{:>4d}{:6}{:1}{:2}{:>15s}{:2}{:1}{:2}{:>15s}{:2}{:1}' \
                    '{:2}{:>15s}{:2}{:1}{:5}{:>9d}\n'
            form += ('',j, \
                        '','|','',_time(exp, 'mbe', i), \
                        '','|','',_time(exp, 'screen', i), \
                        '','|','',_time(exp, 'sum', i), \
                        '','|','',calc_i,)

        string += DIVIDER[:98]+'\n'
        string += '{:8}{:5s}{:4}{:1}{:2}{:>15s}{:2}{:1}{:2}{:>15s}{:2}{:1}' \
                '{:2}{:>15s}{:2}{:1}{:5}{:>9d}\n'
        form += ('','total', \
                    '','|','',_time(exp, 'tot_mbe', -1), \
                    '','|','',_time(exp, 'tot_screen', -1), \
                    '','|','',_time(exp, 'tot_sum', -1), \
                    '','|','',calcs,)

        string += DIVIDER[:98]+'\n'

        return string.format(*form)


def _energy_prt(calc, exp):
        """
        this function returns the energies table

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        string = DIVIDER[:66]+'\n'
        string_in = 'MBE energy (root = '+str(calc.state['root'])+')'
        string += '{:^66}\n'
        form = (string_in,)

        string += DIVIDER[:66]+'\n'
        string += '{:6}{:9}{:2}{:1}{:5}{:12}{:5}{:1}{:4}{:}\n'
        form += ('','MBE order','','|','','total energy','','|','','correlation energy',)

        string += DIVIDER[:66]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:5}{:>11.6f}{:6}{:1}{:7}{:11.4e}\n'
        form += ('','ref','','|','',calc.prop['hf']['energy'] + calc.prop['ref']['energy'], \
                    '','|','',calc.prop['ref']['energy'],)

        string += DIVIDER[:66]+'\n'
        energy = _energy(calc, exp)

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:5}{:>11.6f}{:6}{:1}{:7}{:11.4e}\n'
            form += ('',j, \
                        '','|','',energy[i], \
                        '','|','',energy[i] - calc.prop['hf']['energy'],)

        string += DIVIDER[:66]+'\n'

        return string.format(*form)


def _energies_plot(calc, exp):
        """
        this function plots the energies

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                _energy(calc, exp), marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
                linestyle='-', label='state {:}'.format(calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Energy (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1)

        # save plot
        plt.savefig(output.OUT+'/energy_state_{:}.pdf'. \
                        format(calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _excitation_prt(calc, exp):
        """
        this function returns the excitation energies table

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        string = DIVIDER[:43]+'\n'
        string_in = 'MBE excitation energy (root = '+str(calc.state['root'])+')'
        string += '{:^46}\n'
        form = (string_in,)

        string += DIVIDER[:43]+'\n'
        string += '{:6}{:9}{:2}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','excitation energy',)

        string += DIVIDER[:43]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:8}{:9.4e}\n'
        form += ('','ref','','|','',calc.prop['ref']['excitation'],)

        string += DIVIDER[:43]+'\n'
        excitation = _excitation(calc, exp)

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:8}{:9.4e}\n'
            form += ('',j,'','|','',excitation[i],)

        string += DIVIDER[:43]+'\n'

        return string.format(*form)


def _excitation_plot(calc, exp):
        """
        this function plots the excitation energies

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                _excitation(calc, exp), marker='x', linewidth=2, mew=1, color='xkcd:dull blue', \
                linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Excitation energy (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1)

        # save plot
        plt.savefig(output.OUT+'/excitation_states_{:}_{:}.pdf'. \
                        format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _dipole_prt(calc, exp):
        """
        this function returns the dipole moments table

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        string = DIVIDER[:82]+'\n'
        string_in = 'MBE dipole moment (root = '+str(calc.state['root'])+')'
        string += '{:^82}\n'
        form = (string_in,)

        string += DIVIDER[:82]+'\n'
        string += '{:6}{:9}{:2}{:1}{:8}{:25}{:9}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','dipole components (x,y,z)','','|','','dipole moment',)

        string += DIVIDER[:82]+'\n'

        dipole, nuc_dipole = _dipole(mol, calc, exp)
        string += '{:9}{:>3s}{:5}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
        form += ('','ref', \
                    '','|','',nuc_dipole[0] - calc.prop['hf']['dipole'][0] + calc.prop['ref']['dipole'][0], \
                    '',nuc_dipole[1] - calc.prop['hf']['dipole'][1] + calc.prop['ref']['dipole'][1], \
                    '',nuc_dipole[2] - calc.prop['hf']['dipole'][2] + calc.prop['ref']['dipole'][2], \
                    '','|','',np.linalg.norm(nuc_dipole - calc.prop['hf']['dipole']),)

        string += DIVIDER[:82]+'\n'

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
            form += ('',j, \
                        '','|','',nuc_dipole[0] - dipole[i, 0], \
                        '',nuc_dipole[1] - dipole[i, 1], \
                        '',nuc_dipole[2] - dipole[i, 2], \
                        '','|','',np.linalg.norm(nuc_dipole - dipole[i, :]),)

        string += DIVIDER[:82]+'\n'

        return string.format(*form)


def _dipole_plot(calc, exp):
        """
        this function plots the dipole moments

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # array of total MBE dipole moment
        dipole, nuc_dipole = _dipole(mol, calc, exp)
        dipole_arr = np.empty(dipole.shape[0], dtype=np.float64)
        for i in range(dipole.size):
            dipole_arr[i] = np.linalg.norm(nuc_dipole - dipole[i, :])

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                dipole_arr, marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
                linestyle='-', label='state {:}'.format(calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Dipole moment (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1)

        # save plot
        plt.savefig(output.OUT+'/dipole_state_{:}.pdf'. \
                        format(calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _trans_prt(calc, exp):
        """
        this function returns the transition dipole moments and oscillator strengths table

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        string = DIVIDER[:109]+'\n'
        string_in = 'MBE transition dipole moment (excitation 0 > '+str(calc.state['root'])+')'
        string += '{:^109}\n'
        form = (string_in,)

        string += DIVIDER[:109]+'\n'

        excitation = _excitation(calc, exp)
        trans = _trans(mol, calc, exp)

        string += '{:6}{:9}{:2}{:1}{:8}{:25}{:9}{:1}{:5}{:13}{:3}{:1}{:4}{:}\n'
        form += ('','MBE order','','|','','dipole components (x,y,z)', \
                    '','|','','dipole moment','','|','','oscillator strength',)

        string += DIVIDER[:109]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}{:6}{:1}{:8}{:9.6f}\n'
        form += ('','ref', \
                    '','|','',calc.prop['ref']['trans'][0], \
                    '',calc.prop['ref']['trans'][1], \
                    '',calc.prop['ref']['trans'][2], \
                    '','|','',np.linalg.norm(calc.prop['ref']['trans'][:]), \
                    '','|','',(2./3.) * calc.prop['ref']['excitation'] * np.linalg.norm(calc.prop['ref']['trans'][:])**2,)

        string += DIVIDER[:109]+'\n'

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}{:6}{:1}{:8}{:9.6f}\n'
            form += ('',j, \
                        '','|','',trans[i, 0], \
                        '',trans[i, 1], \
                        '',trans[i, 2], \
                        '','|','',np.linalg.norm(trans[i, :]), \
                        '','|','',(2./3.) * excitation[i] * np.linalg.norm(trans[i, :])**2,)

        string += DIVIDER[:109]+'\n'

        return string.format(*form)


def _trans_plot(calc, exp):
        """
        this function plots the transition dipole moments

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # array of total MBE transition dipole moment
        trans = _trans(mol, calc, exp)
        trans_arr = np.empty(trans.shape[0], dtype=np.float64)
        for i in range(trans.size):
            trans_arr[i] = np.linalg.norm(trans[i, :])

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                trans_arr, marker='s', linewidth=2, mew=1, color='xkcd:dark magenta', \
                linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Transition dipole (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1)

        # save plot
        plt.savefig(output.OUT+'/trans_dipole_states_{:}_{:}.pdf'. \
                        format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _osc_strength_plot(calc, exp):
        """
        this function plots the oscillator strengths

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: formatted string
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # array of total MBE oscillator strength
        excitation = _excitation(calc, exp)
        trans = _trans(mol, calc, exp)
        osc_strength = np.empty(trans.shape[0], dtype=np.float64)
        for i in range(osc_strength.size):
            osc_strength[i] = (2./3.) * excitation[i] * np.linalg.norm(trans[i, :])**2

        # plot results
        ax.plot(np.arange(exp.min_order, exp.final_order+1), \
                osc_strength, marker='+', linewidth=2, mew=1, color='xkcd:royal blue', \
                linestyle='-', label='excitation {:} -> {:}'.format(0, calc.state['root']))

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Oscillator strength (in au)')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # set legend
        ax.legend(loc=1)

        # save plot
        plt.savefig(output.OUT+'/osc_strength_states_{:}_{:}.pdf'. \
                        format(0, calc.state['root']), bbox_inches = 'tight', dpi=1000)


def _ndets_plot(exp):
        """
        this function plots the number of determinants

        :param exp: pymbe exp object
        :return: formatted string
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # array of max number of determinants at each order
        max_ndets = np.empty(len(exp.ndets), dtype=np.float64)
        for i in range(max_ndets.size):
            ndets = exp.ndets[i]
            if ndets.any():
                max_ndets[i] = np.max(ndets[np.nonzero(ndets)])
            else:
                max_ndets[i] = 0.0

        # plot results
        ax.semilogy(np.arange(exp.min_order, exp.final_order+1), \
                    max_ndets, marker='x', linewidth=2, mew=1, color='red', linestyle='-')

        # set x limits
        ax.set_xlim([0.5, exp.final_order+1 - 0.5])

        # turn off x-grid
        ax.xaxis.grid(False)

        # set labels
        ax.set_xlabel('Expansion order')
        ax.set_ylabel('Number of determinants')

        # force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        # despine
        if SNS_FOUND:
            sns.despine()

        # save plot
        plt.savefig(output.OUT+'/ndets.pdf', bbox_inches = 'tight', dpi=1000)


