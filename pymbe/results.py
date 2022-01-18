#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module
"""

from __future__ import annotations

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, symm
from typing import TYPE_CHECKING
try:
    import matplotlib
    PLT_FOUND = True
except (ImportError, OSError):
    pass
    PLT_FOUND = False
if PLT_FOUND:
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    try:
        import seaborn as sns
        SNS_FOUND = True
    except (ImportError, OSError):
        pass
        SNS_FOUND = False

from pymbe.output import main_header
from pymbe.tools import intervals, time_str, nelec

if TYPE_CHECKING:

    from typing import Tuple, Any, Dict, Union, List, Optional

    from pymbe.parallel import MPICls
    from pymbe.expansion import ExpCls


# results parameters
DIVIDER = '{:^143}'.format('-'*137)
FILL = '{:^143}'.format('|'*137)


def print_results(mol: Optional[gto.Mole], mpi: MPICls, exp: ExpCls) -> str:
        """
        this function handles printing of results
        """
        # print header
        string = main_header()+'\n'

        # print atom info
        if mol and mol.atom:
            string += _atom(mol)+'\n'

        # print summary
        string += _summary_prt(mpi, exp)+'\n'

        # print timings
        string += _timings_prt(exp, exp.method)+'\n'

        # print and plot results
        if exp.target == 'energy' :
            string += _energy_prt(exp.method, exp.fci_state_root, \
                                  exp.prop['energy']['tot'], exp.hf_prop, \
                                  exp.base_prop, exp.ref_prop, exp.min_order, \
                                  exp.final_order)
        if exp.target == 'excitation':
            string += _excitation_prt(exp.fci_state_root, \
                                      exp.prop['excitation']['tot'], \
                                      exp.ref_prop, exp.min_order, \
                                      exp.final_order)
        if exp.target == 'dipole' :
            string += _dipole_prt(exp.fci_state_root, exp.nuc_dipole, \
                                 exp.prop['dipole']['tot'], exp.hf_prop, \
                                 exp.base_prop, exp.ref_prop, exp.min_order, \
                                 exp.final_order)
        if exp.target == 'trans':
            string += _trans_prt(exp.fci_state_root, exp.prop['trans']['tot'], \
                                 exp.ref_prop, exp.min_order, exp.final_order)

        return string


def plot_results(exp: ExpCls) -> matplotlib.figure.Figure:
        """
        this function handles plotting of results
        """
        # check if matplotlib is available
        if not PLT_FOUND:
            raise ModuleNotFoundError('No module named matplotlib')

        # print and plot results
        if exp.target == 'energy':
            fig = _energy_plot(exp.fci_state_root, exp.prop['energy']['tot'], \
                         exp.hf_prop, exp.base_prop, exp.ref_prop, \
                         exp.min_order, exp.final_order)
        if exp.target == 'excitation':
            fig = _excitation_plot(exp.fci_state_root, \
                             exp.prop['excitation']['tot'], exp.ref_prop, \
                             exp.min_order, exp.final_order)
        if exp.target == 'dipole':
            fig = _dipole_plot(exp.fci_state_root, exp.nuc_dipole, \
                         exp.prop['dipole']['tot'], exp.hf_prop, \
                         exp.base_prop, exp.ref_prop, exp.min_order, \
                         exp.final_order)
        if exp.target == 'trans':
            fig = _trans_plot(exp.fci_state_root, exp.prop['trans']['tot'], \
                        exp.ref_prop, exp.min_order, exp.final_order)

        return fig


def _atom(mol: gto.Mole) -> str:
        """
        this function returns the molecular geometry
        """
        # print atom
        string: str = DIVIDER[:39]+'\n'
        string += '{:^43}\n'
        form: Tuple[Any] = ('geometry',)
        string += DIVIDER[:39]+'\n'
        molecule = gto.tostring(mol).split('\n')
        for i in range(len(molecule)):
            atom = molecule[i].split()
            for j in range(1, 4):
                atom[j] = float(atom[j])
            string += '   {:<3s} {:>10.5f} {:>10.5f} {:>10.5f}\n'
            form += (*atom,) # type: ignore
        string += DIVIDER[:39]+'\n'
        return string.format(*form)


def _model(method: str) -> str:
        """
        this function returns the expansion model
        """
        string = '{:}'.format(method.upper())

        return string


def _state(spin: int, root: int) -> str:
        """
        this function returns the state of interest
        """
        string = '{:}'.format(root)
        if spin == 0:
            string += ' (singlet)'
        elif spin == 1:
            string += ' (doublet)'
        elif spin == 2:
            string += ' (triplet)'
        elif spin == 3:
            string += ' (quartet)'
        elif spin == 4:
            string += ' (quintet)'
        else:
            string += ' ({:})'.format(spin+1)
        return string


def _base(base_method: Optional[str]) -> str:
        """
        this function returns the base model
        """
        if base_method is None:
            return 'none'
        else:
            return base_method.upper()


def _system(nocc: int, ncore: int, norb: int) -> str:
        """
        this function returns the system size
        """
        return '{:} e in {:} o'.format(2 * (nocc - ncore), norb - ncore)


def _solver(method: str, fci_solver: str) -> str:
        """
        this function returns the chosen fci solver
        """
        if method != 'fci':
            return 'none'
        else:
            if fci_solver == 'pyscf_spin0':
                return 'PySCF (spin0)'
            elif fci_solver == 'pyscf_spin1':
                return 'PySCF (spin1)'
            else:
                raise NotImplementedError('unknown solver')


def _active_space(occup: np.ndarray, ref_space: np.ndarray) -> str:
        """
        this function returns the active space
        """
        act_n_elec = nelec(occup, ref_space)
        string = '{:} e, {:} o'.format(act_n_elec[0] + act_n_elec[1], ref_space.size)
        return string


def _active_orbs(ref_space: np.ndarray) -> str:
        """
        this function returns the orbitals of the active space
        """
        if ref_space.size == 0:
            return 'none'

        # init string
        string = '['
        # divide ref_space into intervals
        ref_space_ints = [i for i in intervals(ref_space)]

        for idx, i in enumerate(ref_space_ints):
            elms = '{:}-{:}'.format(i[0], i[1]) if len(i) > 1 else '{:}'.format(i[0])
            string += '{:},'.format(elms) if idx < len(ref_space_ints) - 1 else '{:}'.format(elms)
        string += ']'

        return string


def _orbs(orb_type: str) -> str:
        """
        this function returns the choice of orbitals
        """
        if orb_type == 'can':
            return 'canonical'
        elif orb_type == 'ccsd':
            return 'CCSD NOs'
        elif orb_type == 'ccsd(t)':
            return 'CCSD(T) NOs'
        elif orb_type == 'local':
            return 'pipek-mezey'
        elif orb_type == 'casscf':
            return 'casscf'
        else:
            raise NotImplementedError('unknown orbital basis')


def _mpi(num_masters: int, global_size: int) -> str:
        """
        this function returns the mpi information
        """
        return '{:} & {:}'.format(num_masters, global_size - num_masters)


def _point_group(point_group: str) -> str:
        """
        this function returns the point group symmetry
        """
        return point_group


def _symm(method: str, point_group: str, fci_state_sym: int, \
          pi_prune: bool) -> str:
        """
        this function returns the symmetry of the wavefunction in the
        computational point group
        """
        if method == 'fci':
            string = symm.addons.irrep_id2name(point_group, fci_state_sym)+'('+point_group+')'
            if pi_prune:
                string += ' (pi)'
            return string
        else:
            return 'unknown'


def _energy(corr_energy: List[float], hf_energy: float, base_energy: float, \
            ref_energy: float) -> np.ndarray:
        """
        this function returns the final total energy
        """
        tot_energy = np.copy(corr_energy)
        tot_energy += hf_energy
        tot_energy += base_energy
        tot_energy += ref_energy

        return tot_energy


def _excitation(corr_exc: List[float], ref_exc: float) -> np.ndarray:
        """
        this function returns the final excitation energy
        """
        tot_exc = np.copy(corr_exc)
        tot_exc += ref_exc

        return tot_exc


def _dipole(corr_dipole: List[np.ndarray], hf_dipole: np.ndarray, \
            base_dipole: np.ndarray, ref_dipole: np.ndarray) -> np.ndarray:
        """
        this function returns the final molecular dipole moment
        """
        tot_dipole = np.copy(corr_dipole)
        tot_dipole += hf_dipole
        tot_dipole += base_dipole
        tot_dipole += ref_dipole

        return tot_dipole


def _trans(corr_trans: List[np.ndarray], ref_trans: np.ndarray) -> np.ndarray:
        """
        this function returns the final molecular transition dipole moment
        """
        tot_trans = np.copy(corr_trans)
        tot_trans += ref_trans

        return tot_trans


def _time(time: Dict[str, Union[List[float], np.ndarray]], comp: str, \
          idx: int) -> str:
        """
        this function returns the final timings in (HHH : MM : SS) format
        """
        # init time
        if comp in ['mbe', 'purge']:
            req_time = time[comp][idx]
        elif comp == 'sum':
            req_time = time['mbe'][idx] + time['purge'][idx]
        elif comp in ['tot_mbe', 'tot_purge']:
            req_time = np.sum(time[comp[4:]])
        elif comp == 'tot_sum':
            req_time = np.sum(time['mbe']) + np.sum(time['purge'])
        return time_str(req_time)


def _summary_prt(mpi: MPICls, exp: ExpCls) -> str:
        """
        this function returns the summary table
        """
        if exp.target == 'energy':
            hf_prop = exp.hf_prop
            base_prop = exp.hf_prop + exp.base_prop
            mbe_tot_prop = _energy(exp.prop['energy']['tot'], exp.hf_prop, exp.base_prop, exp.ref_prop)[-1].item()
        elif exp.target == 'dipole':
            dipole = _dipole(exp.prop['dipole']['tot'], exp.hf_prop, exp.base_prop, exp.ref_prop)
            hf_prop = np.linalg.norm(exp.nuc_dipole - exp.hf_prop)
            base_prop = np.linalg.norm(exp.nuc_dipole - (exp.hf_prop + exp.base_prop))
            mbe_tot_prop = np.linalg.norm(exp.nuc_dipole - dipole[-1, :])
        elif exp.target == 'excitation':
            hf_prop = 0.
            base_prop = 0.
            mbe_tot_prop = _excitation(exp.prop['excitation']['tot'], exp.ref_prop)[-1].item()
        else:
            hf_prop = 0.
            base_prop = 0.
            mbe_tot_prop = np.linalg.norm(_trans(exp.prop['trans']['tot'], exp.ref_prop)[-1, :])

        string: str = DIVIDER+'\n'
        string += '{:14}{:21}{:12}{:1}{:12}{:21}{:11}{:1}{:13}{:}\n'
        form: Tuple[Any, ...] = ('','molecular information','','|','', \
                    'expansion information','','|','','calculation information',)
        string += DIVIDER+'\n'

        string += '{:7}{:20}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                    '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','system size','','=','',_system(exp.nocc, exp.ncore, exp.norb), \
                    '','|','','expansion model','','=','',_model(exp.method), \
                    '','|','','mpi masters & slaves','','=','',_mpi(mpi.num_masters, mpi.global_size),)

        string += '{:7}{:20}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:23}{:1}{:1}{:2}{:.6f}\n'
        form += ('','state (mult.)','','=','',_state(exp.spin, exp.fci_state_root), \
                    '','|','','reference space','','=','',_active_space(exp.occup, exp.ref_space), \
                    '','|','','Hartree-Fock '+exp.target,'','=','',hf_prop,)

        string += '{:7}{:20}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:.6f}\n'
        form += ('','orbitals','','=','',_orbs(exp.orb_type), \
                    '','|','','reference orbs.','','=','',_active_orbs(exp.ref_space), \
                    '','|','','base model '+exp.target,'','=','',base_prop,)

        string += '{:7}{:20}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:.6f}\n'
        form += ('','point group','','=','',_point_group(exp.point_group), \
                    '','|','','base model','','=','',_base(exp.base_method), \
                    '','|','','MBE total '+exp.target,'','=','',mbe_tot_prop,)

        string += '{:7}{:20}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','FCI solver','','=','',_solver(exp.method, exp.fci_solver), \
                    '','|','','','','','','', \
                    '','|','','total time','','=','',_time(exp.time, 'tot_sum', -1),)

        string += '{:7}{:20}{:2}{:1}{:2}{:<13s}{:2}{:1}{:7}{:15}{:2}{:1}{:2}' \
                '{:<16s}{:1}{:1}{:7}{:21}{:3}{:1}{:2}{:<s}\n'
        form += ('','wave funct. symmetry','','=','',_symm(exp.method, exp.point_group, exp.fci_state_sym, exp.pi_prune), \
                    '','|','','','','','','', \
                    '','|','','','','','','',)

        string += DIVIDER+'\n'+FILL+'\n'+DIVIDER+'\n'

        return string.format(*form)


def _timings_prt(exp: ExpCls, method: str) -> str:
        """
        this function returns the timings table
        """
        string: str = DIVIDER[:112]+'\n'
        string += '{:^112}\n'
        form: Tuple[Any, ...] = ('MBE-{:} timings'.format(method.upper()),)

        string += DIVIDER[:112]+'\n'
        string += '{:6}{:9}{:2}{:1}{:8}{:3}{:8}{:1}{:6}{:7}{:6}' \
                  '{:1}{:8}{:3}{:8}{:1}{:4}{:12s}{:4}{:1}{:5}{:4}\n'
        form += ('','MBE order','','|','','MBE','',
                 '|','','purging','','|','','sum','',\
                 '|','','calculations','','|','','in %',)

        string += DIVIDER[:112]+'\n'

        for i, j in enumerate(range(exp.min_order, exp.final_order+1)):
            calc_i = exp.n_tuples['calc'][i]
            rel_i = exp.n_tuples['calc'][i] / exp.n_tuples['theo'][i] * 100.
            calc_tot = sum(exp.n_tuples['calc'][:i+1])
            rel_tot = calc_tot / sum(exp.n_tuples['theo'][:i+1]) * 100.
            string += '{:7}{:>4d}{:6}{:1}{:2}{:>15s}{:2}{:1}{:2}{:>15s}{:2}' \
                      '{:1}{:2}{:>15s}{:2}{:1}{:5}{:>10d}{:5}{:1}{:4}{:6.2f}\n'
            form += ('',j, \
                     '','|','',_time(exp.time, 'mbe', i), \
                     '','|','',_time(exp.time, 'purge', i), \
                     '','|','',_time(exp.time, 'sum', i), \
                     '','|','',calc_i,'','|','',rel_i,)

        string += DIVIDER[:112]+'\n'
        string += '{:8}{:5s}{:4}{:1}{:2}{:>15s}{:2}{:1}{:2}{:>15s}{:2}' \
                  '{:1}{:2}{:>15s}{:2}{:1}{:5}{:>10d}{:5}{:1}{:4}{:6.2f}\n'
        form += ('','total', \
                 '','|','',_time(exp.time, 'tot_mbe', -1), \
                 '','|','',_time(exp.time, 'tot_purge', -1), \
                 '','|','',_time(exp.time, 'tot_sum', -1), \
                 '','|','',calc_tot,'','|','',rel_tot,)

        string += DIVIDER[:112]+'\n'

        return string.format(*form)


def _energy_prt(method: str, root: int, corr_energy: List[float], \
                hf_energy: float, base_energy: float, ref_energy: float, \
                min_order: int, final_order: int) -> str:
        """
        this function returns the energies table
        """
        string: str = DIVIDER[:66]+'\n'
        string_in = 'MBE-{:} energy (root = {:})'.format(method.upper(), root)
        string += '{:^66}\n'
        form: Tuple[Any, ...] = (string_in,)

        string += DIVIDER[:66]+'\n'
        string += '{:6}{:9}{:2}{:1}{:5}{:12}{:5}{:1}{:4}{:}\n'
        form += ('','MBE order','','|','','total energy','','|','','correlation energy',)

        string += DIVIDER[:66]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:5}{:>11.6f}{:6}{:1}{:6}{:>12.5e}\n'
        form += ('','ref','','|','',hf_energy + ref_energy, \
                    '','|','',ref_energy,)

        string += DIVIDER[:66]+'\n'
        energy = _energy(corr_energy, hf_energy, base_energy, ref_energy)

        for i, j in enumerate(range(min_order, final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:5}{:>11.6f}{:6}{:1}{:6}{:>12.5e}\n'
            form += ('',j, \
                        '','|','',energy[i].item(), \
                        '','|','',energy[i].item() - hf_energy,)

        string += DIVIDER[:66]+'\n'

        return string.format(*form)


def _energy_plot(root: int, corr_energy: List[float], hf_energy: float, \
                 base_energy: float, ref_energy: float, min_order: int, \
                 final_order: int) -> matplotlib.figure.Figure:
        """
        this function plots the energies
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # plot results
        ax.plot(np.arange(min_order, final_order+1), \
                _energy(corr_energy, hf_energy, base_energy, ref_energy), \
                marker='x', linewidth=2, mew=1, color='xkcd:kelly green', \
                linestyle='-', label='state {:}'.format(root))

        # set x limits
        ax.set_xlim([0.5, final_order+1 - 0.5])

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
        ax.legend(loc=1, frameon=False)

        return fig


def _excitation_prt(root: int, corr_exc: List[float], ref_exc: float, \
                    min_order: int, final_order: int) -> str:
        """
        this function returns the excitation energies table
        """
        string: str = DIVIDER[:43]+'\n'
        string_in = 'MBE excitation energy (root = '+str(root)+')'
        string += '{:^46}\n'
        form: Tuple[Any, ...] = (string_in,)

        string += DIVIDER[:43]+'\n'
        string += '{:6}{:9}{:2}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','excitation energy',)

        string += DIVIDER[:43]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:7}{:>.5e}\n'
        form += ('','ref','','|','',root,)

        string += DIVIDER[:43]+'\n'
        excitation = _excitation(corr_exc, ref_exc)

        for i, j in enumerate(range(min_order, final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:7}{:>.5e}\n'
            form += ('',j,'','|','',excitation[i].item(),)

        string += DIVIDER[:43]+'\n'

        return string.format(*form)


def _excitation_plot(root: int, corr_exc: List[float], ref_exc: float, \
                     min_order: int, \
                     final_order: int) -> matplotlib.figure.Figure:
        """
        this function plots the excitation energies
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # plot results
        ax.plot(np.arange(min_order, final_order+1), \
                _excitation(corr_exc, ref_exc), marker='x', linewidth=2, mew=1, color='xkcd:dull blue', \
                linestyle='-', label='excitation {:} -> {:}'.format(0, root))

        # set x limits
        ax.set_xlim([0.5, final_order+1 - 0.5])

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
        ax.legend(loc=1, frameon=False)

        return fig


def _dipole_prt(root: int, nuc_dipole: np.ndarray, \
                corr_dipole: List[np.ndarray], hf_dipole: np.ndarray, \
                base_dipole: np.ndarray, ref_dipole: np.ndarray, \
                min_order: int, final_order: int) -> str:
        """
        this function returns the dipole moments table
        """
        string: str = DIVIDER[:82]+'\n'
        string_in = 'MBE dipole moment (root = {:})'.\
                        format(root)
        string += '{:^82}\n'
        form: Tuple[Any, ...] = (string_in,)

        string += DIVIDER[:82]+'\n'
        string += '{:6}{:9}{:2}{:1}{:8}{:25}{:9}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','dipole components (x,y,z)','','|','','dipole moment',)

        string += DIVIDER[:82]+'\n'

        tot_ref_dipole: np.ndarray = hf_dipole + base_dipole + ref_dipole
        dipole = _dipole(corr_dipole, hf_dipole, base_dipole, ref_dipole)
        string += '{:9}{:>3s}{:5}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
        form += ('','ref', \
                    '','|','',nuc_dipole[0] - tot_ref_dipole[0], \
                    '',nuc_dipole[1] - tot_ref_dipole[1], \
                    '',nuc_dipole[2] - tot_ref_dipole[2], \
                    '','|','',np.linalg.norm(nuc_dipole - tot_ref_dipole),)

        string += DIVIDER[:82]+'\n'

        for i, j in enumerate(range(min_order, final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
            form += ('',j, \
                        '','|','',nuc_dipole[0] - dipole[i, 0], \
                        '',nuc_dipole[1] - dipole[i, 1], \
                        '',nuc_dipole[2] - dipole[i, 2], \
                        '','|','',np.linalg.norm(nuc_dipole - dipole[i, :]),)

        string += DIVIDER[:82]+'\n'

        return string.format(*form)


def _dipole_plot(root: int, nuc_dipole: np.ndarray, \
                 corr_dipole: List[np.ndarray], hf_dipole: np.ndarray, \
                 base_dipole: np.ndarray, ref_dipole: np.ndarray, \
                 min_order: int, final_order: int) -> matplotlib.figure.Figure:
        """
        this function plots the dipole moments
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # array of total MBE dipole moment
        dipole = _dipole(corr_dipole, hf_dipole, base_dipole, \
                                     ref_dipole)
        dipole_arr = np.empty(dipole.shape[0], dtype=np.float64)
        for i in range(dipole.shape[0]):
            dipole_arr[i] = np.linalg.norm(nuc_dipole - dipole[i, :])

        # plot results
        ax.plot(np.arange(min_order, final_order+1), \
                dipole_arr, marker='*', linewidth=2, mew=1, color='xkcd:salmon', \
                linestyle='-', label='state {:}'.format(root))

        # set x limits
        ax.set_xlim([0.5, final_order+1 - 0.5])

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
        ax.legend(loc=1, frameon=False)

        return fig


def _trans_prt(root: int, corr_trans: List[np.ndarray], ref_trans: np.ndarray, \
               min_order: int, final_order: int) -> str:
        """
        this function returns the transition dipole moments and oscillator strengths table
        """
        string: str = DIVIDER[:82]+'\n'
        string_in = 'MBE trans. dipole moment (roots 0 > {:})'.\
                        format(root)
        string += '{:^84}\n'
        form: Tuple[Any, ...] = (string_in,)

        string += DIVIDER[:82]+'\n'

        trans = _trans(corr_trans, ref_trans)

        string += '{:6}{:9}{:2}{:1}{:8}{:25}{:9}{:1}{:5}{:}\n'
        form += ('','MBE order','','|','','dipole components (x,y,z)','','|','','dipole moment',)

        tot_ref_trans: np.ndarray = ref_trans
        string += DIVIDER[:82]+'\n'
        string += '{:9}{:>3s}{:5}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
        form += ('','ref', \
                    '','|','',tot_ref_trans[0], '', tot_ref_trans[1], '', tot_ref_trans[2], \
                    '','|','',np.linalg.norm(tot_ref_trans[:]),)

        string += DIVIDER[:82]+'\n'

        for i, j in enumerate(range(min_order, final_order+1)):
            string += '{:7}{:>4d}{:6}{:1}{:4}{:9.6f}{:^3}{:9.6f}{:^3}{:9.6f}{:5}{:1}{:6}{:9.6f}\n'
            form += ('',j, \
                        '','|','',trans[i, 0], \
                        '',trans[i, 1], \
                        '',trans[i, 2], \
                        '','|','',np.linalg.norm(trans[i, :]),)

        string += DIVIDER[:82]+'\n'

        return string.format(*form)


def _trans_plot(root: int, corr_trans: List[np.ndarray], \
                ref_trans: np.ndarray, min_order: int, \
                final_order: int) -> matplotlib.figure.Figure:
        """
        this function plots the transition dipole moments
        """
        # set seaborn
        if SNS_FOUND:
            sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')

        # set subplot
        fig, ax = plt.subplots()

        # array of total MBE transition dipole moment
        trans = _trans(corr_trans, ref_trans)
        trans_arr = np.empty(trans.shape[0], dtype=np.float64)
        for i in range(trans.shape[0]):
            trans_arr[i] = np.linalg.norm(trans[i, :])

        # plot results
        ax.plot(np.arange(min_order, final_order+1), \
                trans_arr, marker='s', linewidth=2, mew=1, color='xkcd:dark magenta', \
                linestyle='-', label='excitation {:} -> {:}'.format(0, root))

        # set x limits
        ax.set_xlim([0.5, final_order+1 - 0.5])

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
        ax.legend(loc=1, frameon=False)

        return fig
