#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
calculation module containing all calculation attributes
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import re
import sys
import os
import ast
import numpy as np
from pyscf import symm, scf
from typing import Dict, List, Tuple, Union, Any

import tools


# attributes
ATTR = ['model', 'base', 'orbs', 'target', 'prot', 'thres', 'mpi', 'extra', 'misc', 'ref', 'state']


class CalcCls:
        """
        this class contains the pymbe calculation attributes
        """
        def __init__(self, ncore: int, nelectron: int, symmetry: Union[bool, str]) -> None:
                """
                init calculation attributes
                """
                # set defaults
                self.model: Dict[str, str] = {'method': 'fci', 'solver': 'pyscf_spin0'}
                self.target: Dict[str, bool] = {'energy': False, 'excitation': False, 'dipole': False, 'trans': False}
                self.prot: Dict[str, int] = {'scheme': 2}
                self.ref: Dict[str, Any] = {'method': 'casci', 'hf_guess': True, 'active': 'manual', \
                                            'select': [i for i in range(ncore, nelectron // 2)], \
                                            'wfnsym': [symm.addons.irrep_id2name(symmetry, 0) if symmetry else 0]}
                self.base: Dict[str, Union[None, str]] = {'method': None}
                self.state: Dict[str, Any] = {'wfnsym': symm.addons.irrep_id2name(symmetry, 0) if symmetry else 0, 'root': 0}
                self.extra: Dict[str, bool] = {'hf_guess': True, 'pi_prune': False}
                self.thres: Dict[str, float] = {'init': 1.e-10, 'relax': 1., 'start': 3}
                self.misc: Dict[str, Any] = {'order': None, 'rst': True, 'rst_freq': int(1e6)}
                self.orbs: Dict[str, str] = {'type': 'can'}
                self.mpi: Dict[str, int] = {}
                self.prop: Dict[str, Dict[str, Union[float, np.ndarray]]] = {'hf': {}, 'base': {}, 'ref': {}}

                # init attributes
                self.restart: bool = False
                self.target_mbe: str = ''
                self.hf: scf.RHF = None
                self.occup: np.ndarray = None
                self.orbsym: np.ndarray = None
                self.mo_energy: np.ndarray = None
                self.mo_coeff: np.ndarray = None
                self.nelec: Tuple[int, ...] = ()
                self.ref_space: Dict[str, np.ndarray] = {'tot': None}
                self.exp_space: Dict[str, np.ndarray] = {'tot': None}


def set_calc(calc: CalcCls) -> CalcCls:
        """
        this function sets calculation and mpi attributes from input file
        """
        # read input file
        try:

            with open(os.getcwd()+'/input') as f:

                content = f.readlines()

                for i in range(len(content)):

                    if content[i].strip():

                        if content[i].split()[0][0] == '#':
                            continue
                        else:
                            attr = re.split('=',content[i])[0].strip()

                            if attr in ATTR:

                                try:
                                    inp = ast.literal_eval(re.split('=',content[i])[1].strip())
                                except ValueError:
                                    raise ValueError('wrong input -- error in reading in {:} dictionary'.format(attr))

                                # make keys uniformly lower-case
                                keys = [key.lower() for key in inp.keys()]

                                # make string values lower-case as well
                                vals = [val.lower() if isinstance(val, str) else val for val in inp.values()]

                                # recast wfnsym as standard symbol
                                if 'wfnsym' in inp.keys():

                                    if attr == 'state':
                                        inp['wfnsym'] = symm.addons.std_symb(inp['wfnsym'])
                                    elif attr == 'ref':
                                        if not isinstance(inp['wfnsym'], list):
                                            inp['wfnsym'] = list(inp['wfnsym'])
                                        inp['wfnsym'] = [symm.addons.std_symb(sym) for sym in inp['wfnsym']]

                                # update calc attribute
                                setattr(calc, attr, {**getattr(calc, attr), **inp})

        except IOError:

            sys.stderr.write('\nIOError : input file not found\n\n')
            raise

        return calc


def sanity_chk(calc: CalcCls, spin: int, atom: Union[List[str], str], \
                symmetry: Union[bool, str]) -> None:
        """
        this function performs sanity checks of calc and mpi attributes
        """
        # expansion model
        tools.assertion(isinstance(calc.model['method'], str), \
                        'input electronic structure method (method) must be a string')
        tools.assertion(calc.model['method'] in ['ccsd', 'ccsd(t)', 'fci'], \
                        'valid expansion models are: ccsd, ccsd(t), and fci')
        tools.assertion(calc.model['solver'] in ['pyscf_spin0', 'pyscf_spin1'], \
                        'valid FCI solvers are: pyscf_spin0 and pyscf_spin1')
        if calc.model['method'] != 'fci':
            tools.assertion(calc.model['solver'] == 'pyscf_spin0', \
                            'setting a FCI solver for a non-FCI expansion model is not meaningful')
        if spin > 0:
            tools.assertion(calc.model['solver'] != 'pyscf_spin0', \
                            'the pyscf_spin0 FCI solver is designed for spin singlets only')

        # reference model
        tools.assertion(calc.ref['method'] in ['casci', 'casscf'], \
                        'valid reference models are: casci and casscf')
        if calc.ref['method'] == 'casscf':
            tools.assertion(calc.model['method'] == 'fci', \
                            'a casscf reference is only meaningful for an fci expansion model')
        tools.assertion(calc.ref['active'] == 'manual', \
                        'active space choices are currently: manual')
        tools.assertion(isinstance(calc.ref['select'], list), \
                        'select key (select) for active space must be a list of orbitals')
        tools.assertion(isinstance(calc.ref['hf_guess'], bool), \
                        'HF initial guess for CASSCF calc (hf_guess) must be a bool')
        if atom:
            if calc.ref['hf_guess']:
                tools.assertion(len(set(calc.ref['wfnsym'])) == 1, \
                                'illegal choice of ref wfnsym when enforcing hf initial guess')
                tools.assertion(calc.ref['wfnsym'][0] == symm.addons.irrep_id2name(symmetry, 0), \
                                'illegal choice of ref wfnsym when enforcing hf initial guess')
            for i in range(len(calc.ref['wfnsym'])):
                try:
                    calc.ref['wfnsym'][i] = symm.addons.irrep_name2id(symmetry, calc.ref['wfnsym'][i])
                except Exception as err:
                    raise ValueError('illegal choice of ref wfnsym -- PySCF error: {:}'.format(err))

        # base model
        if calc.base['method'] is not None:
            tools.assertion(calc.ref['method'] == 'casci', \
                            'use of base model is only permitted for casci expansion references')
            tools.assertion(calc.target['energy'], \
                            'use of base model is only permitted for target energies')
            tools.assertion(calc.base['method'] in ['ccsd', 'ccsd(t)'], \
                            'valid base models are currently: ccsd, and ccsd(t)')

        # state
        if atom:
            try:
                calc.state['wfnsym'] = symm.addons.irrep_name2id(symmetry, calc.state['wfnsym'])
            except Exception as err:
                raise ValueError('illegal choice of state wfnsym -- PySCF error: {:}'.format(err))
            tools.assertion(calc.state['root'] >= 0, \
                            'choice of target state (root) must be an int >= 0')
            if calc.model['method'] != 'fci':
                tools.assertion(calc.state['wfnsym'] == 0, \
                                'illegal choice of wfnsym for chosen expansion model')
                tools.assertion(calc.state['root'] == 0, \
                                'excited states not implemented for chosen expansion model')

        # targets
        tools.assertion(any(calc.target.values()) and len([x for x in calc.target.keys() if calc.target[x]]) == 1, \
                        'one and only one target property must be requested')
        tools.assertion(all(isinstance(i, bool) for i in calc.target.values()), \
                        'values in target input (target) must be bools')
        tools.assertion(set(list(calc.target.keys())) <= set(['energy', 'excitation', 'dipole', 'trans']), \
                        'invalid choice for target property. valid choices are: '
                        'energy, excitation energy (excitation), dipole, and transition dipole (trans)')
        if calc.target['excitation']:
            tools.assertion(calc.state['root'] > 0, \
                            'calculation of excitation energy (excitation) requires target state root >= 1')
        if calc.target['trans']:
            tools.assertion(calc.state['root'] > 0, \
                            'calculation of transition dipole moment (trans) requires target state root >= 1')

        # extra
        tools.assertion(isinstance(calc.extra['hf_guess'], bool), \
                        'HF initial guess for FCI calcs (hf_guess) must be a bool')
        tools.assertion(isinstance(calc.extra['pi_prune'], bool), \
                        'pruning of pi-orbitals (pi_prune) must be a bool')
        if calc.extra['pi_prune']:
            tools.assertion(symm.addons.std_symb(symmetry) == 'D2h', \
                            'pruning of pi-orbitals (pi_prune) is only implemented for D2h symmetry')

        # screening protocol
        tools.assertion(isinstance(calc.prot['scheme'], int), \
                        'screening protocol scheme (scheme) must be an int')
        tools.assertion(0 < calc.prot['scheme'] < 4, \
                        'valid screening protocol schemes (scheme) are: 1 (1st gen), 2 (2nd) gen, 3 (3rd gen)')

        # expansion thresholds
        tools.assertion(set(list(calc.thres.keys())) <= set(['init', 'relax', 'start']), \
                        'valid input in thres dict is: init, relax, and start')
        tools.assertion(isinstance(calc.thres['init'], float), \
                        'initial threshold (init) must be a float')
        tools.assertion(calc.thres['init'] >= 0., \
                        'initial threshold (init) must be a float >= 0.0')
        tools.assertion(isinstance(calc.thres['relax'], float), \
                        'initial threshold (init) must be a float')
        tools.assertion(calc.thres['relax'] >= 1., \
                        'threshold relaxation (relax) must be a float >= 1.0')
        tools.assertion(isinstance(calc.thres['start'], int), \
                        'start threshold parameter (start) must be an int')
        tools.assertion(calc.thres['start'] >= 1, \
                        'start threshold parameter (start) must be an int >= 1')

        # orbital representation
        tools.assertion(calc.orbs['type'] in ['can', 'local', 'ccsd', 'ccsd(t)'], \
                        'valid occupied orbital representations (occ) are currently: '
                        'canonical (can), pipek-mezey (local), or natural orbs (ccsd or ccsd(t))')
        if calc.orbs['type'] != 'can':
            tools.assertion(calc.ref['method'] == 'casci', \
                            'non-canonical orbitals requires casci expansion reference')
        if atom and calc.orbs['type'] == 'local':
            tools.assertion(symmetry == 'C1', \
                            'the combination of local orbs and point group symmetry '
                            'different from c1 is not allowed')

        # misc
        tools.assertion(isinstance(calc.misc['order'], (int, type(None))), \
                        'maximum expansion order (order) must be an int >= 1')
        if calc.misc['order'] is not None:
            tools.assertion(calc.misc['order'] >= 1, \
                            'maximum expansion order (order) must be an int >= 1')
            if len(calc.ref['select']) == 0:
                tools.assertion(calc.misc['order'] >= 2, \
                                'maximum expansion order (order) must be an int >= 2 '
                                'for vacuum reference spaces')
        tools.assertion(isinstance(calc.misc['rst'], bool), \
                        'restart logical (rst) must be a bool')
        tools.assertion(isinstance(calc.misc['rst_freq'], int), \
                        'restart freqeuncy (rst_freq) must be an int')
        tools.assertion(calc.misc['rst_freq'] >= 1, \
                        'restart frequency (rst_freq) must be an int >= 1')


