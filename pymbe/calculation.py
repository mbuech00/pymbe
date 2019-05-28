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
from pyscf import symm

import tools


# attributes
ATTR = ['model', 'base', 'orbs', 'target', 'prot', 'thres', 'mpi', 'extra', 'misc', 'ref', 'state']


class CalcCls(object):
        """
        this class contains the pymbe calculation attributes
        """
        def __init__(self, mol):
                """
                init calculation attributes

                :param mpi: pymbe mpi object
                :param mol: pymbe mol object
                """
                # set defaults
                self.model = {'method': 'fci', 'solver': 'pyscf_spin0'}
                self.target = {'energy': False, 'excitation': False, 'dipole': False, 'trans': False}
                self.prot = {'scheme': 2, 'seed': 'occ'}
                self.ref = {'method': 'casci', 'hf_guess': True, 'active': 'manual', \
                            'select': [i for i in range(mol.ncore, mol.nelectron // 2)], \
                            'wfnsym': [symm.addons.irrep_id2name(mol.symmetry, 0) if mol.symmetry else 0]}
                self.base = {'method': None}
                self.state = {'wfnsym': symm.addons.irrep_id2name(mol.symmetry, 0) if mol.symmetry else 0, 'root': 0}
                self.extra = {'hf_guess': True, 'pi_prune': False}
                self.thres = {'init': 1.0e-10, 'relax': 1.0}
                self.misc = {'order': None}
                self.orbs = {'type': 'can'}
                self.mpi = {'task_size': 5}
                self.prop = {'hf': {}, 'base': {}, 'ref': {}}


def set_calc(calc):
        """
        this function sets calculation and mpi attributes from input file

        :param calc: pymbe calc object
        :return: updated calc object
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

            restart.rm()
            sys.stderr.write('\nIOError : input file not found\n\n')
            raise

        return calc


def sanity_chk(mol, calc):
        """
        this function performs sanity checks of calc and mpi attributes

        :param mol: pymbe mol object
        :param calc: pymbe calc object
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
        if mol.spin > 0:
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
        if mol.atom:
            if calc.ref['hf_guess']:
                tools.assertion(len(set(calc.ref['wfnsym'])) == 1, \
                                'illegal choice of ref wfnsym when enforcing hf initial guess')
                tools.assertion(calc.ref['wfnsym'][0] == symm.addons.irrep_id2name(mol.symmetry, 0), \
                                'illegal choice of ref wfnsym when enforcing hf initial guess')
            for i in range(len(calc.ref['wfnsym'])):
                try:
                    calc.ref['wfnsym'][i] = symm.addons.irrep_name2id(mol.symmetry, calc.ref['wfnsym'][i])
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
        if mol.atom:
            try:
                calc.state['wfnsym'] = symm.addons.irrep_name2id(mol.symmetry, calc.state['wfnsym'])
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
            tools.assertion(calc.target['excitation'], \
                            'calculation of transition dipole moment (trans) '
                            'requires calculation of excitation energy (excitation)')

        # extra
        tools.assertion(isinstance(calc.extra['hf_guess'], bool), \
                        'HF initial guess for FCI calcs (hf_guess) must be a bool')
        tools.assertion(isinstance(calc.extra['pi_prune'], bool), \
                        'pruning of pi-orbitals (pi_prune) must be a bool')
        if calc.extra['pi_prune']:
            tools.assertion(symm.addons.std_symb(mol.symmetry) == 'D2h', \
                            'pruning of pi-orbitals (pi_prune) is only implemented for D2h symmetry')

        # screening protocol
        tools.assertion(isinstance(calc.prot['scheme'], int), \
                        'screening protocol scheme (scheme) must be an int')
        tools.assertion(0 < calc.prot['scheme'] < 4, \
                        'valid screening protocol schemes (scheme) are: 1 (1st gen), 2 (2nd) gen, 3 (3rd gen)')
        tools.assertion(isinstance(calc.prot['seed'], str), \
                        'screening seed space (seed) must be a string')
        tools.assertion(calc.prot['seed'] in ['occ', 'virt'], \
                        'valid screening seed spaces are: occ and virt')

        # expansion thresholds
        tools.assertion(all(isinstance(i, float) for i in calc.thres.values()), \
                        'values in threshold input (thres) must be floats')
        tools.assertion(set(list(calc.thres.keys())) <= set(['init', 'relax']), \
                        'valid input in thres dict is: init and relax')
        tools.assertion(calc.thres['init'] >= 0.0, \
                        'initial threshold (init) must be a float >= 0.0')
        tools.assertion(calc.thres['relax'] >= 1.0, \
                        'threshold relaxation (relax) must be a float >= 1.0')

        # orbital representation
        tools.assertion(calc.orbs['type'] in ['can', 'local', 'ccsd', 'ccsd(t)'], \
                        'valid occupied orbital representations (occ) are currently: '
                        'canonical (can), pipek-mezey (local), or natural orbs (ccsd or ccsd(t))')
        if calc.orbs['type'] != 'can':
            tools.assertion(calc.ref['method'] == 'casci', \
                            'non-canonical orbitals requires casci expansion reference')
        if mol.atom and calc.orbs['type'] == 'local':
            tools.assertion(mol.symmetry == 'C1', \
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

        # mpi parameters
        tools.assertion(all(isinstance(i, int) for i in calc.mpi.values()), \
                        'values in mpi input (mpi) must be ints')
        tools.assertion(calc.mpi['task_size'] >= 1, \
                        'mpi task size (task_size) must be an int >= 1')



