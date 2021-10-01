#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
system module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import re
import sys
import os
from ast import literal_eval
from math import floor
from pyscf import gto, symm, ao2mo
from typing import List, Tuple, Dict, Union, Any, Callable

from tools import assertion


class MolCls(gto.Mole):
        """
        this class contains all molecule attributes (inherited from the pyscf gto.Mole class)
        """
        def __init__(self) -> None:
                """
                init molecule attributes
                """
                super().__init__()


        def defaults(self) -> None:
                """
                set mol defaults
                """
                self.atom: Union[List[str], str] = ''
                self.system: Dict[str, Any] = {'charge': 0, 'spin': 0, 'symmetry': 'c1', 'x2c': False, \
                                               'basis': 'sto-3g', 'cart': False, 'unit': 'ang', \
                                               'frozen': False, 'ncore': 0, 'debug': 0, 'gauge': 'zero', \
                                               'u': 1., 'n': 1., 'matrix': (1, 6), 'pbc': True}
                self.max_memory: float = 1e10
                self.incore_anyway: bool = True


        def make(self) -> None:
                """
                this function builds the pyscf Mole object
                """
                try:
                    self.build(dump_input=False, parse_arg=False, verbose=0)
                except RuntimeWarning as err:
                    try:
                        raise RuntimeError
                    except RuntimeError:
                        sys.stderr.write('\nValueError: non-sensible system input\n'
                                            'PySCF error : {:}\n\n'.format(err))
                        raise
                # set core region
                if self.frozen:
                    self.ncore: int = _set_ncore(self.natm, self.atom_charge)


def _set_ncore(natm: int, atom_charge: Callable[[int], int]) -> int:
        """
        this function sets ncore
        """
        # init ncore
        ncore: int = 0
        # loop over atoms
        for i in range(natm):
            if atom_charge(i) > 2:
                ncore += 1
            if atom_charge(i) > 12:
                ncore += 4
            if atom_charge(i) > 20:
                ncore += 4
            if atom_charge(i) > 30:
                ncore += 6
        return ncore


def set_system(mol: MolCls) -> MolCls:
        """
        this function sets system attributes from input file
        """
        # read input file
        try:
            with open(os.getcwd()+'/input') as f:
                content = f.readlines()
                for i in range(len(content)):
                    if content[i].strip():
                        if content[i].split()[0][0] == '#':
                            continue
                        elif re.split('=',content[i])[0].strip() == 'atom':
                            mol.atom = ''
                            for j in range(i+1, len(content)):
                                if content[j][:3] == "'''" or content[j][:3] == '"""':
                                    break
                                else:
                                    mol.atom += content[j]
                        elif re.split('=',content[i])[0].strip() == 'system':
                            try:
                                inp = literal_eval(re.split('=',content[i])[1].strip())
                            except ValueError:
                                raise ValueError('wrong input -- error in reading in system dictionary')
                            # update system
                            mol.system = {**mol.system, **inp}
        except IOError:
            sys.stderr.write('\nIOError : input file not found\n\n')
            raise
        return mol


def translate_system(mol: MolCls) -> MolCls:
        """
        this function translates system input to mol attributes
        """
        # copy all attributes
        for key, val in mol.system.items():
            setattr(mol, key, val)
        if not mol.atom:
            # hubbard hamiltonian
            mol.atom = []
            mol.symmetry = mol.hf_symmetry = False
            mol.nelectron = floor(mol.matrix[0] * mol.matrix[1] * mol.n)
        else:
            # backwards compatibility for sym <-> symmetry
            if hasattr(mol, 'sym'):
                mol.symmetry = mol.sym
            # recast symmetries as standard symbols
            if isinstance(mol.symmetry, str):
                mol.symmetry = symm.addons.std_symb(mol.symmetry)
        return mol


def sanity_check(mol: MolCls) -> None:
        """
        this function performs sanity checks of mol attributes
        """
        # charge
        assertion(isinstance(mol.charge, int), \
                        'charge input in system dict (charge) must be an int')
        # x2c
        assertion(isinstance(mol.x2c, bool), \
                        'x2c input in system dict (x2c) must be a bool')
        # spin
        assertion(isinstance(mol.spin, int) and mol.spin >= 0, \
                        'spin input (2S) in system dict (spin) must be an int >= 0')
        # symmetry
        assertion(isinstance(mol.symmetry, (str, bool)), \
                        'symmetry input in system dict (symmetry) must be a str or bool')
        # basis
        assertion(isinstance(mol.basis, (str, dict)), \
                        'basis set input in system dict (basis) must be a str or a dict')
        # cart
        assertion(isinstance(mol.cart, bool), \
                        'cartesian gto basis input in system dict (cart) must be a bool')
        # unit
        assertion(isinstance(mol.unit, str), \
                        'unit input in system dict (unit) must be a str')
        # frozen
        assertion(isinstance(mol.frozen, bool), \
                        'frozen core input in system dict (frozen) must be a bool')
        # gauge origin
        assertion(isinstance(mol.gauge, str), \
                        'gauge origin input in system dict (gauge) must be a string')
        assertion(mol.gauge in ['zero', 'charge'], \
                        'valid gauge origins (gauge) are: zero and charge')
        # debug
        assertion(type(mol.debug) is int, \
                        'debug input in system dict (debug) must be an int')
        assertion(mol.debug >= 0, \
                        'debug input in system dict (debug) must be an int >= 0')
        # hubbard
        if not mol.atom:
            # matrix
            assertion(isinstance(mol.matrix, tuple), \
                            'hubbard matrix input in system dict (matrix) must be a tuple')
            assertion(len(mol.matrix) == 2, \
                            'hubbard matrix input in system dict (matrix) must have a dimension of 2')
            assertion(isinstance(mol.matrix[0], int) and isinstance(mol.matrix[1], int), \
                            'hubbard matrix input in system dict (matrix) must be a tuple of ints')
            # u parameter
            assertion(isinstance(mol.u, float), \
                            'hubbard on-site repulsion parameter (u) must be a float')
            assertion(mol.u > 0., \
                            'only repulsive hubbard models are implemented (u > 0.0)')
            # n parameter
            assertion(isinstance(mol.n, float), \
                            'hubbard model filling parameter (n) must be a float')
            assertion(mol.n > 0. and mol.n < 2., \
                            'hubbard model filling parameter (n) must be a float between 0.0 < n < 2.0')
            # periodic boundary conditions
            assertion(isinstance(mol.pbc, bool), \
                            'hubbard model pbc parameter (pbc) must be a bool')
