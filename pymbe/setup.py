#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
setup module containing all initialization functions
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import numpy as np
try:
    from pyscf import lib, scf
except ImportError:
    sys.stderr.write('\nImportError : pyscf module not found\n\n')

import parallel
import system
import calculation
import expansion
import kernel
import restart
import tools


def main():
        """
        this function initializes and broadcasts mpi, mol, calc, and exp objects

        :return: pymbe mpi object,
                 pymbe mol object,
                 pymbe calc object,
                 pymbe exp object
        """
        # mpi object
        mpi = parallel.MPICls()

        # mol object
        mol = _mol(mpi)

        # calc object
        calc = _calc(mpi, mol)

        # exp object
        mol, calc, exp = _exp(mpi, mol, calc)

        return mpi, mol, calc, exp


def _mol(mpi):
        """
        this function initializes mol object

        :param mpi: pymbe mpi object
        :return: pymbe mol object
        """
        # mol object
        mol = system.MolCls()

        # input handling
        if mpi.global_master:

            # read input
            mol = system.set_system(mol)

            # translate input
            mol = system.translate_system(mol)

            # sanity check
            system.sanity_chk(mol)

        # bcast info from master to slaves
        mol = parallel.mol(mpi, mol)

        # make pyscf mol object
        mol.make(mpi)

        return mol


def _calc(mpi, mol):
        """
        this function initializes calc object

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :return: pymbe calc object
        """
        # calc object
        calc = calculation.CalcCls(mol)

        # input handling
        if mpi.global_master:

            # read input
            calc = calculation.set_calc(calc)

            # sanity check
            calculation.sanity_chk(mol, calc)

            # set target
            calc.target = [x for x in calc.target.keys() if calc.target[x]][0]

            # restart logical
            calc.restart = restart.restart()

        # bcast info from master to slaves
        calc = parallel.calc(mpi, calc)

        return calc


def _exp(mpi, mol, calc):
        """
        this function initializes exp object

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: updated mol object,
                 updated calc object,
                 pymbe exp object
        """
        # get dipole integrals
        mol.dipole = kernel.dipole_ints(mol) if calc.target in ['dipole', 'trans'] else None

        # nuclear repulsion energy
        mol.e_nuc = np.asscalar(mol.energy_nuc()) if mol.atom else 0.0

        if mpi.global_master:

            if calc.restart:

                # read fundamental info
                mol, calc = restart.read_fund(mol, calc)

                # read properties
                mol, calc = restart.read_prop(mol, calc)

            else:

                # hf calculation
                mol.nocc, mol.nvirt, mol.norb, calc.hf, \
                    calc.prop['hf']['energy'], calc.prop['hf']['dipole'], \
                    calc.occup, calc.orbsym, calc.mo_energy, calc.mo_coeff = kernel.hf(mol, calc.target)

                # reference and expansion spaces and mo coefficients
                calc.mo_energy, calc.mo_coeff, \
                    calc.nelec, calc.ref_space, calc.exp_space = kernel.ref_mo(mol, calc)

        # bcast fundamental info
        mol, calc = parallel.fund(mpi, mol, calc)

        # get handles to all integral windows
        mol.hcore, mol.vhf, mol.eri = kernel.ints(mpi, mol, calc.mo_coeff)

        # write fundamental info
        if mpi.global_master and calc.misc['rst']:
            restart.write_fund(mol, calc)

        # pyscf hf object not needed anymore
        if mpi.global_master and not calc.restart:
            del calc.hf

        if mpi.global_master:

            # base energy
            if calc.base['method'] is not None:
                calc.prop['base']['energy'] = kernel.base(mol, calc.occup, calc.base['method'])
            else:
                calc.prop['base']['energy'] = 0.0

            # reference space properties
            calc.prop['ref'][calc.target] = kernel.ref_prop(mol, calc)

        # mo_coeff not needed anymore
        if mpi.global_master:
            del calc.mo_coeff

        # bcast properties
        calc = parallel.prop(mpi, calc)

        # write properties
        if mpi.global_master and calc.misc['rst']:
            restart.write_prop(mol, calc)

        # exp object
        exp = expansion.ExpCls(mol, calc)

        # init hashes, n_tasks, and tuples
        exp.hashes, exp.tuples, exp.n_tasks, exp.min_order = expansion.init_tup(mpi, mol, calc)

        # possible restart
        if calc.restart:
            exp.start_order = restart.main(mpi, calc, exp)
        else:
            exp.start_order = exp.min_order

        return mol, calc, exp


def settings():
        """
        this function sets and asserts some general settings
        """
        # only run with python3+
        tools.assertion(3 <= sys.version_info[0], 'PyMBE only runs under python3+')

        # force OMP_NUM_THREADS = 1
        lib.num_threads(1)

        # mute scf checkpoint files
        scf.hf.MUTE_CHKFILE = True

        # PYTHONHASHSEED = 0
        pythonhashseed = os.environ.get('PYTHONHASHSEED', -1)
        tools.assertion(int(pythonhashseed) == 0, \
                        'environment variable PYTHONHASHSEED must be set to zero')


