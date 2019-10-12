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
import os.path
import numpy as np
import json
from mpi4py import MPI
try:
    from pyscf import lib, symm, scf
except ImportError:
    sys.stderr.write('\nImportError : pyscf module not found\n\n')
from typing import Tuple

import parallel
import system
import calculation
import expansion
import kernel
import tools


# restart folder
RST = os.getcwd()+'/rst'


def main() -> Tuple[parallel.MPICls, system.MolCls, calculation.CalcCls, expansion.ExpCls]:
        """
        this function initializes and broadcasts mpi, mol, calc, and exp objects
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


def _mol(mpi: parallel.MPICls) -> system.MolCls:
        """
        this function initializes a mol object
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
        mol.make()

        return mol


def _calc(mpi: parallel.MPICls, mol: system.MolCls) -> calculation.CalcCls:
        """
        this function initializes a calc object
        """
        # calc object
        calc = calculation.CalcCls(mol.ncore, mol.nelectron, mol.symmetry)

        # input handling
        if mpi.global_master:

            # read input
            calc = calculation.set_calc(calc)

            # set target
            calc.target_mbe = [x for x in calc.target.keys() if calc.target[x]][0]

            # sanity check
            calculation.sanity_chk(calc, mol.spin, mol.atom, mol.symmetry)

            # restart folder and logical
            if not os.path.isdir(RST):
                os.mkdir(RST)
                calc.restart = False
            else:
                calc.restart = True

        # bcast info from master to slaves
        calc = parallel.calc(mpi, calc)

        return calc


def _exp(mpi: parallel.MPICls, mol: system.MolCls, \
            calc: calculation. CalcCls) -> Tuple[system.MolCls, calculation.CalcCls, expansion.ExpCls]:
        """
        this function initializes an exp object
        """
        # get dipole integrals
        mol.dipole = kernel.dipole_ints(mol) if calc.target_mbe in ['dipole', 'trans'] else None

        # nuclear repulsion energy
        mol.e_nuc = np.asscalar(mol.energy_nuc()) if mol.atom else 0.

        if mpi.global_master:

            if calc.restart:

                # read fundamental info
                mol, calc = restart_read_fund(mol, calc)

                # read properties
                mol, calc = restart_read_prop(mol, calc)

            else:

                # hf calculation
                mol.nocc, mol.nvirt, mol.norb, calc.hf, \
                    calc.prop['hf']['energy'], calc.prop['hf']['dipole'], \
                    calc.occup, calc.orbsym, calc.mo_coeff = kernel.hf(mol, calc.target_mbe)

                # reference and expansion spaces and mo coefficients
                calc.mo_coeff, calc.nelec, calc.ref_space, calc.exp_space = kernel.ref_mo(mol, calc)

        # bcast fundamental info
        mol, calc = parallel.fund(mpi, mol, calc)

        # get handles to all integral windows
        mol.hcore, mol.vhf, mol.eri = kernel.ints(mol, calc.mo_coeff, mpi.global_master, mpi.local_master, \
                                                    mpi.global_comm, mpi.local_comm, mpi.master_comm, mpi.num_masters)

        # write fundamental info
        if not calc.restart and mpi.global_master and calc.misc['rst']:
            restart_write_fund(mol, calc)

        # pyscf hf object not needed anymore
        if mpi.global_master and not calc.restart:
            del calc.hf

        if mpi.global_master:

            # base energy
            if calc.base['method'] is not None:
                calc.prop['base']['energy'], \
                    calc.prop['base']['dipole'] = kernel.base(mol, calc.occup, calc.base['method'], calc.target_mbe, \
                                                               mol.dipole, calc.mo_coeff, calc.prop['hf']['dipole'])
            else:
                calc.prop['base']['energy'] = 0.
                calc.prop['base']['dipole'] = np.zeros(3, dtype=np.float64)

            # reference space properties
            calc.prop['ref'][calc.target_mbe] = kernel.ref_prop(mol, calc)

        # mo_coeff not needed anymore
        if mpi.global_master:
            del calc.mo_coeff

        # bcast properties
        calc = parallel.prop(mpi, calc)

        # write properties
        if not calc.restart and mpi.global_master and calc.misc['rst']:
            restart_write_prop(mol, calc)

        # exp object
        exp = expansion.ExpCls(calc)

        # init hashes, n_tasks, and tuples
        exp.hashes, exp.tuples, exp.n_tasks, \
            exp.min_order = expansion.init_tup(calc.occup, calc.ref_space, calc.exp_space, \
                                                mpi.local_master, mpi.local_comm, calc.extra['pi_prune'])

        # possible restart
        if calc.restart:
            exp.start_order = restart_main(mpi, calc, exp)
        else:
            exp.start_order = exp.min_order

        return mol, calc, exp


def restart_main(mpi: parallel.MPICls, calc: calculation.CalcCls, exp: expansion.ExpCls) -> int:
        """
        this function reads in all expansion restart files and returns the start order
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key=tools.natural_keys)

        # loop over n_tasks files
        if mpi.global_master:

            for i in range(len(files)):

                # read n_tasks
                if 'mbe_n_tasks' in files[i]:
                    exp.n_tasks.append(np.load(os.path.join(RST, files[i])))

            mpi.global_comm.bcast(exp.n_tasks, root=0)

        else:

            exp.n_tasks = mpi.global_comm.bcast(None, root=0)

        # loop over all other files
        for i in range(len(files)):

            # read tuples
            if 'mbe_tup' in files[i]:
                n_tasks = exp.n_tasks[-1]
                order = len(exp.n_tasks) + exp.min_order - 1
                if mpi.local_master:
                    exp.tuples = MPI.Win.Allocate_shared(2 * n_tasks * order, 2, comm=mpi.local_comm)
                else:
                    exp.tuples = MPI.Win.Allocate_shared(0, 2, comm=mpi.local_comm)
                buf = exp.tuples.Shared_query(0)[0]
                tuples = np.ndarray(buffer=buf, dtype=np.int16, shape=(n_tasks, order))
                if mpi.global_master:
                    tuples[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    tuples[:] = parallel.bcast(mpi.master_comm, tuples)
                mpi.local_comm.Barrier()

            # read hashes
            elif 'mbe_hash' in files[i]:
                n_tasks = exp.n_tasks[len(exp.hashes)]
                if mpi.local_master:
                    exp.hashes.append(MPI.Win.Allocate_shared(8 * n_tasks, 8, comm=mpi.local_comm))
                else:
                    exp.hashes.append(MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm))
                buf = exp.hashes[-1].Shared_query(0)[0]
                hashes = np.ndarray(buffer=buf, dtype=np.int64, shape=(n_tasks,))
                if mpi.global_master:
                    hashes[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    hashes[:] = parallel.bcast(mpi.master_comm, hashes)
                mpi.local_comm.Barrier()

            # read increments
            elif 'mbe_inc' in files[i]:
                n_tasks = exp.n_tasks[len(exp.prop[calc.target_mbe]['inc'])]
                if mpi.local_master:
                    if calc.target_mbe in ['energy', 'excitation']:
                        exp.prop[calc.target_mbe]['inc'].append(MPI.Win.Allocate_shared(8 * n_tasks, 8, comm=mpi.local_comm))
                    elif calc.target_mbe in ['dipole', 'trans']:
                        exp.prop[calc.target_mbe]['inc'].append(MPI.Win.Allocate_shared(8 * n_tasks * 3, 8, comm=mpi.local_comm))
                else:
                    exp.prop[calc.target_mbe]['inc'].append(MPI.Win.Allocate_shared(0, 8, comm=mpi.local_comm))
                buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0] # type: ignore
                if calc.target_mbe in ['energy', 'excitation']:
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tasks,))
                elif calc.target_mbe in ['dipole', 'trans']:
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tasks, 3))
                if mpi.global_master:
                    inc[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master:
                    inc[:] = parallel.bcast(mpi.master_comm, inc)
                mpi.local_comm.Barrier()

            if mpi.global_master:

                # read total properties
                if 'mbe_tot' in files[i]:
                    exp.prop[calc.target_mbe]['tot'].append(np.load(os.path.join(RST, files[i])).tolist())

                # read ndets statistics
                elif 'mbe_mean_ndets' in files[i]:
                    exp.mean_ndets.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_min_ndets' in files[i]:
                    exp.min_ndets.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_max_ndets' in files[i]:
                    exp.max_ndets.append(np.load(os.path.join(RST, files[i])))

                # read inc statistics
                elif 'mbe_mean_inc' in files[i]:
                    exp.mean_inc.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_min_inc' in files[i]:
                    exp.min_inc.append(np.load(os.path.join(RST, files[i])))
                elif 'mbe_max_inc' in files[i]:
                    exp.max_inc.append(np.load(os.path.join(RST, files[i])))

                # read timings
                elif 'mbe_time_mbe' in files[i]:
                    exp.time['mbe'].append(np.load(os.path.join(RST, files[i])).tolist())
                elif 'mbe_time_screen' in files[i]:
                    exp.time['screen'].append(np.load(os.path.join(RST, files[i])).tolist())

        # mpi barrier
        mpi.global_comm.Barrier()

        return tuples.shape[1]


def restart_write_fund(mol: system.MolCls, calc: calculation.CalcCls) -> None:
        """
        this function writes all fundamental info restart files
        """
        # write dimensions
        dims = {'nocc': mol.nocc, 'nvirt': mol.nvirt, 'norb': mol.norb, 'nelec': calc.nelec}
        with open(os.path.join(RST, 'dims.rst'), 'w') as f:
            json.dump(dims, f)

        # write expansion spaces
        np.save(os.path.join(RST, 'ref_space_occ'), calc.ref_space['occ'])
        np.save(os.path.join(RST, 'ref_space_virt'), calc.ref_space['virt'])
        np.save(os.path.join(RST, 'ref_space_tot'), calc.ref_space['tot'])
        np.save(os.path.join(RST, 'exp_space_occ'), calc.exp_space['occ'])
        np.save(os.path.join(RST, 'exp_space_virt'), calc.exp_space['virt'])
        np.save(os.path.join(RST, 'exp_space_seed'), calc.exp_space['seed'])
        np.save(os.path.join(RST, 'exp_space_tot'), calc.exp_space['tot'])
        np.save(os.path.join(RST, 'exp_space_pi_orbs'), calc.exp_space['pi_orbs'])
        np.save(os.path.join(RST, 'exp_space_pi_hashes'), calc.exp_space['pi_hashes'])

        # occupation
        np.save(os.path.join(RST, 'occup'), calc.occup)

        # write orbital coefficients
        np.save(os.path.join(RST, 'mo_coeff'), calc.mo_coeff)


def restart_read_fund(mol: system.MolCls, calc: calculation.CalcCls) -> Tuple[system.MolCls, calculation.CalcCls]:
        """
        this function reads all fundamental info restart files
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key=tools.natural_keys)

        # loop over files
        for i in range(len(files)):

            # read dimensions
            if 'dims' in files[i]:
                with open(os.path.join(RST, files[i]), 'r') as f:
                    dims = json.load(f)
                mol.nocc = dims['nocc']; mol.nvirt = dims['nvirt']
                mol.norb = dims['norb']; calc.nelec = dims['nelec']

            # read expansion spaces
            elif 'ref_space_occ' in files[i]:
                calc.ref_space['occ'] = np.load(os.path.join(RST, files[i]))
            elif 'ref_space_virt' in files[i]:
                calc.ref_space['virt'] = np.load(os.path.join(RST, files[i]))
            elif 'ref_space_tot' in files[i]:
                calc.ref_space['tot'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_occ' in files[i]:
                calc.exp_space['occ'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_virt' in files[i]:
                calc.exp_space['virt'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_seed' in files[i]:
                calc.exp_space['seed'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_tot' in files[i]:
                calc.exp_space['tot'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_pi_orbs' in files[i]:
                calc.exp_space['pi_orbs'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_pi_hashes' in files[i]:
                calc.exp_space['pi_hashes'] = np.load(os.path.join(RST, files[i]))

            # read occupation
            elif 'occup' in files[i]:
                calc.occup = np.load(os.path.join(RST, files[i]))

            # read orbital coefficients
            elif 'mo_coeff' in files[i]:
                calc.mo_coeff = np.load(os.path.join(RST, files[i]))
                if mol.atom:
                    calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
                else:
                    calc.orbsym = np.zeros(mol.norb, dtype=np.int)

        return mol, calc


def restart_write_prop(mol: system.MolCls, calc: calculation.CalcCls) -> None:
        """
        this function writes all property restart files
        """
        # write hf, reference, and base properties
        energies = {'hf': calc.prop['hf']['energy']}

        if calc.target_mbe == 'energy':

            energies['base'] = calc.prop['base']['energy']
            energies['ref'] = calc.prop['ref']['energy']

        with open(os.path.join(RST, 'energies.rst'), 'w') as f:
            json.dump(energies, f)

        if calc.target_mbe == 'excitation':

            excitations = {'ref': calc.prop['ref']['excitation']}
            with open(os.path.join(RST, 'excitations.rst'), 'w') as f:
                json.dump(excitations, f)

        elif calc.target_mbe == 'dipole':

            dipoles = {'hf': calc.prop['hf']['dipole'].tolist()} # type: ignore
            dipoles['base'] = calc.prop['base']['dipole'].tolist() # type: ignore
            dipoles['ref'] = calc.prop['ref']['dipole'].tolist() # type: ignore
            with open(os.path.join(RST, 'dipoles.rst'), 'w') as f:
                json.dump(dipoles, f)

        elif calc.target_mbe == 'trans':

            transitions = {'ref': calc.prop['ref']['trans'].tolist()} # type: ignore
            with open(os.path.join(RST, 'transitions.rst'), 'w') as f:
                json.dump(transitions, f)


def restart_read_prop(mol: system.MolCls, calc: calculation.CalcCls) -> Tuple[system.MolCls, calculation.CalcCls]:
        """
        this function reads all property restart files
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key=tools.natural_keys)

        # loop over files
        for i in range(len(files)):

            # read hf and base properties
            if 'energies' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    energies = json.load(f)
                calc.prop['hf']['energy'] = energies['hf']

                if calc.target_mbe == 'energy':
                    calc.prop['base']['energy'] = energies['base']
                    calc.prop['ref']['energy'] = energies['ref']

            elif 'excitations' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    excitations = json.load(f)
                calc.prop['ref']['excitation'] = excitations['ref']

            elif 'dipoles' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    dipoles = json.load(f)
                calc.prop['hf']['dipole'] = np.asarray(dipoles['hf'])
                calc.prop['base']['dipole'] = np.asarray(dipoles['base'])
                calc.prop['ref']['dipole'] = np.asarray(dipoles['ref'])

            elif 'transitions' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    transitions = json.load(f)
                calc.prop['ref']['trans'] = np.asarray(transitions['ref'])

        return mol, calc


def settings() -> None:
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


