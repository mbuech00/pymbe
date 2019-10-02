#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
restart module containing all functions related to writing and reading restart files
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import json
import os
import os.path
import shutil
import re
from pyscf import symm

import parallel


# restart folder
RST = os.getcwd()+'/rst'


def restart():
        """
        this function returns the restart logical

        :return: bool
        """
        if not os.path.isdir(RST):
            os.mkdir(RST)
            return False
        else:
            return True


def rm():
        """
        this function removes the rst directory in case pymbe successfully terminates
        """
        shutil.rmtree(RST)


def main(mpi, calc, exp):
        """
        this function reads in all expansion restart files and returns the start order

        :param calc: pymbe calc object
        :param exp: pymbe exp object
        :return: integer
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key=_natural_keys)

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
                buf = exp.prop[calc.target_mbe]['inc'][-1].Shared_query(0)[0]
                if calc.target_mbe in ['energy', 'excitation']:
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tasks,))
                elif calc.target_mbe in ['dipole', 'trans']:
                    inc = np.ndarray(buffer=buf, dtype=np.float64, shape=(n_tasks, 3))
                if mpi.global_master:
                    inc[:] = np.load(os.path.join(RST, files[i]))
                if mpi.num_masters > 1 and mpi.local_master and len(exp.prop[calc.target_mbe]['inc']) < len(exp.n_tasks):
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


def write_gen(order, arr, string):
        """
        this function writes a general restart file corresponding to input string

        :param order: current mbe order. integer or None
        :param arr: saved quantity. numpy array of arbitrary shape, integer, or float
        :param string: specifier. string
        """
        if order is None:
            np.save(os.path.join(RST, '{:}'.format(string)), arr)
        else:
            np.save(os.path.join(RST, '{:}_{:}'.format(string, order)), arr)


def read_gen(order, string):
        """
        this function reads a general restart file corresponding to input string
        """
        if order is None:
            return np.load(os.path.join(RST, '{:}.npy'.format(string)))
        else:
            return np.load(os.path.join(RST, '{:}_{:}.npy'.format(string, order)))


def write_fund(mol, calc):
        """
        this function writes all fundamental info restart files

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        """
        # write dimensions
        dims = {'nocc': mol.nocc, 'nvirt': mol.nvirt, 'norb': mol.norb, 'nelec': calc.nelec}
        with open(os.path.join(RST, 'dims.rst'), 'w') as f:
            json.dump(dims, f)

        # write expansion spaces
        np.save(os.path.join(RST, 'ref_space'), calc.ref_space)
        np.save(os.path.join(RST, 'exp_space_tot'), calc.exp_space['tot'])
        np.save(os.path.join(RST, 'exp_space_occ'), calc.exp_space['occ'])
        np.save(os.path.join(RST, 'exp_space_virt'), calc.exp_space['virt'])
        np.save(os.path.join(RST, 'exp_space_pi_orbs'), calc.exp_space['pi_orbs'])
        np.save(os.path.join(RST, 'exp_space_pi_hashes'), calc.exp_space['pi_hashes'])

        # occupation
        np.save(os.path.join(RST, 'occup'), calc.occup)

        # write orbital energies
        np.save(os.path.join(RST, 'mo_energy'), calc.mo_energy)

        # write orbital coefficients
        np.save(os.path.join(RST, 'mo_coeff'), calc.mo_coeff)


def write_prop(mol, calc):
        """
        this function writes all property restart files

        :param mol: pymbe mol object
        :param calc: pymbe calc object
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

            dipoles = {'hf': calc.prop['hf']['dipole'].tolist(), \
                        'ref': calc.prop['ref']['dipole'].tolist()}
            with open(os.path.join(RST, 'dipoles.rst'), 'w') as f:
                json.dump(dipoles, f)

        elif calc.target_mbe == 'trans':

            transitions = {'ref': calc.prop['ref']['trans'].tolist()}
            with open(os.path.join(RST, 'transitions.rst'), 'w') as f:
                json.dump(transitions, f)


def read_fund(mol, calc):
        """
        this function reads all fundamental info restart files

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: updated mol object,
                 updated calc object
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key=_natural_keys)

        # init exp_space
        calc.exp_space = {}

        # loop over files
        for i in range(len(files)):

            # read dimensions
            if 'dims' in files[i]:
                with open(os.path.join(RST, files[i]), 'r') as f:
                    dims = json.load(f)
                mol.nocc = dims['nocc']; mol.nvirt = dims['nvirt']
                mol.norb = dims['norb']; calc.nelec = dims['nelec']

            # read expansion spaces
            elif 'ref_space' in files[i]:
                calc.ref_space = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_tot' in files[i]:
                calc.exp_space['tot'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_occ' in files[i]:
                calc.exp_space['occ'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_virt' in files[i]:
                calc.exp_space['virt'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_pi_orbs' in files[i]:
                calc.exp_space['pi_orbs'] = np.load(os.path.join(RST, files[i]))
            elif 'exp_space_pi_hashes' in files[i]:
                calc.exp_space['pi_hashes'] = np.load(os.path.join(RST, files[i]))

            # read occupation
            elif 'occup' in files[i]:
                calc.occup = np.load(os.path.join(RST, files[i]))

            # read orbital energies
            elif 'mo_energy' in files[i]:
                calc.mo_energy = np.load(os.path.join(RST, files[i]))

            # read orbital coefficients
            elif 'mo_coeff' in files[i]:
                calc.mo_coeff = np.load(os.path.join(RST, files[i]))
                if mol.atom:
                    calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
                else:
                    calc.orbsym = np.zeros(mol.norb, dtype=np.int)

        return mol, calc


def read_prop(mol, calc):
        """
        this function reads all property restart files

        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: updated mol object,
                 updated calc object
        """
        # list filenames in files list
        files = [f for f in os.listdir(RST) if os.path.isfile(os.path.join(RST, f))]

        # sort the list of files
        files.sort(key=_natural_keys)

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
                calc.prop['ref']['dipole'] = np.asarray(dipoles['ref'])

            elif 'transitions' in files[i]:

                with open(os.path.join(RST, files[i]), 'r') as f:
                    transitions = json.load(f)
                calc.prop['ref']['trans'] = np.asarray(transitions['ref'])

        return mol, calc


def _natural_keys(txt):
        """
        this function return keys to sort a string in human order (as alist.sort(key=natural_keys))
        see: http://nedbatchelder.com/blog/200712/human_sorting.html
        see: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

        :param txt: text. string
        :return: list of keys
        """
        return [_convert(c) for c in re.split('(\d+)', txt)]


def _convert(txt):
        """
        this function converts strings with numbers in them

        :param txt: text. string
        :return: integer or string depending on txt
        """
        return int(txt) if txt.isdigit() else txt


