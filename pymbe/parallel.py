#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
parallel module containing all mpi operations in pymbe
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
try:
    from mpi4py import MPI
except ImportError:
    sys.stderr.write('\nImportError : mpi4py module not found\n\n')
try:
    import numpy as np
except ImportError:
    sys.stderr.write('\nImportError : numpy module not found\n\n')
from pyscf import symm, lib

import tools
import restart


# parameters for tiled mpi operations
INT_MAX = 2147483647
BLKSIZE = INT_MAX // 32 + 1


class MPICls(object):
        """
        this class contains the pymbe mpi attributes
        """
        def __init__(self):
                """
                init mpi attributes
                """
                # general
                self.host = MPI.Get_processor_name()
                self.stat = MPI.Status()
                # global communicator
                self.global_comm = MPI.COMM_WORLD
                self.global_size = self.global_comm.Get_size()
                if self.global_master:
                    tools.assertion(self.global_size >= 2, 'PyMBE requires two or more MPI processes')
                self.global_rank = self.global_comm.Get_rank()
                self.global_master = (self.rank == 0)
                self.global_slave = not self.global_master
                # local node communicator (memory sharing)
                self.local_comm = self.global_comm.Split_type(MPI.COMM_TYPE_SHARED)
                self.local_size = self.local_comm.Get_size()
                self.local_rank = self.local_comm.Get_rank()
                self.local_master = self.local_rank == 0
                # master communicator
                mpi.master_comm = mpi.global_comm.Split(1 if self.local_master else MPI.UNDEFINED)
                if self.local_master:
                    self.master_size = self.num_masters = self.master_comm.Get_size()
                    self.master_rank = self.master_comm.Get_rank()
                if self.global_master:
                    mpi.global_comm.bcast(self.num_masters, root=0)
                else:
                    self.num_masters = mpi.global_comm.bcast(None, root=0)


def mol(mpi, mol):
        """
        this function bcast all standard mol info to slaves

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :return: updated mol object
        """
        if mpi.master:

            # collect standard info (must be updated with new future attributes)
            info = {'atom': mol.atom, 'charge': mol.charge, 'spin': mol.spin, 'ncore': mol.ncore, \
                    'symmetry': mol.symmetry, 'irrep_nelec': mol.irrep_nelec, 'basis': mol.basis, \
                    'cart': mol.cart, 'unit': mol.unit, 'frozen': mol.frozen, 'debug': mol.debug}

            # add hubbard info if relevant (also needs to be updated with new future attributes)
            if not mol.atom:

                info['u'] = mol.u
                info['n'] = mol.n
                info['matrix'] = mol.matrix
                info['nsites'] = mol.nsites
                info['pbc'] = mol.pbc
                info['nelectron'] = mol.nelectron

            # bcast to slaves
            mpi.global_comm.bcast(info, root=0)

        else:

            # receive info from master
            info = mpi.global_comm.bcast(None, root=0)

            # set mol attributes from info dict
            for key, val in info.items():
                setattr(mol, key, val)

        return mol


def calc(mpi, calc):
        """
        this function bcast all standard calc info to slaves

        :param mpi: pymbe mpi object
        :param calc: pymbe calc object
        :return: updated calc object
        """
        if mpi.master:

            # collect standard info (must be updated with new future attributes)
            info = {'model': calc.model, 'target': calc.target, 'base': calc.base, \
                    'thres': calc.thres, 'prot': calc.prot, 'state': calc.state, \
                    'extra': calc.extra, 'misc': calc.misc, 'mpi': calc.mpi, \
                    'orbs': calc.orbs, 'restart': calc.restart}

            # bcast to slaves
            mpi.global_comm.bcast(info, root=0)

        else:

            # receive info from master
            info = mpi.global_comm.bcast(None, root=0)

            # set calc attributes from info dict
            for key, val in info.items():
                setattr(calc, key, val)

        return calc


def fund(mpi, mol, calc):
        """
        this function bcasts all fundamental mol and calc info to slaves

        :param mpi: pymbe mpi object
        :param mol: pymbe mol object
        :param calc: pymbe calc object
        :return: updated mol object,
                 updated calc object
        """
        if mpi.master:

            # collect standard info (must be updated with new future attributes)
            info = {'norb': mol.norb, 'nocc': mol.nocc, 'nvirt': mol.nvirt, 'e_nuc': mol.e_nuc}

            # bcast to slaves
            mpi.global_comm.bcast(info, root=0)

            # collect standard info (must be updated with new future attributes)
            info = {'occup': calc.occup, 'mo_energy': calc.mo_energy, \
                    'ref_space': calc.ref_space, 'exp_space': calc.exp_space}

            # bcast to slaves
            mpi.global_comm.bcast(info, root=0)

            # bcast mo coefficients
            calc.mo_coeff = bcast(mpi, calc.mo_coeff)

            # update orbsym
            if mol.atom:
                calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
            else:
                calc.orbsym = np.zeros(mol.norb, dtype=np.int)

        else:

            # receive info from master
            info = mpi.global_comm.bcast(None, root=0)

            # set mol attributes from info dict
            for key, val in info.items():
                setattr(mol, key, val)

            # receive info from master
            info = mpi.global_comm.bcast(None, root=0)

            # set calc attributes from info dict
            for key, val in info.items():
                setattr(calc, key, val)

            # receive mo coefficients
            calc.mo_coeff = np.zeros([mol.norb, mol.norb], dtype=np.float64)
            calc.mo_coeff = bcast(mpi, calc.mo_coeff)

            # update orbsym
            if mol.atom:
                calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
            else:
                calc.orbsym = np.zeros(mol.norb, dtype=np.int)

        return mol, calc


def prop(mpi, calc):
        """
        this function bcasts properties to slaves

        :param mpi: pymbe mpi object
        :param calc: pymbe calc object
        :return: updated calc object
        """
        if mpi.master:

            # bcast to slaves
            mpi.global_comm.bcast(calc.prop, root=0)

        else:

            # receive prop from master
            calc.prop = mpi.global_comm.bcast(None, root=0)

        return calc


def probe(mpi, tag_ready):
        """
        this function probes for available slaves

        :param mpi: pymbe mpi object
        :param tag_ready: mpi ready tag. enum
        """
        # probe for available slaves
        mpi.global_comm.Probe(source=MPI.ANY_SOURCE, tag=tag_ready, status=mpi.stat)

        # receive slave status
        mpi.global_comm.recv(None, source=mpi.stat.source, tag=tag_ready)


def bcast(mpi, buff):
        """
        this function performs a tiled Bcast operation
        inspired by: https://github.com/pyscf/mpi4pyscf/blob/master/tools/mpi.py

        :param mpi: pymbe mpi object
        :param buff: buffer. numpy array of any kind of shape and dtype (may not be allocated on slave procs)
        :return: numpy array of same shape and dtype as master buffer
        """
        # init buff_tile
        buff_tile = np.ndarray(buff.size, dtype=buff.dtype, buffer=buff)

        # bcast all tiles
        for p0, p1 in lib.prange(0, buff.size, BLKSIZE):
            mpi.global_comm.Bcast(buff_tile[p0:p1], root=0)

        return buff


def gatherv(mpi, send_buff, counts):
        """
        this function performs a gatherv operation using point-to-point operations
        inspired by: https://github.com/pyscf/mpi4pyscf/blob/master/tools/mpi.py

        :param mpi: pymbe mpi object
        :param send_buff: send buffer. numpy array of any kind of shape and dtype
        :param counts: number of elements from individual processes. numpy array of shape (n_procs,)
        :return: numpy array of shape (n_child_tuples * (order+1),)
        """
        if mpi.master:

            # init recv_buff
            recv_buff = np.empty(np.sum(counts), dtype=np.int32)

            # gatherv all tiles
            for slave in range(1, mpi.size):

                slave_idx = np.sum(counts[:slave])

                for p0, p1 in lib.prange(0, counts[slave], BLKSIZE):
                    mpi.global_comm.Recv(recv_buff[slave_idx+p0:slave_idx+p1], source=slave)

            return recv_buff

        else:

            # gatherv all tiles
            for p0, p1 in lib.prange(0, counts[mpi.rank], BLKSIZE):
                mpi.global_comm.Send(send_buff[p0:p1], dest=0)

            return send_buff


def abort():
        """
        this function aborts mpi in case of a pymbe error
        """
        MPI.COMM_WORLD.Abort()


def finalize(mpi):
        """
        this function terminates a successful pymbe calculation

        :param mpi: pymbe mpi object
        """
        # wake up slaves
        if mpi.master:
            restart.rm()
            mpi.global_comm.bcast({'task': 'exit'}, root=0)

        # finalize
        mpi.global_comm.Barrier()
        MPI.Finalize()


