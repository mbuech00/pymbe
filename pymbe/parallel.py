#!/usr/bin/env python
# -*- coding: utf-8 -*

""" parallel.py: mpi class """

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.6'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from pyscf import symm, lib

import tools
import restart


INT_MAX = 2147483647
BLKSIZE = INT_MAX // 32 + 1


class MPICls(object):
		""" mpi parameters """
		def __init__(self):
				""" init parameters """
				self.comm = MPI.COMM_WORLD
				self.size = self.comm.Get_size()
				self.rank = self.comm.Get_rank()
				self.master = (self.rank == 0)
				self.slave = not self.master
				self.host = MPI.Get_processor_name()
				self.stat = MPI.Status()


def mol(mpi, mol):
		""" bcast mol info """
		if mpi.master:
			info = {'atom': mol.atom, 'charge': mol.charge, 'spin': mol.spin, \
					'symmetry': mol.symmetry, 'irrep_nelec': mol.irrep_nelec, 'basis': mol.basis, \
					'cart': mol.cart, 'unit': mol.unit, 'frozen': mol.frozen, 'debug': mol.debug}
			if not mol.atom:
				info['u'] = mol.u
				info['n'] = mol.n
				info['matrix'] = mol.matrix
				info['nsites'] = mol.nsites
				info['pbc'] = mol.pbc
				info['nelec'] = mol.nelectron
			mpi.comm.bcast(info, root=0)
		else:
			info = mpi.comm.bcast(None, root=0)
			mol.atom = info['atom']; mol.charge = info['charge']; mol.spin = info['spin']
			mol.symmetry = info['symmetry']; mol.irrep_nelec = info['irrep_nelec']
			mol.basis = info['basis']; mol.cart = info['cart']
			mol.unit = info['unit']; mol.frozen = info['frozen']
			mol.debug = info['debug']
			if not mol.atom:
				mol.u = info['u']; mol.n = info['n']
				mol.matrix = info['matrix']; mol.nsites = info['nsites'] 
				mol.pbc = info['pbc']; mol.nelectron = info['nelec']


def calc(mpi, calc):
		""" bcast calc info """
		if mpi.master:
			info = {'model': calc.model, 'target': calc.target, \
					'base': calc.base, \
					'thres': calc.thres, 'prot': calc.prot, \
					'state': calc.state, 'extra': calc.extra, \
					'misc': calc.misc, 'mpi': calc.mpi, \
					'orbs': calc.orbs, 'restart': calc.restart}
			mpi.comm.bcast(info, root=0)
		else:
			info = mpi.comm.bcast(None, root=0)
			calc.model = info['model']; calc.target = info['target']
			calc.base = info['base']
			calc.thres = info['thres']; calc.prot = info['prot']
			calc.state = info['state']; calc.extra = info['extra']
			calc.misc = info['misc']; calc.mpi = info['mpi']
			calc.orbs = info['orbs']; calc.restart = info['restart']


def fund(mpi, mol, calc):
		""" bcast fundamental info """
		if mpi.master:
			info = {'prop': calc.prop, \
						'norb': mol.norb, 'nocc': mol.nocc, 'nvirt': mol.nvirt, \
						'ref_space': calc.ref_space, 'exp_space': calc.exp_space, \
						'occup': calc.occup, 'mo_energy': calc.mo_energy, 'e_nuc': mol.e_nuc}
			mpi.comm.bcast(info, root=0)
			# bcast mo coefficients
			mpi.comm.Bcast([calc.mo_coeff, MPI.DOUBLE], root=0)
			# update orbsym
			if mol.atom:
				calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
			else:
				calc.orbsym = np.zeros(mol.norb, dtype=np.int)
			# mo_coeff not needed on master anymore
			del calc.mo_coeff
			# bcast core hamiltonian (MO basis)
			mpi.comm.Bcast([mol.hcore, MPI.DOUBLE], root=0)
			# hcore not needed on master anymore
			del mol.hcore
			# bcast effective fock potentials (MO basis)
			mpi.comm.Bcast([mol.vhf, MPI.DOUBLE], root=0)
			# vhf not needed on master anymore
			del mol.vhf
		else:
			info = mpi.comm.bcast(None, root=0)
			calc.prop = info['prop']
			mol.norb = info['norb']; mol.nocc = info['nocc']; mol.nvirt = info['nvirt']
			calc.ref_space = info['ref_space']; calc.exp_space = info['exp_space']
			calc.occup = info['occup']; calc.mo_energy = info['mo_energy']; mol.e_nuc = info['e_nuc']
			# receive mo coefficients
			buff = np.zeros([mol.norb, mol.norb], dtype=np.float64)
			mpi.comm.Bcast([buff, MPI.DOUBLE], root=0)
			calc.mo_coeff = buff
			# receive hcore
			buff = np.zeros([mol.norb, mol.norb], dtype=np.float64)
			mpi.comm.Bcast([buff, MPI.DOUBLE], root=0)
			mol.hcore = buff
			# receive fock potentials
			buff = np.zeros([mol.nocc, mol.norb, mol.norb], dtype=np.float64)
			mpi.comm.Bcast([buff, MPI.DOUBLE], root=0)
			mol.vhf = buff
			# update orbsym
			if mol.atom:
				calc.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, calc.mo_coeff)
			else:
				calc.orbsym = np.zeros(mol.norb, dtype=np.int)


def exp(mpi, calc, exp):
		""" bcast exp info """
		if mpi.master:
			# collect info
			info = {'min_order': exp.min_order}
			if calc.restart:
				info['len_tup'] = [exp.tuples[i].shape[0] for i in range(len(exp.tuples))]
				info['len_prop'] = [exp.prop[calc.target]['inc'][i].shape[0] for i in range(len(exp.prop[calc.target]['inc']))]
			# bcast info
			mpi.comm.bcast(info, root=0)
			if calc.restart:
				# bcast hashes
				for i in range(1, len(exp.hashes)):
					mpi.comm.Bcast([exp.hashes[i], MPI.LONG], root=0)
				# bcast increments
				for i in range(len(exp.prop[calc.target]['inc'])):
					mpi.comm.Bcast([exp.prop[calc.target]['inc'][i], MPI.DOUBLE], root=0)
		else:
			# receive info
			info = mpi.comm.bcast(None, root=0)
			# receive min_order
			exp.min_order = info['min_order']
			if calc.restart:
				# receive hashes
				for i in range(1, len(info['len_tup'])):
					buff = np.empty(info['len_tup'][i], dtype=np.int64)
					mpi.comm.Bcast([buff, MPI.LONG], root=0)
					exp.hashes.append(buff)
				# receive increments
				if calc.target in ['energy', 'excitation']:
					for i in range(len(info['len_prop'])):
						buff = np.zeros(info['len_prop'][i], dtype=np.float64)
						mpi.comm.Bcast([buff, MPI.DOUBLE], root=0)
						exp.prop[calc.target]['inc'].append(buff)
				else:
					for i in range(len(info['len_prop'])):
						buff = np.zeros([info['len_prop'][i], 3], dtype=np.float64)
						mpi.comm.Bcast([buff, MPI.DOUBLE], root=0)
						exp.prop[calc.target]['inc'].append(buff)


def bcast(mpi, buff):
		"""
		this function performs a tiled Bcast operation

		:param mpi: pymbe mpi object
		:param buff: buffer. numpy array of any kind of shape and dtype (may not be allocated on slave procs)
		:return: numpy array of same shape and dtype as master buffer
		"""
		# init buff_tile
		buff_tile = np.ndarray(buff.size, dtype=buff.dtype, buffer=buff)

		# bcast all tiles
		for p0, p1 in lib.prange(0, buff.size, BLKSIZE):
			mpi.comm.Bcast(buff_tile[p0:p1], root=0)

		return buff


def allreduce(mpi, send_buff):
		"""
		this function performs a tiled Allreduce operation

		:param mpi: pymbe mpi object
		:param send_buff: send buffer. numpy array of any kind of shape and dtype
		:return: numpy array of same shape and dtype as send_buff
		"""
		# bcast shape and dtype of send_buff to slaves
		shape, mpi_dtype = mpi.comm.bcast((send_buff.shape, send_buff.dtype.char))

		# init recv_buff		
		recv_buff = np.zeros_like(send_buff)

		# init send_tile and recv_tile
		send_tile = np.ndarray(send_buff.size, dtype=send_buff.dtype, buffer=send_buff)
		recv_tile = np.ndarray(recv_buff.size, dtype=recv_buff.dtype, buffer=recv_buff)

		# allreduce all tiles
		for p0, p1 in lib.prange(0, send_buff.size, BLKSIZE):
			mpi.comm.Allreduce(send_tile[p0:p1], recv_tile[p0:p1], op=MPI.SUM)

		return recv_buff


def recv_counts(mpi, n_elms):
		"""
		this function performs an allgather operation to return an array with n_elms from all procs

		:param mpi: pymbe mpi object
		:param n_elms: number of elements. integer
		:return: numpy array of shape (n_procs,)
		"""
		return np.array(mpi.comm.allgather(n_elms))


def gatherv(mpi, send_buff):
		"""
		this function performs a tiled gatherv operation

		:param mpi: pymbe mpi object
		:param send_buff: send buffer. numpy array of any kind of shape and dtype
		:return: numpy array of shape (n_child_tuples * (order+1),)
		"""
		# allgather shape and dtype of send_buff from slaves
		shape = send_buff.shape
		size_dtype = mpi.comm.allgather((shape, send_buff.dtype.char))
		rshape = [x[0] for x in size_dtype]
		mpi_dtype = np.result_type(*[x[1] for x in size_dtype]).char

		# compute counts
		counts = np.array([np.prod(x) for x in rshape])

		if mpi.master:

			# compute displacements
			displs = np.append(0, np.cumsum(counts[:-1]))

			# init recv_buff
			recv_buff = np.empty(sum(counts), dtype=mpi_dtype)

			# gatherv all tiles
			for p0, p1 in lib.prange(0, np.max(counts), BLKSIZE):
				counts_tile = _tile_counts(counts, p0, p1)
				mpi.comm.Gatherv([send_buff[p0:p1], mpi_dtype], \
									[recv_buff, counts_tile, displs+p0, mpi_dtype], root=0)

			return recv_buff#.reshape((-1,) + shape[1:])

		else:

			# gatherv all tiles
			for p0, p1 in lib.prange(0, np.max(counts), BLKSIZE):
				mpi.comm.Gatherv([send_buff[p0:p1], mpi_dtype], None, root=0)

			return send_buff


def _tile_counts(counts, p0, p1):
		"""
		this function counts the individual tiles

		:param counts: main counts. numpy array of shape (n_procs,)
		:param p0: start index. integer
		:param p1: end index. integer
		:return: tile counter. numpy array of shape (n_procs,)
		"""
		# compute counts_tile
		counts_tile = counts - p0
		counts_tile[counts <= p0] = 0
		counts_tile[counts > p1] = p1 - p0

		return counts_tile


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
			mpi.comm.bcast({'task': 'exit'}, root=0)

		# finalize
		mpi.comm.Barrier()
		MPI.Finalize()


