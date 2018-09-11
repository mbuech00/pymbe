#!/usr/bin/env python
# -*- coding: utf-8 -*

""" mbe.py: mbe module """

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__license__ = '???'
__version__ = '0.10'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
from pyscf import symm
import sys
import itertools
import scipy.misc

import kernel
import output
import expansion
import driver
import parallel
import tools


# mbe parameters
TAGS = tools.enum('start', 'ready', 'exit', 'collect')


def main(mpi, mol, calc, exp):
		""" mbe phase """
		# print header
		if mpi.global_master: output.mbe_header(exp)
		# init increments
		if calc.target['energy']:
			if len(exp.prop['energy']['inc']) < exp.order - (exp.start_order - 1):
				exp.prop['energy']['inc'].append(np.zeros(len(exp.tuples[-1]), dtype=np.float64))
		if calc.target['excitation']:
			if len(exp.prop['excitation']['inc']) < exp.order - (exp.start_order - 1):
				exp.prop['excitation']['inc'].append(np.zeros(len(exp.tuples[-1]), dtype=np.float64))
		if calc.target['dipole']:
			if len(exp.prop['dipole']['inc']) < exp.order - (exp.start_order - 1):
				exp.prop['dipole']['inc'].append(np.zeros([len(exp.tuples[-1]), 3], dtype=np.float64))
		if calc.target['trans']:
			if len(exp.prop['trans']['inc']) < exp.order - (exp.start_order - 1):
				exp.prop['trans']['inc'].append(np.zeros([len(exp.tuples[-1]), 3], dtype=np.float64))
		# mpi parallel or serial version
		if mpi.parallel:
			if mpi.global_master:
				# master
				_master(mpi, mol, calc, exp)
			else:
				# slaves
				_slave(mpi, mol, calc, exp)
				return
		else:
			_serial(mpi, mol, calc, exp)
		# sum up total quantities
		if mol.debug and exp.order == 1:
			if calc.target['energy']:
				print('')
				a = np.abs(exp.prop['energy']['inc'][0])
				for j in range(exp.tuples[0].shape[0]):
					print('o = {0:} , root {1:} energy = {2:.2e}'.format(exp.tuples[0][np.argsort(a)[::-1]][j], \
																			calc.state['root'], a[np.argsort(a)[::-1]][j]))
			if calc.target['excitation']:
				print('')
				a = np.abs(exp.prop['excitation']['inc'][0])
				for j in range(exp.tuples[0].shape[0]):
					print('o = {0:} , roots 0 -> {1:} excitation energy = {2:.2e}'.format(exp.tuples[0][np.argsort(a)[::-1]][j], \
																			calc.state['root'], a[np.argsort(a)[::-1]][j]))
			if calc.target['dipole']:
				print('')
				a = np.linalg.norm(exp.prop['dipole']['inc'][0], axis=1)
				for j in range(exp.tuples[0].shape[0]):
					print('o = {0:} , root {1:} dipole = {2:.2e}'.format(exp.tuples[0][np.argsort(a)[::-1]][j], \
																			calc.state['root'], a[np.argsort(a)[::-1]][j]))
			if calc.target['trans']:
				print('')
				a = np.linalg.norm(exp.prop['trans']['inc'][0], axis=1)
				for j in range(exp.tuples[0].shape[0]):
					print('o = {0:} , roots 0 -> {1:} trans = {2:.2e}'.format(exp.tuples[0][np.argsort(a)[::-1]][j], \
																				calc.state['root'], a[np.argsort(a)[::-1]][j]))
			print('')
		if calc.target['energy']:
			exp.prop['energy']['tot'].append(tools.fsum(exp.prop['energy']['inc'][-1]))
		if calc.target['excitation']:
			exp.prop['excitation']['tot'].append(tools.fsum(exp.prop['excitation']['inc'][-1]))
		if calc.target['dipole']:
			exp.prop['dipole']['tot'].append(tools.fsum(exp.prop['dipole']['inc'][-1]))
		if calc.target['trans']:
			exp.prop['trans']['tot'].append(tools.fsum(exp.prop['trans']['inc'][-1]))
		if exp.order > exp.start_order:
			if calc.target['energy']:
				exp.prop['energy']['tot'][-1] += exp.prop['energy']['tot'][-2]
			if calc.target['excitation']:
				exp.prop['excitation']['tot'][-1] += exp.prop['excitation']['tot'][-2]
			if calc.target['dipole']:
				exp.prop['dipole']['tot'][-1] += exp.prop['dipole']['tot'][-2]
			if calc.target['trans']:
				exp.prop['trans']['tot'][-1] += exp.prop['trans']['tot'][-2]


def _serial(mpi, mol, calc, exp):
		""" serial version """
		# start time
		time = MPI.Wtime()
		# init counter
		exp.count.append(0)
		# loop over tuples
		for i in range(len(exp.tuples[-1])):
			# calculate increments
			if not calc.extra['lz_sym'] or (calc.extra['lz_sym'] and _lz_check(mol, calc, exp, i)):
				_calc(mpi, mol, calc, exp, i)
				exp.count[-1] += 1
				# print status
				output.mbe_status(exp, float(i+1) / float(len(exp.tuples[-1])))
		# collect time
		exp.time['mbe'].append(MPI.Wtime() - time)


def _master(mpi, mol, calc, exp):
		""" master function """
		# set communicator
		comm = mpi.local_comm
		# wake up slaves
		msg = {'task': 'mbe', 'order': exp.order}
		comm.bcast(msg, root=0)
		# start time
		time = MPI.Wtime()
		num_slaves = slaves_avail = min(mpi.local_size - 1, len(exp.tuples[-1]))
		# start index
		i = 0
		# init request
		req = MPI.Request()
		# loop until no tasks left
		while True:
			# probe for available slaves
			if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAGS.ready, status=mpi.stat):
				# receive slave status
				req = comm.Irecv([None, MPI.INT], source=mpi.stat.source, tag=TAGS.ready)
				# any tasks left?
				if i < len(exp.tuples[-1]):
					# send index
					comm.Isend([np.array([i], dtype=np.int32), MPI.INT], dest=mpi.stat.source, tag=TAGS.start)
					# increment index
					i += 1
					# wait for completion
					req.Wait()
				else:
					# send exit signal
					comm.Isend([None, MPI.INT], dest=mpi.stat.source, tag=TAGS.exit)
					# remove slave
					slaves_avail -= 1
					# wait for completion
					req.Wait()
					# any slaves left?
					if slaves_avail == 0:
						# exit loop
						break
		# init counter
		exp.count.append(0)
		# allreduce properties
		if calc.target['energy']:
			parallel.mbe(exp.prop['energy']['inc'][-1], comm)
			if exp.count[-1] == 0:
				exp.count[-1] = np.count_nonzero(exp.prop['energy']['inc'][-1])
		if calc.target['excitation']:
			parallel.mbe(exp.prop['excitation']['inc'][-1], comm)
			if exp.count[-1] == 0:
				exp.count[-1] = np.count_nonzero(exp.prop['excitation']['inc'][-1])
		if calc.target['dipole']:
			parallel.mbe(exp.prop['dipole']['inc'][-1], comm)
			if exp.count[-1] == 0:
				exp.count[-1] = np.count_nonzero(np.count_nonzero(exp.prop['dipole']['inc'][-1], axis=1))
		if calc.target['trans']:
			parallel.mbe(exp.prop['trans']['inc'][-1], comm)
			if exp.count[-1] == 0:
				exp.count[-1] = np.count_nonzero(np.count_nonzero(exp.prop['trans']['inc'][-1], axis=1))
		# collect time
		exp.time['mbe'].append(MPI.Wtime() - time)


def _slave(mpi, mol, calc, exp):
		""" slave function """
		# set communicator
		comm = mpi.local_comm
		# init idx
		idx = np.empty(1, dtype=np.int32)
		# send availability to master
		if mpi.local_rank <= len(exp.tuples[-1]):
			comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
		# receive work from master
		while True:
			# early exit in case of large proc count
			if mpi.local_rank > len(exp.tuples[-1]):
				break
			# receive index
			comm.Recv([idx, MPI.INT], source=0, status=mpi.stat)
			# do job
			if mpi.stat.tag == TAGS.start:
				# send availability to master
				comm.Isend([None, MPI.INT], dest=0, tag=TAGS.ready)
				# calculate increments
				if not calc.extra['lz_sym'] or (calc.extra['lz_sym'] and _lz_check(mol, calc, exp, idx[0])):
					_calc(mpi, mol, calc, exp, idx[0])
			elif mpi.stat.tag == TAGS.exit:
				break
		# allreduce properties
		if calc.target['energy']:
			parallel.mbe(exp.prop['energy']['inc'][-1], comm)
		if calc.target['excitation']:
			parallel.mbe(exp.prop['excitation']['inc'][-1], comm)
		if calc.target['dipole']:
			parallel.mbe(exp.prop['dipole']['inc'][-1], comm)
		if calc.target['trans']:
			parallel.mbe(exp.prop['trans']['inc'][-1], comm)


def _calc(mpi, mol, calc, exp, idx):
		""" calculate increments """
		res = _inc(mpi, mol, calc, exp, exp.tuples[-1][idx])
		if calc.target['energy']:
			exp.prop['energy']['inc'][-1][idx] = res['energy']
		if calc.target['excitation']:
			exp.prop['excitation']['inc'][-1][idx] = res['excitation']
		if calc.target['dipole']:
			exp.prop['dipole']['inc'][-1][idx] = res['dipole']
		if calc.target['trans']:
			exp.prop['trans']['inc'][-1][idx] = res['trans']


def _inc(mpi, mol, calc, exp, tup):
		""" calculate increments corresponding to tup """
		# generate input
		exp.core_idx, exp.cas_idx = kernel.core_cas(mol, exp, tup)
		# perform calc
		res = kernel.main(mol, calc, exp, calc.model['method'])
		inc = {}
		if calc.target['energy']:
			inc['energy'] = res['energy'] - calc.prop['ref']['energy']
			if calc.base['method'] is None:
				e_base = 0.0
			else:
				res = kernel.main(mol, calc, exp, calc.base['method'])
				e_base = res['energy']
			inc['energy'] -= e_base
		if calc.target['excitation']:
			inc['excitation'] = res['excitation'] - calc.prop['ref']['excitation']
		if calc.target['dipole']:
			if res['dipole'] is None:
				inc['dipole'] = np.zeros(3, dtype=np.float64)
			else:
				inc['dipole'] = res['dipole'] - calc.prop['ref']['dipole']
		if calc.target['trans']:
			if res['trans'] is None:
				inc['trans'] = np.zeros(3, dtype=np.float64)
			else:
				inc['trans'] = res['trans'] - calc.prop['ref']['trans']
		if exp.order > exp.start_order:
			if calc.target['energy']:
				if inc['energy'] != 0.0:
					res = _sum(calc, exp, tup, 'energy')
					inc['energy'] = inc['energy'] - res['energy']
			if calc.target['excitation']:
				if inc['excitation'] != 0.0:
					res = _sum(calc, exp, tup, 'excitation')
					inc['excitation'] = inc['excitation'] - res['excitation']
			if calc.target['dipole']:
				if np.any(inc['dipole'] != 0.0):
					res = _sum(calc, exp, tup, 'dipole')
					inc['dipole'] = inc['dipole'] - res['dipole']
			if calc.target['trans']:
				if np.any(inc['trans'] != 0.0):
					res = _sum(calc, exp, tup, 'trans')
					inc['trans'] = inc['trans'] - res['trans']
		# debug print
		if mol.debug:
			core_lst = exp.core_idx.tolist()
			if core_lst:
				core_sym = [symm.addons.irrep_id2name(mol.symmetry, i) for i in calc.orbsym[exp.core_idx]]
			else:
				core_sym = []
			cas_lst = [calc.map[i] for i in exp.cas_idx]
			cas_sym = [symm.addons.irrep_id2name(mol.symmetry, i) for i in calc.orbsym[exp.cas_idx]]
			string = ' INC: proc = {:} , core = {:} , cas = {:}\n'
			string += '      core-sym = {:} , cas-sym = {:}\n'
			form = (mpi.local_rank, core_lst, cas_lst, core_sym, cas_sym)
			if calc.target['energy']:
				string += '      correlation energy increment for state {:} = {:.4e}\n'
				form += (calc.state['root'], inc['energy'],)
			if calc.target['excitation']:
				string += '      excitation energy increment for root {:} = {:.4e}\n'
				form += (calc.state['root'], inc['excitation'],)
			if calc.target['dipole']:
				string += '      dipole moment increment for root {:} = ({:.4e}, {:.4e}, {:.4e})\n'
				form += (calc.state['root'], *inc['dipole'],)
			if calc.target['trans']:
				string += '      transition dipole moment increment for excitation 0 -> {:} = ({:.4e}, {:.4e}, {:.4e})\n'
				form += (calc.state['root'], *inc['trans'],)
			print(string.format(*form))
		return inc


def _sum(calc, exp, tup, prop):
		""" recursive summation """
		# init res
		res = {}
		if prop == 'energy':
			res['energy'] = 0.0
		elif prop == 'excitation':
			res['excitation'] = 0.0
		elif prop == 'dipole':
			res['dipole'] = np.zeros(3, dtype=np.float64)
		elif prop == 'trans':
			res['trans'] = np.zeros(3, dtype=np.float64)
		# compute contributions from lower-order increments
		for i in range(exp.order-exp.start_order, 0, -1):
			# generate array with all subsets of particular tuple (manually adding active orbs)
			if calc.no_exp > 0:
				if calc.model['type'] == 'occ':
					combs = np.array([comb+tuple(exp.tuples[0][0]) for comb in itertools.\
										combinations(tup[:-calc.no_exp], i-1)], dtype=np.int32)
				elif calc.model['type'] == 'virt':
					combs = np.array([tuple(exp.tuples[0][0])+comb for comb in itertools.\
										combinations(tup[calc.no_exp:], i-1)], dtype=np.int32)
			else:
				combs = np.array([comb for comb in itertools.combinations(tup, i)], dtype=np.int32)
			# convert to sorted hashes
			combs = tools.hash_2d(combs)
			combs.sort()
			# get index
			diff, left, right = tools.hash_compare(exp.hashes[i-1], combs)
			assert diff.size == combs.size, \
						('\nmbe.py:_sum()\ndiff  = {:}\nleft = {:}\nright = {:}\n'.format(diff, left, right))
			indx = left
			# add up lower-order increments
			if prop == 'energy':
				res['energy'] += tools.fsum(exp.prop['energy']['inc'][i-1][indx])
			elif prop == 'excitation':
				res['excitation'] += tools.fsum(exp.prop['excitation']['inc'][i-1][indx])
			elif prop == 'dipole':
				res['dipole'] += tools.fsum(exp.prop['dipole']['inc'][i-1][indx, :])
			elif prop == 'trans':
				res['trans'] += tools.fsum(exp.prop['trans']['inc'][i-1][indx, :])
		return res


def _lz_check(mol, calc, exp, idx):
		""" lz symmetry check """
		allow = True
		exp.cas_idx = kernel.core_cas(mol, exp, exp.tuples[-1][idx])[-1]
		lz_orbs = np.array([tools.LZMAP[x] for x in calc.orbsym[exp.cas_idx[(calc.ref_space.size+calc.no_exp):]]])
		pi_orbs_g = lz_orbs[np.where(np.abs(lz_orbs) == 5)]
		if pi_orbs_g.size % 2 > 0:
			allow = False
		elif pi_orbs_g.size > 0:
			g_orbs = np.array([calc.map[x] for x in exp.cas_idx[(calc.ref_space.size+calc.no_exp):][np.where(np.abs(lz_orbs) == 5)]])
			if np.where(np.ediff1d(g_orbs) == 1)[0].size < g_orbs.size // 2:
				allow = False
		pi_orbs_u = lz_orbs[np.where(np.abs(lz_orbs) == 6)]
		if pi_orbs_u.size % 2 > 0:
			allow = False
		elif pi_orbs_u.size > 0:
			u_orbs = np.array([calc.map[x] for x in exp.cas_idx[(calc.ref_space.size+calc.no_exp):][np.where(np.abs(lz_orbs) == 6)]])
			if np.where(np.ediff1d(u_orbs) == 1)[0].size < u_orbs.size // 2:
				allow = False
		return allow



