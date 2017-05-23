#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_screening_mpi.py: MPI screening routines for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

from mpi4py import MPI
import numpy as np
from itertools import combinations

from bg_mpi_time import timer_mpi
from bg_mpi_utils import enum


def bcast_tuples(molecule, buff, tup, order):
		""" master/slave routine for bcasting total number of tuples """
		if (molecule['mpi_master']):
			# start comm time
			timer_mpi(molecule,'mpi_time_comm_screen',order)
			# init bcast dict
			tup_info = {'tup_len': len(buff)}
			# bcast
			molecule['mpi_comm'].bcast(tup_info,root=0)
		# start idle time
		timer_mpi(molecule,'mpi_time_idle_screen',order)
		# all meet at barrier
		molecule['mpi_comm'].Barrier()
		# start comm time
		timer_mpi(molecule,'mpi_time_comm_screen',order)
		# bcast buffer
		molecule['mpi_comm'].Bcast([buff,MPI.INT],root=0)
		# start work time
		timer_mpi(molecule,'mpi_time_work_screen',order)
		# append tup[-1] with buff
		if (len(buff) >= 1): tup.append(buff)
		# end work time
		timer_mpi(molecule,'mpi_time_work_screen',order,True)
		#
		return


def tuple_generation_master(molecule, tup, e_inc, thres, l_limit, u_limit, order, level):
		""" master routine for generating input tuples """
		# start idle time
		timer_mpi(molecule,'mpi_time_idle_screen',order)
		# wake up slaves
		msg = {'task': 'tuple_generation_par', 'thres': thres, 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'level': level}
		# bcast
		molecule['mpi_comm'].bcast(msg,root=0)
		# start work time
		timer_mpi(molecule,'mpi_time_work_screen',order)
		# init job_info dictionary
		job_info = {}
		# number of slaves
		num_slaves = molecule['mpi_size']-1
		# number of available slaves
		slaves_avail = num_slaves
		# define mpi message tags
		tags = enum('ready','done','exit','start')
		# init job index, tmp list, and screen_count
		i = 0; tmp = []; molecule['screen_count'] = 0
		# loop until no slaves left
		while (slaves_avail >= 1):
			# start idle time
			timer_mpi(molecule,'mpi_time_idle_screen',order)
			# receive data dict
			data = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=molecule['mpi_stat'])
			# start work time
			timer_mpi(molecule,'mpi_time_work_screen',order)
			# probe for source and tag
			source = molecule['mpi_stat'].Get_source(); tag = molecule['mpi_stat'].Get_tag()
			# slave is ready
			if (tag == tags.ready):
				# jobs left
				if (i <= len(tup[-1])-1):
					# start comm time
					timer_mpi(molecule,'mpi_time_comm_screen',order)
					# save parent tuple index
					job_info['index'] = i
					# send parent tuple index
					molecule['mpi_comm'].send(job_info,dest=source,tag=tags.start)
					# start work time
					timer_mpi(molecule,'mpi_time_work_screen',order)
					# increment job index
					i += 1
				else:
					# start comm time#
					timer_mpi(molecule,'mpi_time_comm_screen',order)
					# send None info
					molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
					# start work time
					timer_mpi(molecule,'mpi_time_work_screen',order)
			# receive result from slave
			elif (tag == tags.done):
				# write tmp child tuple list
				tmp += data['child_tuple'] 
				# increment number of screened tuples
				molecule['screen_count'] += data['screen_count']
			# put slave to sleep
			elif (tag == tags.exit):
				# remove slave
				slaves_avail -= 1
		# finally we sort the tuples or mark expansion as converged 
		if (len(tmp) >= 1):
			tmp.sort()
		else:
			molecule['conv_orb'].append(True)
		# make numpy array out of tmp
		buff = np.array(tmp,dtype=np.int32)
		# bcast buff
		bcast_tuples(molecule,buff,tup,order)
		# delete tmp list
		del tmp
		#
		return


def tuple_generation_slave(molecule, tup, e_inc, thres, l_limit, u_limit, order, level):
		""" slave routine for generating input tuples """
		# start work time
		timer_mpi(molecule,'mpi_time_work_screen',order)
		# define mpi message tags
		tags = enum('ready','done','exit','start')
		# init data dict and combs list
		data = {'child_tuple': [], 'screen_count': 0}; combs = []
		# determine which tuples have contributions larger than the threshold
		allow_tuple = tup[-1][np.where(np.abs(e_inc[-1]) >= thres)]
		# receive work from master
		while True:
			# start comm time
			timer_mpi(molecule,'mpi_time_comm_screen',order)
			# send status to master
			molecule['mpi_comm'].send(None,dest=0,tag=tags.ready)
			# receive parent tuple
			job_info = molecule['mpi_comm'].recv(source=0,tag=MPI.ANY_SOURCE,status=molecule['mpi_stat'])
			# start work time
			timer_mpi(molecule,'mpi_time_work_screen',order)
			# recover tag
			tag = molecule['mpi_stat'].Get_tag()
			# do job
			if (tag == tags.start):
				# init child tuple list and screen counter
				data['child_tuple'][:] = []; data['screen_count'] = 0
				# generate list with all subsets of particular tuple
				combs = list(list(comb) for comb in combinations(tup[-1][job_info['index']],order-1))
				# loop through possible orbitals to augment the combinations with
				for m in range(tup[-1][job_info['index']][-1]+1,(l_limit+u_limit)+1):
					# init screening logical
					screen = False
					# loop over subset combinations
					for j in range(0,len(combs)):
						# check whether or not the particular tuple is actually allowed
						if (not np.equal(combs[j]+[m],tup[-1]).all(axis=1).any()):
							# screen away
							screen = True
							break
					if (not screen):
	                    # loop over subset combinations
						for j in range(0,len(combs)):
							# check whether the particular tuple among negligible tuples
							if (not np.equal(combs[j]+[m],allow_tuple).all(axis=1).any()):
								# screen away
								screen = True
								break
					# if tuple is allowed, add to child tuple list, otherwise screen away
					if (not screen):
						data['child_tuple'].append(tup[-1][job_info['index']].tolist()+[m])
					else:
						data['screen_count'] += 1
				# start comm time
				timer_mpi(molecule,'mpi_time_comm_screen',order)
				# send data back to master
				molecule['mpi_comm'].send(data,dest=0,tag=tags.done)
				# start work time
				timer_mpi(molecule,'mpi_time_work_screen',order)
			# exit
			elif (tag == tags.exit):
				break
		# start comm time
		timer_mpi(molecule,'mpi_time_comm_screen',order)
		# send None info to master
		molecule['mpi_comm'].send(None,dest=0,tag=tags.exit)
		# start work time
		timer_mpi(molecule,'mpi_time_work_screen',order)
		# init buffer
		tup_info = molecule['mpi_comm'].bcast(None,root=0)
		buff = np.empty([tup_info['tup_len'],order+1],dtype=np.int32)
		# receive buffer
		bcast_tuples(molecule,buff,tup,order)
		# delete combs list and clear data dict
		del combs; data.clear()
		#
		return


