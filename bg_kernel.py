#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_kernel.py: kernel class for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI


class KernCls():
		""" kernel class """
		def kernel_main(molecule, tup, e_inc, l_limit, u_limit, order, level):
				""" energy kernel phase """
				# mpi parallel version
				if (molecule['mpi_parallel']):
					energy_kernel_master(molecule,tup,e_inc,l_limit,u_limit,order,level)
					collect_kernel_mpi_time(molecule,order)
				else:
					# init string dict
					string = {'drop': ''}
					# determine start index
					if (molecule['rst'] and (order == molecule['min_order'])):
						start = np.argmax(e_inc[order-1] == 0.0)
					else:
						start = 0
					# loop over tuples
					for i in range(start,len(tup[order-1])):
						# start work time
						timer_mpi(molecule,'mpi_time_work_kernel',order)
						# write string
						orb_string(molecule,l_limit,u_limit,tup[order-1][i],string)
						# run correlated calc
						run_calc_corr(molecule,string['drop'],level)
						# write tuple energy
						e_inc[order-1][i] = molecule['e_tmp']
						# print status
						print_status(float(i+1)/float(len(tup[order-1])),level)
						# error handling
						if (molecule['error'][-1]):
							molecule['error_rank'] = 0
							molecule['error_drop'] = string['drop']
							term_calc(molecule)
						# collect work time
						timer_mpi(molecule,'mpi_time_work_kernel',order,True)
						# write restart files
						if (((i+1) % molecule['rst_freq']) == 0): rst_write_kernel(molecule,e_inc,order)
				#
				return
		
		
		def kernel_master(molecule, tup, e_inc, l_limit, u_limit, order, level):
				""" master function energy kernel phase """
				# start idle time
				timer_mpi(molecule,'mpi_time_idle_kernel',order)
				# wake up slaves
				msg = {'task': 'energy_kernel_par', 'l_limit': l_limit, 'u_limit': u_limit, 'order': order, 'level': level}
				# bcast
				molecule['mpi_comm'].bcast(msg,root=0)
				# start work time
				timer_mpi(molecule,'mpi_time_work_kernel',order)
				# init job_info dictionary
				job_info = {}
				# number of slaves
				num_slaves = molecule['mpi_size'] - 1
				# number of available slaves
				slaves_avail = num_slaves
				# define mpi message tags
				tags = enum('ready','done','data','exit','start')
				# init job index
				i = np.argmax(e_inc[-1] == 0.0)
				# init stat counter
				counter = i
				# init timings
				if ((not molecule['rst']) or (order != molecule['min_order'])):
					for j in range(0,molecule['mpi_size']):
						molecule['mpi_time_work'][0][j].append(0.0)
						molecule['mpi_time_comm'][0][j].append(0.0)
						molecule['mpi_time_idle'][0][j].append(0.0)
				# print status for START
				print_status(float(counter)/float(len(tup[-1])),level)
				# loop until no slaves left
				while (slaves_avail >= 1):
					# start idle time
					timer_mpi(molecule,'mpi_time_idle_kernel',order)
					# receive data dict
					stat = molecule['mpi_comm'].recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=molecule['mpi_stat'])
					# start work time
					timer_mpi(molecule,'mpi_time_work_kernel',order)
					# probe for source and tag
					source = molecule['mpi_stat'].Get_source(); tag = molecule['mpi_stat'].Get_tag()
					# slave is ready
					if (tag == tags.ready):
						# any jobs left?
						if (i <= (len(tup[-1])-1)):
							# store job index
							job_info['index'] = i
							# start comm time
							timer_mpi(molecule,'mpi_time_comm_kernel',order)
							# send string dict
							molecule['mpi_comm'].send(job_info,dest=source,tag=tags.start)
							# start work time
							timer_mpi(molecule,'mpi_time_work_kernel',order)
							# increment job index
							i += 1
						else:
							# start comm time
							timer_mpi(molecule,'mpi_time_comm_kernel',order)
							# send exit signal
							molecule['mpi_comm'].send(None,dest=source,tag=tags.exit)
							# start work time
							timer_mpi(molecule,'mpi_time_work_kernel',order)
					# receive result from slave
					elif (tag == tags.done):
						# start comm time
						timer_mpi(molecule,'mpi_time_comm_kernel',order)
						# receive data
						data = molecule['mpi_comm'].recv(source=source,tag=tags.data,status=molecule['mpi_stat'])
						# start work time
						timer_mpi(molecule,'mpi_time_work_kernel',order)
						# write to e_inc
						e_inc[-1][data['index']] = data['energy']
						# store timings
						molecule['mpi_time_work'][0][source][-1] = data['t_work']
						molecule['mpi_time_comm'][0][source][-1] = data['t_comm']
						molecule['mpi_time_idle'][0][source][-1] = data['t_idle']
						# write restart files
						if (((data['index']+1) % int(molecule['rst_freq'])) == 0):
							molecule['mpi_time_work'][0][0][-1] = molecule['mpi_time_work_kernel'][-1]
							molecule['mpi_time_comm'][0][0][-1] = molecule['mpi_time_comm_kernel'][-1]
							molecule['mpi_time_idle'][0][0][-1] = molecule['mpi_time_idle_kernel'][-1]
							rst_write_kernel(molecule,e_inc,order)
						# increment stat counter
						counter += 1
						# print status
						if (((data['index']+1) % 1000) == 0): print_status(float(counter)/float(len(tup[-1])),level)
						# error handling
						if (data['error']):
							molecule['error'].append(True)
							molecule['error_code'] = data['error_code']
							molecule['error_msg'] = data['error_msg']
							molecule['error_rank'] = source
							molecule['error_drop'] = data['error_drop']
							term_calc(molecule)
					# put slave to sleep
					elif (tag == tags.exit):
						slaves_avail -= 1
				# print 100.0 %
				print_status(1.0,level)
				# collect work time
				timer_mpi(molecule,'mpi_time_work_kernel',order,True)
				#
				return
		
		
		def energy_kernel_slave(molecule, tup, e_inc, l_limit, u_limit, order, level):
				""" slave function energy kernel phase """
				# start work time
				timer_mpi(molecule,'mpi_time_work_kernel',order)
				# init e_inc list
				if (len(e_inc) != order): e_inc.append(np.zeros(len(tup[-1]),dtype=np.float64))
				# define mpi message tags
				tags = enum('ready','done','data','exit','start')
				# init string dict
				string = {'drop': ''}
				# init data dict
				data = {}
				# receive work from master
				while True:
					# start comm time
					timer_mpi(molecule,'mpi_time_comm_kernel',order)
					# ready for task
					molecule['mpi_comm'].send(None,dest=0,tag=tags.ready)
					# receive drop string
					job_info = molecule['mpi_comm'].recv(source=0,tag=MPI.ANY_SOURCE,status=molecule['mpi_stat'])
					# start work time
					timer_mpi(molecule,'mpi_time_work_kernel',order)
					# recover tag
					tag = molecule['mpi_stat'].Get_tag()
					# do job
					if (tag == tags.start):
						# write string
						orb_string(molecule,l_limit,u_limit,tup[-1][job_info['index']],string)
						# run correlated calc
						run_calc_corr(molecule,string['drop'],level)
						# write tuple energy
						e_inc[-1][job_info['index']] = molecule['e_tmp']
						# start comm time
						timer_mpi(molecule,'mpi_time_comm_kernel',order)
						# report status back to master
						molecule['mpi_comm'].send(None,dest=0,tag=tags.done)
						# start work time
						timer_mpi(molecule,'mpi_time_work_kernel',order)
						# write info into data dict
						data['index'] = job_info['index']
						data['energy'] = molecule['e_tmp']
						data['t_work'] = molecule['mpi_time_work_kernel'][-1]
						data['t_comm'] = molecule['mpi_time_comm_kernel'][-1]
						data['t_idle'] = molecule['mpi_time_idle_kernel'][-1]
						data['error'] = molecule['error'][-1]
						data['error_code'] = molecule['error_code']
						data['error_msg'] = molecule['error_msg']
						data['error_drop'] = string['drop']
						# start comm time
						timer_mpi(molecule,'mpi_time_comm_kernel',order)
						# send data back to master
						molecule['mpi_comm'].send(data,dest=0,tag=tags.data)
						# start work time
						timer_mpi(molecule,'mpi_time_work_kernel',order)
					# exit
					elif (tag == tags.exit):
						break
				# start comm time
				timer_mpi(molecule,'mpi_time_comm_kernel',order)
				# send exit signal to master
				molecule['mpi_comm'].send(None,dest=0,tag=tags.exit)
				# collect comm time
				timer_mpi(molecule,'mpi_time_comm_kernel',order,True)
				#
				return


