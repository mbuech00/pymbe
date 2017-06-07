#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst.py: restart utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from os import mkdir, listdir
from os.path import join, isfile, isdir
from re import search
from copy import deepcopy


class RstCls():
		""" restart class """
		def __init__(self, wrk_dir):
				""" init restart env and parameters """
				self.rst_dir = wrk_dir+'/rst'
				if (not isdir(self.rst_dir)):
					self.restart = False
					mkdir(_rst_dir)
				else:
					self.restart = False
				#
				return self


		def rst_main(self, mpi, calc, exp):
				""" main restart driver """
				if (not self.restart):
					# set start order for expansion
					calc.min_exp_order = 1
				else:
					# read in restart files
					self.read_main(mpi, exp)
					# distribute expansion data to slaves
					if (mpi.parallel): self.dist_master(molecule)
					# update threshold
					self.update(calc)
				#
				return
		
		
		def update(self, calc):
				""" update expansion thres and restart freq according to start order """
				for i in range(1,calc.min_exp_order):
					calc.exp_thres = calc.exp_damp ** i * calc.exp_thres_init
					calc.rst_freq /= 2.
				#
				return


		def write_kernel(self, mpi, exp, time, order):
				""" write energy kernel restart files """
				# write e_inc
				np.save(join(self.rst_dir, 'e_inc_'+str(order)),
						exp.energy_inc[order-1])
				# write timings
				if (mpi.parallel):
					np.save(join(self.rst_dir, 'mpi_time_work_kernel'),
							np.asarray(time.mpi_time_work[0]))
					np.save(join(self.rst_dir, 'mpi_time_comm_kernel'),
							np.asarray(time.mpi_time_comm[0]))
					np.save(join(self.rst_dir, 'mpi_time_idle_kernel'),
							np.asarray(time.mpi_time_idle[0]))
				else:
					np.save(join(self.rst_dir, 'mpi_time_work_kernel'),
							np.asarray(time.mpi_time_work_kernel))
				#
				return
		
		
		def write_summation(self, mpi, exp, time, order):
				""" write energy summation restart files """
				# write e_inc and e_tot
				np.save(join(self.rst_dir, 'e_inc_'+str(order)),
						exp.energy_inc[order-1])
				np.save(join(self.rst_dir, 'e_tot_'+str(order)),
						np.asarray(exp.energy_tot[order-1]))
				# write timings
				if (mpi.parallel):
					np.save(join(self.rst_dir, 'mpi_time_work_summation'),
							np.asarray(time.mpi_time_work[1]))
					np.save(join(self.rst_dir, 'mpi_time_comm_summation'),
							np.asarray(time.mpi_time_comm[1]))
					np.save(join(self.rst_dir, 'mpi_time_idle_summation'),
							np.asarray(time.mpi_time_idle[1]))
				else:
					np.save(join(self.rst_dir, 'mpi_time_work_summation'),
							np.asarray(time.mpi_time_work_summation))
				#
				return
		
		
		def write_screen(self, mpi, exp, time, order):
				""" write screening restart files """
				# write tuples
				np.save(join(self.rst_dir, 'tup_'+str(order+1)),
						exp.tuples[order])
				# write orb_con_abs and orb_con_rel
				np.save(join(self.rst_dir, 'orb_con_abs_'+str(order)),
						np.asarray(exp.orb_con_abs[order-1]))
				np.save(join(self.rst_dir, 'orb_con_rel_'+str(order)),
						np.asarray(exp.orb_con_rel[order-1]))
				# write timings
				if (mpi.parallel):
					np.save(join(self.rst_dir, 'mpi_time_work_screen'),
							np.asarray(time.mpi_time_work[2]))
					np.save(join(self.rst_dir, 'mpi_time_comm_screen'),
							np.asarray(time.mpi_time_comm[2]))
					np.save(join(self.rst_dir, 'mpi_time_idle_screen'),
							np.asarray(time.mpi_time_idle[2]))
				else:
					np.save(join(self.rst_dir, 'mpi_time_work_screen'),
							np.asarray(time.mpi_time_work_screen))
				# write orb_ent_abs and orb_ent_rel
				if (order >= 2):
					np.save(join(self.rst_dir, 'orb_ent_abs_'+str(order)),
							exp.orb_ent_abs[order-2])
					np.save(join(self.rst_dir, 'orb_ent_rel_'+str(order)),
							exp.orb_ent_rel[order-2])
				#
				return


		def rst_read_main(self, mpi, exp, calc, time):
				""" driver for reading of restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort()
				# loop over files
				for i in range(len(files)):
					# read tuples
					if ('tup' in files[i]):
						exp.tuples.append(np.load(join(self.rst_dir,
											files[i])))
					# read orbital entanglement matrices
					elif ('orb_ent' in files[i]):
						if ('abs' in files[i]):
							exp.orb_ent_abs.append(np.load(join(self.rst_dir,
															files[i])))
						elif ('rel' in files[i]):
							exp.orb_ent_rel.append(np.load(join(self.rst_dir,
																files[i])))
					# read orbital contributions
					elif ('orb_con' in files[i]):
						if ('abs' in files[i]):
							exp.orb_con_abs.append(np.load(join(self.rst_dir,
															files[i])).tolist())
						elif ('rel' in files[i]):
							exp.orb_con_rel.append(np.load(join(self.rst_dir,
															files[i])).tolist())
					# read e_inc
					elif ('e_inc' in files[i]):
						exp.energy_inc.append(np.load(join(self.rst_dir,
														files[i])))
					# read e_tot
					elif ('e_tot' in files[i]):
						exp.energy_tot.append(np.load(join(self.rst_dir,
														files[i])).tolist())
					# read timings
					elif ('time' in files[i]):
						if ('kernel' in files[i]):
							if ('work' in files[i]):
								if (mpi.parallel):
									time.mpi_time_work[0] = np.load(join(self.rst_dir,
																		files[i])).tolist()
									time.mpi_time_work_kernel = deepcopy(time.mpi_time_work[0][0])
								else:
									time.mpi_time_work_kernel = np.load(join(self.rst_dir,	
																			files[i])).tolist()
							elif ('comm' in files[i]):
								time.mpi_time_comm[0] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								mpi_time_comm_kernel = deepcopy(time.mpi_time_comm[0][0])
							elif ('idle' in files[i]):
								time.mpi_time_idle[0] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								time.mpi_time_idle_kernel = deepcopy(time.mpi_time_idle[0][0])
						elif ('summation' in files[i]):
							if ('work' in files[i]):
								if (mpi.parallel):
									time.mpi_time_work[1] = np.load(join(self.rst_dir,
																		files[i])).tolist()
									time.mpi_time_work_summation = deepcopy(time.mpi_time_work[1][0])
								else:
									time.mpi_time_work_summation = np.load(join(self.rst_dir,
																			files[i])).tolist()
							elif ('comm' in files[i]):
								time.mpi_time_comm[1] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								time.mpi_time_comm_summation = deepcopy(time.mpi_time_comm[1][0])
							elif ('idle' in files[i]):
								time.mpi_time_idle[1] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								time.mpi_time_idle_summation = deepcopy(time.mpi_time_idle[1][0])
						elif ('screen' in files[i]):
							if ('work' in files[i]):
								if (mpi.parallel):
									time.mpi_time_work[2] = np.load(join(self.rst_dir,
																		files[i])).tolist()
									time.mpi_time_work_screen = deepcopy(time.mpi_time_work[2][0])
								else:
									time.mpi_time_work_screen = np.load(join(self.rst_dir,
																			files[i])).tolist()
							elif ('comm' in files[i]):
								time.mpi_time_comm[2] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								time.mpi_time_comm_screen = deepcopy(time.mpi_time_comm[2][0])
							elif ('idle' in files[i]):
								time.mpi_time_idle[2] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								time.mpi_time_idle_screen = deepcopy(time.mpi_time_idle[2][0])
				# set start order for expansion
				calc.exp_min_order = len(exp.tuples)
				#
				return


