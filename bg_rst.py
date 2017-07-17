#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_rst.py: restart utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np
from os import mkdir, listdir
from os.path import join, isfile, isdir
from shutil import rmtree
from re import search
from copy import deepcopy


class RstCls():
		""" restart class """
		def __init__(self, _out, _mpi):
				""" init restart env and parameters """
				if (_mpi.master):
					self.rst_dir = _out.wrk_dir+'/rst'
					self.rst_freq = 50000.0
					if (not isdir(self.rst_dir)):
						self.restart = False
						mkdir(self.rst_dir)
					else:
						self.restart = True
				_mpi.bcast_rst_info(self)
				#
				return


		def rm_rst(self):
				""" remove rst directory in case of successful calc """
				rmtree(self.rst_dir, ignore_errors=True)
				#
				return


		def rst_main(self, _mpi, _calc, _exp, _time):
				""" main restart driver """
				if (not self.restart):
					# set start order for expansion
					_calc.exp_min_order = 1
				else:
					# read in restart files
					if (_mpi.master): self.read_main(_mpi, _calc, _exp, _time)
					# distribute expansion data to slaves
					if (_mpi.parallel): _mpi.bcast_rst(_calc, _exp, _time)
					# update restart frequency
					for _ in range(1, _calc.exp_min_order): self.rst_freq = self.update()
				#
				return
		
		
		def update(self):
				""" update restart freq according to start order """
				#
				return self.rst_freq / 2.


		def write_kernel(self, _mpi, _exp, _time):
				""" write energy kernel restart files """
				# write e_inc
				np.save(join(self.rst_dir, 'e_inc_' + str(_exp.order)),
						_exp.energy_inc[_exp.order - 1])
				# write timings
				self.write_time(_mpi, _time, 'kernel')
				#
				return
		
		
		def write_summation(self, _mpi, _exp, _time):
				""" write energy summation restart files """
				# write e_inc and e_tot
				np.save(join(self.rst_dir, 'e_inc_' + str(_exp.order)),
						_exp.energy_inc[_exp.order - 1])
				np.save(join(self.rst_dir, 'e_tot_' + str(_exp.order)),
						np.asarray(_exp.energy_tot[_exp.order - 1]))
				# write timings
				self.write_time(_mpi, _time, 'summation')
				#
				return
		
		
		def write_screen(self, _mpi, _exp, _time):
				""" write screening restart files """
				# write tuples
				np.save(join(self.rst_dir, 'tup_' + str(_exp.order + 1)),
						_exp.tuples[_exp.order])
				# write orb_con_abs and orb_con_rel
				np.save(join(self.rst_dir, 'orb_con_abs_'+str(_exp.order)),
						np.asarray(_exp.orb_con_abs[_exp.order - 1]))
				np.save(join(self.rst_dir, 'orb_con_rel_'+str(_exp.order)),
						np.asarray(_exp.orb_con_rel[_exp.order - 1]))
				# write timings
				self.write_time(_mpi, _time, 'screen')
				# write orb_ent_abs and orb_ent_rel
				if (_exp.order >= 2):
					np.save(join(self.rst_dir, 'orb_ent_abs_' + str(_exp.order)),
							_exp.orb_ent_abs[_exp.order - 2])
					np.save(join(self.rst_dir, 'orb_ent_rel_' + str(_exp.order)),
							_exp.orb_ent_rel[_exp.order - 2])
				#
				return


		def write_time(self, _mpi, _time, _phase):
				""" write timings """
				# set phase index
				if (_phase == 'kernel'):
					idx = 0
				elif (_phase == 'summation'):
					idx = 1
				elif (_phase == 'screen'):
					idx = 2
				# write timings
				if (_mpi.parallel):
					np.save(join(self.rst_dir, 'time_work_' + str(_phase)),
							np.asarray(_time.time_work[idx]))
					np.save(join(self.rst_dir, 'time_comm_' + str(_phase)),
							np.asarray(_time.time_comm[idx]))
					np.save(join(self.rst_dir, 'time_idle_' + str(_phase)),
							np.asarray(_time.time_idle[idx]))
				else:
					np.save(join(self.rst_dir, 'time_work_' + str(_phase)),
							np.asarray(_time.timings['work_' + str(_phase)]))
				#
				return


		def read_main(self, _mpi, _calc, _exp, _time):
				""" driver for reading of restart files """
				# list filenames in files list
				files = [f for f in listdir(self.rst_dir) if isfile(join(self.rst_dir, f))]
				# sort the list of files
				files.sort()
				# loop over files
				for i in range(len(files)):
					# read tuples
					if ('tup' in files[i]):
						_exp.tuples.append(np.load(join(self.rst_dir,
											files[i])))
					# read orbital entanglement matrices
					elif ('orb_ent' in files[i]):
						if ('abs' in files[i]):
							_exp.orb_ent_abs.append(np.load(join(self.rst_dir,
															files[i])))
						elif ('rel' in files[i]):
							_exp.orb_ent_rel.append(np.load(join(self.rst_dir,
																files[i])))
					# read orbital contributions
					elif ('orb_con' in files[i]):
						if ('abs' in files[i]):
							_exp.orb_con_abs.append(np.load(join(self.rst_dir,
															files[i])).tolist())
						elif ('rel' in files[i]):
							_exp.orb_con_rel.append(np.load(join(self.rst_dir,
															files[i])).tolist())
					# read e_inc
					elif ('e_inc' in files[i]):
						_exp.energy_inc.append(np.load(join(self.rst_dir,
														files[i])))
					# read e_tot
					elif ('e_tot' in files[i]):
						_exp.energy_tot.append(np.load(join(self.rst_dir,
														files[i])).tolist())
					# read timings
					elif ('time' in files[i]):
						if ('kernel' in files[i]):
							if ('work' in files[i]):
								if (_mpi.parallel):
									_time.time_work[0] = np.load(join(self.rst_dir,
																		files[i])).tolist()
									_time.timings['work_kernel'] = deepcopy(_time.time_work[0][0])
								else:
									_time.timings['work_kernel'] = np.load(join(self.rst_dir,	
																			files[i])).tolist()
							elif ('comm' in files[i]):
								_time.time_comm[0] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								_time.timings['comm_kernel'] = deepcopy(_time.time_comm[0][0])
							elif ('idle' in files[i]):
								_time.time_idle[0] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								_time.timings['idle_kernel'] = deepcopy(_time.time_idle[0][0])
						elif ('summation' in files[i]):
							if ('work' in files[i]):
								if (_mpi.parallel):
									_time.time_work[1] = np.load(join(self.rst_dir,
																		files[i])).tolist()
									_time.timings['work_summation'] = deepcopy(_time.time_work[1][0])
								else:
									_time.timings['work_summation'] = np.load(join(self.rst_dir,
																			files[i])).tolist()
							elif ('comm' in files[i]):
								_time.time_comm[1] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								_time.timings['comm_summation'] = deepcopy(_time.time_comm[1][0])
							elif ('idle' in files[i]):
								_time.time_idle[1] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								_time.timings['idle_summation'] = deepcopy(_time.time_idle[1][0])
						elif ('screen' in files[i]):
							if ('work' in files[i]):
								if (_mpi.parallel):
									_time.time_work[2] = np.load(join(self.rst_dir,
																		files[i])).tolist()
									_time.timings['work_screen'] = deepcopy(_time.time_work[2][0])
								else:
									_time.timings['work_screen'] = np.load(join(self.rst_dir,
																			files[i])).tolist()
							elif ('comm' in files[i]):
								_time.time_comm[2] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								_time.timings['comm_screen'] = deepcopy(_time.time_comm[2][0])
							elif ('idle' in files[i]):
								_time.time_idle[2] = np.load(join(self.rst_dir,
																	files[i])).tolist()
								_time.timings['idle_screen'] = deepcopy(_time.time_idle[2][0])
				# set start order for expansion
				_calc.exp_min_order = len(_exp.tuples)
				#
				return


