#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_driver.py: driver routines for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

import numpy as np

from bg_mpi_utils import prepare_calc
from bg_print import print_mono_exp_header, print_mono_exp_end, print_kernel_header, print_kernel_end, print_results,\
                      print_summation_header, print_summation_end, print_screen_header, print_screen_end
from bg_kernel import energy_kernel
from bg_summation import energy_summation
from bg_entanglement import entanglement_main
from bg_screening import screening_main
from bg_rst_main import rst_main
from bg_rst_write import rst_write_summation, rst_write_screen


class DrvCls():
		""" driver class """
		def driver(self, _mpi, _mol, _calc, _exp, _prt, _time, _err):
				""" main driver routine """
				# print header for mono expansion
				_prt.exp_header()
				# call mono expansion driver function 
				mono_exp_drv(molecule,molecule['min_order'],molecule['max_order'],'MACRO')
				# print end for mono expansion
				print_mono_exp_end(molecule)
				#
				return
		
		
		def mono_exp_drv(molecule, start, end, level):
				""" driver for mono expansions """
				for k in range(start,end+1):
					# mono expansion energy kernel
					mono_exp_kernel(molecule,k,level)
					# mono expansion energy summation
					mono_exp_summation(molecule,k,level)
					# mono expansion screening
					mono_exp_screen(molecule,k,level)
					# return if converged
					if (molecule['conv_orb'][-1] or molecule['conv_energy'][-1]): break
				#
				return
		
		
		def mono_exp_kernel(molecule, order, level):
				""" driver for the energy kernel (mono expansions) """
				# set local variables
				if (level == 'MACRO'):
					tup = molecule['prim_tuple']
					e_inc = molecule['prim_energy_inc']
					e_tot = molecule['prim_energy']
				# print kernel header
				print_kernel_header(molecule,tup[-1],order,level)
				# init e_int list
				if (len(e_inc) != order): e_inc.append(np.zeros(len(tup[-1]),dtype=np.float64))
				# run the calculations
				energy_kernel(molecule,tup,e_inc,molecule['l_limit'],molecule['u_limit'],order,level)
				# print kernel end
				print_kernel_end(molecule,tup,order,level)
				#
				return
		
		
		def mono_exp_summation(molecule, order, level):
				""" driver for the energy summation (mono expansions) """
				# set local variables
				if (level == 'MACRO'):
					tup = molecule['prim_tuple']
					e_inc = molecule['prim_energy_inc']
					e_tot = molecule['prim_energy']
					thres = molecule['prim_energy_thres']
				# print summation header
				print_summation_header(molecule,order,level)
				# calculate the energy at current order
				energy_summation(molecule,tup,e_inc,e_tot,thres,order,level)
				# write restart files
				rst_write_summation(molecule,e_inc,e_tot,order)
				# print summation end
				print_summation_end(molecule,e_inc,thres,order,level)
				# print results
				print_results(molecule,tup,e_inc,level)
				#
				return
		
		
		def mono_exp_screen(molecule, order, level):
				""" driver for the screening (mono expansions) """
				# set local variables
				if (level == 'MACRO'):
					#
					tup = molecule['prim_tuple']
					e_inc = molecule['prim_energy_inc']
					thres = molecule['prim_exp_thres']
					orb_con_abs = molecule['prim_orb_con_abs']
					orb_con_rel = molecule['prim_orb_con_rel']
					orb_ent_abs = molecule['prim_orb_ent_abs']
					orb_ent_rel = molecule['prim_orb_ent_rel']
				# print screen header
				print_screen_header(molecule,order,level)
				# orbital entanglement
				entanglement_main(molecule,molecule['l_limit'],molecule['u_limit'],order,molecule['conv_energy'][-1])
				# orbital screening (using info from previous order)
				if (not molecule['conv_energy'][-1]):
					# perform screening
					screening_main(molecule,tup,e_inc,thres,molecule['l_limit'],molecule['u_limit'],order,level)
					# write restart files
					if (not molecule['conv_orb'][-1]): rst_write_screen(molecule,tup,orb_con_abs,orb_con_rel,orb_ent_abs,orb_ent_rel,order)
				# print screen end
				print_screen_end(molecule,order,level)
				#
				return


