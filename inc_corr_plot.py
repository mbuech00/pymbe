#!/usr/bin/env python

#
# python plotting program for inc.-corr. calculations 
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def abs_energy_plot(molecule):
   #
   fig, ax = plt.subplots()
   #
   dim_list = []
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      dim_list.append(i+1.0)
   #
   ax.set_title('Total '+molecule['model']+' energy')
   #
   ax.plot(dim_list,molecule['e_fin'],marker='x',linewidth=2,color='red',linestyle='-')
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      ax.set_xlim([0.5,(molecule['nocc']-molecule['core'])+0.5])
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      ax.set_xlim([0.5,molecule['nvirt']+0.5])
   #
   ax.yaxis.grid(True)
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Energy (in Hartree)')
   #
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/abs_energy_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def n_contrib_plot(molecule):
   #
   fig, ax = plt.subplots()
   #
   width = 0.6
   #
   dim_list = []
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      dim_list.append(i+0.7)
   #
   ax.set_title('Total number of '+molecule['model']+' tuples')
   #
   ax.set_yscale('log')
   #
   ax.bar(dim_list,molecule['n_contrib'][0:len(molecule['e_fin'])],width,color='blue',alpha=0.3,log=True)
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      ax.set_xlim([0.5,(molecule['nocc']-molecule['core'])+0.5])
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      ax.set_xlim([0.5,molecule['nvirt']+0.5])
   #
   ax.yaxis.grid(True)
   #
   ax.set_ylim(bottom=0.7)
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Number of correlated tuples')
   #
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/n_contrib_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def e_contrib_plot(molecule):
   #
   fig, ax = plt.subplots()
   #
   orb_contrib = []
   orb_contrib_sum = []
   dim_list = []
   #
   width = 0.6
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      l_limit = molecule['core']
      u_limit = molecule['nocc']-molecule['core']
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      l_limit = molecule['nocc']
      u_limit = molecule['nvirt']
   #
   for i in range(0,u_limit):
      #
      dim_list.append(i+0.7)
      #
   for _ in range(0,u_limit):
      #
      orb_contrib.append([])
   #
   for j in range(l_limit,l_limit+u_limit):
      # 
      for k in range(0,len(molecule['e_contrib'])):
         #
         for l in range(0,len(molecule['e_contrib'][k])):
            #
            if (set([j+1]) <= set(molecule['e_contrib'][k][l][0])):
               #
               orb_contrib[j-l_limit].append(molecule['e_contrib'][k][l][1])
   #
   for i in range(0,u_limit):
      #
      orb_contrib_sum.append(sum(orb_contrib[i]))
   #
   ax.bar(dim_list,orb_contrib_sum,width,color='green',alpha=0.3)
   #
   ax.set_xlim([0.5,u_limit+0.5])
   #
   ax.set_ylim(top=0.0)
   #
   ax.set_xlabel('Orbital')
   ax.set_ylabel('Energy contribution')
   #
   plt.gca().invert_yaxis()
   #
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/e_contrib_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def dev_ref_plot(molecule):
   #
   fig, ( ax1, ax2 ) = plt.subplots(2, 1, sharex='col', sharey='row')
   #
   kcal_mol = 0.001594
   #
   if (molecule['exp_ctrl']):
      #
      if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
         #
         error_abs = (molecule['thres'][0]/kcal_mol)/2.0
         error_rel_p = ((molecule['e_ref']+(molecule['thres'][0]/2.0))/molecule['e_ref'])*100.
         error_rel_m = ((molecule['e_ref']-(molecule['thres'][0]/2.0))/molecule['e_ref'])*100.
      #
      elif (molecule['exp'] == 'VIRT'):
         #
         error_abs = (molecule['thres'][1]/kcal_mol)/2.0
         error_rel_p = ((molecule['e_ref']+(molecule['thres'][1]/2.0))/molecule['e_ref'])*100.
         error_rel_m = ((molecule['e_ref']-(molecule['thres'][1]/2.0))/molecule['e_ref'])*100.
   else:
      #
      error_abs = 0.0
      error_rel_p = 0.0
      error_rel_m = 0.0
   #
   dim_list = []
   #
   e_diff_abs = []
   e_diff_rel = []
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      dim_list.append(i+1.0)
      #
      e_diff_abs.append((molecule['e_fin'][i]-molecule['e_ref'])/kcal_mol)
      e_diff_rel.append((molecule['e_fin'][i]/molecule['e_ref'])*100.)
   #
   ax1.set_title('Absolute difference from E('+molecule['model']+')')
   #
   ax1.axhline(0.0,color='black',linewidth=2)
   #
   ax1.plot(dim_list,e_diff_abs,marker='x',linewidth=2,color='red',linestyle='-')
   #
   ax1.axhspan(-error_abs,error_abs,color='green',alpha=0.2)
   #
   ax1.xaxis.grid(True)
   #
   ax1.set_ylim([-3.4,3.4])
   #
   ax1.grid()
   #
   ax1.set_ylabel('Difference (in kcal/mol)')
   #
   ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax2.set_title('Relative recovery of E('+molecule['model']+')')
   #
   ax2.axhline(100.0,color='black',linewidth=2)
   #
   ax2.plot(dim_list,e_diff_rel,marker='x',linewidth=2,color='red',linestyle='-')
   #
   ax2.axhspan(error_rel_m,error_rel_p,color='green',alpha=0.2)
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      ax2.set_xlim([0.5,(molecule['nocc']-molecule['core'])+0.5])
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      ax2.set_xlim([0.5,molecule['nvirt']+0.5])
   #
   ax2.xaxis.grid(True)
   #
   ax2.grid()
   #
   ax2.set_ylim([93.,107.])
   #
   ax2.set_ylabel('Recovery (in %)')
   ax2.set_xlabel('Expansion order')
   #
   ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   plt.savefig(molecule['wrk']+'/output/dev_ref_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def ic_plot(molecule):
   #
   #  ---  plot total energies  ---
   #
   abs_energy_plot(molecule)
   #
   #  ---  plot number of calculations from each orbital  ---
   #
   n_contrib_plot(molecule)
   #
   #  ---  plot energy contributions from each orbital  ---
   #
   e_contrib_plot(molecule)
   #
   #  ---  plot deviation from reference calc  ---
   #
   if (molecule['ref']):
      #
      dev_ref_plot(molecule)
   #
   return molecule


