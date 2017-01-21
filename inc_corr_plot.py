#!/usr/bin/env python

#
# python plotting program for inc.-corr. calculations 
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import seaborn as sns

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def abs_energy_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('Total '+molecule['model']+' energy')
   #
   e_tot = []
   #
   for i in range(0,len(molecule['e_tot'][0])):
      #
      if (molecule['est']):
         #
         e_tot.append(molecule['e_tot'][0][i][1])
      #
      else:
         #
         e_tot.append(molecule['e_tot'][0][i][0])
   #
   ax.plot(list(range(1,len(e_tot)+1)),e_tot,marker='x',linewidth=2,color='red',linestyle='-')
   #
   ax.set_xlim([0.5,molecule['u_limit'][0]+0.5])
   #
   ax.xaxis.grid(False)
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Energy (in Hartree)')
   #
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   sns.despine()
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/abs_energy_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def n_tuples_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('Total number of '+molecule['model']+' tuples')
   #
   sns.barplot(list(range(1,len(molecule['e_tot'][0])+1)),molecule['n_tuples'][0][0:len(molecule['e_tot'][0])],palette='BuGn_d',log=True)
   #
   ax.xaxis.grid(False)
   #
   ax.set_ylim(bottom=0.7)
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Number of correlated tuples')
   #
   sns.despine()
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/n_tuples_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def e_contrib_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title(str(molecule['exp'])+' scheme: orbital entanglement matrix (order = '+str(len(molecule["e_tot"][0]))+')')
   #
   orbital_arr = np.asarray(molecule['orbital'])
   #
   # calculate realtive contributions
   #
   tot_sum = 0.0
   #
   for i in range(0,len(molecule['e_tot'][0])):
      #
      tot_sum += sum(orbital_arr[i,0:])
   #
   for i in range(0,len(molecule['e_tot'][0])):
      #
      for j in range(0,molecule['u_limit'][0]):
         #
         orbital_arr[i,j] = (orbital_arr[i,j] / tot_sum) * 100.0
   #
   ax = sns.heatmap(orbital_arr,linewidths=.5,xticklabels=range(1,u_limit+1),\
                    yticklabels=range(1,len(molecule['e_tot'][0])+1),cmap='coolwarm',cbar=False,\
                    annot=True,fmt='.1f',vmin=-np.amax(orbital_arr),vmax=np.amax(orbital_arr))
   #
   ax.set_xlabel('Orbital')
   ax.set_ylabel('Order')
   #
   plt.yticks(rotation=0)
   #
   sns.despine(left=True,bottom=True)
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/e_contrib_plot_{0:}_{1:}.pdf'.format(molecule['exp'],len(molecule['e_tot'][0])), bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def dev_ref_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   sns.despine()
   #
   fig, ( ax1, ax2 ) = plt.subplots(2, 1, sharex='col', sharey='row')
   #
   kcal_mol = 0.001594
   #
   e_diff_abs = []
   e_diff_rel = []
   #
   for i in range(0,len(molecule['e_tot'][0])):
      #
      if (molecule['est']):
         #
         e_diff_abs.append((molecule['e_tot'][0][i][1]-molecule['e_ref'])/kcal_mol)
         e_diff_rel.append((molecule['e_tot'][0][i][1]/molecule['e_ref'])*100.)
      #
      else:
      #
         e_diff_abs.append((molecule['e_tot'][0][i][0]-molecule['e_ref'])/kcal_mol)
         e_diff_rel.append((molecule['e_tot'][0][i][0]/molecule['e_ref'])*100.)
   #
   ax1.set_title('Absolute difference from E('+molecule['model']+')')
   #
   ax1.axhline(0.0,color='black',linewidth=2)
   #
   ax1.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_abs,marker='x',linewidth=2,color='red',linestyle='-')
   #
   ax1.set_ylim([-3.4,3.4])
   #
   ax1.xaxis.grid(False)
   #
   ax1.set_ylabel('Difference (in kcal/mol)')
   #
   ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax2.set_title('Relative recovery of E('+molecule['model']+')')
   #
   ax2.axhline(100.0,color='black',linewidth=2)
   #
   ax2.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_rel,marker='x',linewidth=2,color='red',linestyle='-')
   #
   ax2.set_xlim([0.5,molecule['u_limit'][0]+0.5])
   #
   ax2.xaxis.grid(False)
   #
   ax2.set_ylim([93.,107.])
   #
   ax2.set_ylabel('Recovery (in %)')
   ax2.set_xlabel('Expansion order')
   #
   ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   sns.despine()
   #
   fig.tight_layout()
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
#   #  ---  plot number of calculations from each orbital  ---
#   #
#   n_tuples_plot(molecule)
   #
   #  ---  plot deviation from reference calc  ---
   #
   if (molecule['ref']):
      #
      dev_ref_plot(molecule)
   #
   return molecule


