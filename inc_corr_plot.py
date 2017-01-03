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
   ax.plot(list(range(1,len(molecule['e_fin'])+1)),molecule['e_fin'],marker='x',linewidth=2,color='red',linestyle='-')
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      ax.set_xlim([0.5,(molecule['nocc']-molecule['core'])+0.5])
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      ax.set_xlim([0.5,molecule['nvirt']+0.5])
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

def n_contrib_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('Total number of '+molecule['model']+' tuples')
   #
   sns.barplot(list(range(1,len(molecule['e_fin'])+1)),molecule['n_contrib'][0:len(molecule['e_fin'])],palette='BuGn_d',log=True)
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
   plt.savefig(molecule['wrk']+'/output/n_contrib_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def e_contrib_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title(str(molecule['exp'])+' scheme: orbital entanglement matrix (order = '+str(len(molecule["e_fin"]))+')')
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      u_limit = molecule['nocc']
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      u_limit = molecule['nvirt']
   #
   orb_contrib_arr = np.asarray(molecule['orb_contrib'])
   #
   # calculate realtive contributions
   #
   tot_sum = 0.0
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      tot_sum += sum(orb_contrib_arr[i,0:])
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      for j in range(0,u_limit):
         #
         orb_contrib_arr[i,j] = (orb_contrib_arr[i,j] / tot_sum) * 100.0
   #
   ax = sns.heatmap(orb_contrib_arr,linewidths=.5,xticklabels=range(1,u_limit+1),\
                    yticklabels=range(1,len(molecule['e_fin'])+1),cmap='coolwarm',cbar=False,\
                    annot=True,fmt='.1f',vmin=-np.amax(orb_contrib_arr),vmax=np.amax(orb_contrib_arr))
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
   plt.savefig(molecule['wrk']+'/output/e_contrib_plot_{0:}_{1:}.pdf'.format(molecule['exp'],len(molecule['e_fin'])), bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def e_contrib_comb_plot(molecule):
   #
   sns.set(style='darkgrid')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title(str(molecule['exp'])+' scheme: orbital entanglement matrix (order = '+str(len(molecule['e_fin']))+')')
   #
   orb_contrib_comb = []
   #
   for i in range(molecule['core'],molecule['nocc']):
      #
      orb_contrib_comb.append([0.0] * molecule['nvirt'])
      #
      for j in range(0,len(molecule['e_contrib'])):
         #
         for k in range(0,len(molecule['e_contrib'][j])):
            #
            if (set([i+1]) <= set(molecule['e_contrib'][j][k][0])):
               #
               for l in range(0,molecule['nvirt']):
                  #
                  orb_contrib_comb[i-molecule['core']][l] += molecule['e_contrib'][j][k][2][l]
   #
   orb_contrib_arr = np.asarray(orb_contrib_comb)
   #
   ax = sns.heatmap(orb_contrib_arr,linewidths=.5,xticklabels=range(1,molecule['nvirt']+1),\
                    yticklabels=range(molecule['core']+1,molecule['nocc']+1),cmap='RdYlBu',cbar=False)
   #
   cbar = ax.figure.colorbar(ax.collections[0],orientation='horizontal')
   cbar.solids.set_rasterized(True)
   cbar.set_ticks([np.amin(orb_contrib_arr),(np.amin(orb_contrib_arr)+np.amax(orb_contrib_arr))/2.0,np.amax(orb_contrib_arr)])
   cbar.set_ticklabels(['high','medium','low'])
   #
   ax.set_xlabel('Virtual orbital')
   ax.set_ylabel('Occupied orbital')
   #
   plt.yticks(rotation=0)
   #
   sns.despine(left=True,bottom=True)
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/e_contrib_plot_{0:}_{1:}.pdf'.format(molecule['exp'],len(molecule['e_fin'])), bbox_inches = 'tight', dpi=1000)
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
   e_diff_abs = []
   e_diff_rel = []
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      e_diff_abs.append((molecule['e_fin'][i]-molecule['e_ref'])/kcal_mol)
      e_diff_rel.append((molecule['e_fin'][i]/molecule['e_ref'])*100.)
   #
   ax1.set_title('Absolute difference from E('+molecule['model']+')')
   #
   ax1.axhline(0.0,color='black',linewidth=2)
   #
   ax1.plot(list(range(1,len(molecule['e_fin'])+1)),e_diff_abs,marker='x',linewidth=2,color='red',linestyle='-')
   #
   ax1.axhspan(-error_abs,error_abs,color='green',alpha=0.2)
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
   ax2.plot(list(range(1,len(molecule['e_fin'])+1)),e_diff_rel,marker='x',linewidth=2,color='red',linestyle='-')
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
#   n_contrib_plot(molecule)
#   #
#   if (molecule['exp'] == 'COMB'):
#      #
#      e_contrib_comb_plot(molecule)
   #
   #  ---  plot deviation from reference calc  ---
   #
   if (molecule['ref']):
      #
      dev_ref_plot(molecule)
   #
   return molecule


