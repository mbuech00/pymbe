#!/usr/bin/env python

#
# python plotting program for inc.-corr. calculations 
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall 2016, Mainz, Germnay.
#

import copy
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
   ax.set_title('Total '+molecule['model'].upper()+' energy')
   #
   u_limit = molecule['u_limit'][0]
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'] == 'conv')):
      #
      u_limit -= molecule['ncore']
   #
   if (molecule['corr']):
      #
      colours = ['red','green','orange','lightblue','limegreen']
      #
      e_corr = []
      #
      for i in range(0,molecule['corr_order']):
         #
         e_corr.append([])
         #
         if (i == 0):
            #
            e_corr[i] = copy.deepcopy(molecule['e_tot'][0])
         #
         else:
            #
            e_corr[i] = copy.deepcopy(e_corr[i-1])
         #
         for j in range((molecule['min_corr_order']+i)-1,len(e_corr[i])):
            #
            e_corr[i][j] += (molecule['e_corr'][0][(molecule['min_corr_order']+i)-1]-molecule['e_corr'][0][(molecule['min_corr_order']+i)-2])
   #
   ax.plot(list(range(1,len(molecule['e_tot'][0])+1)),molecule['e_tot'][0],marker='x',linewidth=2,\
           label='BG('+molecule['model'].upper()+')')
   #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         ax.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_corr[i],\
                 marker='x',linewidth=2,label='BG('+molecule['model'].upper()+')-'+str(i+1))
   #
   ax.set_xlim([0.5,u_limit+0.5])
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
   with sns.axes_style("whitegrid"):
      #
      insert = plt.axes([.35, .50, .50, .30],frameon=True)
      #
      insert.plot(list(range(2,len(molecule['e_tot'][0])+1)),molecule['e_tot'][0][1:],marker='x',linewidth=2)
      #
      if (molecule['corr']):
         #
         for i in range(0,molecule['corr_order']):
            #
            insert.plot(list(range(2,len(molecule['e_tot'][0])+1)),e_corr[i][1:],\
                     marker='x',linewidth=2)
      #
      plt.setp(insert,xticks=list(range(3,len(molecule['e_tot'][0])+1)))
      #
      insert.set_xlim([2.5,len(molecule['e_tot'][0])+0.5])
      #
      insert.locator_params(axis='y',nbins=6)
      #
      insert.set_ylim([molecule['e_tot'][0][-1]-0.01,\
                       molecule['e_tot'][0][-1]+0.01])
      #
      insert.xaxis.grid(False)
   #
   ax.legend(loc=1,ncol=molecule['corr_order']+1)
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
   ax.set_title('Total number of '+molecule['model'].upper()+' tuples')
   #
   u_limit = molecule['u_limit'][0]
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'] == 'conv')):
      #
      u_limit -= molecule['ncore']
   #
   if (molecule['corr']):
      #
      sec_prim = []
      theo_sec = []
      #
      for i in range(0,u_limit):
         #
         if (molecule['prim_n_tuples'][0][i] < molecule['theo_work'][0][i]):
            #
            if (molecule['sec_n_tuples'][0][i] > 0):
               #
               if (molecule['sec_n_tuples'][0][i] == molecule['theo_work'][0][i]):
                  #
                  sec_prim.append(molecule['sec_n_tuples'][0][i]-molecule['prim_n_tuples'][0][i])
               #
               else:
                  #
                  sec_prim.append(molecule['theo_work'][0][i]-molecule['prim_n_tuples'][0][i])
               #
               theo_sec.append(0)
            #
            else:
               #
               sec_prim.append(0)
               theo_sec.append(molecule['theo_work'][0][i]-molecule['prim_n_tuples'][0][i])
         #
         else:
            #
            sec_prim.append(0)
            theo_sec.append(0)
      #
      sns.barplot(list(range(1,u_limit+1)),theo_sec,bottom=[(i + j) for i,j in zip(sec_prim,molecule['prim_n_tuples'][0])],\
                  palette='BuGn_d',label='Theoretical number',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),sec_prim,bottom=molecule['prim_n_tuples'][0],palette='Reds_r',\
                  label='Energy corr.',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),molecule['prim_n_tuples'][0],palette='Blues_r',\
                  label='BG('+molecule['model'].upper()+') expansion',log=True)
   #
   else:
      #
      sns.barplot(list(range(1,u_limit+1)),[(i - j) for i,j in zip(molecule['theo_work'][0],molecule['prim_n_tuples'][0])],\
                  bottom=molecule['prim_n_tuples'][0],palette='BuGn_d',label='Theoretical number',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),molecule['prim_n_tuples'][0],palette='Blues_r',\
                  label='BG('+molecule['model'].upper()+') expansion',log=True)
   #
   ax.xaxis.grid(False)
   #
   ax.set_ylim(bottom=0.7)
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Number of correlated tuples')
   #
   plt.legend(loc=1)
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
   if (molecule['corr']):
      #
      e_corr = []
      #
      for i in range(0,molecule['corr_order']):
         #
         e_corr.append([])
         #
         if (i == 0):
            #
            e_corr[i] = copy.deepcopy(molecule['e_tot'][0])
         #
         else:
            #
            e_corr[i] = copy.deepcopy(e_corr[i-1])
         #
         for j in range((molecule['min_corr_order']+i)-1,len(e_corr[i])):
            #
            e_corr[i][j] += (molecule['e_corr'][0][(molecule['min_corr_order']+i)-1]-molecule['e_corr'][0][(molecule['min_corr_order']+i)-2])
   #
   e_diff_tot_abs = []
   e_diff_corr_abs = []
   e_diff_tot_rel = []
   e_diff_corr_rel = []
   #
   for j in range(0,len(molecule['e_tot'][0])):
      #
      e_diff_tot_abs.append((molecule['e_tot'][0][j]-molecule['e_ref'])/kcal_mol)
      e_diff_tot_rel.append((molecule['e_tot'][0][j]/molecule['e_ref'])*100.)
      #
   for i in range(0,molecule['corr_order']):
      #
      e_diff_corr_abs.append([])
      e_diff_corr_rel.append([])
      #
      for j in range(0,len(molecule['e_tot'][0])):
         #
         e_diff_corr_abs[i].append((e_corr[i][j]-molecule['e_ref'])/kcal_mol)
         e_diff_corr_rel[i].append((e_corr[i][j]/molecule['e_ref'])*100.)
   #
   ax1.set_title('Absolute difference from E('+molecule['model'].upper()+')')
   #
   u_limit = molecule['u_limit'][0]
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and (molecule['frozen'] == 'conv')):
      #
      u_limit -= molecule['ncore']
   #
   ax1.axhline(0.0,color='black',linewidth=2)
   #
   ax1.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_tot_abs,marker='x',linewidth=2,\
            label='BG('+molecule['model'].upper()+')')
   #
   for i in range(0,molecule['corr_order']):
      #
      ax1.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_corr_abs[i],marker='x',linewidth=2,\
               label='BG('+molecule['model'].upper()+')-'+str(i+1))
   #
   ax1.set_ylim([-3.4,3.4])
   #
   ax1.xaxis.grid(False)
   #
   ax1.set_ylabel('Difference (in kcal/mol)')
   #
   ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax1.legend(loc=1)
   #
   ax2.set_title('Relative recovery of E('+molecule['model'].upper()+')')
   #
   ax2.axhline(100.0,color='black',linewidth=2)
   #
   ax2.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_tot_rel,marker='x',linewidth=2,\
            label='BG('+molecule['model'].upper()+')')
   #
   for i in range(0,molecule['corr_order']):
      #
      ax2.plot(list(range(1,len(molecule['e_tot'][0])+1)),e_diff_corr_rel[i],marker='x',linewidth=2,\
               label='BG('+molecule['model'].upper()+')-'+str(i+1))
   #
   ax2.set_xlim([0.5,u_limit+0.5])
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
   ax2.legend(loc=1)
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
   #  ---  plot number of calculations from each orbital  ---
   #
   n_tuples_plot(molecule)
   #
   #  ---  plot deviation from reference calc  ---
   #
   if (molecule['ref']):
      #
      dev_ref_plot(molecule)
   #
   return molecule


