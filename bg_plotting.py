#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_plotting.py: plotting utilities for Bethe-Goldstone correlation calculations."""

from copy import deepcopy
from numpy import asarray, amax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.4'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def abs_energy_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('Total '+molecule['model'].upper()+' energy')
   #
   u_limit = molecule['u_limit']
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']):
      #
      u_limit -= molecule['ncore']
   #
   if (molecule['corr']):
      #
      styles = ['--','-.','-','--','-.','-','--','-.','-','--','-.']
      #
      corr_energy = []
      #
      for i in range(0,molecule['corr_order']):
         #
         corr_energy.append([])
         #
         if (i == 0):
            #
            corr_energy[i] = deepcopy(molecule['prim_energy'])
         #
         else:
            #
            corr_energy[i] = deepcopy(corr_energy[i-1])
         #
         for j in range((molecule['min_corr_order']+i)-1,len(corr_energy[i])):
            #
            corr_energy[i][j] += (molecule['corr_energy'][(molecule['min_corr_order']+i)-1]-molecule['corr_energy'][(molecule['min_corr_order']+i)-2])
   #
   ax.plot(list(range(1,len(molecule['prim_energy'])+1)),molecule['prim_energy'],marker='x',linewidth=2,linestyle='-',\
           label='BG('+molecule['model'].upper()+')')
   #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         ax.plot(list(range(1,len(molecule['prim_energy'])+1)),corr_energy[i],\
                 marker='x',linestyle=styles[i],linewidth=2,label='BG('+molecule['model'].upper()+')-'+str(i+1))
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
      insert.plot(list(range(2,len(molecule['prim_energy'])+1)),molecule['prim_energy'][1:],marker='x',linewidth=2,linestyle='-')
      #
      if (molecule['corr']):
         #
         for i in range(0,molecule['corr_order']):
            #
            insert.plot(list(range(2,len(molecule['prim_energy'])+1)),corr_energy[i][1:],\
                     marker='x',linewidth=2,linestyle=styles[i])
      #
      plt.setp(insert,xticks=list(range(3,len(molecule['prim_energy'])+1)))
      #
      insert.set_xlim([2.5,len(molecule['prim_energy'])+0.5])
      #
      insert.locator_params(axis='y',nbins=6)
      #
      insert.set_ylim([molecule['prim_energy'][-1]-0.01,\
                       molecule['prim_energy'][-1]+0.01])
      #
      insert.xaxis.grid(False)
   #
   if (molecule['corr']):
      #
      ax.legend(loc=1,ncol=molecule['corr_order']+1)
   #
   else:
      #
      ax.legend(loc=1)
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
   u_limit = molecule['u_limit']
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']):
      #
      u_limit -= molecule['ncore']
   #
   if (molecule['corr']):
      #
      corr_prim = []
      theo_corr = []
      #
      for i in range(0,u_limit):
         #
         corr_prim.append(molecule['corr_n_tuples'][i])
         theo_corr.append(molecule['theo_work'][i]-(molecule['prim_n_tuples'][i]+molecule['corr_n_tuples'][i]))
      #
      sns.barplot(list(range(1,u_limit+1)),theo_corr,bottom=[(i + j) for i,j in zip(corr_prim,molecule['prim_n_tuples'])],\
                  palette='BuGn_d',label='Theoretical number',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),corr_prim,bottom=molecule['prim_n_tuples'],palette='Reds_r',\
                  label='Energy corr.',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),molecule['prim_n_tuples'],palette='Blues_r',\
                  label='BG('+molecule['model'].upper()+') expansion',log=True)
   #
   else:
      #
      sns.barplot(list(range(1,u_limit+1)),[(i - j) for i,j in zip(molecule['theo_work'],molecule['prim_n_tuples'])],\
                  bottom=molecule['prim_n_tuples'],palette='BuGn_d',label='Theoretical number',log=True)
      #
      sns.barplot(list(range(1,u_limit+1)),molecule['prim_n_tuples'],palette='Blues_r',\
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

def orb_con_plot(molecule):
   #
   sns.set(style='whitegrid')
   #
   cmap = sns.cubehelix_palette(as_cmap=True)
   #
   if ((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')):
      #
      orbital_type = 'occupied'
   #
   elif ((molecule['exp'] == 'virt') or (molecule['exp'] == 'comb-vo')):
      #
      orbital_type = 'virtual'
   #
   if (molecule['corr']):
      #
      fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
   #
   else:
      #
      fig, ax1 = plt.subplots()
   #
   if (molecule['corr']):
      #
      # primary expansion
      #
      orb_arr = 100.0 * asarray(molecule['prim_orb_con_rel'])
      #
      sns.heatmap(orb_arr,ax=ax1,cmap=cmap,cbar_kws={'format':'%.0f'},\
                       xticklabels=False,\
                       yticklabels=range(1,len(molecule['prim_orb_con_rel'])+1),cbar=True,\
                       annot=False,fmt='.1f',vmin=0.0,vmax=amax(orb_arr))
      #
      ax1.set_yticklabels(ax1.get_yticklabels(),rotation=0)
      #
      ax1.set_title('Primary BG expansion')
      #
      # energy correction
      #
      orb_arr_corr = 100.0 * asarray(molecule['corr_orb_con_rel'])
      #
      diff_arr = orb_arr_corr - orb_arr
      #
      mask_arr = (diff_arr == 0.0)
      #
      sns.heatmap(diff_arr,ax=ax2,mask=mask_arr,cmap='coolwarm',cbar_kws={'format':'%.1f'},\
                       xticklabels=False,\
                       yticklabels=range(1,len(molecule['corr_orb_con_rel'])+1),cbar=True,\
                       annot=False,fmt='.1f',vmax=amax(diff_arr))
      #
      ax2.set_yticklabels(ax2.get_yticklabels(),rotation=0)
      #
      ax2.set_title('Energy correction')
      #
      fig.text(0.42,0.0,'Relative contribution (in %) from individual {0:} orbitals'.format(orbital_type),ha='center',va='center')
      fig.text(0.0,0.5,'Bethe-Goldstone order',ha='center',va='center',rotation='vertical')
   #
   else:
      #
      # primary expansion
      #
      orb_arr = 100.0 * asarray(molecule['prim_orb_con_rel'])
      #
      sns.heatmap(orb_arr,ax=ax1,cmap=cmap,cbar_kws={'format':'%.0f'},\
                       xticklabels=False,\
                       yticklabels=range(1,len(molecule['prim_orb_con_rel'])+1),cbar=True,\
                       annot=False,fmt='.1f',vmin=0.0,vmax=amax(orb_arr))
      #
      ax1.set_yticklabels(ax1.get_yticklabels(),rotation=0)
      #
      ax1.set_title('Primary BG expansion')
      #
      ax1.set_xlabel('Relative contribution (in %) from individual {0:} orbitals'.format(orbital_type))
      ax1.set_ylabel('Bethe-Goldstone order')
   #
   sns.despine(left=True,bottom=True)
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/orb_con_plot.pdf', bbox_inches = 'tight', dpi=1000)
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
      styles = ['--','-.','-','--','-.','-','--','-.','-','--','-.']
      #
      corr_energy = []
      #
      for i in range(0,molecule['corr_order']):
         #
         corr_energy.append([])
         #
         if (i == 0):
            #
            corr_energy[i] = deepcopy(molecule['prim_energy'])
         #
         else:
            #
            corr_energy[i] = deepcopy(corr_energy[i-1])
         #
         for j in range((molecule['min_corr_order']+i)-1,len(corr_energy[i])):
            #
            corr_energy[i][j] += (molecule['corr_energy'][(molecule['min_corr_order']+i)-1]-molecule['corr_energy'][(molecule['min_corr_order']+i)-2])
   #
   e_diff_tot_abs = []
   e_diff_corr_abs = []
   e_diff_tot_rel = []
   e_diff_corr_rel = []
   #
   for j in range(0,len(molecule['prim_energy'])):
      #
      e_diff_tot_abs.append((molecule['prim_energy'][j]-molecule['e_ref'])/kcal_mol)
      e_diff_tot_rel.append((molecule['prim_energy'][j]/molecule['e_ref'])*100.)
      #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         e_diff_corr_abs.append([])
         e_diff_corr_rel.append([])
         #
         for j in range(0,len(molecule['prim_energy'])):
            #
            e_diff_corr_abs[i].append((corr_energy[i][j]-molecule['e_ref'])/kcal_mol)
            e_diff_corr_rel[i].append((corr_energy[i][j]/molecule['e_ref'])*100.)
   #
   ax1.set_title('Absolute difference from E('+molecule['model'].upper()+')')
   #
   u_limit = molecule['u_limit']
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']):
      #
      u_limit -= molecule['ncore']
   #
   ax1.axhline(0.0,color='black',linewidth=2)
   #
   ax1.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_abs,marker='x',linewidth=2,linestyle='-',\
            label='BG('+molecule['model'].upper()+')')
   #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         ax1.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_corr_abs[i],marker='x',linewidth=2,linestyle=styles[i],\
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
   ax2.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_rel,marker='x',linewidth=2,linestyle='-',\
            label='BG('+molecule['model'].upper()+')')
   #
   if (molecule['corr']):
      #
      for i in range(0,molecule['corr_order']):
         #
         ax2.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_corr_rel[i],marker='x',linewidth=2,linestyle=styles[i],\
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

def mpi_time_plot(molecule):
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('MPI timings')
   #
   labels = 'Idle ({0:.0f} %)'.format(molecule['mpi_time_idle'][1]),'Comm ({0:.0f} %)'.format(molecule['mpi_time_comm'][1]),'Work ({0:.0f} %)'.format(molecule['mpi_time_work'][1])
   #
   sizes = [molecule['mpi_time_idle'][1],molecule['mpi_time_comm'][1],molecule['mpi_time_work'][1]]
   #
   ax.pie(sizes,colors=sns.color_palette("Set2",3),shadow=True,startangle=45)
   #
   plt.legend(labels,frameon=True,fancybox=True,shadow=True,loc=6,bbox_to_anchor=(0.3, 0.4))
   #
   ax.axis('equal')
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk']+'/output/mpi_time_plot.pdf', bbox_inches = 'tight', dpi=1000)
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
   #  ---  plot total orbital contribution matrix  ---
   #
   orb_con_plot(molecule)
   #
   #  ---  plot deviation from reference calc  ---
   #
   if (molecule['ref']): dev_ref_plot(molecule)
   #
   #  ---  plot mpi timings  ---
   #
#   if (molecule['mpi_parallel']): mpi_time_plot(molecule)
   #
   return molecule

