#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_plotting.py: plotting utilities for Bethe-Goldstone correlation calculations."""

from copy import deepcopy
import numpy as np
from itertools import cycle
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.7'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

def abs_energy_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('Total '+molecule['model']+' energy')
   #
   u_limit = molecule['u_limit']
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']):
      #
      u_limit -= molecule['ncore']
   #
   ax.plot(list(range(1,len(molecule['prim_energy'])+1)),molecule['prim_energy'],marker='x',linewidth=2,linestyle='-',\
           label='BG('+molecule['model']+')')
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
   ax.legend(loc=1)
   #
   plt.savefig(molecule['wrk_dir']+'/output/abs_energy_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def n_tuples_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
   #
   fig, ax = plt.subplots()
   #
   ax.set_title('Total number of '+molecule['model']+' tuples')
   #
   u_limit = molecule['u_limit']
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']):
      #
      u_limit -= molecule['ncore']
   #
   prim = []
   #
   for i in range(0,u_limit):
      #
      if (i < len(molecule['prim_tuple'])):
         #
         prim.append(len(molecule['prim_tuple'][i]))
      #
      else:
         #
         prim.append(0)
   #
   sns.barplot(list(range(1,u_limit+1)),molecule['theo_work'],\
               palette='Greens',label='Theoretical number',log=True)
   #
   sns.barplot(list(range(1,u_limit+1)),prim,palette='Blues_r',\
               label='BG('+molecule['model']+') expansion',log=True)
   #
   ax.xaxis.grid(False)
   #
   ax.set_xlim([-0.5,u_limit-0.5])
   ax.set_ylim(bottom=0.7)
   #
   if (u_limit < 8):
      #
      ax.set_xticks(list(range(0,u_limit)))
      ax.set_xticklabels(list(range(1,u_limit+1)))
   #
   else:
      #
      ax.set_xticks(list(range(0,u_limit,u_limit//8)))
      ax.set_xticklabels(list(range(1,u_limit+1,u_limit//8)))
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Number of correlated tuples')
   #
   plt.legend(loc=1)
   leg = ax.get_legend()
   leg.legendHandles[0].set_color(sns.color_palette('Greens')[-1])
   leg.legendHandles[1].set_color(sns.color_palette('Blues_r')[0])
   #
   sns.despine()
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk_dir']+'/output/n_tuples_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   del prim
   #
   return molecule

def orb_ent_all_plot(molecule):
   #
   sns.set(style='white',font='DejaVu Sans')
   #
   cmap = sns.cubehelix_palette(as_cmap=True)
   #
   h_length = len(molecule['prim_orb_ent_rel'])//2
   #
   if (len(molecule['prim_orb_ent_rel']) % 2 != 0): h_length += 1
   #
   ratio = 0.98/float(h_length)
   #
   fig, ax = plt.subplots(h_length+1, 2, sharex='col', sharey='row', gridspec_kw = {'height_ratios': [ratio]*h_length+[0.02]})
   #
   fig.set_size_inches([8.268,11.693])
   #
   cbar_ax = fig.add_axes([0.06,0.02,0.88,0.03])
   #
   mask_arr = np.zeros_like(molecule['prim_orb_ent_rel'][0],dtype=np.bool)
   #
   fig.suptitle('Entanglement matrices')
   #
   for i in range(0,len(molecule['prim_orb_ent_rel'])):
      #
      mask_arr = (molecule['prim_orb_ent_rel'][i] == 0.0)
      #
      sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][i]),ax=ax.flat[i],mask=np.transpose(mask_arr),cmap=cmap,\
                       xticklabels=False,yticklabels=False,cbar=True,cbar_ax=cbar_ax,cbar_kws={'format':'%.0f', 'orientation': 'horizontal'},\
                       annot=False,vmin=0.0,vmax=100.0)
      #
      ax.flat[i].set_title('BG order = '+str(i+2))
   #
   ax[-1,0].set_yticklabels([]); ax[-1,1].set_yticklabels([])
   #
   sns.despine(left=True,right=True,top=True,bottom=True)
   #
   fig.tight_layout()
   #
   plt.subplots_adjust(top=0.95)
   #
   plt.savefig(molecule['wrk_dir']+'/output/orb_ent_all_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   del mask_arr
   #
   return

def orb_ent_plot(molecule):
   #
   sns.set(style='white',font='DejaVu Sans')
   #
   cmap = sns.cubehelix_palette(as_cmap=True)
   #
   fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1.0,1.0,0.06]})
   #
   ax1.get_shared_y_axes().join(ax2)
   #
   mask_arr = (molecule['prim_orb_ent_rel'][0] == 0.0)
   #
   sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][0]),ax=ax1,mask=np.transpose(mask_arr),cmap=cmap,\
                    xticklabels=False,yticklabels=False,cbar=False,\
                       annot=False,vmin=0.0,vmax=100.0)
   #
   ax1.set_title('Entanglement matrix, order = 2')
   #
   mask_arr = (molecule['prim_orb_ent_rel'][-1] == 0.0)
   #
   sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][-1]),ax=ax2,mask=np.transpose(mask_arr),cmap=cmap,\
                    xticklabels=False,yticklabels=False,cbar=True,cbar_ax=cbar_ax,cbar_kws={'format':'%.0f'},\
                       annot=False,vmin=0.0,vmax=100.0)
   #
   ax2.set_title('Entanglement matrix, order = '+str(len(molecule['prim_energy'])))
   #
   sns.despine(left=True,right=True,top=True,bottom=True)
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk_dir']+'/output/orb_ent_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   del mask_arr
   #
   return

def orb_con_order_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
   #
   fig, ax = plt.subplots()
   #
   orb_con_arr = np.transpose(np.asarray(molecule['prim_orb_con_abs']))
   #
   for i in range(0,len(orb_con_arr)):
      #
      end = len(molecule['prim_energy'])
      #
      for j in range(1,len(orb_con_arr[i])):
         #
         if ((orb_con_arr[i,j]-orb_con_arr[i,j-1]) == 0.0):
            #
            end = j-1
            #
            break
      #
      ax.plot(list(range(1,end+1)),orb_con_arr[i,:end],linewidth=2)
   #
   ax.set_xlim([0.5,len(molecule['prim_energy'])+0.5])
   #
   ax.xaxis.grid(False)
   #
   ax.set_xlabel('Expansion order')
   ax.set_ylabel('Accumulated orbital contribution (in Hartree)')
   #
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   sns.despine()
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk_dir']+'/output/orb_con_order_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   del orb_con_arr
   #
   return

def orb_dist_plot(molecule):
   #
   sns.set(style='white', palette='Set2')
   #
   palette = cycle(sns.color_palette())
   #
   w_length = len(molecule['prim_energy_inc'])//2
   if (len(molecule['prim_energy_inc']) % 2 != 0): w_length += 1
   #
   fig, axes = plt.subplots(2, w_length, figsize=(12, 8), sharex=False, sharey=False)
   #
   fig.suptitle('Distribution of energy contributions')
   #
   thres = molecule['prim_exp_thres_init']
   #
   for i in range(0,len(molecule['prim_energy_inc'])):
      #
      if (len(molecule['prim_energy_inc'][i]) != 1):
         #
         e_inc_sort = np.sort(molecule['prim_energy_inc'][i])
         #
         e_inc_count = np.zeros(len(e_inc_sort),dtype=np.float64)
         #
         for j in range(0,len(e_inc_count)):
            #
            e_inc_count[j] = ((j+1)/len(e_inc_count))*100.0
         #
         e_inc_contrib = np.zeros(len(e_inc_sort),dtype=np.float64)
         #
         for j in range(0,len(e_inc_contrib)):
            #
            e_inc_contrib[j] = np.sum(e_inc_sort[:j+1])
         #
         l1 = axes.flat[i].step(e_inc_sort,e_inc_count,where='post',linewidth=2,linestyle='-',color=sns.xkcd_rgb['salmon'],label='Contributions')
         #
         l2 = axes.flat[i].axvline(x=0.0,ymin=0.0,ymax=100.0,linewidth=2,linestyle='--',color=sns.xkcd_rgb['royal blue'])
         #
         axes.flat[i].axvspan(0.0-thres,0.0+thres,color=sns.xkcd_rgb['amber'],alpha=0.5)
         #
         thres *= (1.0-molecule['prim_exp_scaling']) 
         #
         ax2 = axes.flat[i].twinx()
         #
         l3 = ax2.step(e_inc_sort,e_inc_contrib,where='post',linewidth=2,linestyle='-',color=sns.xkcd_rgb['kelly green'],label='Energy')
         #
         axes.flat[i].set_title('E-{0:} = {1:4.2e}'.format(i+1,np.sum(e_inc_sort)))
         #
         delta = (np.abs(np.max(e_inc_sort)-np.min(e_inc_sort)))*0.05
         axes.flat[i].set_xlim([np.min(e_inc_sort)-delta,np.max(e_inc_sort)+delta])
         axes.flat[i].set_xticks([np.min(e_inc_sort),np.max(e_inc_sort)])
         axes.flat[i].xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
         ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
         axes.flat[i].set_yticks([0.0,25.0,50.0,75.0,100.0])
         #
         axes.flat[i].tick_params('y',colors=sns.xkcd_rgb['salmon'])
         ax2.tick_params('y',colors=sns.xkcd_rgb['kelly green'])
         #
         if (not ((i == 0) or (i == w_length))): axes.flat[i].set_yticks([])
         #
         if (i == 0):
            #
            lns = l1+l3
            labs = [l.get_label() for l in lns]
            plt.legend(lns,labs,loc=2,fancybox=True,frameon=True)
   #
   plt.tight_layout()
   #
   if ((len(molecule['prim_energy_inc']) % 2 != 0) or (len(molecule['prim_energy_inc'][-1]) == 1)):
      #
      axes.flat[-1].set_xticks([])
      axes.flat[-1].set_yticks([])
      axes.flat[-1].set_xticklabels([])
      axes.flat[-1].set_yticklabels([])
      #
      sns.despine(left=True,bottom=True,ax=axes.flat[-1])
   #
   plt.subplots_adjust(top=0.925)
   #
   plt.savefig(molecule['wrk_dir']+'/output/orb_dist_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   del e_inc_sort
   del e_inc_count
   del e_inc_contrib
   #
   return

def orb_con_tot_plot(molecule):
   #
   sns.set(style='whitegrid',font='DejaVu Sans')
   #
   fig, ax = plt.subplots()
   #
   orb_con_arr = 100.0*np.array(molecule['prim_orb_con_rel'])
   #
   mask_arr = np.zeros_like(orb_con_arr,dtype=np.bool)
   #
   mask_arr = (orb_con_arr == 0.0)
   #
   sns.heatmap(orb_con_arr,ax=ax,mask=mask_arr,cmap='coolwarm',cbar_kws={'format':'%.0f'},\
                    xticklabels=False,yticklabels=range(1,len(molecule['prim_orb_con_rel'])+1),cbar=True,\
                    annot=False,vmin=0.0,vmax=np.amax(orb_con_arr))
   #
   ax.set_title('Total orbital contributions (in %)')
   #
   ax.set_ylabel('Expansion order')
   #
   ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
   #
   sns.despine(left=True,bottom=True)
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk_dir']+'/output/orb_con_tot_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   del orb_con_arr
   del mask_arr
   #
   return

def dev_ref_plot(molecule):
   #
   sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
   #
   fig, ( ax1, ax2 ) = plt.subplots(2, 1, sharex='col', sharey='row')
   #
   kcal_mol = 0.001594
   #
   e_diff_tot_abs = []
   e_diff_tot_rel = []
   #
   for j in range(0,len(molecule['prim_energy'])):
      #
      e_diff_tot_abs.append((molecule['prim_energy'][j]-molecule['e_ref'])/kcal_mol)
      e_diff_tot_rel.append((molecule['prim_energy'][j]/molecule['e_ref'])*100.)
   #
   ax1.set_title('Absolute difference from E('+molecule['model']+')')
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
            label='BG('+molecule['model']+')')
   #
   ax1.set_ylim([-1.1,3.4])
   #
   ax1.xaxis.grid(False)
   #
   ax1.set_ylabel('Difference (in kcal/mol)')
   #
   ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax1.legend(loc=4)
   #
   with sns.axes_style("whitegrid"):
      #
      insert = inset_axes(ax1,width='40%',height=1.1,loc=1)
      #
      insert.axhline(0.0,color='black',linewidth=2)
      #
      insert.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_abs,marker='x',linewidth=2,linestyle='-')
      #
      plt.setp(insert,xticks=list(range(3,len(molecule['prim_energy'])+1)))
      #
      insert.set_xlim([3.5,len(molecule['prim_energy'])+0.5])
      #
      insert.locator_params(axis='y',nbins=4)
      #
      insert.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
      #
      insert.set_ylim([-0.6,0.6])
      #
      insert.xaxis.grid(False)
   #
   ax2.set_title('Relative recovery of E('+molecule['model']+')')
   #
   ax2.axhline(100.0,color='black',linewidth=2)
   #
   ax2.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_rel,marker='x',linewidth=2,linestyle='-',\
            label='BG('+molecule['model']+')')
   #
   ax2.set_xlim([0.5,u_limit+0.5])
   #
   ax2.xaxis.grid(False)
   #
   ax2.set_ylim([97.5,107.])
   #
   ax2.set_ylabel('Recovery (in %)')
   ax2.set_xlabel('Expansion order')
   #
   ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax2.legend(loc=4)
   #
   with sns.axes_style("whitegrid"):
      #
      insert = inset_axes(ax2,width='40%',height=1.1,loc=1)
      #
      insert.axhline(100.0,color='black',linewidth=2)
      #
      insert.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_rel,marker='x',linewidth=2,linestyle='-')
      #
      plt.setp(insert,xticks=list(range(3,len(molecule['prim_energy'])+1)))
      #
      insert.set_xlim([3.5,len(molecule['prim_energy'])+0.5])
      #
      insert.locator_params(axis='y',nbins=4)
      #
      insert.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
      #
      insert.set_ylim([99.7,100.3])
      #
      insert.xaxis.grid(False)
   #
   sns.despine()
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk_dir']+'/output/dev_ref_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   return molecule

def time_plot(molecule):
   #
   sns.set(style='whitegrid',palette='Set2',font='DejaVu Sans')
   #
   if (molecule['mpi_parallel']):
      #
      fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
   #
   else:
      #
      fig, ax1 = plt.subplots()
   #
   sns.set_color_codes('pastel')
   #
   u_limit = molecule['u_limit']
   #
   if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']):
      #
      u_limit -= molecule['ncore']
   #
   order = list(range(1,len(molecule['prim_energy'])+2))
   #
   y_labels = list(range(1,len(molecule['prim_energy'])+1))
   y_labels.append('total')
   #
   ax1.set_title('Phase timings')
   #
   kernel_dat = (molecule['time_kernel']/molecule['time_tot'])*100.0
   sum_dat = kernel_dat + (molecule['time_summation']/molecule['time_tot'])*100.0
   screen_dat = sum_dat + (molecule['time_screen']/molecule['time_tot'])*100.0
   #
   screen = sns.barplot(screen_dat,order,ax=ax1,orient='h',label='screen',color=sns.xkcd_rgb['amber'])
   #
   summation = sns.barplot(sum_dat,order,ax=ax1,orient='h',label='summation',color=sns.xkcd_rgb['salmon'])
   #
   kernel = sns.barplot(kernel_dat,order,ax=ax1,orient='h',label='kernel',color=sns.xkcd_rgb['windows blue'])
   #
   ax1.set_ylim([-0.5,(len(molecule['prim_energy'])+1)-0.5])
   ax1.set_xlim([0.0,100.0])
   #
   ax1.set_yticklabels(y_labels)
   #
   handles,labels = ax1.get_legend_handles_labels()
   #
   handles = [handles[2],handles[1],handles[0]]
   labels = [labels[2],labels[1],labels[0]]
   #
   ax1.legend(handles,labels,ncol=3,loc=9,fancybox=True,frameon=True)
   #
   ax1.invert_yaxis()
   #
   if (not molecule['mpi_parallel']):
      #
      ax1.set_xlabel('Distribution (in %)')
      ax1.set_ylabel('Expansion order')
   #
   else:
      #
      ax2.set_title('MPI timings')
      #
      work_dat = molecule['dist_order'][0]
      comm_dat = work_dat + molecule['dist_order'][1]
      idle_dat = comm_dat + molecule['dist_order'][2]
      #
      idle = sns.barplot(idle_dat,order,ax=ax2,orient='h',label='idle',color=sns.xkcd_rgb['sage'])
      #
      comm = sns.barplot(comm_dat,order,ax=ax2,orient='h',label='comm',color=sns.xkcd_rgb['baby blue'])
      #
      work = sns.barplot(work_dat,order,ax=ax2,orient='h',label='work',color=sns.xkcd_rgb['wine'])
      #
      ax2.set_ylim([-0.5,(len(molecule['prim_energy'])+1)-0.5])
      ax2.set_xlim([0.0,100.0])
      #
      ax2.set_yticklabels(y_labels)
      #
      handles,labels = ax2.get_legend_handles_labels()
      #
      handles = [handles[2],handles[1],handles[0]]
      labels = [labels[2],labels[1],labels[0]]
      #
      ax2.legend(handles,labels,ncol=3,loc=9,fancybox=True,frameon=True)
      #
      fig.text(0.52,0.0,'Distribution (in %)',ha='center',va='center')
      fig.text(0.0,0.5,'Expansion order',ha='center',va='center',rotation='vertical')
      #
      ax2.invert_yaxis()
   #
   sns.despine(left=True,bottom=True)
   #
   fig.tight_layout()
   #
   plt.savefig(molecule['wrk_dir']+'/output/time_plot.pdf', bbox_inches = 'tight', dpi=1000)
   #
   del screen_dat
   del kernel_dat
   del sum_dat
   #
   if (molecule['mpi_parallel']):
      #
      del work_dat
      del comm_dat
      del idle_dat
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
   #  ---  plot orbital entanglement matrices  ---
   #
   orb_ent_all_plot(molecule)
   orb_ent_plot(molecule)
   #
   #  ---  plot individual orbital contributions by order  ---
   #
   orb_con_order_plot(molecule)
   #
   #  ---  plot orbital/energy distributions
   #
   orb_dist_plot(molecule)
   #
   #  ---  plot total orbital contributions  ---
   #
   orb_con_tot_plot(molecule)
   #
   #  ---  plot deviation from reference calc  ---
   #
   if (molecule['ref']): dev_ref_plot(molecule)
   #
   #  ---  plot timings  ---
   #
   time_plot(molecule)
   #
   return molecule


