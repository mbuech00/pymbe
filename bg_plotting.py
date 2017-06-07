#!/usr/bin/env python
# -*- coding: utf-8 -*

""" bg_plotting.py: plotting utilities for Bethe-Goldstone correlation calculations."""

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'
__copyright__ = 'Copyright 2017'
__credits__ = ['Prof. Juergen Gauss', 'Dr. Filippo Lipparini']
__license__ = '???'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'jeriksen@uni-mainz.de'
__status__ = 'Development'

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


def abs_energy_plot(molecule):
		""" plot absolute energy """
		# set seaborn
		sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
		# set 1 plot
		fig, ax = plt.subplots()
		# set title
		ax.set_title('Total '+molecule['model']+' energy')
		# set upper limit
		u_limit = molecule['u_limit']
		if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): u_limit -= molecule['ncore']
		# plot results
		ax.plot(list(range(1,len(molecule['prim_energy'])+1)),molecule['prim_energy'],marker='x',linewidth=2,linestyle='-',label='BG('+molecule['model']+')')
		# set x limits
		ax.set_xlim([0.5,u_limit+0.5])
		# turn off x-grid
		ax.xaxis.grid(False)
		# set labels
		ax.set_xlabel('Expansion order')
		ax.set_ylabel('Energy (in Hartree)')
		# force integer ticks on x-axis
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		# despine
		sns.despine()
		# make insert
		with sns.axes_style("whitegrid"):
			# define frame
			insert = plt.axes([.35, .50, .50, .30],frameon=True)
			# plot results
			insert.plot(list(range(2,len(molecule['prim_energy'])+1)),molecule['prim_energy'][1:],marker='x',linewidth=2,linestyle='-')
			# set x limits
			plt.setp(insert,xticks=list(range(3,len(molecule['prim_energy'])+1)))
			insert.set_xlim([2.5,len(molecule['prim_energy'])+0.5])
			# set number of y ticks
			insert.locator_params(axis='y',nbins=6)
			# set y limits
			insert.set_ylim([molecule['prim_energy'][-1]-0.01,molecule['prim_energy'][-1]+0.01])
			# turn off x-grid
			insert.xaxis.grid(False)
		# set legends
		ax.legend(loc=1)
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/abs_energy_plot.pdf', bbox_inches = 'tight', dpi=1000)
		#
		return


def n_tuples_plot(molecule):
		""" plot number of tuples """
		# set seaborn
		sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
		# set 1 plot
		fig, ax = plt.subplots()
		# set title
		ax.set_title('Total number of '+molecule['model']+' tuples')
		# set upper limit
		u_limit = molecule['u_limit']
		if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): u_limit -= molecule['ncore']
		# init prim list
		prim = []
		# set prim list
		for i in range(0,u_limit):
			if (i < len(molecule['prim_tuple'])):
				prim.append(len(molecule['prim_tuple'][i]))
			else:
				prim.append(0)
		# plot results
		sns.barplot(list(range(1,u_limit+1)),molecule['theo_work'],palette='Greens',label='Theoretical number',log=True)
		sns.barplot(list(range(1,u_limit+1)),prim,palette='Blues_r',label='BG('+molecule['model']+') expansion',log=True)
		# turn off x-grid
		ax.xaxis.grid(False)
		# set x- and y-limits
		ax.set_xlim([-0.5,u_limit-0.5])
		ax.set_ylim(bottom=0.7)
		# set x-ticks
		if (u_limit < 8):
			ax.set_xticks(list(range(0,u_limit)))
			ax.set_xticklabels(list(range(1,u_limit+1)))
		else:
			ax.set_xticks(list(range(0,u_limit,u_limit//8)))
			ax.set_xticklabels(list(range(1,u_limit+1,u_limit//8)))
		# set x- and y-labels
		ax.set_xlabel('Expansion order')
		ax.set_ylabel('Number of correlated tuples')
		# set legend
		plt.legend(loc=1)
		leg = ax.get_legend()
		leg.legendHandles[0].set_color(sns.color_palette('Greens')[-1])
		leg.legendHandles[1].set_color(sns.color_palette('Blues_r')[0])
		# despind
		sns.despine()
		# tight layout
		fig.tight_layout()
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/n_tuples_plot.pdf', bbox_inches = 'tight', dpi=1000)
		# del prim list
		del prim
		#
		return


def orb_ent_all_plot(molecule):
		""" plot orbital entanglement (all plots) """
		# set seaborn
		sns.set(style='white',font='DejaVu Sans')
		# set colormap
		cmap = sns.cubehelix_palette(as_cmap=True)
		# set number of subplots
		h_length = len(molecule['prim_orb_ent_rel'])//2
		if (len(molecule['prim_orb_ent_rel']) % 2 != 0): h_length += 1
		ratio = 0.98/float(h_length)
		fig, ax = plt.subplots(h_length+1, 2, sharex='col', sharey='row', gridspec_kw = {'height_ratios': [ratio]*h_length+[0.02]})
		# set figure size
		fig.set_size_inches([8.268,11.693])
		# set location for colorbar
		cbar_ax = fig.add_axes([0.06,0.02,0.88,0.03])
		# init mask array
		mask_arr = np.zeros_like(molecule['prim_orb_ent_rel'][0],dtype=np.bool)
		# set title
		fig.suptitle('Entanglement matrices')
		# plot results
		for i in range(0,len(molecule['prim_orb_ent_rel'])):
			mask_arr = (molecule['prim_orb_ent_rel'][i] == 0.0)
			sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][i]),ax=ax.flat[i],mask=np.transpose(mask_arr),cmap=cmap,\
						xticklabels=False,yticklabels=False,cbar=True,cbar_ax=cbar_ax,cbar_kws={'format':'%.0f', 'orientation': 'horizontal'},\
						annot=False,vmin=0.0,vmax=100.0)
			ax.flat[i].set_title('BG order = '+str(i+2))
		# remove ticks
		ax[-1,0].set_yticklabels([]); ax[-1,1].set_yticklabels([])
		# despine
		sns.despine(left=True,right=True,top=True,bottom=True)
		# tight layout
		fig.tight_layout()
		# adjust subplots (to make room for title)
		plt.subplots_adjust(top=0.95)
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/orb_ent_all_plot.pdf', bbox_inches = 'tight', dpi=1000)
		# del mask array
		del mask_arr
		#
		return


def orb_ent_plot(molecule):
		""" plot orbital entanglement (first and last plot) """
		# set seaborn
		sns.set(style='white',font='DejaVu Sans')
		# set colormap
		cmap = sns.cubehelix_palette(as_cmap=True)
		# make 2 plots + 1 colorbar
		fig, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1.0,1.0,0.06]})
		# set figure size
		fig.set_size_inches([11.0,5.0])
		# fix for colorbar
		ax1.get_shared_y_axes().join(ax2)
		# set mask array
		mask_arr = (molecule['prim_orb_ent_rel'][0] == 0.0)
		# plot results
		sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][0]),ax=ax1,mask=np.transpose(mask_arr),cmap=cmap,\
					xticklabels=False,yticklabels=False,cbar=False,\
					annot=False,vmin=0.0,vmax=100.0)
		# set title
		ax1.set_title('Entanglement matrix, order = 2')
		# set mask array
		mask_arr = (molecule['prim_orb_ent_rel'][-1] == 0.0)
		# plot results
		sns.heatmap(np.transpose(molecule['prim_orb_ent_rel'][-1]),ax=ax2,mask=np.transpose(mask_arr),cmap=cmap,\
					xticklabels=False,yticklabels=False,cbar=True,cbar_ax=cbar_ax,cbar_kws={'format':'%.0f'},\
					annot=False,vmin=0.0,vmax=100.0)
		# set title
		ax2.set_title('Entanglement matrix, order = '+str(len(molecule['prim_energy'])))
		# despine
		sns.despine(left=True,right=True,top=True,bottom=True)
		# tight layout
		fig.tight_layout()
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/orb_ent_plot.pdf', bbox_inches = 'tight', dpi=1000)
		# del mask array
		del mask_arr
		#
		return


def orb_con_order_plot(molecule):
		""" plot orbital contributions (individually, order by order) """
		# set seaborn
		sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
		# set 1 plot
		fig, ax = plt.subplots()
		# transpose orb_con_abs array
		orb_con_arr = np.transpose(np.asarray(molecule['prim_orb_con_abs']))
		# plot results
		for i in range(0,len(orb_con_arr)):
			# determine x-range
			end = len(molecule['prim_energy'])
			for j in range(1,len(orb_con_arr[i])):
				if ((orb_con_arr[i,j]-orb_con_arr[i,j-1]) == 0.0):
					end = j-1
					break
			ax.plot(list(range(1,end+1)),orb_con_arr[i,:end],linewidth=2)
		# set x-limits
		ax.set_xlim([0.5,len(molecule['prim_energy'])+0.5])
		# turn off x-grid
		ax.xaxis.grid(False)
		# set x- and y-labels
		ax.set_xlabel('Expansion order')
		ax.set_ylabel('Accumulated orbital contribution (in Hartree)')
		# for integer ticks on x-axis
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		# despine
		sns.despine()
		# tight layout
		fig.tight_layout()
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/orb_con_order_plot.pdf', bbox_inches = 'tight', dpi=1000)
		# del orb_con array
		del orb_con_arr
		#
		return


def orb_dist_plot(molecule):
		""" plot orbital distribution """
		# set seaborn
		sns.set(style='white', palette='Set2')
		# define color palette
		palette = cycle(sns.color_palette())
		# set number of subplots
		w_length = len(molecule['prim_energy_inc'])//2
		if ((len(molecule['prim_energy_inc']) % 2 != 0) and (len(molecule['prim_energy_inc'][-1]) != 1)): w_length += 1
		fig, axes = plt.subplots(2, w_length, figsize=(12, 8), sharex=False, sharey=False)
		# set title
		fig.suptitle('Distribution of energy contributions')
		# save threshold
		thres = molecule['prim_exp_thres_init']
		# set lists and plot results
		for i in range(0,len(molecule['prim_energy_inc'])):
			if (len(molecule['prim_energy_inc'][i]) != 1):
				# sort energy increments
				e_inc_sort = np.sort(molecule['prim_energy_inc'][i])
				# init counting list
				e_inc_count = np.zeros(len(e_inc_sort),dtype=np.float64)
				# count
				for j in range(0,len(e_inc_count)): e_inc_count[j] = ((j+1)/len(e_inc_count))*100.0
				# init contribution list
				e_inc_contrib = np.zeros(len(e_inc_sort),dtype=np.float64)
				# calc contributions
				for j in range(0,len(e_inc_contrib)): e_inc_contrib[j] = np.sum(e_inc_sort[:j+1])
				# plot contributions
				l1 = axes.flat[i].step(e_inc_sort,e_inc_count,where='post',linewidth=2,linestyle='-',color=sns.xkcd_rgb['salmon'],label='Contributions')
				# plot x = 0.0
				l2 = axes.flat[i].axvline(x=0.0,ymin=0.0,ymax=100.0,linewidth=2,linestyle='--',color=sns.xkcd_rgb['royal blue'])
				# plot threshold span
				axes.flat[i].axvspan(0.0-thres,0.0+thres,color=sns.xkcd_rgb['amber'],alpha=0.5)
				# update thres
				thres = molecule['prim_exp_scaling']**(i+1) * molecule['prim_exp_thres_init']
				# change to second y-axis
				ax2 = axes.flat[i].twinx()
				# plot counts
				l3 = ax2.step(e_inc_sort,e_inc_contrib,where='post',linewidth=2,linestyle='-',color=sns.xkcd_rgb['kelly green'],label='Energy')
				# set title
				axes.flat[i].set_title('E-{0:} = {1:4.2e}'.format(i+1,np.sum(e_inc_sort)))
				# set nice axis formatting
				delta = (np.abs(np.max(e_inc_sort)-np.min(e_inc_sort)))*0.05
				axes.flat[i].set_xlim([np.min(e_inc_sort)-delta,np.max(e_inc_sort)+delta])
				axes.flat[i].set_xticks([np.min(e_inc_sort),np.max(e_inc_sort)])
				axes.flat[i].xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
				ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
				axes.flat[i].set_yticks([0.0,25.0,50.0,75.0,100.0])
				# make ticks coloured and remove y2-axis ticks for most of the subplots
				axes.flat[i].tick_params('y',colors=sns.xkcd_rgb['salmon'])
				ax2.tick_params('y',colors=sns.xkcd_rgb['kelly green'])
				if (not ((i == 0) or (i == w_length))): axes.flat[i].set_yticks([])
				# set legend
				if (i == 0):
					lns = l1+l3
					labs = [l.get_label() for l in lns]
					plt.legend(lns,labs,loc=2,fancybox=True,frameon=True)
		# tight layout
		plt.tight_layout()
		# remove ticks for most of the subplots and despine
		if ((len(molecule['prim_energy_inc']) % 2 != 0) and (len(molecule['prim_energy_inc'][-1]) != 1)):
			axes.flat[-1].set_xticks([])
			axes.flat[-1].set_yticks([])
			axes.flat[-1].set_xticklabels([])
			axes.flat[-1].set_yticklabels([])
			sns.despine(left=True,bottom=True,ax=axes.flat[-1])
		# adjust subplots (to make room for title)
		plt.subplots_adjust(top=0.925)
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/orb_dist_plot.pdf', bbox_inches = 'tight', dpi=1000)
		# del lists
		del e_inc_sort; del e_inc_count; del e_inc_contrib
		#
		return


def orb_con_tot_plot(molecule):
		""" plot total orbital contributions """
		# set seaborn
		sns.set(style='whitegrid',font='DejaVu Sans')
		# set 1 plot
		fig, ax = plt.subplots()
		# set orb_con and mask arrays
		orb_con_arr = 100.0*np.array(molecule['prim_orb_con_rel'])
		mask_arr = np.zeros_like(orb_con_arr,dtype=np.bool)
		mask_arr = (orb_con_arr == 0.0)
		# plot results
		sns.heatmap(orb_con_arr,ax=ax,mask=mask_arr,cmap='coolwarm',cbar_kws={'format':'%.0f'},\
					xticklabels=False,yticklabels=range(1,len(molecule['prim_orb_con_rel'])+1),cbar=True,\
					annot=False,vmin=0.0,vmax=np.amax(orb_con_arr))
		# set title
		ax.set_title('Total orbital contributions (in %)')
		# set y-label and y-ticks
		ax.set_ylabel('Expansion order')
		ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
		# despine
		sns.despine(left=True,bottom=True)
		# tight layout
		fig.tight_layout()
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/orb_con_tot_plot.pdf', bbox_inches = 'tight', dpi=1000)
		# del lists
		del orb_con_arr; del mask_arr
		#
		return


def dev_ref_plot(molecule):
		""" plot deviation from reference """
		# set seaborn
		sns.set(style='darkgrid',palette='Set2',font='DejaVu Sans')
		# set 2 subplots
		fig, ( ax1, ax2 ) = plt.subplots(2, 1, sharex='col', sharey='row')
		# set conversion from Ha to kcal/mol
		kcal_mol = 0.001594
		# init and set deviation lists
		e_diff_tot_abs = []; e_diff_tot_rel = []
		for j in range(0,len(molecule['prim_energy'])):
			e_diff_tot_abs.append((molecule['prim_energy'][j]-molecule['e_ref'])/kcal_mol)
			e_diff_tot_rel.append((molecule['prim_energy'][j]/molecule['e_ref'])*100.)
		# set title
		ax1.set_title('Absolute difference from E('+molecule['model']+')')
		# set upper limit
		u_limit = molecule['u_limit']
		if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): u_limit -= molecule['ncore']
		# plot results
		ax1.axhline(0.0,color='black',linewidth=2)
		ax1.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_abs,marker='x',linewidth=2,linestyle='-',label='BG('+molecule['model']+')')
		# set x-limits
		ax1.set_ylim([-1.1,3.4])
		# turn off x-grid
		ax1.xaxis.grid(False)
		# set y-label
		ax1.set_ylabel('Difference (in kcal/mol)')
		# force integer ticks on x-axis
		ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
		# set legend
		ax1.legend(loc=4)
		# make insert
		with sns.axes_style("whitegrid"):
			# define frame
			insert = inset_axes(ax1,width='40%',height=1.1,loc=1)
			# plot results
			insert.axhline(0.0,color='black',linewidth=2)
			insert.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_abs,marker='x',linewidth=2,linestyle='-')
			# set x-limits
			plt.setp(insert,xticks=list(range(3,len(molecule['prim_energy'])+1)))
			insert.set_xlim([3.5,len(molecule['prim_energy'])+0.5])
			# set number of y-ticks
			insert.locator_params(axis='y',nbins=4)
			# force float ticks on y-axis
			insert.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
			# set y-limits
			insert.set_ylim([-0.6,0.6])
			# turn off x-grid
			insert.xaxis.grid(False)
		# set title
		ax2.set_title('Relative recovery of E('+molecule['model']+')')
		# plot results
		ax2.axhline(100.0,color='black',linewidth=2)
		ax2.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_rel,marker='x',linewidth=2,linestyle='-',label='BG('+molecule['model']+')')
		# set x-limits
		ax2.set_xlim([0.5,u_limit+0.5])
		# turn off x-grid
		ax2.xaxis.grid(False)
		# set y-limits
		ax2.set_ylim([97.5,107.])
		# set x- and y-labels
		ax2.set_ylabel('Recovery (in %)')
		ax2.set_xlabel('Expansion order')
		# force integer ticks on x-axis
		ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		# set legend
		ax2.legend(loc=4)
		# make insert
		with sns.axes_style("whitegrid"):
		   # define frame
		   insert = inset_axes(ax2,width='40%',height=1.1,loc=1)
		   # plot results
		   insert.axhline(100.0,color='black',linewidth=2)
		   insert.plot(list(range(1,len(molecule['prim_energy'])+1)),e_diff_tot_rel,marker='x',linewidth=2,linestyle='-')
		   # set x-limits
		   plt.setp(insert,xticks=list(range(3,len(molecule['prim_energy'])+1)))
		   insert.set_xlim([3.5,len(molecule['prim_energy'])+0.5])
		   # set number of y-ticks
		   insert.locator_params(axis='y',nbins=4)
		   # force float ticks on y-axis
		   insert.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		   # set y-limits
		   insert.set_ylim([99.7,100.3])
		   # turn off x-grid
		   insert.xaxis.grid(False)
		# despine
		sns.despine()
		# tight layout
		fig.tight_layout()
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/dev_ref_plot.pdf', bbox_inches = 'tight', dpi=1000)
		#
		return


def time_plot(molecule):
		""" plot total and mpi timings """
		# set seaborn
		sns.set(style='whitegrid',palette='Set2',font='DejaVu Sans')
		# set number of subplots - 2 with mpi, 1 without
		if (molecule['mpi_parallel']):
			fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
		else:
			fig, ax1 = plt.subplots()
		# set color palette
		sns.set_color_codes('pastel')
		# set upper limit
		u_limit = molecule['u_limit']
		if (((molecule['exp'] == 'occ') or (molecule['exp'] == 'comb-ov')) and molecule['frozen']): u_limit -= molecule['ncore']
		# set x-range
		order = list(range(1,len(molecule['prim_energy'])+2))
		# define y-ticks
		y_labels = list(range(1,len(molecule['prim_energy'])+1))
		y_labels.append('total')
		# set title
		ax1.set_title('Phase timings')
		# set result arrays and plot results
		kernel_dat = (molecule['time_kernel']/molecule['time_tot'])*100.0
		sum_dat = kernel_dat + (molecule['time_summation']/molecule['time_tot'])*100.0
		screen_dat = sum_dat + (molecule['time_screen']/molecule['time_tot'])*100.0
		screen = sns.barplot(screen_dat,order,ax=ax1,orient='h',label='screen',color=sns.xkcd_rgb['salmon'])
		summation = sns.barplot(sum_dat,order,ax=ax1,orient='h',label='summation',color=sns.xkcd_rgb['windows blue'])
		kernel = sns.barplot(kernel_dat,order,ax=ax1,orient='h',label='kernel',color=sns.xkcd_rgb['amber'])
		# set x- and y-limits
		ax1.set_ylim([-0.5,(len(molecule['prim_energy'])+1)-0.5])
		ax1.set_xlim([0.0,100.0])
		# set y-ticks
		ax1.set_yticklabels(y_labels)
		# set legend
		handles,labels = ax1.get_legend_handles_labels()
		handles = [handles[2],handles[1],handles[0]]
		labels = [labels[2],labels[1],labels[0]]
		ax1.legend(handles,labels,ncol=3,loc=9,fancybox=True,frameon=True)
		# invert plot
		ax1.invert_yaxis()
		# if not mpi, set labels. if mpi, plot mpi timings
		if (not molecule['mpi_parallel']):
			ax1.set_xlabel('Distribution (in %)')
			ax1.set_ylabel('Expansion order')
		else:
			# set title
			ax2.set_title('MPI timings')
			# set result arrays and plot results
			work_dat = molecule['dist_order'][0]
			comm_dat = work_dat + molecule['dist_order'][1]
			idle_dat = comm_dat + molecule['dist_order'][2]
			idle = sns.barplot(idle_dat,order,ax=ax2,orient='h',label='idle',color=sns.xkcd_rgb['sage'])
			comm = sns.barplot(comm_dat,order,ax=ax2,orient='h',label='comm',color=sns.xkcd_rgb['baby blue'])
			work = sns.barplot(work_dat,order,ax=ax2,orient='h',label='work',color=sns.xkcd_rgb['wine'])
			# set x- and y-limits
			ax2.set_ylim([-0.5,(len(molecule['prim_energy'])+1)-0.5])
			ax2.set_xlim([0.0,100.0])
			# set y-ticks
			ax2.set_yticklabels(y_labels)
			# set legend
			handles,labels = ax2.get_legend_handles_labels()
			handles = [handles[2],handles[1],handles[0]]
			labels = [labels[2],labels[1],labels[0]]
			ax2.legend(handles,labels,ncol=3,loc=9,fancybox=True,frameon=True)
			# set x- and y-labels
			fig.text(0.52,0.0,'Distribution (in %)',ha='center',va='center')
			fig.text(0.0,0.5,'Expansion order',ha='center',va='center',rotation='vertical')
			# invert plot
			ax2.invert_yaxis()
		# despine
		sns.despine(left=True,bottom=True)
		# tight layout
		fig.tight_layout()
		# save plot
		plt.savefig(molecule['wrk_dir']+'/output/time_plot.pdf', bbox_inches = 'tight', dpi=1000)
		# del lists
		del screen_dat; del kernel_dat; del sum_dat
		if (molecule['mpi_parallel']): del work_dat; del comm_dat; del idle_dat
		#
		return


def plot(molecule):
		""" driver function for result plotting """
		#  plot total energies
		abs_energy_plot(molecule)
		#  plot number of calculations from each orbital
		n_tuples_plot(molecule)
		#  plot orbital entanglement matrices
		orb_ent_all_plot(molecule)
		orb_ent_plot(molecule)
		#  plot individual orbital contributions by order
		orb_con_order_plot(molecule)
		#  plot orbital/energy distributions
		orb_dist_plot(molecule)
		#  plot total orbital contributions
		orb_con_tot_plot(molecule)
		#  plot deviation from reference calc
		if (molecule['ref']): dev_ref_plot(molecule)
		#  plot timings
		time_plot(molecule)
		#
		return


