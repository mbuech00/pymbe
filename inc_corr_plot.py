#!/usr/bin/env python

#
# python plotting program for inc.-corr. calculations 
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), August 2016, Mainz, Germnay.
#

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def ic_plot(molecule):
   #
   fig = plt.figure()
   #
   if (molecule['ref']):
      #
      ax1 = plt.subplot2grid((2,2),(0,0),colspan=2)
      ax2 = plt.subplot2grid((2,2),(1,0))
      ax3 = plt.subplot2grid((2,2),(1,1))
   #
   else:
      #
      ax1 = plt.subplot2grid((1,1),(0,0))
   #
   width = 0.6
   kcal_mol = 0.001594
   #
   if (molecule['ref']):
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
   dim_list_2 = []
   #
   if (molecule['ref']):
      #
      e_diff_abs = []
      e_diff_rel = []
   #
   for i in range(0,len(molecule['e_fin'])):
      #
      dim_list.append(i+1.0)
      dim_list_2.append(i+0.7)
      #
      if (molecule['ref']):
         #
         e_diff_abs.append((molecule['e_fin'][i]-molecule['e_ref'])/kcal_mol)
         e_diff_rel.append((molecule['e_fin'][i]/molecule['e_ref'])*100.)
   #
   ax1.set_title('Total '+molecule['model']+' energy')
   #
   ax1.plot(dim_list,molecule['e_fin'],marker='x',linewidth=2,color='red',linestyle='-')
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      ax1.set_xlim([0.5,(molecule['nocc']-molecule['core'])+0.5])
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      ax1.set_xlim([0.5,molecule['nvirt']+0.5])
   #
   ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax1.set_xlabel('Expansion order')
   ax1.set_ylabel('Energy (in Hartree)')
   #
   ax12 = ax1.twinx()
   #
   ax12.set_yscale('log')
   #
   ax12.bar(dim_list_2,molecule['n_contrib'][0:len(molecule['e_fin'])],width,color='blue',alpha=0.3,log=True)
   #
   if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
      #
      ax12.set_xlim([0.5,(molecule['nocc']-molecule['core'])+0.5])
   #
   elif (molecule['exp'] == 'VIRT'):
      #
      ax12.set_xlim([0.5,molecule['nvirt']+0.5])
   #
   ax12.xaxis.set_major_locator(MaxNLocator(integer=True))
   #
   ax12.set_ylim(bottom=0.7)
   #
   ax12.set_ylabel('Number of correlated tuples')
   #
   if (molecule['ref']):
      #
      ax2.set_title('Absolute difference from E('+molecule['model']+')')
      #
      ax2.axhline(0.0,color='black',linewidth=2)
      #
      ax2.plot(dim_list,e_diff_abs,marker='x',linewidth=2,color='red',linestyle='-')
      #
      ax2.axhspan(-error_abs,error_abs,color='green',alpha=0.2)
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
      ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
      #
      ax2.set_ylim([-3.4,3.4])
      #
      ax2.grid()
      #
      ax2.set_ylabel('Difference (in kcal/mol)')
      ax2.set_xlabel('Expansion order')
      #
      ax3.set_title('Relative recovery of E('+molecule['model']+')')
      #
      ax3.axhline(100.0,color='black',linewidth=2)
      #
      ax3.plot(dim_list,e_diff_rel,marker='x',linewidth=2,color='red',linestyle='-')
      #
      ax3.axhspan(error_rel_m,error_rel_p,color='green',alpha=0.2)
      #
      if ((molecule['exp'] == 'OCC') or (molecule['exp'] == 'COMB')):
         #
         ax3.set_xlim([0.5,(molecule['nocc']-molecule['core'])+0.5])
      #
      elif (molecule['exp'] == 'VIRT'):
         #
         ax3.set_xlim([0.5,molecule['nvirt']+0.5])
      #
      ax3.xaxis.grid(True)
      #
      ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
      #
      ax3.grid()
      #
      ax3.set_ylim([93.,107.])
      #
      ax3.set_ylabel('Recovery (in %)')
      ax3.set_xlabel('Expansion order')
   #
   fig.tight_layout()
   #
   plt.savefig('output.pdf', bbox_inches = 'tight', dpi=1000)

