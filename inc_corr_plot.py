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

def ic_plot(mol_string,dim,core,exp,thres,order,n_tuples,model,basis,e_inc,e_ref,ref,local):
   #
   fig = plt.figure()
   #
   if (ref):
      ax1 = plt.subplot2grid((2,2),(0,0),colspan=2)
      ax2 = plt.subplot2grid((2,2),(1,0))
      ax3 = plt.subplot2grid((2,2),(1,1))
   else:
      ax1 = plt.subplot2grid((1,1),(0,0))
   #
   width = 0.6
   kcal_mol = 0.001594
   #
   if (ref):
      error_abs = (thres/kcal_mol)/2.0
      error_rel_p = ((e_ref[0]+(thres/2.0))/e_ref[0])*100.
      error_rel_m = ((e_ref[0]-(thres/2.0))/e_ref[0])*100.
   #
   dim_list = []
   dim_list_2 = []
   if (ref):
      e_inc_abs = []
      e_inc_rel = []
   #
   for i in range(0,len(e_inc)):
      dim_list.append(i+1.0)
      dim_list_2.append(i+0.7)
      if (ref):
         e_inc_abs.append((e_inc[i]-e_ref[0])/kcal_mol)
         e_inc_rel.append((e_inc[i]/e_ref[0])*100.)
   #
   if (thres > 0.0):
      if (core[0] > 0):
         if (exp[0] == 1):
            ax1.set_title('occ/{0:}/{1:}/FC energy (thres. = {2:6.1e} Hartree)'.format(model,basis,thres))
         elif (exp[0] == 2):
            ax1.set_title('virt/{0:}/{1:}/FC energy (thres. = {2:6.1e} Hartree)'.format(model,basis,thres))
      else:
         if (local):
            if (exp[0] == 1):
               ax1.set_title('occ/{0:}/{1:}/local energy (thres. = {2:6.1e} Hartree)'.format(model,basis,thres))
            elif (exp[0] == 2):
               ax1.set_title('virt/{0:}/{1:}/local energy (thres. = {2:6.1e} Hartree)'.format(model,basis,thres))
         else:
            if (exp[0] == 1):
               ax1.set_title('occ/{0:}/{1:} energy (thres. = {2:6.1e} Hartree)'.format(model,basis,thres))
            elif (exp[0] == 2):
               ax1.set_title('virt/{0:}/{1:} energy (thres. = {2:6.1e} Hartree)'.format(model,basis,thres))
   else:
      if (core[0] > 0):
         if (exp[0] == 1):
            ax1.set_title('occ/{0:}/{1:}/FC energy (order = {2:})'.format(model,basis,order))
         elif (exp[0] == 2):
            ax1.set_title('virt/{0:}/{1:}/FC energy (order = {2:})'.format(model,basis,order))
      else:
         if (local):
            if (exp[0] == 1):
               ax1.set_title('occ/{0:}/{1:}/local energy (order = {2:})'.format(model,basis,order))
            elif (exp[0] == 2):
               ax1.set_title('virt/{0:}/{1:}/local energy (order = {2:})'.format(model,basis,order))
         else:
            if (exp[0] == 1):
               ax1.set_title('occ/{0:}/{1:} energy (order = {2:})'.format(model,basis,order))
            elif (exp[0] == 2):
               ax1.set_title('virt/{0:}/{1:} energy (order = {2:})'.format(model,basis,order))
   #
   ax1.plot(dim_list,e_inc,marker='x',linewidth=2,color='red',linestyle='-')
   if (exp[0] == 1):
      ax1.set_xlim([0.5,(dim[0]-core[0])+0.5])
   elif (exp[0] == 2):
      ax1.set_xlim([0.5,dim[0]+0.5])
   ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
   ax1.set_xlabel('Inc.-corr. order')
   ax1.set_ylabel('Energy (in Hartree)')
   #
   ax12 = ax1.twinx()
   ax12.set_yscale('log')
   ax12.bar(dim_list_2,n_tuples[0:len(e_inc)],width,color='blue',alpha=0.3,log=True)
   if (exp[0] == 1):
      ax12.set_xlim([0.5,(dim[0]-core[0])+0.5])
   elif (exp[0] == 2):
      ax12.set_xlim([0.5,dim[0]+0.5])
   ax12.xaxis.set_major_locator(MaxNLocator(integer=True))
   ax12.set_ylim(bottom=0.7)
   ax12.set_ylabel('Number of correlated tuples')
   #
   if (ref):
      ax2.set_title('Absolute difference from E('+model+')')
      ax2.axhline(0.0,color='black',linewidth=2)
      ax2.plot(dim_list,e_inc_abs,marker='x',linewidth=2,color='red',linestyle='-')
      ax2.axhspan(-error_abs,error_abs,color='green',alpha=0.2)
      if (exp[0] == 1):
         ax2.set_xlim([0.5,(dim[0]-core[0])+0.5])
      elif (exp[0] == 2):
         ax2.set_xlim([0.5,dim[0]+0.5])
      ax2.xaxis.grid(True)
      ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
      ax2.set_ylim([-3.4,3.4])
      ax2.grid()
      ax2.set_ylabel('Difference (in kcal/mol)')
      ax2.set_xlabel('Inc.-corr. order')
      #
      ax3.set_title('Relative recovery of E('+model+')')
      ax3.axhline(100.0,color='black',linewidth=2)
      ax3.plot(dim_list,e_inc_rel,marker='x',linewidth=2,color='red',linestyle='-')
      ax3.axhspan(error_rel_m,error_rel_p,color='green',alpha=0.2)
      if (exp[0] == 1):
         ax3.set_xlim([0.5,(dim[0]-core[0])+0.5])
      elif (exp[0] == 2):
         ax3.set_xlim([0.5,dim[0]+0.5])
      ax3.xaxis.grid(True)
      ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
      ax3.grid()
      ax3.set_ylim([93.,107.])
      ax3.set_ylabel('Recovery (in %)')
      ax3.set_xlabel('Inc.-corr. order')
   #
   fig.tight_layout()
   #
   if (thres > 0.0):
      if (core[0] > 0):
         if (exp[0] == 1):
            plt.savefig(mol_string+'_occ_'+model+'_'+basis+'_FC_thres_{0:6.1e}.pdf'.format(thres), bbox_inches = 'tight', dpi=1000)
         elif (exp[0] == 2):
            plt.savefig(mol_string+'_virt_'+model+'_'+basis+'_FC_thres_{0:6.1e}.pdf'.format(thres), bbox_inches = 'tight', dpi=1000)
      else:
         if (local):
            if (exp[0] == 1):
               plt.savefig(mol_string+'_occ_'+model+'_'+basis+'_thres_{0:6.1e}_LOCAL.pdf'.format(thres), bbox_inches = 'tight', dpi=1000)
            elif (exp[0] == 2):
               plt.savefig(mol_string+'_virt_'+model+'_'+basis+'_thres_{0:6.1e}_LOCAL.pdf'.format(thres), bbox_inches = 'tight', dpi=1000)
         else:
            if (exp[0] == 1):
               plt.savefig(mol_string+'_occ_'+model+'_'+basis+'_thres_{0:6.1e}.pdf'.format(thres), bbox_inches = 'tight', dpi=1000)
            elif (exp[0] == 2):
               plt.savefig(mol_string+'_virt_'+model+'_'+basis+'_thres_{0:6.1e}.pdf'.format(thres), bbox_inches = 'tight', dpi=1000)
   else:
      if (core[0] > 0):
         if (exp[0] == 1):
            plt.savefig(mol_string+'_occ_'+model+'_'+basis+'_FC_order_{0:}.pdf'.format(order), bbox_inches = 'tight', dpi=1000)
         elif (exp[0] == 2):
            plt.savefig(mol_string+'_virt_'+model+'_'+basis+'_FC_order_{0:}.pdf'.format(order), bbox_inches = 'tight', dpi=1000)
      else:
         if (local):
            if (exp[0] == 1):
               plt.savefig(mol_string+'_occ_'+model+'_'+basis+'_order_{0:}_LOCAL.pdf'.format(order), bbox_inches = 'tight', dpi=1000)
            elif (exp[0] == 2):
               plt.savefig(mol_string+'_virt_'+model+'_'+basis+'_order_{0:}_LOCAL.pdf'.format(order), bbox_inches = 'tight', dpi=1000)
         else:
            if (exp[0] == 1):
               plt.savefig(mol_string+'_occ_'+model+'_'+basis+'_order_{0:}.pdf'.format(order), bbox_inches = 'tight', dpi=1000)
            elif (exp[0] == 2):
               plt.savefig(mol_string+'_virt_'+model+'_'+basis+'_order_{0:}.pdf'.format(order), bbox_inches = 'tight', dpi=1000)

def main():
   #
   exp = []
   exp.append(1)
   #
   mol_string = ''
   #
   nocc = []
   nocc.append(1)
   #
   nvirt = []
   nvirt.append(1)
   #
   core = []
   core.append(0)
   #
   thres = 0.0
   #
   order = 0
   #
   n_tuples = []
   n_tuples.append(1)
   #
   model = ''
   #
   basis = ''
   #
   e_inc = []
   e_inc.append(1.0)
   #
   e_ref = []
   e_ref.append(1.0)
   #
   local = []
   local.append(False)
   #
   # here: assume occ expansion
   ic_plot(mol_string,nocc,core,exp,thres,n_tuples,model,basis,e_inc,e_ref,True,local[0])

if __name__ == '__main__':
   #
   main()

