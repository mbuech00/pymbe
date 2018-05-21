#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

cwd = os.getcwd()

if len(sys.argv) != 2:
	sys.exit('\n error: size of expansion space must be specified: python plot.py exp_size\n')

exp_size = int(sys.argv[1])

with open(cwd+'/results.out') as f:
	while 'MBE' not in f.readline():
		continue
	content = f.readlines()

order = []
energy = []

for i in range(len(content)):
	if len(content[i].split()) > 1:
		try:
			int(content[i].split()[0])
		except ValueError:
			continue
		else:
			order.append(int(content[i].split()[0]))
			energy.append(float(content[i].split()[2]))

order = np.asarray(order)
energy = np.asarray(energy)

if exp_size < len(order):
	sys.exit('\n error: exp_size < len(order)\n')

with open(cwd+'/output.out') as f:
	while 'expansion' not in f.readline():
		continue
	content = f.readlines()

inc_mean = np.empty_like(energy)
inc_min = np.empty_like(energy)
inc_max = np.empty_like(energy)
idx = 0

for i in range(len(content)):
	if len(content[i].split()) > 1:
		try:
			float(content[i].split()[1])
		except ValueError:
			continue
		else:
			inc_mean[idx] = float(content[i].split()[1])
			inc_min[idx] = float(content[i].split()[3])
			inc_max[idx] = float(content[i].split()[5])
			idx += 1

# plot correlation energy
sns.set(style='darkgrid', palette='Set2', font='DejaVu Sans')
fig, ax = plt.subplots()
# plot results
ax.plot(order, energy, \
		marker='x', linewidth=2, color='green', \
		linestyle='-', label='MBE-FCI')
# set x limits
ax.set_xlim([0.5, exp_size + 0.5])
# turn off x-grid
ax.xaxis.grid(False)
# set labels
ax.set_xlabel('Expansion order')
ax.set_ylabel('Correlation energy (in Hartree)')
# force integer ticks on x-axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# despine
sns.despine()
# set legends
ax.legend(loc=1)
# tight layout
plt.tight_layout()
# save plot
plt.savefig(cwd+'/energy.pdf', bbox_inches = 'tight', dpi=1000)

# plot maximal increments
fig, ax = plt.subplots()
# plot results
ax.semilogy(order, np.abs(inc_mean), \
			marker='x', linewidth=2, color=sns.xkcd_rgb['salmon'], \
			linestyle='-', label='mean')
ax.semilogy(order, np.abs(inc_min), \
			marker='x', linewidth=2, color=sns.xkcd_rgb['royal blue'], \
			linestyle='-', label='min')
ax.semilogy(order, np.abs(inc_max), \
			marker='x', linewidth=2, color=sns.xkcd_rgb['kelly green'], \
			linestyle='-', label='max')
# set x limits
ax.set_xlim([0.5, exp_size + 0.5])
# turn off x-grid
ax.xaxis.grid(False)
# set labels
ax.set_xlabel('Expansion order')
ax.set_ylabel('Absolute increments (in Hartree)')
# force integer ticks on x-axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
# despine
sns.despine()
# set legends
ax.legend(loc=1)
# tight layout
plt.tight_layout()
# save plot
plt.savefig(cwd+'/increments.pdf', bbox_inches = 'tight', dpi=1000)

