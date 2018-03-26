#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import os
import numpy as np
import subprocess

if len(sys.argv) != 2:
    sys.exit('\n error: output directory must be specified: python plot.py out_dir\n')

out_dir = sys.argv[1]

if not os.path.isdir(out_dir):
    sys.exit('\n error: out_dir argument is not a directory\n')

e_corr = None

with open(out_dir+'/results.out') as f:
	e_corr = float(subprocess.check_output(['tail', '-4', out_dir+'/results.out']).split()[2])

if e_corr is None:
    sys.exit('\n error: correlation energies were not found on file\n')

print('{0:} has e_corr = {1:.5f}'.format(out_dir, e_corr))

