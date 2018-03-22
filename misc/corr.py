#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import os
import numpy as np

if len(sys.argv) != 2:
    sys.exit('\n error: output directory must be specified: python plot.py out_dir\n')

out_dir = sys.argv[1]

if not os.path.isdir(out_dir):
    sys.exit('\n error: out_dir argument is not a directory\n')

e_hf = e_fci = None

with open(out_dir+'/results.out') as f:
    for line in f:
        if 'Hartree-Fock energy' in line:
            e_hf = float(line.split()[-1])
        elif 'final MBE energy' in line:
            e_fci = float(line.split()[-1])
        else:
            continue

if e_hf is None or e_fci is None:
    sys.exit('\n error: hartree-fock and/or mbe-fci energies were not found on file\n')

e_corr = e_fci - e_hf

print('{0:} has e_corr = {1:.5f}'.format(out_dir, e_corr))

