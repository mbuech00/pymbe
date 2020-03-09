#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import sys

from runtest import version_info, get_filter, cli, run

def configure(options, input_files, extra_args):
    """
    This function is used by runtest to configure runtest
    at runtime for code specific launch command and file naming.
    """
    launcher = ''
    full_command = 'mpiexec -np 8 python ../../../src/main.py'
    output_prefix = 'output/pymbe'
    relative_reference_path = 'ref'
    return launcher, full_command, output_prefix, relative_reference_path

assert version_info.major == 2

f = [
    get_filter(from_string = 'MBE order  |        MBE',
               num_lines = 10,
               rel_tolerance = 1.0e-6),
    get_filter(from_string = 'MBE order  |     total energy',
               num_lines = 10,
               rel_tolerance = 1.0e-6)
]

g = [get_filter(rel_tolerance = 1.0e-6)]

options = cli()

ierr = run(options, configure, input_files='input', filters={'results': f, 'output': g})

sys.exit(ierr)

