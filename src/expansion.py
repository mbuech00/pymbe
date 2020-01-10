#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
expansion module containing all expansion attributes
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from mpi4py import MPI
import functools
import copy
from typing import List, Dict, Tuple, Union, Any

import parallel
import system
import calculation
import tools


class ExpCls:
        """
        this class contains the pymbe expansion attributes
        """
        def __init__(self, mol: system.MolCls, calc: calculation.CalcCls) -> None:
                """
                init expansion attributes
                """
                # set expansion model dict
                self.model = copy.deepcopy(calc.model)

                # init prop dict
                self.prop: Dict[str, Dict[str, Union[List[float], MPI.Win]]] = {str(calc.target_mbe): {'inc': [], 'tot': []}}

                # set max_order
                if calc.misc['order'] is not None:
                    self.max_order = min(calc.exp_space.size, calc.misc['order'])
                else:
                    self.max_order = calc.exp_space.size

                # init timings and and statistics lists
                self.time: Dict[str, Union[List[float], np.ndarray]] = {'mbe': [], 'screen': []}
                self.mean_inc: Union[List[float], np.ndarray] = []
                self.min_inc: Union[List[float], np.ndarray] = []
                self.max_inc: Union[List[float], np.ndarray] = []
                self.mean_ndets: Union[List[int], np.ndarray] = []
                self.min_ndets: Union[List[int], np.ndarray] = []
                self.max_ndets: Union[List[int], np.ndarray] = []

                # init order
                self.order: int = 0

                # init attributes
                self.min_order: int = 2 if calc.ref_space.size == 0 else 1
                self.start_order: int = 0
                self.final_order: int = 0
                self.hashes: List[MPI.Win] = []
                self.n_tuples: List[int] = [tools.n_tuples(calc.exp_space[calc.exp_space < mol.nocc], \
                                                           calc.exp_space[mol.nocc <= calc.exp_space], \
                                                           tools.virt_prune(calc.occup, calc.ref_space), \
                                                           tools.occ_prune(calc.occup, calc.ref_space), self.min_order)]


if __name__ == "__main__":
    import doctest
    doctest.testmod()


