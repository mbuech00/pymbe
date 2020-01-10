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
import calculation
import tools


class ExpCls:
        """
        this class contains the pymbe expansion attributes
        """
        def __init__(self, calc: calculation.CalcCls) -> None:
                """
                init expansion attributes
                """
                # set expansion model dict
                self.model = copy.deepcopy(calc.model)

                # init prop dict
                self.prop: Dict[str, Dict[str, Union[List[float], MPI.Win]]] = {str(calc.target_mbe): {'inc': [], 'tot': []}}

                # set max_order
                if calc.misc['order'] is not None:
                    self.max_order = min(calc.exp_space['seed'].size + calc.exp_space['tot'].size, calc.misc['order'])
                else:
                    self.max_order = calc.exp_space['seed'].size + calc.exp_space['tot'].size

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
                self.hashes: List[MPI.Win] = [None]
                self.n_tuples: List[int] = [0]
                self.min_order: int = 0
                self.start_order: int = 0
                self.final_order: int = 0


if __name__ == "__main__":
    import doctest
    doctest.testmod()


