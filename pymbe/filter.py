#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
filter module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"


import numpy as np
from pyscf import symm, ao2mo, fci
from itertools import islice, combinations, groupby, chain, product
from pymbe.tools import tuples, hash_1d, hash_lookup, tuples_with_nocc, valid_tup
from pymbe.parallel import open_shared_win

def filter(
    nocc_tup: int,          
    tup: np.ndarray,       
    M_tot: np.ndarray,       
    order: int,              
    #hashes: List[List[np.ndarray]],                   
) -> float:


    # Initialize 
    M_ia_ab_ij = np.array([], dtype=np.int64)

    # Define virt and occ tuples
    tup_occ = tup[:nocc_tup]
    tup_virt = tup[nocc_tup:]

    # Matrix for virt occ combinations
    M_ia = M_tot[tup_occ.reshape(-1,1),tup_virt]
    M_ia = np.ravel(M_ia)
    
    # Matrix for virt virt combinations
    M_ab = M_tot[tup_virt.reshape(-1,1),tup_virt]
    M_ab=M_ab[np.triu_indices(tup_virt.size, k=1)]

    # Matrix for occ occ combinations
    M_ij = M_tot[tup_occ.reshape(-1,1),tup_occ]
    M_ij=M_ij[np.triu_indices(tup_occ.size, k=1)]
   
    # Array for all Combinations
    M_ia_ab_ij = np.concatenate((M_ia ,M_ab,M_ij))
 
    # Product of all Combinations
    I_1 = np.prod(M_ia_ab_ij)

    # Normalization of product
    I_1 = np.power(I_1,1/(order-1))

    return I_1


# Restart fehlt?

#  # order
#         self.order = 0
#         self.start_order = 1
#         self.min_order = 1
#         self.max_order = mbe.max_order
#         self.final_order = 0

#         self.nocc = exp.nocc

#     # begin mbe expansion depending
#     for self.order in range(self.min_order, self.max_order + 1):

#   # loop over orders
#         for order in range(1, self.order + 1):
#             # order index
#             k = order - 1

#             # loop over number of occupied orbitals
#             for tup_nocc in range(order + 1):
#                 # occupation index
#                 l = tup_nocc

#                 # load k-th order hashes and increments
#                 hashes = open_shared_win(
#                     self.hashes[k][l], np.int64, (self.n_incs[k][l],)
#                 )
#                 inc = self._open_shared_inc(
#                     self.incs[k][l], self.n_incs[k][l], order, tup_nocc
#                 )

#                 # check if hashes are available
#                 if hashes.size == 0:
#                     continue

#                 # loop over tuples
#                 for tup_idx, tup in enumerate(
#                     tuples_with_nocc(
#                         self.exp_space[k], None, self.nocc, order, tup_nocc, cached=True
#                     )
#                 ):