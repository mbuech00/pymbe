#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
interface module
"""

__author__ = 'Jonas Greiner, Johannes Gutenberg-Universität Mainz, Germany'
__license__ = 'MIT'
__version__ = '0.9'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import os
import ctypes
import numpy as np
from pyscf import ao2mo, gto, scf, symm
from typing import Tuple

from tools import idx_tril, nelec

try:
    import settings
except ImportError:
    msg = '''settings.py not found for module interface. Please create %s
    ''' % os.path.join(os.path.dirname(__file__), 'settings.py')
    sys.stderr.write(msg)

MAX_MEM = 1e10
CONV_TOL = 10

def mbecc_interface(method: str, orb_type: str, point_group: str, orbsym: np.ndarray, h1e: np.ndarray, \
                    h2e: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray, \
                    n_elec: Tuple[int, int], debug: int) -> Tuple[float, int]:
        """
        this function returns the results of a cc calculation using the mbecc
        interface

        example:
        >>> mol = gto.Mole()
        >>> _ = mol.build(atom='O 0. 0. 0.10841; H -0.7539 0. -0.47943; H 0.7539 0. -0.47943',
        ...               basis = '631g', symmetry = 'C2v', verbose=0)
        >>> hf = scf.RHF(mol)
        >>> _ = hf.kernel()
        >>> core_idx = np.array([])
        >>> cas_idx = np.array([0, 1, 2, 3, 4, 7, 9])
        >>> orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
        >>> hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
        >>> h1e = np.einsum('pi,pq,qj->ij', hf.mo_coeff, hcore_ao, hf.mo_coeff)
        >>> eri_ao = mol.intor('int2e_sph', aosym=4)
        >>> h2e = ao2mo.incore.full(eri_ao, hf.mo_coeff)
        >>> h1e_cas = h1e[cas_idx[:, None], cas_idx]
        >>> cas_idx_tril = idx_tril(cas_idx)
        >>> h2e_cas = h2e[cas_idx_tril[:, None], cas_idx_tril]
        >>> n_elec = nelec(hf.mo_occ, cas_idx)
        >>> cc_energy, success = mbecc_interface('ccsd', 'can', 'C2v', orbsym, h1e_cas, \
                                                 h2e_cas, core_idx, cas_idx, n_elec)
        >>> np.isclose(cc_energy, -0.014118607610972705)
        True
        """

        # initialize library
        cclib = ctypes.cdll.LoadLibrary(settings.MBECCLIB)

        # method keys in cfour
        method_dict = {'ccsd': 10, 'ccsd(t)': 22, 'ccsdt': 18}

        # point group
        point_group_dict = {'C1': 1, 'C2': 2, 'Ci': 3, 'Cs': 4, 'D2': 5, 'C2v': 6, 'C2h': 7, 'D2h': 8}

        # settings
        method = ctypes.c_int64(method_dict[method])
        point_group = ctypes.c_int64(point_group_dict[point_group])
        non_canonical = ctypes.c_int64(0 if orb_type == 'can' else 1)
        maxcor = ctypes.c_int64(int(MAX_MEM)) # max memory
        conv = ctypes.c_int64(CONV_TOL)
        max_cycle = ctypes.c_int64(500)
        verbose = ctypes.c_int64(1 if debug >= 3 else 0)
        orbsym = orbsym[cas_idx]

        n_act = cas_idx.size
        h2e = ao2mo.restore(1, h2e, n_act)

        # intitialize variables
        n_elec = np.array(n_elec, dtype=np.int64) # number of occupied orbitals
        n_act = ctypes.c_int64(n_act) # number of orbitals
        cc_energy = ctypes.c_double() # cc-energy output
        success = ctypes.c_int64() # success flag

        # perform cc calculation
        cclib.cc_interface(ctypes.byref(method), ctypes.byref(non_canonical),#
            ctypes.byref(maxcor), n_elec.ctypes.data_as(ctypes.c_void_p),#
            ctypes.byref(n_act), orbsym.ctypes.data_as(ctypes.c_void_p),#
            ctypes.byref(point_group), h1e.ctypes.data_as(ctypes.c_void_p),#
            h2e.ctypes.data_as(ctypes.c_void_p), ctypes.byref(conv),#
            ctypes.byref(max_cycle), ctypes.byref(verbose),#
            ctypes.byref(cc_energy), ctypes.byref(success))

        # close library
        dlclose_func = ctypes.CDLL(None).dlclose
        dlclose_func.argtypes = [ctypes.c_void_p]

        dlclose_func(cclib._handle)

        return cc_energy.value, success.value

if __name__ == "__main__":
    import doctest
    doctest.testmod()


