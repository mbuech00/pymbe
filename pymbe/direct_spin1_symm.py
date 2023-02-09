#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
direct_spin1_symm replacement module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import direct_spin1_symm, cistring
from pyscf.fci.direct_spin1 import _unpack_nelec, _unpack, FCIvector


class FCISolver(direct_spin1_symm.FCISolver):

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if nroots is None: nroots = self.nroots
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec

        wfnsym = self.guess_wfnsym(norb, nelec, ci0, orbsym, wfnsym, **kwargs)

        with lib.temporary_env(self, orbsym=orbsym, wfnsym=wfnsym):
            e, c = kernel_ms1(self, h1e, eri, norb, nelec, ci0, None,
                                           tol, lindep, max_cycle, max_space,
                                           nroots, davidson_only, pspace_size,
                                           ecore=ecore, **kwargs)
        self.eci, self.ci = e, c
        return e, c


def kernel_ms1(fci, h1e, eri, norb, nelec, ci0=None, link_index=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               max_memory=None, verbose=None, ecore=0, **kwargs):
    '''
    Args:
        h1e: ndarray
            1-electron Hamiltonian
        eri: ndarray
            2-electron integrals in chemist's notation
        norb: int
            Number of orbitals
        nelec: int or (int, int)
            Number of electrons of the system

    Kwargs:
        ci0: ndarray
            Initial guess
        link_index: ndarray
            A lookup table to cache the addresses of CI determinants in
            wave-function vector
        tol: float
            Convergence tolerance
        lindep: float
            Linear dependence threshold
        max_cycle: int
            Max. iterations for diagonalization
        max_space: int
            Max. trial vectors to store for sub-space diagonalization method
        nroots: int
            Number of states to solve
        davidson_only: bool
            Whether to call subspace diagonlization (davidson solver) or do a
            full diagonlization (lapack eigh) for small systems
        pspace_size: int
            Number of determinants as the threshold of "small systems",

    Note: davidson solver requires more arguments. For the parameters not
    dispatched, they can be passed to davidson solver via the extra keyword
    arguments **kwargs
    '''
    if nroots is None: nroots = fci.nroots
    if davidson_only is None: davidson_only = fci.davidson_only
    if pspace_size is None: pspace_size = fci.pspace_size
    if max_memory is None:
        max_memory = fci.max_memory - lib.current_memory()[0]
    log = logger.new_logger(fci, verbose)

    nelec = _unpack_nelec(nelec, fci.spin)
    assert (0 <= nelec[0] <= norb and 0 <= nelec[1] <= norb)
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]

    if max_memory < na*nb*6*8e-6:
        log.warn('Not enough memory for FCI solver. '
                 'The minimal requirement is %.0f MB', na*nb*60e-6)

    hdiag = fci.make_hdiag(h1e, eri, norb, nelec)
    nroots = min(hdiag.size, nroots)

    try:
        addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag, max(pspace_size,nroots))
        if pspace_size > 0:
            pw, pv = fci.eig(h0)
        else:
            pw = pv = None

        if pspace_size >= na*nb and ci0 is None and not davidson_only:
            # The degenerated wfn can break symmetry.  The davidson iteration with proper
            # initial guess doesn't have this issue
            if na*nb == 1:
                return pw[0]+ecore, pv[:,0].reshape(1,1).view(FCIvector)
            elif nroots > 1:
                civec = numpy.empty((nroots,na*nb))
                civec[:,addr] = pv[:,:nroots].T
                return pw[:nroots]+ecore, [c.reshape(na,nb).view(FCIvector) for c in civec]
            elif abs(pw[0]-pw[1]) > 1e-12:
                civec = numpy.empty((na*nb))
                civec[addr] = pv[:,0]
                return pw[0]+ecore, civec.reshape(na,nb).view(FCIvector)
    except NotImplementedError:
        addr = [0]
        pw = pv = None

    precond = fci.make_precond(hdiag, pw, pv, addr)

    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    t1_addrs_a = numpy.array([cistring.str2addr(norb, nelec[0], x) for x in cistring.tn_strs(norb, nelec[0], 1)])
    t1_addrs_b = numpy.array([cistring.str2addr(norb, nelec[1], x) for x in cistring.tn_strs(norb, nelec[1], 1)])
    def hop(c):
        hc = fci.contract_2e(h2e, c, norb, nelec, (link_indexa,link_indexb))
        if t1_addrs_a.size > 0:
            hc[t1_addrs_a] = 0.
        if t1_addrs_b.size > 0:
            hc[t1_addrs_b * na] = 0.
        return hc.ravel()

    if ci0 is None:
        if callable(getattr(fci, 'get_init_guess', None)):
            ci0 = lambda: fci.get_init_guess(norb, nelec, nroots, hdiag)
        else:
            def ci0():  # lazy initialization to reduce memory footprint
                x0 = []
                for i in range(nroots):
                    x = numpy.zeros(na*nb)
                    x[addr[i]] = 1
                    x0.append(x)
                return x0
    elif not callable(ci0):
        if isinstance(ci0, numpy.ndarray) and ci0.size == na*nb:
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]
        # Add vectors if not enough initial guess is given
        if len(ci0) < nroots:
            if callable(getattr(fci, 'get_init_guess', None)):
                ci0.extend(fci.get_init_guess(norb, nelec, nroots, hdiag)[len(ci0):])
            else:
                for i in range(len(ci0), nroots):
                    x = numpy.zeros(na*nb)
                    x[addr[i]] = 1
                    ci0.append(x)

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    with lib.with_omp_threads(fci.threads):
        #e, c = lib.davidson(hop, ci0, precond, tol=fci.conv_tol, lindep=fci.lindep)
        e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=log, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    if nroots > 1:
        return e+ecore, [ci.reshape(na,nb).view(FCIvector) for ci in c]
    else:
        return e+ecore, c.reshape(na,nb).view(FCIvector)
