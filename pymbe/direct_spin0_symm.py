#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
direct_spin0_symm replacement module
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
from pyscf.fci import direct_spin0_symm, direct_spin1, cistring
from pyscf.fci.direct_spin0 import _unpack, _check_


class FCISolver(direct_spin0_symm.FCISolver):

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
            e, c = kernel_ms0(self, h1e, eri, norb, nelec, ci0, None,
                                           tol, lindep, max_cycle, max_space,
                                           nroots, davidson_only, pspace_size,
                                           ecore=ecore, **kwargs)
        self.eci, self.ci = e, c
        return e, c


def kernel_ms0(fci, h1e, eri, norb, nelec, ci0=None, link_index=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               max_memory=None, verbose=None, ecore=0, **kwargs):
    if nroots is None: nroots = fci.nroots
    if davidson_only is None: davidson_only = fci.davidson_only
    if pspace_size is None: pspace_size = fci.pspace_size
    if max_memory is None:
        max_memory = fci.max_memory - lib.current_memory()[0]
    log = logger.new_logger(fci, verbose)

    assert (fci.spin is None or fci.spin == 0)
    assert (0 <= numpy.sum(nelec) <= norb*2)

    link_index = _unpack(norb, nelec, link_index)
    h1e = numpy.ascontiguousarray(h1e)
    eri = numpy.ascontiguousarray(eri)
    na = link_index.shape[0]

    if max_memory < na**2*6*8e-6:
        log.warn('Not enough memory for FCI solver. '
                 'The minimal requirement is %.0f MB', na**2*60e-6)

    hdiag = fci.make_hdiag(h1e, eri, norb, nelec)
    nroots = min(hdiag.size, nroots)

    try:
        addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag, max(pspace_size,nroots))
        if pspace_size > 0:
            pw, pv = fci.eig(h0)
        else:
            pw = pv = None

        if pspace_size >= na*na and ci0 is None and not davidson_only:
            # The degenerated wfn can break symmetry.  The davidson iteration with proper
            # initial guess doesn't have this issue
            if na*na == 1:
                return pw[0]+ecore, pv[:,0].reshape(1,1).view(direct_spin1.FCIvector)
            elif nroots > 1:
                civec = numpy.empty((nroots,na*na))
                civec[:,addr] = pv[:,:nroots].T
                civec = civec.reshape(nroots,na,na)
                try:
                    return (pw[:nroots]+ecore,
                            [_check_(ci).view(direct_spin1.FCIvector) for ci in civec])
                except ValueError:
                    pass
            elif abs(pw[0]-pw[1]) > 1e-12:
                civec = numpy.empty((na*na))
                civec[addr] = pv[:,0]
                civec = civec.reshape(na,na)
                civec = lib.transpose_sum(civec) * .5
                # direct diagonalization may lead to triplet ground state

                #TODO: optimize initial guess.  Using pspace vector as initial guess may have
                # spin problems.  The 'ground state' of psapce vector may have different spin
                # state to the true ground state.
                try:
                    return (pw[0]+ecore,
                            _check_(civec.reshape(na,na)).view(direct_spin1.FCIvector))
                except ValueError:
                    pass
    except NotImplementedError:
        addr = [0]
        pw = pv = None

    precond = fci.make_precond(hdiag, pw, pv, addr)

    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    t1_addrs = numpy.array([cistring.str2addr(norb, nelec[0], x) for x in cistring.tn_strs(norb, nelec[0], 1)])
    def hop(c):
        hc = fci.contract_2e(h2e, c.reshape(na,na), norb, nelec, link_index)
        hc[t1_addrs, 0] = 0.
        hc[0, t1_addrs] = 0.
        return hc.ravel()

#TODO: check spin of initial guess
    if ci0 is None:
        if callable(getattr(fci, 'get_init_guess', None)):
            ci0 = lambda: fci.get_init_guess(norb, nelec, nroots, hdiag)
        else:
            def ci0():
                x0 = []
                for i in range(nroots):
                    x = numpy.zeros((na,na))
                    addra = addr[i] // na
                    addrb = addr[i] % na
                    if addra == addrb:
                        x[addra,addrb] = 1
                    else:
                        x[addra,addrb] = x[addrb,addra] = numpy.sqrt(.5)
                    x0.append(x.ravel())
                return x0
    elif not callable(ci0):
        if isinstance(ci0, numpy.ndarray) and ci0.size == na*na:
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]

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
        return (e+ecore,
                [_check_(ci.reshape(na,na)).view(direct_spin1.FCIvector) for ci in c])
    else:
        return e+ecore, _check_(c.reshape(na,na)).view(direct_spin1.FCIvector)
