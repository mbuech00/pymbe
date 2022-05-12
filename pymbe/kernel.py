#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
kernel module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from pyscf import gto, scf, cc, fci, ao2mo
from pyscf.cc import ccsd_t
from pyscf.cc import uccsd_t
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import uccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import uccsd_t_rdm
from typing import TYPE_CHECKING

from pymbe.tools import RDMCls, assertion, get_nhole, get_nexc, idx_tril
from pymbe.interface import mbecc_interface

if TYPE_CHECKING:

    from typing import Tuple, Dict, Union, Any, List


MAX_MEM = 1e10
CONV_TOL = 1.0e-10
SPIN_TOL = 1.0e-05


def e_core_h1e(
    hcore: np.ndarray, vhf: np.ndarray, core_idx: np.ndarray, cas_idx: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    this function returns core energy and cas space 1e integrals
    """
    # determine effective core fock potential
    if core_idx.size > 0:
        core_vhf = np.sum(vhf[core_idx], axis=0)
    else:
        core_vhf = 0.0

    # calculate core energy
    e_core = np.trace((hcore + 0.5 * core_vhf)[core_idx[:, None], core_idx]) * 2.0

    # extract cas integrals
    h1e_cas = (hcore + core_vhf)[cas_idx[:, None], cas_idx]

    return e_core, h1e_cas


def main_kernel(
    method: str,
    cc_backend: str,
    orb_type: str,
    spin: int,
    occup: np.ndarray,
    target_mbe: str,
    state_wfnsym: int,
    point_group: str,
    orbsym: np.ndarray,
    hf_guess: bool,
    state_root: int,
    hf_prop: Union[float, np.ndarray, RDMCls],
    e_core: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
    core_idx: np.ndarray,
    cas_idx: np.ndarray,
    nelec: np.ndarray,
    verbose: int,
    higher_amp_extrap: bool = False,
) -> Dict[str, Any]:
    """
    this function return the result property from a given method
    """
    if method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"]:

        res = cc_kernel(
            spin,
            occup,
            core_idx,
            cas_idx,
            method,
            cc_backend,
            nelec,
            orb_type,
            point_group,
            orbsym,
            h1e,
            h2e,
            higher_amp_extrap,
            target_mbe,
            verbose,
            hf_prop,
        )

    elif method == "fci":

        res = fci_kernel(
            spin,
            target_mbe,
            state_wfnsym,
            orbsym,
            hf_guess,
            state_root,
            hf_prop,
            e_core,
            h1e,
            h2e,
            cas_idx,
            nelec,
            verbose,
        )

    return res


def dipole_kernel(
    dipole_ints: np.ndarray,
    occup: np.ndarray,
    cas_idx: np.ndarray,
    cas_rdm1: np.ndarray,
    hf_dipole: np.ndarray = np.zeros(3, dtype=np.float64),
    trans: bool = False,
) -> np.ndarray:
    """
    this function returns an electronic (transition) dipole moment
    """
    # init (transition) rdm1
    if trans:
        rdm1 = np.zeros([occup.size, occup.size], dtype=np.float64)
    else:
        rdm1 = np.diag(occup.astype(np.float64))

    # insert correlated subblock
    rdm1[cas_idx[:, None], cas_idx] = cas_rdm1

    # compute elec_dipole
    elec_dipole = np.einsum("xij,ji->x", dipole_ints, rdm1)

    # 'correlation' dipole
    if not trans:
        elec_dipole -= hf_dipole

    return elec_dipole


def fci_kernel(
    spin: int,
    target_mbe: str,
    wfnsym: int,
    orbsym: np.ndarray,
    hf_guess: bool,
    root: int,
    hf_prop: Any,
    e_core: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
    cas_idx: np.ndarray,
    nelec: np.ndarray,
    verbose: int,
) -> Dict[str, Any]:
    """
    this function returns the results of a fci calculation
    """
    # spin
    spin_cas = abs(nelec[0] - nelec[1])
    assertion(spin_cas == spin, f"casci wrong spin in space: {cas_idx}")

    # init fci solver
    if spin_cas == 0:
        solver = fci.direct_spin0_symm.FCI()
    else:
        solver = fci.direct_spin1_symm.FCI()

    # settings
    solver.conv_tol = CONV_TOL
    if target_mbe in ["dipole", "trans"]:
        solver.conv_tol *= 1.0e-04
        solver.lindep = solver.conv_tol * 1.0e-01
    solver.max_memory = MAX_MEM
    solver.max_cycle = 5000
    solver.max_space = 25
    solver.davidson_only = True
    solver.pspace_size = 0
    if verbose >= 3:
        solver.verbose = 10
    solver.wfnsym = wfnsym
    solver.orbsym = orbsym[cas_idx]
    solver.nroots = root + 1

    # hf starting guess
    if hf_guess:
        na = fci.cistring.num_strings(cas_idx.size, nelec[0])
        nb = fci.cistring.num_strings(cas_idx.size, nelec[1])
        ci0 = np.zeros((na, nb))
        ci0[0, 0] = 1
    else:
        ci0 = None

    # interface
    def _fci_kernel() -> Tuple[List[float], List[np.ndarray]]:
        """
        this function provides an interface to solver.kernel
        """
        # perform calc
        e, c = solver.kernel(h1e, h2e, cas_idx.size, nelec, ecore=e_core, ci0=ci0)

        # collect results
        if solver.nroots == 1:
            return [e], [c]
        else:
            return [e[0], e[-1]], [c[0], c[-1]]

    # perform calc
    energy, civec = _fci_kernel()

    # multiplicity check
    for root in range(len(civec)):

        s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)

        if np.abs((spin_cas + 1) - mult) > SPIN_TOL:

            # fix spin by applying level shift
            sz = np.abs(nelec[0] - nelec[1]) * 0.5
            solver = fci.addons.fix_spin_(solver, shift=0.25, ss=sz * (sz + 1.0))

            # perform calc
            energy, civec = _fci_kernel()

            # verify correct spin
            for root in range(len(civec)):
                s, mult = solver.spin_square(civec[root], cas_idx.size, nelec)
                assertion(
                    np.abs((spin_cas + 1) - mult) < SPIN_TOL,
                    f"spin contamination for root entry = {root}\n"
                    f"2*S + 1 = {mult:.6f}\n"
                    f"cas_idx = {cas_idx}\n"
                    f"cas_sym = {orbsym[cas_idx]}",
                )

    # convergence check
    if solver.nroots == 1:

        assertion(
            solver.converged,
            f"state {root} not converged\n"
            f"cas_idx = {cas_idx}\n"
            f"cas_sym = {orbsym[cas_idx]}",
        )

    else:

        if target_mbe == "excitation":

            for root in [0, solver.nroots - 1]:
                assertion(
                    solver.converged[root],
                    f"state {root} not converged\n"
                    f"cas_idx = {cas_idx}\n"
                    f"cas_sym = {orbsym[cas_idx]}",
                )

        else:

            assertion(
                solver.converged[-1],
                f"state {root} not converged\n"
                f"cas_idx = {cas_idx}\n"
                f"cas_sym = {orbsym[cas_idx]}",
            )

    # collect results
    res: Dict[str, Union[int, float, np.ndarray]] = {}
    if target_mbe == "energy":
        res["energy"] = energy[-1] - hf_prop
    elif target_mbe == "excitation":
        res["excitation"] = energy[-1] - energy[0]
    elif target_mbe == "dipole":
        res["rdm1"] = solver.make_rdm1(civec[-1], cas_idx.size, nelec)
    elif target_mbe == "trans":
        res["t_rdm1"] = solver.trans_rdm1(
            np.sign(civec[0][0, 0]) * civec[0],
            np.sign(civec[-1][0, 0]) * civec[-1],
            cas_idx.size,
            nelec,
        )
    elif target_mbe == "rdm12":
        res["rdm1"], res["rdm2"] = solver.make_rdm12(civec[-1], cas_idx.size, nelec)
        cas_hf_prop = hf_prop[cas_idx]
        res["rdm1"] -= cas_hf_prop.rdm1
        res["rdm2"] -= cas_hf_prop.rdm2

    return res


def cc_kernel(
    spin: int,
    occup: np.ndarray,
    core_idx: np.ndarray,
    cas_idx: np.ndarray,
    method: str,
    cc_backend: str,
    nelec: np.ndarray,
    orb_type: str,
    point_group: str,
    orbsym: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    higher_amp_extrap: bool,
    target: str,
    verbose: int,
    hf_prop: Any,
) -> Dict[str, Any]:
    """
    this function returns the results of a ccsd / ccsd(t) calculation
    """
    spin_cas = abs(nelec[0] - nelec[1])
    assertion(spin_cas == spin, f"cascc wrong spin in space: {cas_idx}")
    singlet = spin_cas == 0

    if cc_backend == "pyscf":

        # init ccsd solver
        mol_tmp = gto.Mole(verbose=0)
        mol_tmp._built = True
        mol_tmp.max_memory = MAX_MEM
        mol_tmp.incore_anyway = True

        if singlet:
            hf = scf.RHF(mol_tmp)
        else:
            hf = scf.UHF(mol_tmp)

        hf.get_hcore = lambda *args: h1e
        hf._eri = h2e

        if singlet:
            ccsd = cc.ccsd.CCSD(
                hf, mo_coeff=np.eye(cas_idx.size), mo_occ=occup[cas_idx]
            )
        else:
            ccsd = cc.uccsd.UCCSD(
                hf,
                mo_coeff=np.array((np.eye(cas_idx.size), np.eye(cas_idx.size))),
                mo_occ=np.array(
                    (occup[cas_idx] > 0.0, occup[cas_idx] == 2.0), dtype=np.double
                ),
            )

        # settings
        ccsd.conv_tol = CONV_TOL
        if target in ["dipole", "rdm12"]:
            ccsd.conv_tol_normt = ccsd.conv_tol
        ccsd.max_cycle = 500
        ccsd.async_io = False
        ccsd.diis_start_cycle = 4
        ccsd.diis_space = 12
        ccsd.incore_complete = True
        eris = ccsd.ao2mo()

        # calculate ccsd energy
        ccsd.kernel(eris=eris)

        # convergence check
        assertion(
            ccsd.converged,
            f"CCSD error: no convergence, core_idx = {core_idx}, cas_idx = {cas_idx}",
        )

        # e_corr
        e_cc = ccsd.e_corr

        # calculate (t) correction
        if method == "ccsd(t)":

            # number of holes in cas space
            nhole = get_nhole(nelec, cas_idx)

            # nexc
            nexc = get_nexc(nelec, nhole)

            # ensure that more than two excitations are possible
            if nexc > 2:
                if singlet:
                    e_cc += ccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)
                else:
                    e_cc += uccsd_t.kernel(ccsd, eris, ccsd.t1, ccsd.t2, verbose=0)

    elif cc_backend in ["ecc", "ncc"]:

        # calculate cc energy
        cc_energy, success = mbecc_interface(
            method,
            cc_backend,
            orb_type,
            point_group,
            orbsym[cas_idx],
            h1e,
            h2e,
            nelec,
            higher_amp_extrap,
            verbose,
        )

        # convergence check
        assertion(
            success == 1,
            f"MBECC error: no convergence, core_idx = {core_idx}, cas_idx = {cas_idx}",
        )

        # e_corr
        e_cc = cc_energy

    # collect results
    res: Dict[str, Union[float, np.ndarray]] = {"energy": e_cc}

    # rdms
    if target in ["dipole", "rdm12"]:
        if method == "ccsd" or nexc <= 2:
            ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
            rdm1 = ccsd.make_rdm1()
            if target == "rdm12":
                rdm2 = ccsd.make_rdm2()
        elif method == "ccsd(t)":
            if singlet:
                l1, l2 = ccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
                if target == "rdm12":
                    rdm2 = ccsd_t_rdm.make_rdm2(
                        ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                    )
            else:
                l1, l2 = uccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
                if target == "rdm12":
                    rdm2 = uccsd_t_rdm.make_rdm2(
                        ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris
                    )
        if singlet:
            res["rdm1"] = rdm1
        else:
            res["rdm1"] = rdm1[0] + rdm1[1]
        if target == "rdm12":
            if singlet:
                res["rdm2"] = rdm2
            else:
                res["rdm2"] = rdm2[0] + rdm2[1] + rdm2[2] + rdm2[3]
            if hf_prop is not None:
                cas_hf_prop = hf_prop[cas_idx]
                res["rdm1"] -= cas_hf_prop.rdm1
                res["rdm2"] -= cas_hf_prop.rdm2
            else:
                res["rdm1"], res["rdm2"] = hf_rdm12_kernel(cas_idx.size, occup[cas_idx])

    return res


def hf_energy_kernel(
    occup: np.ndarray, spin: int, hcore: np.ndarray, eri: np.ndarray, vhf: np.ndarray
) -> float:
    """
    this function constructs the Hartree-Fock electronic energy
    """
    # add one-electron integrals
    hf_energy = np.sum(occup * np.diag(hcore))

    # set closed- and open-shell indices
    cs_idx = np.where(occup == 2)[0]
    os_idx = np.where(occup == 1)[0]

    # add closed-shell and coupling electron repulsion terms
    hf_energy += np.trace((np.sum(vhf, axis=0))[cs_idx.reshape(-1, 1), cs_idx])

    # check if system is open-shell
    if spin > 0:

        # get indices for eris that only include open-shell orbitals
        os_eri_idx = idx_tril(os_idx)

        # retrieve eris of open-shell orbitals and unpack these
        os_eri = ao2mo.restore(
            1, eri[os_eri_idx.reshape(-1, 1), os_eri_idx], os_idx.size
        )

        # add open-shell electron repulsion terms
        hf_energy += 0.5 * (np.einsum("pqrr->", os_eri) - np.einsum("pqrp->", os_eri))

    return hf_energy


def hf_dipole_kernel(occup: np.ndarray, dipole_ints: np.ndarray) -> np.ndarray:
    """
    this function constructs the Hartree-Fock electronic dipole moment
    """
    return np.einsum("p,xpp->x", occup, dipole_ints)


def hf_rdm12_kernel(norb: int, occup: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function constructs the Hartree-Fock reduced density matrices
    """
    rdm1 = np.zeros(2 * (norb,), dtype=np.float64)
    np.einsum("ii->i", rdm1)[...] += occup

    rdm2 = np.zeros(4 * (norb,), dtype=np.float64)
    occup_a = occup.copy()
    occup_a[occup_a > 0.0] = 1.0
    occup_b = occup - occup_a
    # d_ppqq = k_pa*k_qa + k_pb*k_qb + k_pa*k_qb + k_pb*k_qa = k_p*k_q
    np.einsum("iijj->ij", rdm2)[...] += np.einsum("i,j", occup, occup)
    # d_pqqp = - (k_pa*k_qa + k_pb*k_qb)
    np.einsum("ijji->ij", rdm2)[...] -= np.einsum("i,j", occup_a, occup_a) + np.einsum(
        "i,j", occup_b, occup_b
    )

    return rdm1, rdm2
