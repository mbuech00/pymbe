#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
wrapper module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from pyscf import gto, scf, symm, lo, cc, mcscf, fci, ao2mo
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import uccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import uccsd_t_rdm
from copy import copy
from typing import TYPE_CHECKING, cast, Union
from warnings import catch_warnings, simplefilter

from pymbe.kernel import dipole_kernel, cc_kernel
from pymbe.tools import (
    assertion,
    mat_idx,
    near_nbrs,
    core_cas,
    get_vhf,
    get_nelec,
    idx_tril,
    ground_state_sym,
    get_occup,
)

if TYPE_CHECKING:

    from typing import Tuple, Dict, List, Optional, Any


CONV_TOL = 1.0e-10
SPIN_TOL = 1.0e-05


def ints(
    mol: gto.Mole,
    mo_coeff: np.ndarray,
    x2c: bool = False,
    u: float = 1.0,
    matrix: Tuple[int, int] = (1, 6),
    pbc: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns 1e and 2e mo integrals
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "ints: mol (first argument) must be a gto.Mole object",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "ints: mo coefficients (second argument) must be a np.ndarray",
    )
    # x2c
    assertion(
        isinstance(x2c, bool),
        "ints: spin-free x2c relativistic hamiltonian option (x2c keyword argument) "
        "must be a bool",
    )
    # hubbard
    if not mol.atom:
        # matrix
        assertion(
            isinstance(matrix, tuple)
            and len(matrix) == 2
            and isinstance(matrix[0], int)
            and isinstance(matrix[1], int),
            "ints: hubbard matrix (matrix keyword argument) must be a tuple of ints "
            "with a dimension of 2",
        )
        # u parameter
        assertion(
            isinstance(u, float),
            "ints: hubbard on-site repulsion parameter (u keyword argument) must be a "
            "float",
        )
        assertion(
            u > 0.0,
            "ints: only repulsive hubbard models are implemented, hubbard on-site "
            "repulsion parameter (u keyword argument) must be > 0.",
        )
        # periodic boundary conditions
        assertion(
            isinstance(pbc, bool),
            "ints: hubbard model periodic boundary conditions (pbc keyword argument) "
            "must be a bool",
        )

    # norb
    norb = mol.nao.item()

    # hcore_ao and eri_ao w/o symmetry
    hcore_ao, eri_ao = _ao_ints(mol, x2c, u, matrix, pbc)

    # compute hcore
    hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

    # eri_mo w/o symmetry
    eri = ao2mo.incore.full(eri_ao, mo_coeff)

    # restore 4-fold symmetry in eri
    eri = ao2mo.restore(4, eri, norb)

    return hcore, eri


def _ao_ints(
    mol: gto.Mole, x2c: bool, u: float, matrix: Tuple[int, int], pbc: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns 1e and 2e ao integrals
    """
    if mol.atom:

        # hcore_ao
        if x2c:
            hf = scf.ROHF(mol).x2c()
        else:
            hf = scf.ROHF(mol)
        hcore = hf.get_hcore()
        # eri_ao w/o symmetry
        if mol.cart:
            eri = mol.intor("int2e_cart", aosym=1)
        else:
            eri = mol.intor("int2e_sph", aosym=1)

    else:

        # hcore_ao
        hcore = _hubbard_h1e(matrix, pbc)
        # eri_ao
        eri = _hubbard_eri(matrix, u)

    return hcore, eri


def _hubbard_h1e(matrix: Tuple[int, int], pbc: bool = False) -> np.ndarray:
    """
    this function returns the hubbard hopping hamiltonian
    """
    # dimension
    if 1 in matrix:
        ndim = 1
    else:
        ndim = 2

    # nsites
    nsites = matrix[0] * matrix[1]

    # init h1e
    h1e = np.zeros([nsites] * 2, dtype=np.float64)

    if ndim == 1:

        # adjacent neighbours
        for i in range(nsites - 1):
            h1e[i, i + 1] = h1e[i + 1, i] = -1.0

        if pbc:
            h1e[-1, 0] = h1e[0, -1] = -1.0

    elif ndim == 2:

        # number of x- and y-sites
        nx, ny = matrix[0], matrix[1]

        # adjacent neighbours
        for site_1 in range(nsites):

            site_1_xy = mat_idx(site_1, nx, ny)
            nbrs = near_nbrs(site_1_xy, nx, ny)

            for site_2 in range(site_1):

                site_2_xy = mat_idx(site_2, nx, ny)

                if site_2_xy in nbrs:
                    h1e[site_1, site_2] = h1e[site_2, site_1] = -1.0

    return h1e


def _hubbard_eri(matrix: Tuple[int, int], u: float) -> np.ndarray:
    """
    this function returns the hubbard two-electron hamiltonian
    """
    # nsites
    nsites = matrix[0] * matrix[1]

    # init eri
    eri = np.zeros([nsites] * 4, dtype=np.float64)

    # compute eri
    for i in range(nsites):
        eri[i, i, i, i] = u

    return eri


def dipole_ints(
    mol: gto.Mole, mo_coeff: np.ndarray, gauge_origin: np.ndarray
) -> np.ndarray:
    """
    this function returns dipole integrals (in AO basis)
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "dipole_ints: mol (first argument) must be a gto.Mole object",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "dipole_ints: mo coefficients (second argument) must be a np.ndarray",
    )
    # gauge origin
    assertion(
        isinstance(gauge_origin, np.ndarray) and gauge_origin.size == 3,
        "dipole_ints: gauge origin (gauge_origin keyword argument) must be a "
        "np.ndarray of size 3",
    )

    with mol.with_common_origin(gauge_origin):
        dipole = mol.intor_symmetric("int1e_r", comp=3)

    return np.einsum("pi,xpq,qj->xij", mo_coeff, dipole, mo_coeff)


def hf(
    mol: gto.Mole,
    target: str = "energy",
    init_guess: str = "minao",
    newton: bool = False,
    irrep_nelec: Dict[str, Any] = {},
    x2c: bool = False,
    u: float = 1.0,
    matrix: Tuple[int, int] = (1, 6),
    pbc: bool = False,
    gauge_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> Tuple[
    scf.hf.SCF,
    Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]:
    """
    this function returns the results of a restricted (open-shell) hartree-fock
    calculation
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole), "hf: mol (first argument) must be a gto.Mole object"
    )
    # init_guess
    assertion(
        isinstance(init_guess, str),
        "hf: hf initial guess (init_guess keyword argument) must be a str",
    )
    assertion(
        init_guess in ["minao", "atom", "1e"],
        "hf: valid hf initial guesses (init_guess keyword argument) are: "
        "minao, atom, and 1e",
    )
    # newton
    assertion(
        isinstance(newton, bool),
        "hf: newton option (newton keyword argument) must be a bool",
    )
    # irrep_nelec
    assertion(
        isinstance(irrep_nelec, dict),
        "hf: irreducible representation occupation (irrep_nelec keyword argument) must "
        "be a dict",
    )
    # x2c
    assertion(
        isinstance(x2c, bool),
        "hf: spin-free x2c relativistic hamiltonian option (x2c keyword argument) must "
        "be a bool",
    )
    # hubbard
    if not mol.atom:
        # matrix
        assertion(
            isinstance(matrix, tuple)
            and len(matrix) == 2
            and isinstance(matrix[0], int)
            and isinstance(matrix[1], int),
            "hf: hubbard matrix (matrix keyword argument) must be a tuple of ints with "
            "a dimension of 2",
        )
        # u parameter
        assertion(
            isinstance(u, float),
            "hf: hubbard on-site repulsion parameter (u keyword argument) must be a "
            "float",
        )
        assertion(
            u > 0.0,
            "hf: only repulsive hubbard models are implemented, hubbard on-site "
            "repulsion parameter (u keyword argument) must be > 0.",
        )

        # periodic boundary conditions
        assertion(
            isinstance(pbc, bool),
            "hf: hubbard model periodic boundary conditions (pbc keyword argument) "
            "must be a bool",
        )
    # gauge origin
    assertion(
        isinstance(gauge_origin, np.ndarray) and gauge_origin.size == 3,
        "hf: gauge origin (gauge_origin keyword argument) must be a np.ndarray of size "
        "3",
    )

    # initialize restricted hf calc
    if x2c:
        hf = scf.RHF(mol).x2c()
    else:
        hf = scf.RHF(mol)

    hf.init_guess = init_guess
    if newton:
        hf.conv_tol = 1.0e-01
    else:
        hf.conv_tol = CONV_TOL
    hf.max_cycle = 1000

    if mol.atom:
        # ab initio hamiltonian
        hf.irrep_nelec = irrep_nelec
    else:
        # model hamiltonian
        hf.get_ovlp = lambda *args: np.eye(matrix[0] * matrix[1])
        hf.get_hcore = lambda *args: _hubbard_h1e(matrix, pbc)
        hf._eri = _hubbard_eri(matrix, u)

    # hf calc
    with catch_warnings():
        simplefilter("ignore")
        hf.kernel()

    if newton:

        # initial mo coefficients and occupation
        mo_coeff = hf.mo_coeff
        mo_occ = hf.mo_occ

        # new so-hf object
        hf = hf.newton()
        hf.conv_tol = CONV_TOL

        with catch_warnings():
            simplefilter("ignore")
            hf.kernel(mo_coeff, mo_occ)

    # store occupation and orbsym
    occup = hf.mo_occ
    norb = occup.size
    if mol.symmetry:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
    else:
        orbsym = np.zeros(norb, dtype=np.int64)

    hf_prop: Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]

    if target == "energy":

        hf_prop = hf.e_tot.item()

    elif target == "excitation":

        hf_prop = 0.0

    elif target == "dipole":

        if mol.atom:
            dm = hf.make_rdm1()
            if mol.spin > 0:
                dm = dm[0] + dm[1]
            with mol.with_common_orig(gauge_origin):
                ao_dip = mol.intor_symmetric("int1e_r", comp=3)
            hf_prop = np.einsum("xij,ji->x", ao_dip, dm)
        else:
            hf_prop = np.zeros(3, dtype=np.float64)

    elif target == "trans":

        hf_prop = np.zeros(3, dtype=np.float64)

    elif target == "rdm12":

        rdm1 = np.zeros(2 * (norb,), dtype=np.float64)
        np.einsum("ii->i", rdm1)[...] += occup

        rdm2 = np.zeros(4 * (norb,), dtype=np.float64)
        occup_a = occup.copy()
        occup_a[occup_a > 0.0] = 1.0
        occup_b = occup - occup_a
        # d_ppqq = k_pa*k_qa + k_pb*k_qb + k_pa*k_qb + k_pb*k_qa = k_p*k_q
        np.einsum("iijj->ij", rdm2)[...] += np.einsum("i,j", occup, occup)
        # d_pqqp = - (k_pa*k_qa + k_pb*k_qb)
        np.einsum("ijji->ij", rdm2)[...] += np.einsum(
            "i,j", occup_a, occup_a
        ) + np.einsum("i,j", occup_b, occup_b)

        hf_prop = (rdm1, rdm2)

    return hf, hf_prop, orbsym, np.asarray(hf.mo_coeff, order="C")


def ref_mo(
    orbs: str,
    mol: gto.Mole,
    hf: scf.hf.SCF,
    mo_coeff: np.ndarray,
    orbsym: np.ndarray,
    ncore: int,
    ref_space: np.ndarray = np.array([]),
    wfnsym: Optional[List[Union[str, int]]] = None,
    weights: List[float] = [1.0],
    hf_guess: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function returns a set of reference mo coefficients and symmetries plus the
    associated spaces
    """
    # orbs
    assertion(
        isinstance(orbs, str),
        "ref_mo: orbital representation (first argument) must be a str",
    )
    assertion(
        orbs in ["ccsd", "ccsd(t)", "local", "casscf"],
        "ref_mo: valid orbital representations (first argument) are: natural (ccsd or "
        "ccsd(t)), pipek-mezey (local), or casscf orbs (casscf)",
    )
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "ref_mo: mol (second argument) must be a gto.Mole object",
    )
    # hf
    assertion(
        isinstance(hf, scf.hf.SCF),
        "ref_mo: hf (third argument) must be a scf.hf.SCF object",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "ref_mo: mo coefficients (fourth argument) must be a np.ndarray",
    )
    # orbsym
    assertion(
        isinstance(orbsym, np.ndarray),
        "ref_mo: orbital symmetry (fifth argument) must be a np.ndarray",
    )
    # ncore
    assertion(
        isinstance(ncore, int),
        "ref_mo: number of core orbitals (sixth argument) must be an int",
    )
    # casscf
    if orbs == "casscf":
        # set default casscf reference symmetry
        if wfnsym is None:
            wfnsym = (
                [symm.addons.irrep_id2name(mol.groupname, 0)] if mol.groupname else [0]
            )
        # ref_space
        assertion(
            isinstance(ref_space, np.ndarray),
            "ref_mo: reference space (ref_space keyword argument) must be a np.ndarray "
            "of orbital indices",
        )
        assertion(
            np.any(np.isin(np.arange(max(mol.nelec)), ref_space)),
            "ref_mo: no singly/doubly occupied orbitals in cas space (ref_space "
            "keyword argument) of casscf calculation",
        )
        assertion(
            np.any(np.isin(np.arange(min(mol.nelec), mol.nao), ref_space)),
            "ref_mo: no singly occupied/virtual orbitals in cas space (ref_space "
            "keyword argument) of casscf calculation",
        )
        # wfnsym
        assertion(
            isinstance(wfnsym, list) and all(isinstance(i, str) for i in wfnsym),
            "ref_mo: casscf wavefunction symmetries (wfnsym keyword argument) must be "
            "a list of str",
        )
        wfnsym_int: List[int] = []
        for i in range(len(wfnsym)):
            try:
                wfnsym_int.append(
                    symm.addons.irrep_name2id(mol.groupname, wfnsym[i])
                    if mol.groupname
                    else 0
                )
            except Exception as err:
                raise ValueError(
                    "ref_mo: illegal choice of ref wfnsym (wfnsym keyword argument) "
                    f"-- PySCF error: {err}"
                )
        # weights
        assertion(
            isinstance(weights, list) and all(isinstance(i, float) for i in weights),
            "ref_mo: casscf weights (weights keyword argument) must be a list of "
            "floats",
        )
        assertion(
            len(wfnsym_int) == len(weights),
            "ref_mo: list of wfnsym (wfnsym keyword argument) and weights (weights "
            "keyword argument) for casscf calculation must be of same length",
        )
        assertion(
            all(isinstance(i, float) for i in weights),
            "ref_mo: casscf weights (weights keyword argument) must be floats",
        )
        assertion(
            abs(sum(weights) - 1.0) < 1.0e-3,
            "ref_mo: sum of weights for casscf calculation (weights keyword argument) "
            "must be equal to 1.",
        )
        # hf_guess
        assertion(
            isinstance(hf_guess, bool),
            "ref_mo: hf initial guess (hf_guess keyword argument) must be a bool",
        )
        if hf_guess:
            assertion(
                len(set(wfnsym_int)) == 1,
                "ref_mo: illegal choice of reference wfnsym (wfnsym keyword argument) "
                "when enforcing hf initial guess (hf_guess keyword argument) because "
                "wfnsym should be limited to one state",
            )
            assertion(
                wfnsym_int[0] == ground_state_sym(orbsym, mol.nelec, mol.groupname),
                "ref_mo: illegal choice of reference wfnsym (wfnsym keyword argument) "
                "when enforcing hf initial guess (hf_guess keyword argument) because "
                "wfnsym does not equal hf state symmetry",
            )

    # nocc
    nocc = max(mol.nelec)

    # norb
    norb = mol.nao.item()

    # occup
    occup = get_occup(norb, mol.nelec)

    # copy mo coefficients
    mo_coeff_out = np.copy(mo_coeff)

    # set core and cas spaces
    core_idx, cas_idx = core_cas(nocc, np.arange(ncore, nocc), np.arange(nocc, norb))

    # NOs
    if orbs in ["ccsd", "ccsd(t)"]:

        # compute rmd1
        ccsd = cc.CCSD(hf)
        frozen_orbs = np.asarray(
            [i for i in range(hf.mo_coeff.shape[1]) if i not in cas_idx]
        )
        if frozen_orbs.size > 0:
            ccsd.frozen = frozen_orbs

        # settings
        ccsd.conv_tol = 1.0e-10
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

        # rdm1
        if orbs == "ccsd":
            ccsd.l1, ccsd.l2 = ccsd.solve_lambda(ccsd.t1, ccsd.t2, eris=eris)
            rdm1 = ccsd.make_rdm1()
        elif orbs == "ccsd(t)":
            if mol.spin == 0:
                l1, l2 = ccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = ccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)
            else:
                l1, l2 = uccsd_t_lambda.kernel(ccsd, eris=eris, verbose=0)[1:]
                rdm1 = uccsd_t_rdm.make_rdm1(ccsd, ccsd.t1, ccsd.t2, l1, l2, eris=eris)

        if mol.spin > 0:
            rdm1 = rdm1[0] + rdm1[1]

        # occupied - occupied block
        mask = occup == 2.0
        mask[:ncore] = False
        if np.any(mask):
            no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
            mo_coeff_out[:, mask] = np.einsum(
                "ip,pj->ij", mo_coeff[:, mask], no[:, ::-1]
            )

        # singly occupied - singly occupied block
        mask = occup == 1.0
        mask[:ncore] = False
        if np.any(mask):
            no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
            mo_coeff_out[:, mask] = np.einsum(
                "ip,pj->ij", mo_coeff[:, mask], no[:, ::-1]
            )

        # virtual - virtual block
        mask = occup == 0.0
        mask[:ncore] = False
        if np.any(mask):
            no = symm.eigh(rdm1[np.ix_(mask, mask)], orbsym[mask])[-1]
            mo_coeff_out[:, mask] = np.einsum(
                "ip,pj->ij", mo_coeff[:, mask], no[:, ::-1]
            )

        # orbital symmetries
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_out)

    # pipek-mezey localized orbitals
    elif orbs == "local":

        # occupied - occupied block
        mask = occup == 2.0
        mask[:ncore] = False
        if np.any(mask):
            if mol.atom:
                loc = lo.PM(mol, mo_coeff[:, mask])
            else:
                loc = _hubbard_PM(mol, mo_coeff[:, mask])
            loc.conv_tol = CONV_TOL
            mo_coeff_out[:, mask] = loc.kernel()

        # singly occupied - singly occupied block
        mask = occup == 1.0
        mask[:ncore] = False
        if np.any(mask):
            if mol.atom:
                loc = lo.PM(mol, mo_coeff[:, mask])
            else:
                loc = _hubbard_PM(mol, mo_coeff[:, mask])
            loc.conv_tol = CONV_TOL
            mo_coeff_out[:, mask] = loc.kernel()

        # virtual - virtual block
        mask = occup == 0.0
        mask[:ncore] = False
        if np.any(mask):
            if mol.atom:
                loc = lo.PM(mol, mo_coeff[:, mask])
            else:
                loc = _hubbard_PM(mol, mo_coeff[:, mask])
            loc.conv_tol = CONV_TOL
            mo_coeff_out[:, mask] = loc.kernel()

        # orbital symmetries
        if mol.symmetry:
            orbsym = np.zeros(norb, dtype=np.int64)

    # casscf
    elif orbs == "casscf":

        # electrons in active space
        act_nelec = get_nelec(occup, ref_space)

        # sorter for active space
        n_core_inact = np.array(
            [i for i in range(nocc) if i not in ref_space], dtype=np.int64
        )
        n_virt_inact = np.array(
            [a for a in range(nocc, norb) if a not in ref_space], dtype=np.int64
        )
        sort_casscf = np.concatenate((n_core_inact, ref_space, n_virt_inact))
        mo_coeff_casscf = mo_coeff_out[:, sort_casscf]

        # update orbsym
        if mol.symmetry:
            orbsym_casscf = symm.label_orb_symm(
                mol, mol.irrep_id, mol.symm_orb, mo_coeff_casscf
            )

        # run casscf
        mo_coeff_out = _casscf(
            mol,
            wfnsym_int,
            weights,
            orbsym_casscf,
            hf_guess,
            hf,
            mo_coeff_casscf,
            ref_space,
            act_nelec,
            ncore,
        )

        # reorder mo_coeff
        mo_coeff_out = mo_coeff_out[:, np.argsort(sort_casscf)]

        # orbital symmetries
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff_out)

    return np.asarray(mo_coeff_out, order="C"), orbsym


class _hubbard_PM(lo.pipek.PM):
    """
    this class constructs the site-population tensor for each orbital-pair density
    see: pyscf example - 40-hubbard_model_PM_localization.py
    """

    def atomic_pops(
        self, mol: gto.Mole, mo_coeff: np.ndarray, method: Optional[str] = None
    ) -> np.ndarray:
        """
        this function overwrites the tensor used in the pm cost function and its gradients
        """
        return np.einsum("pi,pj->pij", mo_coeff, mo_coeff)


def _casscf(
    mol: gto.Mole,
    wfnsym: List[int],
    weights: List[float],
    orbsym: np.ndarray,
    hf_guess: bool,
    hf: scf.hf.SCF,
    mo_coeff: np.ndarray,
    ref_space: np.ndarray,
    nelec: np.ndarray,
    ncore: int,
) -> np.ndarray:
    """
    this function returns the results of a casscf calculation
    """
    # init casscf
    cas = mcscf.CASSCF(hf, ref_space.size, nelec)

    # casscf settings
    cas.conv_tol = CONV_TOL
    cas.max_cycle_macro = 500
    cas.frozen = ncore

    # init fcisolver
    if nelec[0] == nelec[1]:
        fcisolver = fci.direct_spin0_symm.FCI(mol)
    else:
        fcisolver = fci.direct_spin1_symm.FCI(mol)

    # create unique list of wfnsym while maintaining order
    unique_wfnsym = list(dict.fromkeys(wfnsym))

    # fci settings
    fcisolver.conv_tol = CONV_TOL
    fcisolver.orbsym = orbsym[ref_space]
    fcisolver.wfnsym = unique_wfnsym[0]
    cas.fcisolver = fcisolver

    # state-averaged casscf
    if len(wfnsym) > 1:

        if len(unique_wfnsym) == 1:

            # state average over all states of same symmetry
            cas.state_average_(weights)

        else:

            # nroots for first fcisolver
            fcisolver.nroots = np.count_nonzero(np.asarray(wfnsym) == unique_wfnsym[0])

            # init list of fcisolvers
            fcisolvers = [fcisolver]

            # loop over symmetries
            for i in range(1, len(unique_wfnsym)):

                # copy fcisolver
                fcisolver_ = copy(fcisolver)

                # wfnsym for fcisolver_
                fcisolver_.wfnsym = unique_wfnsym[i]

                # nroots for fcisolver_
                fcisolver_.nroots = np.count_nonzero(
                    np.asarray(wfnsym) == unique_wfnsym[i]
                )

                # append to fcisolvers
                fcisolvers.append(fcisolver_)

            # state average
            mcscf.state_average_mix_(cas, fcisolvers, weights)

    # hf starting guess
    if hf_guess:
        na = fci.cistring.num_strings(ref_space.size, nelec[0])
        nb = fci.cistring.num_strings(ref_space.size, nelec[1])
        ci0 = np.zeros((na, nb))
        ci0[0, 0] = 1
    else:
        ci0 = None

    # run casscf calc
    cas.kernel(mo_coeff, ci0=ci0)

    # collect ci vectors
    if len(wfnsym) == 1:
        c = [cas.ci]
    else:
        c = cas.ci

    # multiplicity check
    for root in range(len(c)):

        s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)

        if abs((mol.spin + 1) - mult) > SPIN_TOL:

            # fix spin by applyting level shift
            sz = abs(nelec[0] - nelec[1]) * 0.5
            cas.fix_spin_(shift=0.25, ss=sz * (sz + 1.0))

            # run casscf calc
            cas.kernel(mo_coeff, ci0=ci0)

            # collect ci vectors
            if len(wfnsym) == 1:
                c = [cas.ci]
            else:
                c = cas.ci

            # verify correct spin
            for root in range(len(c)):
                s, mult = fcisolver.spin_square(c[root], ref_space.size, nelec)
                assertion(
                    abs((mol.spin + 1) - mult) < SPIN_TOL,
                    f"spin contamination for root entry = {root}, 2*S + 1 = {mult:.6f}",
                )

    # convergence check
    assertion(cas.converged, "CASSCF error: no convergence")

    return np.asarray(cas.mo_coeff, order="C")


def base(
    method: str,
    mol: gto.Mole,
    hf: scf.hf.SCF,
    mo_coeff: np.ndarray,
    orbsym: np.ndarray,
    ncore: int,
    cc_backend: str = "pyscf",
    target: str = "energy",
    hf_prop: Optional[Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
    gauge_origin: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    this function returns base model energy
    """
    # method
    assertion(
        isinstance(method, str),
        "base: base model electronic structure method (first argument) must be a str",
    )
    assertion(
        method in ["ccsd", "ccsd(t)", "ccsdt", "ccsdtq"],
        "base: valid base model electronic structure methods (first argument) are: "
        "ccsd, ccsd(t), ccsdt, and ccsdtq",
    )
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "base: mol (second argument) must be a gto.Mole object",
    )
    # hf
    assertion(
        isinstance(hf, scf.hf.SCF),
        "base: hf (third argument) must be a scf.hf.SCF object",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "base: mo coefficients (fourth argument) must be a np.ndarray",
    )
    # orbsym
    assertion(
        isinstance(orbsym, np.ndarray),
        "base: orbital symmetry (fifth argument) must be a np.ndarray",
    )
    # ncore
    assertion(
        isinstance(ncore, int),
        "base: number of core orbitals (sixth argument) must be an int",
    )
    # cc_backend
    assertion(
        isinstance(cc_backend, str),
        "base: coupled-cluster backend (cc_backend keyword argument) must be a str",
    )
    assertion(
        cc_backend in ["pyscf", "ecc", "ncc"],
        "base: valid coupled-cluster backends (cc_backend keyword argument) are: "
        "pyscf, ecc and ncc",
    )
    if method == "ccsdt":
        assertion(
            cc_backend != "pyscf",
            "base: ccsdt (first argument) is not available with pyscf coupled-cluster "
            "backend (cc_backend keyword argument)",
        )
    if method == "ccsdtq":
        assertion(
            cc_backend == "ncc",
            "base: ccsdtq (first argument) is not available with pyscf and ecc "
            "coupled-cluster backends (cc_backend keyword argument)",
        )
    if mol.spin > 0:
        assertion(
            cc_backend == "pyscf",
            "base: open-shell systems are not available with ecc and ncc "
            "coupled-cluster backends (cc_backend keyword argument)",
        )
    # target
    assertion(
        isinstance(target, str),
        "base: target property (target keyword argument) must be str",
    )
    assertion(
        target in ["energy", "dipole", "rdm12"],
        "base: valid target properties (keyword argument) with coupled-cluster base "
        "methods are: energy (energy) and dipole moment (dipole)",
    )
    if cc_backend in ["ecc", "ncc"]:
        assertion(
            target == "energy",
            "base: calculation of targets (target keyword argument) other than energy "
            "are not possible with ecc and ncc coupled-cluster backends (cc_backend "
            "keyword argument)",
        )
    if target == "dipole":
        # hf_dipole
        assertion(
            isinstance(hf_prop, np.ndarray),
            "base: hartree-fock dipole moment (hf_prop keyword argument) must be a "
            "np.ndarray",
        )
        # gauge_dipole
        assertion(
            isinstance(gauge_origin, np.ndarray),
            "base: gauge origin (gauge_origin keyword argument) must be a np.ndarray",
        )

    # nocc
    nocc = max(mol.nelec)

    # norb
    norb = mol.nao.item()

    # occup
    occup = get_occup(norb, mol.nelec)

    # hcore_ao and eri_ao with 8-fold symmetry
    hcore_ao = hf.get_hcore()
    eri_ao = hf._eri

    # remove symmetry from eri_ao
    eri_ao = ao2mo.restore(1, eri_ao, norb)

    # compute hcore
    hcore = np.einsum("pi,pq,qj->ij", mo_coeff, hcore_ao, mo_coeff)

    # eri_mo w/o symmetry
    eri = ao2mo.incore.full(eri_ao, mo_coeff)

    # compute vhf for core orbitals
    vhf = get_vhf(eri, ncore, norb)

    # restore 4-fold symmetry in eri_mo
    eri = ao2mo.restore(4, eri, norb)

    # compute dipole integrals
    if target == "dipole" and mol.atom:
        dip_ints = dipole_ints(mol, mo_coeff, cast(np.ndarray, gauge_origin))
    else:
        dip_ints = None

    # set core and correlated spaces
    core_idx, corr_idx = core_cas(nocc, np.arange(ncore, nocc), np.arange(nocc, norb))

    # get correlated space h2e
    corr_idx_tril = idx_tril(corr_idx)
    h2e_corr = eri[corr_idx_tril[:, None], corr_idx_tril]

    # determine effective core fock potential
    if core_idx.size > 0:
        core_vhf = np.sum(vhf, axis=0)
    else:
        core_vhf = 0.0

    # get effective h1e for correlated space
    h1e_corr = (hcore + core_vhf)[corr_idx[:, None], corr_idx]

    # nelec
    nelec = get_nelec(occup, corr_idx)

    # run calc
    res = cc_kernel(
        mol.spin,
        occup,
        core_idx,
        corr_idx,
        method,
        cc_backend,
        nelec,
        "can",
        mol.groupname,
        orbsym,
        h1e_corr,
        h2e_corr,
        False,
        target,
        0,
    )

    # collect results
    if target == "energy":
        base_prop = res["energy"]
    elif target == "dipole":
        base_prop = dipole_kernel(
            cast(np.ndarray, dip_ints),
            occup,
            corr_idx,
            res["rdm1"],
            hf_dipole=cast(np.ndarray, hf_prop),
        )
    elif target == "rdm12":
        base_prop = res["rdm1"], res["rdm2"]

    return base_prop


def linear_orbsym(mol: gto.Mole, mo_coeff: np.ndarray) -> np.ndarray:
    """
    returns orbsym in linear point groups for pi pruning
    """
    # mol
    assertion(
        isinstance(mol, gto.Mole),
        "linear orbsym: mol (first argument) must be a gto.Mole object",
    )
    assertion(
        symm.addons.std_symb(mol.groupname) in ["D2h", "C2v"],
        "linear orbsym: only works for linear D2h and C2v symmetries",
    )
    # mo_coeff
    assertion(
        isinstance(mo_coeff, np.ndarray),
        "linear orbsym: mo coefficients (second argument) must be a np.ndarray",
    )

    # recast mol in parent point group (dooh/coov)
    mol_parent = mol.copy()
    parent_group = "Dooh" if mol.groupname == "D2h" else "Coov"
    mol_parent = mol_parent.build(0, 0, symmetry=parent_group)

    orbsym_parent = symm.label_orb_symm(
        mol_parent, mol_parent.irrep_id, mol_parent.symm_orb, mo_coeff
    )

    return orbsym_parent
