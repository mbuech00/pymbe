#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from pyscf import gto, symm
from typing import TYPE_CHECKING

try:
    import matplotlib

    PLT_FOUND = True
except (ImportError, OSError):
    pass
    PLT_FOUND = False
if PLT_FOUND:
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    try:
        import seaborn as sns

        SNS_FOUND = True
    except (ImportError, OSError):
        pass
        SNS_FOUND = False

from pymbe.output import main_header
from pymbe.tools import intervals, time_str, nelec

if TYPE_CHECKING:

    from typing import Tuple, Any, Dict, Union, List, Optional

    from pymbe.parallel import MPICls
    from pymbe.expansion import ExpCls


# results parameters
DIVIDER = f"{('-' * 137):^143}"


def tot_prop(exp: ExpCls) -> Union[float, np.ndarray]:
    """
    this function returns the total property
    """
    if exp.target == "energy":
        prop = _energy(exp.mbe_tot_prop, exp.hf_prop, exp.base_prop, exp.ref_prop)[-1]
    elif exp.target == "excitation":
        prop = _excitation(exp.mbe_tot_prop, exp.ref_prop)[-1]
    elif exp.target == "dipole":
        prop = (
            exp.nuc_dipole
            - _dipole(exp.mbe_tot_prop, exp.hf_prop, exp.base_prop, exp.ref_prop)[-1, :]
        )
    elif exp.target == "trans":
        prop = _trans(exp.mbe_tot_prop, exp.ref_prop)[-1, :]

    return prop


def print_results(mol: Optional[gto.Mole], mpi: MPICls, exp: ExpCls) -> str:
    """
    this function handles printing of results
    """
    # print header
    string = main_header() + "\n\n"

    # print atom info
    if mol and mol.atom:
        string += _atom(mol) + "\n\n"

    # print summary
    string += _summary_prt(mpi, exp) + "\n\n"

    # print timings
    string += _timings_prt(exp, exp.method) + "\n\n"

    # print and plot results
    if exp.target == "energy":
        string += _energy_prt(
            exp.method,
            exp.fci_state_root,
            exp.mbe_tot_prop,
            exp.hf_prop,
            exp.base_prop,
            exp.ref_prop,
            exp.min_order,
            exp.final_order,
        )
    elif exp.target == "excitation":
        string += _excitation_prt(
            exp.fci_state_root,
            exp.mbe_tot_prop,
            exp.ref_prop,
            exp.min_order,
            exp.final_order,
        )
    elif exp.target == "dipole":
        string += _dipole_prt(
            exp.fci_state_root,
            exp.nuc_dipole,
            exp.mbe_tot_prop,
            exp.hf_prop,
            exp.base_prop,
            exp.ref_prop,
            exp.min_order,
            exp.final_order,
        )
    elif exp.target == "trans":
        string += _trans_prt(
            exp.fci_state_root,
            exp.mbe_tot_prop,
            exp.ref_prop,
            exp.min_order,
            exp.final_order,
        )

    return string


def plot_results(exp: ExpCls) -> matplotlib.figure.Figure:
    """
    this function handles plotting of results
    """
    # check if matplotlib is available
    if not PLT_FOUND:
        raise ModuleNotFoundError("No module named matplotlib")

    # print and plot results
    if exp.target == "energy":
        fig = _energy_plot(
            exp.fci_state_root,
            exp.mbe_tot_prop,
            exp.hf_prop,
            exp.base_prop,
            exp.ref_prop,
            exp.min_order,
            exp.final_order,
        )
    elif exp.target == "excitation":
        fig = _excitation_plot(
            exp.fci_state_root,
            exp.mbe_tot_prop,
            exp.ref_prop,
            exp.min_order,
            exp.final_order,
        )
    elif exp.target == "dipole":
        fig = _dipole_plot(
            exp.fci_state_root,
            exp.nuc_dipole,
            exp.mbe_tot_prop,
            exp.hf_prop,
            exp.base_prop,
            exp.ref_prop,
            exp.min_order,
            exp.final_order,
        )
    elif exp.target == "trans":
        fig = _trans_plot(
            exp.fci_state_root,
            exp.mbe_tot_prop,
            exp.ref_prop,
            exp.min_order,
            exp.final_order,
        )

    return fig


def _atom(mol: gto.Mole) -> str:
    """
    this function returns the molecular geometry
    """
    # print atom
    string: str = DIVIDER[:39] + "\n"
    string += f"{'geometry':^45}\n"
    string += DIVIDER[:39] + "\n"
    molecule = gto.tostring(mol).split("\n")
    for i in range(len(molecule)):
        atom = molecule[i].split()
        for j in range(1, 4):
            atom[j] = float(atom[j])
        string += (
            f"   {atom[0]:<3s} {atom[1]:>10.5f} {atom[2]:>10.5f} {atom[3]:>10.5f}\n"
        )
    string += DIVIDER[:39] + "\n"
    return string


def _model(method: str) -> str:
    """
    this function returns the expansion model
    """
    string = f"{method.upper()}"

    return string


def _state(spin: int, root: int) -> str:
    """
    this function returns the state of interest
    """
    string = f"{root}"
    if spin == 0:
        string += " (singlet)"
    elif spin == 1:
        string += " (doublet)"
    elif spin == 2:
        string += " (triplet)"
    elif spin == 3:
        string += " (quartet)"
    elif spin == 4:
        string += " (quintet)"
    else:
        string += f" ({spin + 1})"
    return string


def _base(base_method: Optional[str]) -> str:
    """
    this function returns the base model
    """
    if base_method is None:
        return "none"
    else:
        return base_method.upper()


def _system(nocc: int, ncore: int, norb: int) -> str:
    """
    this function returns the system size
    """
    return f"{2 * (nocc - ncore)} e in {norb - ncore} o"


def _solver(method: str, fci_solver: str) -> str:
    """
    this function returns the chosen fci solver
    """
    if method != "fci":
        return "none"
    else:
        if fci_solver == "pyscf_spin0":
            return "PySCF (spin0)"
        elif fci_solver == "pyscf_spin1":
            return "PySCF (spin1)"
        else:
            raise NotImplementedError("unknown solver")


def _active_space(occup: np.ndarray, ref_space: np.ndarray) -> str:
    """
    this function returns the active space
    """
    act_n_elec = nelec(occup, ref_space)
    string = f"{act_n_elec[0] + act_n_elec[1]} e, {ref_space.size} o"
    return string


def _active_orbs(ref_space: np.ndarray) -> str:
    """
    this function returns the orbitals of the active space
    """
    if ref_space.size == 0:
        return "none"

    # init string
    string = "["
    # divide ref_space into intervals
    ref_space_ints = [i for i in intervals(ref_space)]

    for idx, i in enumerate(ref_space_ints):
        elms = f"{i[0]}-{i[1]}" if len(i) > 1 else f"{i[0]}"
        string += f"{elms}," if idx < len(ref_space_ints) - 1 else f"{elms}"
    string += "]"

    return string


def _orbs(orb_type: str) -> str:
    """
    this function returns the choice of orbitals
    """
    if orb_type == "can":
        return "canonical"
    elif orb_type == "ccsd":
        return "CCSD NOs"
    elif orb_type == "ccsd(t)":
        return "CCSD(T) NOs"
    elif orb_type == "local":
        return "pipek-mezey"
    elif orb_type == "casscf":
        return "casscf"
    else:
        raise NotImplementedError("unknown orbital basis")


def _mpi(num_masters: int, global_size: int) -> str:
    """
    this function returns the mpi information
    """
    return f"{num_masters} & {global_size - num_masters}"


def _point_group(point_group: str) -> str:
    """
    this function returns the point group symmetry
    """
    return point_group


def _symm(method: str, point_group: str, fci_state_sym: int, pi_prune: bool) -> str:
    """
    this function returns the symmetry of the wavefunction in the computational point
    group
    """
    if method == "fci":
        string = (
            symm.addons.irrep_id2name(point_group, fci_state_sym)
            + "("
            + point_group
            + ")"
        )
        if pi_prune:
            string += " (pi)"
        return string
    else:
        return "unknown"


def _energy(
    corr_energy: List[np.ndarray],
    hf_energy: np.ndarray,
    base_energy: np.ndarray,
    ref_energy: np.ndarray,
) -> np.ndarray:
    """
    this function returns the final total energy
    """
    tot_energy = np.array(corr_energy)
    tot_energy += hf_energy
    tot_energy += base_energy
    tot_energy += ref_energy

    return tot_energy.flatten()


def _excitation(corr_exc: List[np.ndarray], ref_exc: np.ndarray) -> np.ndarray:
    """
    this function returns the final excitation energy
    """
    tot_exc = np.array(corr_exc)
    tot_exc += ref_exc

    return tot_exc.flatten()


def _dipole(
    corr_dipole: List[np.ndarray],
    hf_dipole: np.ndarray,
    base_dipole: np.ndarray,
    ref_dipole: np.ndarray,
) -> np.ndarray:
    """
    this function returns the final molecular dipole moment
    """
    tot_dipole = np.array(corr_dipole)
    tot_dipole += hf_dipole
    tot_dipole += base_dipole
    tot_dipole += ref_dipole

    return tot_dipole


def _trans(corr_trans: List[np.ndarray], ref_trans: np.ndarray) -> np.ndarray:
    """
    this function returns the final molecular transition dipole moment
    """
    tot_trans = np.array(corr_trans)
    tot_trans += ref_trans

    return tot_trans


def _time(time: Dict[str, List[float]], comp: str, idx: int) -> str:
    """
    this function returns the final timings in (HHH : MM : SS) format
    """
    # init time
    if comp in ["mbe", "purge"]:
        req_time = time[comp][idx]
    elif comp == "sum":
        req_time = time["mbe"][idx] + time["purge"][idx]
    elif comp in ["tot_mbe", "tot_purge"]:
        req_time = np.sum(time[comp[4:]])
    elif comp == "tot_sum":
        req_time = np.sum(time["mbe"]) + np.sum(time["purge"])
    return time_str(req_time)


def _summary_prt(mpi: MPICls, exp: ExpCls) -> str:
    """
    this function returns the summary table
    """
    hf_prop: Union[float, np.floating]
    base_prop: Union[float, np.floating]
    mbe_tot_prop: Union[float, np.floating]

    if exp.target == "energy":
        hf_prop = exp.hf_prop.item()
        base_prop = exp.hf_prop.item() + exp.base_prop.item()
        energy = _energy(exp.mbe_tot_prop, exp.hf_prop, exp.base_prop, exp.ref_prop)
        mbe_tot_prop = energy[-1]
    elif exp.target == "dipole":
        hf_prop = np.linalg.norm(exp.nuc_dipole - exp.hf_prop)
        base_prop = np.linalg.norm(exp.nuc_dipole - (exp.hf_prop + exp.base_prop))
        dipole = _dipole(exp.mbe_tot_prop, exp.hf_prop, exp.base_prop, exp.ref_prop)
        mbe_tot_prop = np.linalg.norm(exp.nuc_dipole - dipole[-1, :])
    elif exp.target == "excitation":
        hf_prop = 0.0
        base_prop = 0.0
        mbe_tot_prop = _excitation(exp.mbe_tot_prop, exp.ref_prop)[-1]
    else:
        hf_prop = 0.0
        base_prop = 0.0
        mbe_tot_prop = np.linalg.norm(_trans(exp.mbe_tot_prop, exp.ref_prop)[-1, :])

    string: str = DIVIDER + "\n"
    string += (
        f"{'':3}{'molecular information':^45}{'|':1}"
        f"{'expansion information':^45}{'|':1}"
        f"{'calculation information':^45}\n"
    )

    string += DIVIDER + "\n"
    string += (
        f"{'':5}{'system size':<24}{'=':1}{'':2}"
        f"{_system(exp.nocc, exp.ncore, exp.norb):<16s}{'|':1}{'':2}"
        f"{'expansion model':<24}{'=':1}{'':2}"
        f"{_model(exp.method):<16s}{'|':1}{'':2}"
        f"{'mpi masters & slaves':<24}{'=':1}{'':2}"
        f"{_mpi(mpi.num_masters, mpi.global_size):<16s}\n"
    )
    string += (
        f"{'':5}{'state (mult.)':<24}{'=':1}{'':2}"
        f"{_state(exp.spin, exp.fci_state_root):<16s}{'|':1}{'':2}"
        f"{'reference space':<24}{'=':1}{'':2}"
        f"{_active_space(exp.occup, exp.ref_space):<16s}{'|':1}{'':2}"
        f"{('Hartree-Fock ' + exp.target):<24}{'=':1}{'':2}"
        f"{hf_prop:<16.6f}\n"
    )
    string += (
        f"{'':5}{'orbitals':<24}{'=':1}{'':2}"
        f"{_orbs(exp.orb_type):<16s}{'|':1}{'':2}"
        f"{'reference orbs.':<24}{'=':1}{'':2}"
        f"{_active_orbs(exp.ref_space):<16s}{'|':1}{'':2}"
        f"{('base model ' + exp.target):<24}{'=':1}{'':2}"
        f"{base_prop:<16.6f}\n"
    )
    string += (
        f"{'':5}{'point group':<24}{'=':1}{'':2}"
        f"{_point_group(exp.point_group):<16s}{'|':1}{'':2}"
        f"{'base model':<24}{'=':1}{'':2}"
        f"{_base(exp.base_method):<16s}{'|':1}{'':2}"
        f"{('MBE total ' + exp.target):<24}{'=':1}{'':2}"
        f"{mbe_tot_prop:<16.6f}\n"
    )
    string += (
        f"{'':5}{'FCI solver':<24}{'=':1}{'':2}"
        f"{_solver(exp.method, exp.fci_solver):<16s}{'|':1}{'':2}"
        f"{'':<24}{'':1}{'':2}"
        f"{'':<16s}{'|':1}{'':2}"
        f"{('total time'):<24}{'=':1}{'':2}"
        f"{_time(exp.time, 'tot_sum', -1):<16s}\n"
    )
    string += (
        f"{'':5}{'wave funct. symmetry':<24}{'=':1}{'':2}"
        f"{_symm(exp.method, exp.point_group, exp.fci_state_sym, exp.pi_prune):<16s}"
        f"{'|':1}{'':2}"
        f"{'':<24}{'':1}{'':2}"
        f"{'':<16s}{'|':1}{'':2}"
        f"{(''):<24}{'':1}{'':2}"
        f"{'':<16s}\n"
    )

    string += DIVIDER + "\n"

    return string


def _timings_prt(exp: ExpCls, method: str) -> str:
    """
    this function returns the timings table
    """
    string: str = DIVIDER[:106] + "\n"
    string += f"{f'MBE-{method.upper()} timings':^106}\n"

    string += DIVIDER[:106] + "\n"
    string += (
        f"{'':3}{'MBE order':^14}{'|':1}{'MBE':^18}{'|':1}{'purging':^18}{'|':1}"
        f"{'sum':^18}{'|':1}{'calculations':^18}{'|':1}{'in %':^13}\n"
    )

    string += DIVIDER[:106] + "\n"
    for i, j in enumerate(range(exp.min_order, exp.final_order + 1)):
        calc_i = exp.n_tuples["calc"][i]
        rel_i = exp.n_tuples["calc"][i] / exp.n_tuples["theo"][i] * 100.0
        calc_tot = sum(exp.n_tuples["calc"][: i + 1])
        rel_tot = calc_tot / sum(exp.n_tuples["theo"][: i + 1]) * 100.0
        string += (
            f"{'':3}{j:>8d}{'':6}{'|':1}"
            f"{_time(exp.time, 'mbe', i):>16s}{'':2}{'|':1}"
            f"{_time(exp.time, 'purge', i):>16s}{'':2}{'|':1}"
            f"{_time(exp.time, 'sum', i):>16s}{'':2}{'|':1}"
            f"{calc_i:>16d}{'':2}{'|':1}"
            f"{rel_i:>10.2f}\n"
        )

    string += DIVIDER[:106] + "\n"
    string += (
        f"{'':3}{'total':^14s}{'|':1}"
        f"{_time(exp.time, 'tot_mbe', -1):>16s}{'':2}{'|':1}"
        f"{_time(exp.time, 'tot_purge', -1):>16s}{'':2}{'|':1}"
        f"{_time(exp.time, 'tot_sum', -1):>16s}{'':2}{'|':1}"
        f"{calc_tot:>16d}{'':2}{'|':1}"
        f"{rel_tot:>10.2f}\n"
    )

    string += DIVIDER[:106] + "\n"

    return string


def _energy_prt(
    method: str,
    root: int,
    corr_energy: List[np.ndarray],
    hf_energy: np.ndarray,
    base_energy: np.ndarray,
    ref_energy: np.ndarray,
    min_order: int,
    final_order: int,
) -> str:
    """
    this function returns the energies table
    """
    string: str = DIVIDER[:67] + "\n"
    string += f"{f'MBE-{method.upper()} energy (root = {root})':^73}\n"

    string += DIVIDER[:67] + "\n"
    string += (
        f"{'':3}{'MBE order':^14}{'|':1}"
        f"{'total energy':^22}{'|':1}"
        f"{'correlation energy':^26}\n"
    )

    string += DIVIDER[:67] + "\n"
    string += (
        f"{'':3}{'ref':^14s}{'|':1}"
        f"{(hf_energy.item() + ref_energy.item()):^22.6f}{'|':1}"
        f"{ref_energy.item():^26.5e}\n"
    )

    string += DIVIDER[:67] + "\n"
    energy = _energy(corr_energy, hf_energy, base_energy, ref_energy)
    for i, j in enumerate(range(min_order, final_order + 1)):
        string += (
            f"{'':3}{j:>8d}{'':6}{'|':1}"
            f"{energy[i]:^22.6f}{'|':1}"
            f"{(energy[i] - hf_energy):^26.5e}\n"
        )

    string += DIVIDER[:67] + "\n"

    return string


def _energy_plot(
    root: int,
    corr_energy: List[np.ndarray],
    hf_energy: np.ndarray,
    base_energy: np.ndarray,
    ref_energy: np.ndarray,
    min_order: int,
    final_order: int,
) -> matplotlib.figure.Figure:
    """
    this function plots the energies
    """
    # set seaborn
    if SNS_FOUND:
        sns.set(style="darkgrid", palette="Set2", font="DejaVu Sans")

    # set subplot
    fig, ax = plt.subplots()

    # plot results
    ax.plot(
        np.arange(min_order, final_order + 1),
        _energy(corr_energy, hf_energy, base_energy, ref_energy),
        marker="x",
        linewidth=2,
        mew=1,
        color="xkcd:kelly green",
        linestyle="-",
        label=f"state {root}",
    )

    # set x limits
    ax.set_xlim([0.5, final_order + 1 - 0.5])

    # turn off x-grid
    ax.xaxis.grid(False)

    # set labels
    ax.set_xlabel("Expansion order")
    ax.set_ylabel("Energy (in au)")

    # force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # despine
    if SNS_FOUND:
        sns.despine()

    # set legend
    ax.legend(loc=1, frameon=False)

    return fig


def _excitation_prt(
    root: int,
    corr_exc: List[np.ndarray],
    ref_exc: np.ndarray,
    min_order: int,
    final_order: int,
) -> str:
    """
    this function returns the excitation energies table
    """
    string: str = DIVIDER[:43] + "\n"
    string += f"{f'MBE excitation energy (roots = 0 > {root})':^49}\n"

    string += DIVIDER[:43] + "\n"
    string += f"{'':3}{'MBE order':^14}{'|':1}{'excitation energy':^25}\n"

    string += DIVIDER[:43] + "\n"
    string += f"{'':3}{'ref':^14s}{'|':1}{ref_exc.item():^25.5e}\n"

    string += DIVIDER[:43] + "\n"
    excitation = _excitation(corr_exc, ref_exc)
    for i, j in enumerate(range(min_order, final_order + 1)):
        string += f"{'':3}{j:>8d}{'':6}{'|':1}{excitation[i]:^25.5e}\n"

    string += DIVIDER[:43] + "\n"

    return string


def _excitation_plot(
    root: int,
    corr_exc: List[np.ndarray],
    ref_exc: np.ndarray,
    min_order: int,
    final_order: int,
) -> matplotlib.figure.Figure:
    """
    this function plots the excitation energies
    """
    # set seaborn
    if SNS_FOUND:
        sns.set(style="darkgrid", palette="Set2", font="DejaVu Sans")

    # set subplot
    fig, ax = plt.subplots()

    # plot results
    ax.plot(
        np.arange(min_order, final_order + 1),
        _excitation(corr_exc, ref_exc),
        marker="x",
        linewidth=2,
        mew=1,
        color="xkcd:dull blue",
        linestyle="-",
        label=f"excitation 0 -> {root}",
    )

    # set x limits
    ax.set_xlim([0.5, final_order + 1 - 0.5])

    # turn off x-grid
    ax.xaxis.grid(False)

    # set labels
    ax.set_xlabel("Expansion order")
    ax.set_ylabel("Excitation energy (in au)")

    # force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # despine
    if SNS_FOUND:
        sns.despine()

    # set legend
    ax.legend(loc=1, frameon=False)

    return fig


def _dipole_prt(
    root: int,
    nuc_dipole: np.ndarray,
    corr_dipole: List[np.ndarray],
    hf_dipole: np.ndarray,
    base_dipole: np.ndarray,
    ref_dipole: np.ndarray,
    min_order: int,
    final_order: int,
) -> str:
    """
    this function returns the dipole moments table
    """
    string: str = DIVIDER[:82] + "\n"
    string += f"{f'MBE dipole moment (root = {root})':^86}\n"

    string += DIVIDER[:82] + "\n"
    string += (
        f"{'':3}{'MBE order':^14}{'|':1}"
        f"{'dipole components (x,y,z)':^42}{'|':1}"
        f"{'dipole moment':^21}\n"
    )

    string += DIVIDER[:82] + "\n"
    tot_ref_dipole: np.ndarray = hf_dipole + base_dipole + ref_dipole
    string += (
        f"{'':3}{'ref':^14s}{'|':1}"
        f"{(nuc_dipole[0] - tot_ref_dipole[0]):>13.6f}"
        f"{(nuc_dipole[1] - tot_ref_dipole[1]):^16.6f}"
        f"{(nuc_dipole[2] - tot_ref_dipole[2]):<13.6f}{'|':1}"
        f"{np.linalg.norm(nuc_dipole - tot_ref_dipole):^21.6f}\n"
    )

    string += DIVIDER[:82] + "\n"
    dipole = _dipole(corr_dipole, hf_dipole, base_dipole, ref_dipole)
    for i, j in enumerate(range(min_order, final_order + 1)):
        string += (
            f"{'':3}{j:>8d}{'':6}{'|':1}"
            f"{(nuc_dipole[0] - dipole[i, 0]):>13.6f}"
            f"{(nuc_dipole[1] - dipole[i, 1]):^16.6f}"
            f"{(nuc_dipole[2] - dipole[i, 2]):<13.6f}{'|':1}"
            f"{np.linalg.norm(nuc_dipole - dipole[i, :]):^21.6f}\n"
        )

    string += DIVIDER[:82] + "\n"

    return string


def _dipole_plot(
    root: int,
    nuc_dipole: np.ndarray,
    corr_dipole: List[np.ndarray],
    hf_dipole: np.ndarray,
    base_dipole: np.ndarray,
    ref_dipole: np.ndarray,
    min_order: int,
    final_order: int,
) -> matplotlib.figure.Figure:
    """
    this function plots the dipole moments
    """
    # set seaborn
    if SNS_FOUND:
        sns.set(style="darkgrid", palette="Set2", font="DejaVu Sans")

    # set subplot
    fig, ax = plt.subplots()

    # array of total MBE dipole moment
    dipole = _dipole(corr_dipole, hf_dipole, base_dipole, ref_dipole)
    dipole_arr = np.empty(dipole.shape[0], dtype=np.float64)
    for i in range(dipole.shape[0]):
        dipole_arr[i] = np.linalg.norm(nuc_dipole - dipole[i, :])

    # plot results
    ax.plot(
        np.arange(min_order, final_order + 1),
        dipole_arr,
        marker="*",
        linewidth=2,
        mew=1,
        color="xkcd:salmon",
        linestyle="-",
        label=f"state {root}",
    )

    # set x limits
    ax.set_xlim([0.5, final_order + 1 - 0.5])

    # turn off x-grid
    ax.xaxis.grid(False)

    # set labels
    ax.set_xlabel("Expansion order")
    ax.set_ylabel("Dipole moment (in au)")

    # force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # despine
    if SNS_FOUND:
        sns.despine()

    # set legend
    ax.legend(loc=1, frameon=False)

    return fig


def _trans_prt(
    root: int,
    corr_trans: List[np.ndarray],
    ref_trans: np.ndarray,
    min_order: int,
    final_order: int,
) -> str:
    """
    this function returns the transition dipole moments and oscillator strengths table
    """
    string: str = DIVIDER[:82] + "\n"
    string += f"{f'MBE trans. dipole moment (roots 0 > {root})':^86}\n"

    string += DIVIDER[:82] + "\n"
    string += (
        f"{'':3}{'MBE order':^14}{'|':1}"
        f"{'dipole components (x,y,z)':^42}{'|':1}"
        f"{'dipole moment':^21}\n"
    )

    string += DIVIDER[:82] + "\n"
    tot_ref_trans: np.ndarray = ref_trans
    string += (
        f"{'':3}{'ref':^14s}{'|':1}"
        f"{tot_ref_trans[0]:>13.6f}"
        f"{tot_ref_trans[1]:^16.6f}"
        f"{tot_ref_trans[2]:<13.6f}{'|':1}"
        f"{np.linalg.norm(tot_ref_trans[:]):^21.6f}\n"
    )

    string += DIVIDER[:82] + "\n"
    trans = _trans(corr_trans, ref_trans)
    for i, j in enumerate(range(min_order, final_order + 1)):
        string += (
            f"{'':3}{j:>8d}{'':6}{'|':1}"
            f"{trans[i, 0]:>13.6f}"
            f"{trans[i, 1]:^16.6f}"
            f"{trans[i, 2]:<13.6f}{'|':1}"
            f"{np.linalg.norm(trans[i, :]):^21.6f}\n"
        )

    string += DIVIDER[:82] + "\n"

    return string


def _trans_plot(
    root: int,
    corr_trans: List[np.ndarray],
    ref_trans: np.ndarray,
    min_order: int,
    final_order: int,
) -> matplotlib.figure.Figure:
    """
    this function plots the transition dipole moments
    """
    # set seaborn
    if SNS_FOUND:
        sns.set(style="darkgrid", palette="Set2", font="DejaVu Sans")

    # set subplot
    fig, ax = plt.subplots()

    # array of total MBE transition dipole moment
    trans = _trans(corr_trans, ref_trans)
    trans_arr = np.empty(trans.shape[0], dtype=np.float64)
    for i in range(trans.shape[0]):
        trans_arr[i] = np.linalg.norm(trans[i, :])

    # plot results
    ax.plot(
        np.arange(min_order, final_order + 1),
        trans_arr,
        marker="s",
        linewidth=2,
        mew=1,
        color="xkcd:dark magenta",
        linestyle="-",
        label=f"excitation 0 -> {root}",
    )

    # set x limits
    ax.set_xlim([0.5, final_order + 1 - 0.5])

    # turn off x-grid
    ax.xaxis.grid(False)

    # set labels
    ax.set_xlabel("Expansion order")
    ax.set_ylabel("Transition dipole (in au)")

    # force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # despine
    if SNS_FOUND:
        sns.despine()

    # set legend
    ax.legend(loc=1, frameon=False)

    return fig
