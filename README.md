![](doc/logo/pymbe_logo.png "PyMBE")

PyMBE: A Many-Body Expanded Correlation Code 
============================================

News
----

* **October 2019**: The PyMBE code is now open source under the MIT license.


Authors
-------

* Dr. Janus Juul Eriksen (Technical University of Denmark, main author)
* Jonas Greiner (Johannes Gutenberg University Mainz, author). 

Prerequisites
-------------

* [Python](https://www.python.org/) 3.7 or higher
* [PySCF](https://pyscf.github.io/) 1.6 or higher and its requirements
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/) 3.0.0 or higher built upon an MPI-3 library
* [pytest](https://docs.pytest.org/) for unit and system tests


Features
--------

* Many-Body Expanded Full Configuration Interaction.
    - Calculations of Ground State Correlation Energies, Excitation Energies, and (Transition) Dipole Moments.
    - Coupled Cluster Intermediate Base Models: CCSD & CCSD(T).
    - Arbitrary Reference Spaces.
    - Full Abelian Point Group Symmetry.
    - Closed- and Open-Shell Implementations.
    - Restricted Hartree-Fock (RHF/ROHF) and Complete Active Space Self-Consistent Field (CASSCF) Molecular Orbitals.
* Massively Parallel Implementation.
* MPI-3 Shared Memory Parallelism.


Usage
-----

PyMBE expects both itself and PySCF to be properly exported to Python. This can either
be achieved by installing through pip (in the case of PySCF), having a 
pymbe.pth/pyscf.pth file with the corresponding path in the lib/Python3.X/site-packages 
directory of your Python distribution or by including the paths in the environment 
variable `$PYTHONPATH`. Furthermore, the mpi4py implementation of the MPI standard 
needs to be installed, built upon an MPI-3 library.\
Once these requirements are satisfied, PyMBE can be started by importing the 
MBE class and creating a MBE object while passing input data and keywords as 
keyword arguments. Possible keyword arguments are:

* **expansion model**
    * method: electronic structure method (fci, ccsdtq, ccsdt, ccsd(t), ccsd)
    * cc_backend: coupled-cluster backend (pyscf, ecc, ncc)
    * hf_guess: hartree-fock initial guess
- **target property**
    - target: expansion target property (energy, dipole, excitation, trans, rdm12)
* **system**
    * mol: [pyscf](https://pyscf.github.io/) gto.Mole object
    * nuc_energy: nuclear energy
    * nuc_dipole: nuclear dipole moment
    * ncore: number of core orbitals
    * norb: number of orbitals
    * nelec: number of electrons
    * point_group: point group
    * orbsym: orbital symmetry
    * fci_state_sym: state wavefunction symmetry
    * fci_state_root: target state
- **hf calculation**
    - hf_prop: hartree-fock property
* **orbital representation**
    * orb_type: orbital representation
- **integrals**
    - hcore: core hamiltonian integrals
    - vhf: hartree-fock potential
    - eri: electron repulsion integrals
    - dipole_ints: dipole integrals
* **reference space**
    * ref_space: reference space
    * ref_prop: reference space property
- **base model**
    - base_method: base model electronic structure method (ccsdtq, ccsdt, ccsd(t), ccsd)
    - base_prop: base model property
* **screening**
    * screen_start: screening start order
    * screen_perc: screening threshold
    * max_order: maximum expansion order
- **restart**
    - rst: restart logical
    - rst_freq: restart frequency
* **verbose**
    * verbose: verbose option (0: only error output, 1: standard output, 2: MBE debug output, 3: backend debug output)
- **pi-pruning**
    - pi_prune: pruning of pi-orbitals
    - orbsym_linear: linear point group orbital symmetry

Many of these arguments have default values set or are optional depending on 
the calculation type. See the [examples](pymbe/examples/) section for a range of 
example scripts.\
The calculation is started by calling the kernel() member function of the MBE 
object. Restart files are automatically generated (unless otherwise requested 
through the rst keyword argument) in a dedicated directory `rst` within 
`$WORKDIR`, which is deleted in case of successful termination of PyMBE. When 
restarting a calulation from the restart files, the kernel function can be 
called without passing any keyword arguments to the MBE object. The kernel() 
function returns the total target property. The program can also be called in 
parallel by calling the kernel function in multiple MPI processes (e.g. using 
the mpiexec command). Only the keyword arguments of the MBE object on the 
global master will be used during the calculation.\
The results of a PyMBE caluculation can be printed using the results() member
function of the MBE object. This function returns the calculation parameters 
summarized in a string.\
The results can also be plotted using the plot() member function of the MBE
object. This function returns a [matplotlib](https://matplotlib.org) 
figure.Figure object.

Warning: All open-shell CC calculations with the pyscf backend directly estimate
the unrestricted CC property on the basis of a ROHF reference function instead
of the fully restricted CC property.

Documentation
-------------

The PyMBE code is deliberately left undocumented, but the 
[pytest](https://docs.pytest.org/) unit tests and the type hints using the 
Python [typing](https://docs.python.org/3/library/typing.html) module can be 
used to comprehend the usage of the different functions and their arguments.


Tutorials
---------

None at the moment, but please have a look at the various 
[examples](pymbe/examples/) that accompany the code.


Citing PyMBE
------------

The following papers document the development of the PyMBE code and the theory 
it implements:

* Incremental Treatments of the Full Configuration Interaction Problem\
Eriksen, J. J., Gauss, J.\
[WIREs Comput. Mol. Sci., e1525 (2021)](https://onlinelibrary.wiley.com/doi/full/10.1002/wcms.1525?af=R) ([arXiv:2012.07371](https://arxiv.org/abs/2012.07371))

* Ground and Excited State First-Order Properties in Many-Body Expanded Full Configuration Interaction\
Eriksen, J. J., Gauss, J.\
[J. Chem. Phys., 153, 154107 (2020)](https://aip.scitation.org/doi/10.1063/5.0024791) ([arXiv:2008.03610](https://arxiv.org/abs/2008.03610))

* Generalized Many-Body Expanded Full Configuration Interaction Theory\
Eriksen, J. J., Gauss, J.\
[J. Phys. Chem. Lett., 10, 7910 (2019)](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.9b02968) ([arXiv:1910.03527](https://arxiv.org/abs/1910.03527))

* Many-Body Expanded Full Configuration Interaction. II. Strongly Correlated Regime\
Eriksen, J. J., Gauss, J.\
[J. Chem. Theory Comput., 15, 4873 (2019)](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00456) ([arXiv: 1905.02786](https://arxiv.org/abs/1905.02786))

* Many-Body Expanded Full Configuration Interaction. I. Weakly Correlated Regime\
Eriksen, J. J., Gauss, J.\
[J. Chem. Theory Comput., 14, 5180 (2018)](https://pubs.acs.org/doi/10.1021/acs.jctc.8b00680) ([arXiv: 1807.01328](https://arxiv.org/abs/1807.01328))

* Virtual Orbital Many-Body Expansions: A Possible Route Towards the Full Configuration Interaction Limit\
Eriksen, J. J., Lipparini, F.; Gauss, J.\
[J. Phys. Chem. Lett., 8, 4633 (2017)](https://pubs.acs.org/doi/10.1021/acs.jpclett.7b02075) ([arXiv: 1708.02103](https://arxiv.org/abs/1708.02103))

Bug reports
-----------

Janus Juul Eriksen: janus [at] kemi [dot] dtu [dot] dk

