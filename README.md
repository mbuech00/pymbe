![](doc/logo/pymbe_logo.png "PyMBE")

PyMBE: A Many-Body Expanded Correlation Code 
============================================

News
----

* **October 2019**: The PyMBE code is now open source under the MIT license.


Authors
-------

* Dr. Janus Juul Eriksen (Technical University of Denmark, main author)
* Jonas Greiner (Johannes Gutenberg University Mainz, contributor). 

Prerequisites
-------------

* [Python](https://www.python.org/) 3.7 or higher
* [PySCF](https://pyscf.github.io/) 1.6 or higher and its requirements
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/) 3.0.0 or higher built upon an MPI-3 library
* [pytest](https://docs.pytest.org/) for unit tests


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

PyMBE expects PySCF to be properly exported to Python, for which one needs to set environment variable `$PYTHONPATH`. Furthermore, the mpi4py implementation of the MPI standard needs to be installed, built upon an MPI-3 library.\
Once these requirements are satisfied, PyMBE (located in `$PYMBEPATH`) is simply invoked by the following command:

```
mpiexec -np N $PYMBEPATH/src/main.py
```

with an input file `input` placed within the same directory. See the [examples](examples/) section for a range of example inputs.\
Restart files are automatically generated unless otherwise requested in a dedicated directory `rst` within `$WORKDIR`, 
which is deleted in case of successful termination of PyMBE. 
The output and results of a calculation are stored in a dedicated directory `output` within `$WORKDIR`.

Documentation
-------------

The PyMBE code is deliberately left undocumented, but the [pytest](https://docs.pytest.org/) unit tests and the type hints using the Python [typing](https://docs.python.org/3/library/typing.html) module can be used to comprehend the usage of the different functions and their arguments.


Tutorials
---------

None at the moment, but please have a look at the various [examples](examples/) that accompany the code.


Citing PyMBE
------------

The following papers document the development of the PyMBE code and the theory it implements:

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

