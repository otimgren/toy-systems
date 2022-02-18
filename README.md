

# toy-systems
Code for making quantum physics toy models.

![image](https://user-images.githubusercontent.com/34794187/154632686-02fa6343-01ec-4526-94a5-393ebc365d39.png)
![image](https://user-images.githubusercontent.com/34794187/154633117-424604a1-2efd-479e-8eed-1760a86d15b8.png)


# Getting started
- Make sure you have Python installed, e.g. as part of [Anaconda](https://www.anaconda.com/products/individual)
- I suggest creating a clean virtual environment, e.g. using an Anaconda Prompt and conda by running `conda create --name [new environment name] python==3.9` (many of the examples use QuTiP which isn't compatible with python 3.10 as of 2/10/2022).
- [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository, [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the forked repository to you local machine, and then run `python setup.py install` in the root folder of the repository to install the  package and its dependencies.
- Install [jupyter/jupyterLab](https://jupyter.org/install). If using conda you can run `conda install jupyterlab --channel conda-forge`.
- You should then be able to run the example Jupyter notebooks in `./examples/`. This is the best place to start learning how to use the package. Recommended order:
  1. *Rabi oscillations in 2-level system.ipynb*. Shows the basics: how to set up a `QuantumSystem` using `Coupling`s, `Decay`s and a `Basis`, and then time-evolve the system.
  2. Any of *Rapid adiabatic passage in a 2-level system.ipynb*, *Landau-Zener transitions.ipynb*, *STIRAP in a 3-level system.ipynb*. In addition to the physics, these illustrate how to set up time-dependent couplings and energies and how to do parameter scans.
  3. *Coherent dark states and polarization switching.ipynb* for a more complicated system with angular momentum.

## Need for speed?
All of the example notebooks use [`QuTiP`](https://qutip.org/) to time-evolve the quantum system. A minimal installation of `QuTiP` is included by default in the installation of `toy_systems`. However, I strongly recommend additionally installing [`cython`](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) which allows `QuTiP` to compile the time-evolution functions to C - this significantly speeds things up. Also make sure to have a C-compiler installed on your machine. Should be included on Mac and Linux but on Windows you'll need to install the [Visual Studio build tools for C](https://visualstudio.microsoft.com/vs/features/cplusplus/).

# Contributing
To contribute, submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) from your fork of the repository. Bug fixes, new examples, new features, new quantum number bases are all very much welcome.
## Examples wanted
If you use the package to simulate an interesting toy model, it would be nice to add it to the examples. Specific ideas:
- Electromagnetically induced transparency
- AC Stark shift
- Saturated absorption

# Structure of the package
The package is divided into several modules that contain the building blocks that can in principle be used to model any quantum system, although `toy_models` is geared towards manually building small toy systems, where the user has easy control over the parameters.

## States

### `BasisState`
The `states` module provides a representation for the basis states of a quantum system in the form of the `states.BasisState`-object, which can essentially be thought of as basis kets. 

### `State`
Composite states made out of multiple `BasisState`s are called `State`s (can again think of these as kets). `State`s have various utility functions which are currently only documented in the source code

### `Basis`
We typically want to convert quantum states and operators to the matrix formalism for computations. To do this so we need to provide a basis using `Basis`, which contains a tuple of `BasisStates` whose ordering determines the indexing of the matrix representations.

### `QuantumNumbers`
Each `BasisState` requires a `QuantumNumbers`-object when constructed to ...specify its quantum numbers. There are various subclasses of `QuantumNumbers` to represent the various possible quantum numbers that might be used to specify a quantum state. 

## Couplings
`Coupling`s between states are used to generate the matrix elements of the Hamiltonian. Each subclass of `Coupling` has a method `calculate_ME` which takes as arguments two `BasisStates` and returns the matrix element between them. Energies are also treated as couplings in this package.

## Hamiltonian
The couplings and the states allow us to generate a `Hamiltonian` which can be used to calculate the eigenstates and to time-evolve the system. The `Hamiltonian` has a method to convert it to a `qutip.QobjEvo` and so the time-evolution is typically performed using QuTiP; `toy_systems` does not implement any time-evolution methods.

## Decays
Some quantum states are unstable, and can decay. This is represented by the `Decay` object which can be used to specify the decay channels of a quantum system. `Decay`s also come with methods to convert them to `qutip.Qobj`s for easy time-evolution.
