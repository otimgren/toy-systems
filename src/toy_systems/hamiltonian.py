"""
Contains class used to represent Hamiltonians.
"""
from typing import List

import numpy as np
from sympy import Symbol

from .basis import Basis
from .couplings import Coupling


class Hamiltonian:
    """
    Class used to represent Hamiltonians.
    """

    def __init__(self, basis: Basis, couplings: List[Coupling], matrix=None):
        self.basis = basis
        self.couplings = couplings
        self.matrix = matrix

    def __repr__(self) -> str:
        if self.matrix is not None:
            return f"H =\n{self.matrix.__repr__()}\nin basis {self.basis.__repr__()}"

        else:
            return_string = "Hamiltonian with:\n"
            couplings_str = f"\tCouplings: {self.couplings.__repr__()}\n"
            basis_str = f"\t{self.basis.__repr__()}"

            return return_string + couplings_str + basis_str

    def generate_matrix(self) -> None:
        """
        Generates a matrix reperesentation of the hamiltonian.
        """
        # Define a container for the matrix
        H = np.zeros((self.basis.dim, self.basis.dim), dtype="object")

        # Set a flag that tracks if any of the couplings are symbolic
        symbolic = False

        # Loop over basis states and calculate couplings between them
        for i, state1 in enumerate(self.basis[:-1]):
            for j in range(i + 1, self.basis.dim):
                state2 = self.basis[j]
                for coupling in self.couplings:
                    H[i, j] += coupling.calculate_ME(state1, state2)

                    # Check for symbolic matrix elements
                    if not symbolic:
                        symbolic = isinstance(H[i, j], Symbol)

        # Loop over basis states and calculate diagonal elements
        energies = np.zeros((self.basis.dim, self.basis.dim), dtype="object")
        for i, state in enumerate(self.basis[:]):
            for coupling in self.couplings:
                energies[i, i] += coupling.calculate_ME(state, state)

                # Check for symbolic matrix elements
                if not symbolic:
                    symbolic = isinstance(energies[i, i], Symbol)

        # Make sure H is hermitian
        H = H + H.conj().T + energies

        # If no symbolic couplings, convert matrix datatype to float
        if not symbolic:
            H = H.astype(float)

        self.matrix = H
