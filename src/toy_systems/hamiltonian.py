"""
Contains class used to represent Hamiltonians.
"""
import copy
from typing import List

import numpy as np
import qutip
from sympy import Matrix, Symbol

from .couplings import Coupling
from .states import Basis


class Hamiltonian:
    """
    Class used to represent Hamiltonians.
    """

    def __init__(
        self,
        basis: Basis,
        couplings: List[Coupling],
        matrix: np.ndarray = None,
        qobj: qutip.QobjEvo = None,
    ) -> None:
        self.basis = basis
        self.couplings = couplings
        self.matrix = matrix
        self.qobj = None

        # Generate the matrix and Qobj
        self.generate_matrix()
        self.generate_qobj()

    def __repr__(self) -> str:
        if self.matrix is not None:
            return f"H =\n{Matrix(self.matrix).__repr__()[7:-1]}\n\nin basis {self.basis.__repr__()}"

        else:
            return_string = "Hamiltonian with:\n\n"
            couplings_str = f"Couplings: {self.couplings.__repr__()}\n\n"
            basis_str = f"{self.basis.__repr__()}"

            return return_string + couplings_str + basis_str

    def copy(self):
        """
        Return a copy of the Hamiltonian
        """
        return copy.deepcopy(self)

    def generate_matrix(self) -> None:
        """
        Generates a matrix reperesentation of the hamiltonian.

        Note: Time-dependence of couplings is not shown. The matrix is meant
        primarily for checking that the couplings look correct.
        """
        # Define a container for the matrix
        H = np.zeros((self.basis.dim, self.basis.dim), dtype="object")

        # Set a flag that tracks if any of the couplings are symbolic
        symbolic = False

        # Loop over couplings and calculate their matrix representations
        for coupling in self.couplings:
            M = coupling.M(self.basis)
            H += M

            if M.dtype == "object":
                symbolic = True

        # If no symbolic couplings, convert matrix datatype to float
        if not symbolic:
            H = H.astype(complex)

        # Make sure H is hermitian
        if H.dtype != "object":
            if not np.allclose(H, H.conj().T):
                print("Warning: Hamiltonian is not hermitian!")

        self.matrix = H

    def generate_qobj(self) -> None:
        """
        Generates a qutip.QobjEvo of the Hamiltonian that can be used for time-evolution.
        """
        qobjs = []
        args = {}

        for coupling in self.couplings:
            qobjs.append(coupling.qobj(self.basis))
            args.update(coupling.time_args)

        self.qobj = qutip.QobjEvo(Q_object=qobjs, args=args)
