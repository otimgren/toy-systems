import enum
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
import qutip
from sympy import Expr, Symbol

from .states import Basis, BasisState


@dataclass
class Decay:
    """
    Class that represents a decay channel from excited state to ground state at
    rate gamma.
    """

    excited: BasisState
    ground: Union[BasisState, None]
    gamma: Union[float, Symbol]

    def __post_init__(
        self,
        matrix: np.ndarray = None,
        matrix_sym: np.ndarray = None,
        qobj: qutip.Qobj = None,
    ):
        self.matrix = matrix
        self.matrix_sym = matrix_sym
        self._qobj = qobj

    def generate_decay_matrix(self, basis: Basis) -> None:
        """
        Generates a matrix that represents this decay in the given basis.
        """
        # Initialize a container for the matrix
        C = np.zeros((basis.dim, basis.dim), dtype="object")

        # Set a flag that tracks if any of the couplings are symbolic
        symbolic = False

        # If ground state is specified, make sure decays go there
        if self.ground:
            # Loop over basis states
            for i, state1 in enumerate(basis[:]):
                for j, state2 in enumerate(basis[:]):
                    if state1 == self.excited and state2 == self.ground:
                        C[j, 1] = self.gamma ** (1 / 2)

                        # Check for symbolic matrix elements
                        if not symbolic:
                            symbolic = isinstance(C[j, i], (Symbol, Expr))

        # If no ground state specified, the time-evolution is made non-unitary
        # by adding the decay matrix to the Hamiltonian
        else:
            # Loop over basis states
            for i, state1 in enumerate(basis[:]):
                if state1 == self.excited:
                    C[i, i] = -1j * self.gamma / 2

                    # Check for symbolic matrix elements
                    if not symbolic:
                        symbolic = isinstance(C[i, i], (Symbol, Expr))

        # If no symbolic couplings, convert matrix datatype to float
        if not symbolic:
            C = C.astype(complex)
            self.matrix = C

        # Otherwise, store the symbolic matrix but also make a non-symbolic version
        else:
            self.matrix_sym = C
            self._generate_C_complex()

    def _generate_C_complex(self) -> None:
        """
        Returns a version of the decay matrix with dtype = complex 
        """
        # Check if the matrix already exists
        if self.matrix is not None:
            pass

        # Otherwise convert symbolic matrix to complex
        elif self.matrix_sym is not None:
            # Update time_dep and time_args
            expr = self.gamma
            symbol = list(expr.free_symbols)[0]

            # Convert symbolic version to complex version
            C = self.matrix_sym.copy()
            rows, columns = C.nonzero()
            for i in rows:
                for j in columns:
                    if isinstance(C[i, j], (Symbol, Expr)):
                        C[i, j] = C[i, j].subs(symbol, 1)

            C = C.astype(complex)

            self.matrix = C

    def generate_qobj(self, gamma: float, basis: Basis = None) -> None:
        """
        Generates a qutip.Qobj representation of the collapse operator.
        """
        if self.matrix is None:
            self.generate_decay_matrix(basis)

        self._qobj = qutip.Qobj(gamma * self.matrix, type="oper")

    def qobj(self, gamma: float) -> qutip.Qobj:
        """
        Returns the qobj attribute with specified value of gamma.
        """
        self.generate_qobj(gamma)
        return self._qobj

