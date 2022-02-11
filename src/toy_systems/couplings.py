"""
Couplings used to generate hamiltonians
"""

from abc import ABC, abstractmethod
from ast import Call
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
import qutip
from sympy import Expr, Symbol

from .core import QuantumObject
from .states import Basis, BasisState


@dataclass
class Coupling(QuantumObject):
    """
    Abstract parent class for couplings
    """

    def __post_init__(
        self,
        matrix: np.ndarray = None,
        matrix_sym: np.ndarray = None,
        qobj: Union[qutip.Qobj, qutip.QobjEvo] = None,
    ) -> None:
        super().__init__(matrix, matrix_sym, qobj)

        if isinstance(self.mag, (Symbol, Expr)):
            # Check that self.mag contains a maximum of 1 symbol
            error_msg = "ERROR: each coupling magnitude can contain only one Symbol"
            assert len(self.mag.free_symbols) <= 1, error_msg

    @abstractmethod
    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol, Expr]:
        """
        Calculates the matrix element between states 1 and 2
        """

    def generate_matrix(self, basis: Basis) -> None:
        """
        Generates the matrix representation of the coupling stored in self.matrix
        """
        # Define a container for the matrix
        M = np.zeros((basis.dim, basis.dim), dtype="object")

        # Set a flag that tracks if any of the couplings are symbolic
        symbolic = False

        # Loop over basis states and calculate couplings between them
        for i, state1 in enumerate(basis[:]):
            for j, state2 in enumerate(basis[:]):
                M[i, j] += self.calculate_ME(state1, state2)

                # Check for symbolic matrix elements
                if not symbolic:
                    symbolic = isinstance(M[i, j], (Symbol, Expr))

        # If no symbolic couplings, convert matrix datatype to float
        if not symbolic:
            M = M.astype(complex)
            self.matrix = M

        # Otherwise store the symbolic matrix, but also make a non-symbolic version
        else:
            self.matrix_sym = M
            self._generate_matrix_complex()

    def _generate_matrix_complex(self) -> np.ndarray:
        """
        Generates the matrix representation as an ndarray with dtype = complex.
        Adjusts self.time_dep and time_attrs to include self.mag as a constant
        multiplier.
        """
        # Check if the matrix already exists
        if self.matrix is not None:
            return

        # Otherwise convert symbolic matrix to complex
        elif self.matrix_sym is not None:
            # Update time_dep and time_args
            expr = self.mag
            symbol = list(expr.free_symbols)[0]
            symbol_name = symbol.__repr__()
            self.time_args[symbol_name] = 1

            if isinstance(self.time_dep, str):
                self.time_dep = f"{symbol_name}*({self.time_dep})"

            elif isinstance(self.time_dep, Callable):
                old_time_dep = self.time_dep
                self.time_dep = lambda x, attrs: attrs[symbol_name] * old_time_dep(x)

            else:
                self.time_dep = symbol_name

            # Convert symbolic version to complex version
            M = self.matrix_sym.copy()
            rows, columns = M.nonzero()
            for i in rows:
                for j in columns:
                    if isinstance(M[i, j], (Symbol, Expr)):
                        M[i, j] = M[i, j].subs(symbol, 1)

            M = M.astype(complex)

            self.matrix = M

    def generate_qobj(self, basis: Basis = None) -> None:
        """
        Generates a qutip.Qobj representation of the coupling stored in self.qobj.
        """
        if self.matrix is None:
            self.generate_matrix(basis)

        M = self.matrix

        qobj = qutip.Qobj(inpt=M, type="oper")

        if self.time_dep:
            self.qobj = (qobj, self.time_dep)
        else:
            self.qobj = qobj


@dataclass
class ToyEnergy(Coupling):
    """
    Class used to generate diagonal matrix elements for Hamiltonian
    """

    state: BasisState
    mag: Union[complex, Symbol, Expr]
    time_dep: Union[str, Callable] = None
    time_args: dict = field(default_factory=dict)

    def __post_init__(
        self,
        matrix: np.ndarray = None,
        matrix_sym: np.ndarray = None,
        qobj: Union[qutip.Qobj, qutip.QobjEvo] = None,
    ) -> None:
        super().__post_init__(matrix, matrix_sym, qobj)

    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol, Expr]:
        """
        Returns self.mag (i.e. energy) if two provided states both match self.state.
        """
        if (self.state == state1) & (self.state == state2):
            return self.mag
        else:
            return 0


@dataclass
class ToyCoupling(Coupling):
    """
    Generic toy coupling between state_a and state_b of strength mag:
    Omega = <state_b|H|state_a>.
    """

    state_a: BasisState
    state_b: BasisState
    mag: Union[complex, Symbol, Expr]
    time_dep: Union[str, Callable] = None
    time_args: dict = field(default_factory=dict)

    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol]:
        """
        Calculates matrix element between states 1 and 2
        """
        if (self.state_a == state1) & (self.state_b == state2):
            return self.mag

        elif (self.state_b == state1) & (self.state_a == state2):
            return np.conj(self.mag)

        else:
            return 0.0


@dataclass
class FirstRankCouplingJ(Coupling):
    """
    Coupling of two states by a 1st rank tensor
    """

    # TO DO
