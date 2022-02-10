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

from .states import Basis, BasisState


@dataclass
class Coupling(ABC):
    """
    Abstract parent class for couplings
    """

    state_a: BasisState
    state_b: BasisState
    mag: Union[complex, Symbol, Expr]
    time_dep: Union[str, Callable] = None
    time_args: dict = field(default_factory=dict)

    def __post_init__(
        self,
        _matrix: np.ndarray = None,
        _M_complex: np.ndarray = None,
        _qobj: Union[qutip.Qobj, qutip.QobjEvo] = None,
    ) -> None:
        self._matrix = _matrix
        self._qobj = _qobj
        self._M_complex = _M_complex

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

        self._matrix = M

    def M(self, basis: Basis = None) -> np.ndarray:
        """
        Gets the matrix representation of the coupling.
        """
        if self._matrix is None:
            self.generate_matrix(basis)

        return self._matrix

    def generate_qobj(self, basis: Basis = None) -> None:
        """
        Generates a qutip.Qobj representation of the coupling stored in self.qobj.
        """
        M = self.M_complex()

        qobj = qutip.Qobj(inpt=M, type="oper")

        if self.time_dep:
            self._qobj = (qobj, self.time_dep)
        else:
            self._qobj = qobj

    def M_complex(self, basis=None) -> np.ndarray:
        """
        Returns the matrix representation as an ndarray with dtype = complex.
        Adjusts self.time_dep and time_attrs to include self.mag as a constant
        multiplier.
        """
        # If already have a dtype==complex M, return that
        if self.M().dtype == complex:
            return self.M()

        # If complex matrix has already been generated, return that
        if self._M_complex is not None:
            return self._M_complex

        # Otherwise convert symbolic matrix to complex
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

        # Get complex version of M
        M = self.M(basis).copy()
        rows, columns = M.nonzero()
        for i in rows:
            for j in columns:
                if isinstance(M[i, j], (Symbol, Expr)):
                    M[i, j] = M[i, j].subs(symbol, 1)

        M = M.astype(complex)

        self._M_complex = M

        return M

    def qobj(self, basis: Basis = None) -> Union[qutip.Qobj, qutip.QobjEvo]:
        """
        Gets the qutip representation of the object.
        """
        if self._qobj is None:
            self.generate_qobj(basis)

        return self._qobj


@dataclass
class ToyEnergy(Coupling):
    """
    Class used to generate diagonal matrix elements for Hamiltonian
    """

    def __post_init__(
        self, _matrix: np.ndarray = None, _qobj: Union[qutip.Qobj, qutip.QobjEvo] = None
    ) -> None:
        super().__post_init__(_matrix, _qobj)
        assert (
            self.state_a == self.state_b
        ), "States provided to ToyEnergy must be the same!"

    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol, Expr]:
        if (self.state_a == state1) & (self.state_a == state2):
            return self.mag
        else:
            return 0


@dataclass
class ToyCoupling(Coupling):
    """
    Generic toy coupling between state_a and state_b of strength mag:
    Omega = <state_b|H|state_a>.
    """

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
