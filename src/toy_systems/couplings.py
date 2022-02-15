"""
Couplings used to generate hamiltonians
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numexpr as ne
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
        Generates the matrix representation of the coupling and stores in self.matrix
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

    def _generate_matrix_complex(self) -> None:
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
                self.time_dep = f"{expr.__repr__()}*({self.time_dep})"

            elif isinstance(self.time_dep, Callable):
                old_time_dep = self.time_dep
                self.time_dep = lambda x, attrs: attrs[symbol_name] * old_time_dep(x)

            else:
                self.time_dep = expr.__repr__()

            # Convert symbolic version to complex version
            M = self.matrix_sym.copy()
            rows, columns = M.nonzero()
            for i in rows:
                for j in columns:
                    if isinstance(M[i, j], (Symbol, Expr)):
                        M[i, j] = M[i, j].subs(expr, 1)

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

    def eval(self, times: np.array, time_args: dict, mag: float = 1) -> np.ndarray:
        """
        Evaluates the strength of the coupling at the given times t.
        """
        if isinstance(self.time_dep, str):
            values = ne.evaluate(self.time_dep, time_args | {"t": times})
            # If there is no time dependence, need to convert to array
            if not values.shape:
                values = values * np.ones(times.shape)

            return mag * values

        elif isinstance(self.time_dep, Callable):
            return mag * self.time_dep(times, time_args)

    def plot_time_dep(
        self, times: np.ndarray, time_args: dict, mag=1, ax: plt.Axes = None, **kwargs
    ) -> List[plt.Line2D]:
        """
        Plots the time dependence of the coupling.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(16, 9))

        ln = ax.plot(times, self.eval(times, time_args, mag), **kwargs)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Magnitude of coupling", fontsize=16)

        return ln


@dataclass
class ToyEnergy(Coupling):
    """
    Class used to generate diagonal matrix elements for Hamiltonian
    """

    states: List[BasisState]
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
        if (state1 in self.states) & (state1 == state2):
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
    Coupling of states in the JQuantumNumbers-basis by a 1st rank tensor.

    An example of this would be an electric or magnetic dipole coupling, where
    the matrix elements would be given by:

    <s1, J, mJ|d_q^(k)|s2, J', mJ'> = <J', mJ'; k, q| J, mJ>/sqrt(2J+1)
    """

    mag: Union[complex, Symbol, Expr]
    p_car: np.ndarray = None  # Polarization vector in cartesian basis (x,y,z,)
    p_sph: dict = None  # Polarization vector in spherical basis (-1,0,1)
    rm_func: Callable = (
        None  # Function for calculating reduced matrix element given J and J'
    )
    other_conds: List[
        Callable
    ] = None  # Functions for checking that matrix element is non-zero
    time_dep: Union[str, Callable] = None
    time_args: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Check that a polarization vector has been given
        if self.p_sph is None:
            if self.p_car is None:
                raise ValueError("No polarization vector given")

            # Convert cartesian vector to spherical basis
            p_car = self.p_car
            self.p_sph = {}
            self.p_sph[-1] = (p_car[0] + 1j * p_car[1]) / np.sqrt(2)
            self.p_sph[0] = p_car[2]
            self.p_sph[+1] = (-p_car[0] + 1j * p_car[1]) / np.sqrt(2)

        # Check normalization of polarization vector
        norm = (
            np.abs(self.p_sph[-1]) ** 2
            + np.abs(self.p_sph[0]) ** 2
            + np.abs(self.p_sph[+1]) ** 2
        )
        assert (
            np.abs(norm - 1) < 1e-6
        ), f"Error: polarization vector not normalized: norm ={norm}"

    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol, Expr]:
        """
        Calculates the matrix element between states 1 and 2.
        """
        # Initialize matrix element
        ME = 0

        # Check that any provided conditions are satisfied
        if not self.check_other_conds(state1, state2):
            return 0

        # Extract quantum numbers
        J = state1.qn.J
        mJ = state1.qn.mJ
        Jp = state2.qn.J
        mJp = state2.qn.mJ

        # If J, J' and k=1 don't satisfy triangle condition, ME = 0
        if J < np.abs(Jp - 1) and J > Jp + 1:
            return 0

        # Loop over components of polarization
        for q, amp in self.p_sph.items():
            # If amplitude is zero, move to next component
            if amp == 0:
                continue

            # If projections don't add up, no contribution for this q
            if mJ != mJp + q:
                continue

            # At this point have to calculate the Clebsch-Gordan coefficient
            ME += amp * qutip.clebsch(Jp, 1, J, mJp, q, mJ) / np.sqrt(2 * J + 1)

        # Finally multiply by reduced part of ME if provided
        if self.rm_func:
            ME *= self.rm_func(state1, state2)

        return self.mag * ME

    def check_other_conds(self, state1: BasisState, state2: BasisState) -> bool:
        """
        Checks that the quantum numbers other than J and mJ satisfy the given
        conditions.
        """

        for func in self.other_conds:
            if not func(state1, state2):
                return False

        return True
