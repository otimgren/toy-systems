from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union

import numpy as np
import qutip
from sympy import Expr, Symbol

from .couplings import Coupling, FirstRankCouplingJ
from .states import Basis, BasisState


class Decay:
    """
    Class that represents a decay channel from excited state to ground state at
    rate gamma.
    """

    def __post_init__(
        self,
        matrix: np.ndarray = None,
        matrix_sym: np.ndarray = None,
        qobj: qutip.Qobj = None,
    ):
        self.matrix = matrix
        self.matrix_sym = matrix_sym
        self.qobj = qobj

    @abstractmethod
    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol, Expr]:
        """
        Calculates the matrix element between states 1 and 2
        """

    def generate_qobj(self, basis: Basis = None) -> None:
        """
        Generates a qutip.Qobj representation of the collapse operator.
        """
        if self.matrix is None:
            self.generate_decay_matrix(basis)

        qobj = qutip.Qobj(inpt=self.matrix, type="oper")

        if self.time_dep:
            self.qobj = [qobj, self.time_dep]
        else:
            self.qobj = qobj


@dataclass
class ToyDecay:
    """
    Class that represents a decay channel from excited state to ground state at
    rate gamma.
    """

    excited: BasisState
    ground: Union[BasisState, None]
    gamma: Union[float, Symbol]
    time_dep: Union[str, Callable] = None
    time_args: dict = field(default_factory=dict)

    def __post_init__(
        self,
        matrix: np.ndarray = None,
        matrix_sym: np.ndarray = None,
        qobj: List[qutip.Qobj] = None,
    ):
        self.matrix = matrix
        self.matrix_sym = matrix_sym
        self.qobj = qobj

    def generate_matrix(self, basis: Basis) -> None:
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
                        C[j, i] = self.gamma ** (1 / 2)

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
            self._generate_matrix_complex()

    def _generate_matrix_complex(self) -> None:
        """
        Returns a version of the decay matrix with dtype = complex
        """
        # Check if the matrix already exists
        if self.matrix is not None:
            return

        # Otherwise convert symbolic matrix to complex
        elif self.matrix_sym is not None:
            # Update time_dep and time_args
            expr = self.gamma
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
            C = self.matrix_sym.copy()
            rows, columns = C.nonzero()
            for i in rows:
                for j in columns:
                    if isinstance(C[i, j], (Symbol, Expr)):
                        # Important to maintain relative phase
                        C[i, j] = C[i, j].subs(np.conj(symbol), symbol)
                        C[i, j] = C[i, j] / symbol

            C = C.astype(complex)

            self.matrix = C

    def generate_qobj(self, basis: Basis = None) -> None:
        """
        Generates a qutip.Qobj representation of the collapse operator.
        """
        if self.matrix is None:
            self.generate_matrix(basis)

        qobj = qutip.Qobj(inpt=self.matrix, type="oper")

        if self.time_dep:
            self.qobj = [[qobj, self.time_dep]]
        else:
            self.qobj = [qobj]


@dataclass
class CouplingDecay(Decay):
    """
    Decay from an excited state via provided couplings (e.g. electric dipole).
    """

    excited: BasisState
    gamma: Union[float, Symbol]
    couplings: List[Coupling]
    ground: List[BasisState] = None
    time_dep: Union[str, Callable] = None
    time_args: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.MEs = []
        self.brs = []
        self.matrix = None
        self.matrix_sym = None
        self.qobj = []

    def calculate_MEs(self, basis: Basis = None, tol=0) -> None:
        """
        Calculates the matrix elements between all states in basis if provided,
        or between self.ground and self.excited
        """
        # Case 1: Basis is provided
        if basis is not None:
            self.ground = []
            for gs in basis[:]:
                ME = 0
                for coupling in self.couplings:
                    ME += coupling.calculate_ME(self.excited, gs)
                if np.abs(ME) > tol:
                    self.ground.append(gs)
                    self.MEs.append(ME)

        # Case 2: No basis provided but self.ground is provided
        elif (basis is None) and (self.ground is not None):
            for gs in self.ground:
                for coupling in self.couplings:
                    self.MEs.append(coupling.calculate_ME(self.excited, gs))

        # Case 3: Nothing provided, raise error
        else:
            raise ValueError("Need to provide a basis or ground states")

    def generate_matrix(self, basis: Basis) -> None:
        # Calculate matrix elements between excited state and each
        # basis state
        self.calculate_MEs(basis)

        # Calculate index of excited state in the basis
        i_e = basis.find_state_idx(self.excited)

        # Calculate sum of all the MEs for the couplings
        MEsqr_sum = np.sum(np.abs(self.MEs) ** 2)

        # Loop over the ground states of the decay and calculate decay matrices
        for gs, ME in zip(self.ground, self.MEs):
            i_g = basis.find_state_idx(gs)

            # Calculate branching ratio and add to list
            br = (np.abs(ME) ** 2) / MEsqr_sum
            self.brs.append(br)

            # Generate the decay matrix
            c = np.zeros((basis.dim, basis.dim), dtype=object)
            c[i_g, i_e] = br * self.gamma
            if self.matrix_sym is None:
                self.matrix_sym = []
            self.matrix_sym.append(c)

        # Convert decay matrices to float
        self._generate_float_matrices()

    def _generate_float_matrices(self) -> None:
        """
        Converts the symbolic matrices to float dtype
        """
        # Check if the matrices already exists
        if self.matrix is not None:
            return

        # Otherwise convert symbolic matrix to complex
        elif self.matrix_sym is not None:
            # Initialize the storage for matrices
            self.matrix = []

            # Update time_dep and time_args
            expr = self.gamma
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

            for matrix_sym in self.matrix_sym:
                # Convert symbolic versions to complex version
                M = matrix_sym.copy()
                rows, columns = M.nonzero()
                for i in rows:
                    for j in columns:
                        if isinstance(M[i, j], (Symbol, Expr)):
                            # Important to maintain relative phase
                            M[i, j] = M[i, j].subs(np.conj(symbol), symbol)
                            M[i, j] = M[i, j] / symbol

                self.matrix.append(M.astype(complex))

    def generate_qobj(self, basis: Basis = None) -> None:
        """
        Generates a qobj-list in the given basis if provided, otherwise
        use stored matrices.
        """
        if self.matrix is None:
            self.generate_matrix(basis)

        for matrix in self.matrix:
            qobj = qutip.Qobj(inpt=matrix, type="oper")

            if self.time_dep:
                self.qobj.append([qobj, self.time_dep])

            else:
                self.qobj.append(qobj)


@dataclass
class DecayList:
    """
    Class for conveniently storing Decays and passing their qobjs to QuTiP for time-evolution.
    """

    decays: List[Decay]
    basis: Basis = None

    def __post_init__(self, qobjs: List = None, args: dict = None):
        self.qobjs = qobjs
        self.args = args
        if self.basis is not None:
            self.generate_qobjs(self.basis)

    def generate_qobjs(self, basis: Basis) -> None:
        """
        Generates a list of tuples of Qobjs and their time dependences, and a dictionary of
        arguments for the time dependence strings or functions
        """
        self.qobjs = []
        self.args = {}

        for decay in self.decays:
            decay.generate_qobj(basis)
            self.qobjs += decay.qobj
            self.args.update(decay.time_args)
