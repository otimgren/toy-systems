"""
Defining some quantum operators.
"""
from dataclasses import dataclass
from typing import Union

import numpy as np
import qutip
import scipy

from .core import QuantumObject
from .states import Basis, BasisState, JQuantumNumbers


class Operator(QuantumObject):
    """
    Parent class for operators
    """

    def __post_init__(
        self,
        matrix: np.ndarray = None,
        matrix_sym: np.ndarray = None,
        qobj: Union[qutip.Qobj, qutip.QobjEvo] = None,
    ) -> None:
        super().__init__(matrix, matrix_sym, qobj)

    def generate_matrix(self, basis: Basis) -> None:
        """
        Generates the matrix representation of the coupling and stores in self.matrix
        """
        # Define a container for the matrix
        M = np.zeros((basis.dim, basis.dim), dtype="complex")

        # Loop over basis states and calculate couplings between them
        for i, state1 in enumerate(basis[:]):
            for j, state2 in enumerate(basis[:]):
                M[i, j] += self.calculate_ME(state1, state2)

        self.matrix = M

    def generate_qobj(self, basis: Basis = None) -> None:
        """
        Generates a qutip.Qobj representation of the coupling stored in self.qobj.
        """
        if self.matrix is None:
            self.generate_matrix(basis)

        M = self.matrix

        self.qobj = qutip.Qobj(inpt=M, type="oper")


class Jz_op(Operator):
    """
    Representation of Jz.
    """

    def calculate_ME(self, state1: BasisState, state2: BasisState) -> float:
        """
        Calculates <state1|Jz|state2>
        """
        return state1.qn.mJ * (state1 @ state2)


class Jplus_op(Operator):
    """
    Representation of J+
    """

    def calculate_ME(self, state1: BasisState, state2: BasisState) -> float:
        """
        Calculates <state1|J+|state2>.
        """
        J2 = state2.qn.J
        m2 = state2.qn.mJ
        ME = np.sqrt(J2 * (J2 + 1) - m2 * (m2 + 1))

        try:
            new_state = BasisState(
                JQuantumNumbers(
                    label=state2.qn.label, J=state2.qn.J, mJ=state2.qn.mJ + 1,
                )
            )

        except ValueError:
            # If mJ is out of bounds return 0
            return 0

        return ME * (state1 @ new_state)


class Jminus_op(Operator):
    """
    Representation of J-
    """

    def calculate_ME(self, state1: BasisState, state2: BasisState) -> float:
        """
        Calculates <state1|J+|state2>.
        """
        J2 = state2.qn.J
        m2 = state2.qn.mJ
        ME = np.sqrt(J2 * (J2 + 1) - m2 * (m2 - 1))

        try:
            new_state = BasisState(
                JQuantumNumbers(
                    label=state2.qn.label, J=state2.qn.J, mJ=state2.qn.mJ - 1,
                )
            )
        except ValueError:
            # If mJ is out of bounds return 0
            return 0

        return ME * (state1 @ new_state)


class Jx_op(Operator):
    """
    Representation of Jx
    """

    def calculate_ME(self, state1: BasisState, state2: BasisState) -> float:
        plus = Jplus_op().calculate_ME(state1, state2)
        minus = Jminus_op().calculate_ME(state1, state2)
        return (plus + minus) / 2


class Jy_op(Operator):
    """
    Representation of Jy
    """

    def calculate_ME(self, state1: BasisState, state2: BasisState) -> float:
        plus = Jplus_op().calculate_ME(state1, state2)
        minus = Jminus_op().calculate_ME(state1, state2)
        return (plus - minus) / 2j


@dataclass
class JRotation:
    """
    Class for rotation operators.
    """

    theta: float  # Rotation angle
    n: np.ndarray  # Rotation axis

    def __post_init__(self):
        self.matrix = None
        self.qobj = None

        # Generate matrices for angular momentum in given basis
        self.Jx = Jx_op()
        self.Jy = Jy_op()
        self.Jz = Jz_op()

    def generate_matrix(self, basis):

        self.Jx.generate_matrix(basis)
        self.Jy.generate_matrix(basis)
        self.Jz.generate_matrix(basis)

        self.matrix = scipy.linalg.expm(
            -1j
            * self.theta
            * (
                self.n[0] * self.Jx.matrix
                + self.n[1] * self.Jy.matrix
                + self.n[2] * self.Jz.matrix
            )
        )

    def generate_qobj(self, basis: Basis = None) -> None:
        """
        Generates a qutip.Qobj representation of the coupling stored in self.qobj.
        """
        if self.matrix is None:
            self.generate_matrix(basis)

        M = self.matrix

        self.qobj = qutip.Qobj(inpt=M, type="oper")

