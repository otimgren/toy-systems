from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import qutip
from sympy import Expr, Symbol

from .states import Basis, BasisState


class QuantumObject(ABC):
    """
    Abstract parent class for all quantum objects that can be represented as matrices.

    This includes: Couplings, Decays, Hamiltonians and Operators
    """

    def __init__(
        self,
        matrix: np.ndarray = None,
        matrix_sym: np.ndarray = None,
        qobj: Union[qutip.Qobj, qutip.QobjEvo] = None,
    ) -> None:
        self.matrix = matrix
        self.qobj = qobj
        self.matrix_sym = matrix_sym

    @abstractmethod
    def generate_matrix(self, basis: Basis) -> None:
        """
        Generates a matrix representation of the QuantumObject in the provided basis.

        Stored in self.matrix and self.matrix_sym (symbolic version)
        """

    @abstractmethod
    def generate_qobj(self, basis: Basis = None) -> None:
        """
        Generates a qutip.Qobj representation of the coupling stored in self.qobj.
        """
