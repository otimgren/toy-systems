"""
Couplings used to generate hamiltonians
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
from sympy import Symbol

from .states import BasisState


class Coupling(ABC):
    """
    Abstract parent class for couplings
    """

    @abstractmethod
    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol]:
        """
        Calculates the matrix element between states 1 and 2
        """


@dataclass
class ToyEnergy(Coupling):
    """
    Class used to generate diagonal matrix elements for Hamiltonian
    """

    state: BasisState
    energy: Union[float, Symbol]

    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol]:
        if (self.state == state1) & (self.state == state2):
            return self.energy
        else:
            return 0


@dataclass
class ToyCoupling(Coupling):
    """
    Generic toy coupling between state_a and state_b of strength Omega:
    Omega = <state_b|H|state_a>.
    """

    state_a: BasisState
    state_b: BasisState
    Omega: complex

    def calculate_ME(
        self, state1: BasisState, state2: BasisState
    ) -> Union[float, Symbol]:
        """
        Calculates matrix element between states 1 and 2
        """
        if (self.state_a == state1) & (self.state_b == state2):
            return self.Omega

        elif (self.state_b == state1) & (self.state_a == state2):
            return np.conj(self.Omega)

        else:
            return 0.0


@dataclass
class FirstRankCouplingJ(Coupling):
    """
    Coupling of two states by a 1st rank tensor
    """

    state1: BasisState
    state2: BasisState
