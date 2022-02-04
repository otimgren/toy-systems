import enum
from dataclasses import dataclass
from typing import Union

import numpy as np
from sympy import Symbol

from .basis import Basis
from .states import BasisState


@dataclass
class Decay:
    """
    Class that represents a decay channel from excited state to ground state at
    rate gamma.
    """

    excited: BasisState
    ground: BasisState
    gamma: Union[float, Symbol]

    def __post_init__(self):
        self.matrix = None

    def generate_decay_matrix(self, basis: Basis) -> None:
        """
        Generates a matrix that represents this decay in the given basis.
        """
        # Initialize a container for the matrix
        C = np.zeros((basis.dim, basis.dim), dtype="object")

        # Set a flag that tracks if any of the couplings are symbolic
        symbolic = False

        # Loop over basis states
        for i, state1 in enumerate(basis[:]):
            for j, state2 in enumerate(basis[:]):
                if state1 == self.excited and state2 == self.ground:
                    C[i, j] = self.gamma ** (1 / 2)

                    print(type(C[i, j]))

                    # Check for symbolic matrix elements
                    if not symbolic:
                        symbolic = isinstance(C[i, j], Symbol)

        # If no symbolic couplings, convert matrix datatype to float
        if not symbolic:
            C = C.astype(float)

        self.matrix = C
