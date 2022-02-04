from typing import List

from .states import BasisState


class Basis:
    """
    Class used to represent a quantum basis
    """

    def __init__(self, basis_states: List[BasisState], name=None) -> None:
        self.basis_states = basis_states
        self.dim = len(basis_states)
        self.name = name

    def __repr__(self) -> str:
        return f"Basis: name = {self.name}"

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        x = self.basis_states
        self.index += 1
        return x

    def __getitem__(self, i):
        return self.basis_states[i]

    def print_basis(self):
        for i, basis_state in enumerate(self.basis_states):
            print(f"|{i}> = {basis_state.__repr__()}")
