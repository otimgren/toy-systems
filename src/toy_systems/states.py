"""
Contains classes for making simple toy models with states that are simpler than
TlF eigenstates.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class QuantumNumbers:
    """
    Abstract parnet class for storing the quantum numbers of a state.
    """

    def state_string(self):
        """
        Generates a string of the quantum numbers as a ket.
        """


@dataclass
class JQuantumNumbers(QuantumNumbers):
    J: float
    mJ: float
    name: str = None


@dataclass
class BasisState(ABC):
    """
    Class for  basis states.
    """

    qn: QuantumNumbers

    # equality testing
    def __eq__(self, other):
        """
        Determine if two basis states are the same, i.e. have the same
        quantum numbers
        """
        # Check that both states are in the same basis and have the same
        # quantum numbers
        if type(self.qn) != type(other.qn):
            raise ValueError("Error: Comparing states in different bases")

        else:
            return self.qn == other.qn

    def __matmul__(self, other):
        """
        Inner product of basis states.
        """
        if self == other:
            return 1
        else:
            return 0

    def __add__(self, other):
        """
        Addition of basis states.
        """
        if self == other:
            return State([(2, self)])
        else:
            return State([(1, self), (1, other)])

    def __sub__(self, other):
        """
        Subtraction of basis states.
        """
        return self + (-1) * other

    def __mul__(self, a):
        """
        Scalar product of basis state and constant
        """
        return State([(a, self)])

    def __rmul__(self, a):
        """
        Right scalar product for basis state and constant.
        """
        return self * a


class State:
    """
    Class for composite states that are superpositions of basis states.
    """

    def __init__(
        self,
        data=[],
        remove_zero_amp_cpts=True,
        name=None,
    ):
        # check for duplicates
        for i in range(len(data)):
            amp1, cpt1 = data[i][0], data[i][1]
            for amp2, cpt2 in data[i + 1 :]:
                if cpt1 == cpt2:
                    raise AssertionError("duplicate components!")
        # remove components with zero amplitudes
        if remove_zero_amp_cpts:
            self.data = [(amp, cpt) for amp, cpt in data if amp != 0]
        else:
            self.data = data
        # for iteration over the State
        self.index = len(self.data)
        # Give the state a name if desired
        self.name = name

    def __add__(self, other):
        """
        Adds two states together (doesn't normalize resulting state).
        """
        data = []
        # add components that are in self but not in other
        for amp1, cpt1 in self.data:
            only_in_self = True
            for amp2, cpt2 in other.data:
                if cpt2 == cpt1:
                    only_in_self = False
            if only_in_self:
                data.append((amp1, cpt1))
        # add components that are in other but not in self
        for amp1, cpt1 in other:
            only_in_other = True
            for amp2, cpt2 in self.data:
                if cpt2 == cpt1:
                    only_in_other = False
            if only_in_other:
                data.append((amp1, cpt1))
        # add components that are both in self and in other
        for amp1, cpt1 in self.data:
            for amp2, cpt2 in other.data:
                if cpt2 == cpt1:
                    data.append((amp1 + amp2, cpt1))
        return State(data)

    def __sub__(self, other):
        """
        Subtract two states (doesn't normalize resulting state).
        """
        return self + -1 * other

    def __mul__(self, a):
        """
        Scalar multiplication of state.
        """
        return State([(a * amp, psi) for amp, psi in self.data])

    def __rmul__(self, a):
        """
        Scalar multiplies state from right.
        """
        return self * a

    def __truediv__(self, a):
        """
        Scalar division.
        """
        return self * (1 / a)

    def __neg__(self):
        """
        Negative of state.
        """
        return -1 * self

    def __matmul__(self, other):
        """
        Takes the inner product of two states.
        """
        result = 0
        for amp1, psi1 in self:
            for amp2, psi2 in other:
                result += amp1.conjugate() * amp2 * (psi1 @ psi2)
        return result

    def __iter__(self):
        """
        Iterate over basis state amplitudes and basis states.
        """
        return ((amp, state) for amp, state in self.data)

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

    # direct access to a component
    def __getitem__(self, i):
        return self.data[i]

    def __repr__(self):
        ordered = self.order_by_amp()
        idx = 0
        string = ""
        if self.name:
            string += self.name + "="
        amp_max = np.max(np.abs(list(zip(*ordered))[0]))
        for amp, state in ordered:
            if np.abs(amp) < amp_max * 1e-2:
                continue
            string += f"{amp:.2f} x {state}"
            idx += 1
            if (idx > 4) or (idx == len(ordered.data)):
                break
            string += "\n"
        if idx == 0:
            return ""
        else:
            return string

    #########################
    ### Utility functions ###
    #########################
    def normalize(self):
        """
        Normalizes the state to have norm = 1
        """
        data = []
        N = np.sqrt(self @ self)
        for amp, basis_state in self.data:
            data.append([amp / N, basis_state])

        return State(data)

    def print_state(self, tol=0.1, probabilities=False):
        """
        Prints the state.
        """
        for amp, basis_state in self.data:
            if np.abs(amp) > tol:
                if probabilities:
                    amp = np.abs(amp) ** 2
                if np.real(complex(amp)) > 0:
                    print("+", end="")
                string = basis_state.print_quantum_numbers(printing=False)
                string = "{:.4f}".format(complex(amp)) + " x " + string
                print(string)

    def state_vector(self, QN):
        """
        Returns the state as a vector in the provided basis
        """
        state_vector = [1 * state @ self for state in QN]
        return np.array(state_vector, dtype=complex)

    def density_matrix(self, QN):
        """
        Generates a density matrix based on the state given and the provided basis.
        """
        # Get state vector
        state_vec = self.state_vector(QN)

        # Generate density matrix from state vector
        density_matrix = np.tensordot(state_vec.conj(), state_vec, axes=0)

        return density_matrix

    def remove_small_components(self, tol=1e-3):
        """
        Removes components that are smaller than tol from state
        """
        purged_data = []
        for amp, basis_state in self.data:
            if np.abs(amp) > tol:
                purged_data.append((amp, basis_state))

        return State(purged_data)

    def order_by_amp(self):
        """
        Ordes state components in descending order of |amp|^2
        """
        data = self.data
        amp_array = np.zeros(len(data))

        # Make an numpy array of the amplitudes
        for i, d in enumerate(data):
            amp_array[i] = np.abs((data[i][0])) ** 2

        # Find ordering of array in descending order
        index = np.argsort(-1 * amp_array)

        # Reorder data
        reordered_data = data
        reordered_data = [reordered_data[i] for i in index]

        return State(reordered_data)

    def print_largest_components(self, n=1):
        """
        Prints largest n component states
        """
        # Order the state by amplitude
        state = self.order_by_amp()

        # Initialize an empty string
        string = ""

        for i in range(0, n):
            basis_state = state.data[i][1]
            amp = state.data[i][0]
            basis_state.print_quantum_numbers()

        return string

    def find_largest_component(self):
        """
        Returns the largest component in state
        """
        # Order the state by amplitude
        state = self.order_by_amp()

        return state.data[0][1]
