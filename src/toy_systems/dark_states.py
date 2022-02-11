"""
Utility functions for dealing with dark states.
"""
from typing import List, Tuple

from .couplings import Coupling
from .hamiltonian import Hamiltonian
from .quantum_system import QuantumSystem
from .states import Basis, BasisState, State


def get_dark_states(
    ground_states: List[BasisState],
    excited_state: BasisState,
    couplings: List[Coupling],
) -> Tuple[State, List[State]]:
    """
    Determines the bright and dark states for a given list of ground and excited
    states with given couplings
    """
    # Define a basis
    basis = Basis(ground_states + [excited_state])

    # Convert the couplings into a Hamiltonian
    H = Hamiltonian(couplings, basis)

    # Determine the couplings to the excited state for each ground state
    mags = []
    bright = State()
    exc_vec = (1 * excited_state).state_vector(basis)
    for ground_state in ground_states:
        mags.append(
            exc_vec.T.conj() @ H.matrix @ (1 * ground_state).state_vector(basis)
        )
        bright += mags[-1] * ground_state

    # Normalize the bright staet
    bright = bright.normalize()

    # Generate a non-orthogonal basis of dark states
    dark_non = [
        (mags[0] * ground_states[0] - mags[i] * gs).normalize()
        for i, gs in enumerate(ground_states[1:])
    ]

    # Generate orthogonal basis of dark states by using Gram-Schmidt process
    dark_ortho = []
    for v in dark_non:
        for u in dark_ortho:
            v -= v @ u

        dark_ortho.append(v)

    return bright, dark_ortho
