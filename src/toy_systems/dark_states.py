"""
Utility functions for dealing with dark states.
"""
from typing import List, Tuple

import numpy as np

from .couplings import Coupling
from .hamiltonian import Hamiltonian
from .quantum_system import QuantumSystem
from .states import Basis, BasisState, State


def get_dark_states(
    ground_states: List[BasisState],
    excited_state: BasisState,
    couplings: List[Coupling],
    tol: float = 1e-6,
) -> Tuple[List[State], List[State], List[State]]:
    """
    Determines the bright and dark states for a given list of ground and excited
    states with given couplings.

    """
    # Define a basis
    basis = Basis(ground_states + [excited_state])

    # Convert the couplings into a Hamiltonian
    H = Hamiltonian(couplings, basis)

    # Determine the couplings to the excited state for each ground state
    mags = []
    bright = State()
    exc_vec = (1 * excited_state).get_state_vector(basis)
    for ground_state in ground_states:
        mags.append(
            exc_vec.T.conj() @ H.matrix @ (1 * ground_state).get_state_vector(basis)
        )
        bright += mags[-1] * ground_state

    # Normalize the bright state
    bright = bright.normalize()

    mags = np.array(mags)

    # Add states that have no coupling to excited state to list of polarization
    # dark_states
    index_pol_dark = np.abs(mags) < tol
    pol_dark_states = (1 * np.array(ground_states)[index_pol_dark]).tolist()

    # Remove polarization dark states from list of ground states
    ground_states = (np.array(ground_states)[~index_pol_dark]).tolist()

    # If all ground states were polarization dark, return
    if len(ground_states) == 0:
        return [], [], pol_dark_states

    # Generate a non-orthogonal basis of dark states
    dark_non = [
        (mags[i + 1] * ground_states[0] - mags[0] * gs).normalize()
        for i, gs in enumerate(ground_states[1:])
    ]

    # Generate orthogonal basis of dark states by using Gram-Schmidt process
    dark_ortho = [dark_non[0]]
    for v in dark_non[1:]:
        for u in dark_ortho:
            v -= v @ u

        dark_ortho.append(v)

    return [bright], dark_ortho, pol_dark_states
