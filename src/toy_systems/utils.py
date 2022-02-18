"""
Utility functions stored here
"""
from operator import ne
from typing import List

import numpy as np
import qutip
import regex as re

from .constants import qutip_to_numpy, visualization_subs
from .states import Basis, State


def threej(j1, j2, j3, m1, m2, m3) -> float:
    """
    Calculates the 3j-symbol as defined in Wikipedia:
    https://en.wikipedia.org/wiki/3-j_symbol
    """
    # Convert everything to float
    j1 = float(j1)
    j2 = float(j2)
    j3 = float(j3)
    m1 = float(m1)
    m2 = float(m2)
    m3 = float(m3)

    return (
        (-1) ** (j1 - j2 - m3)
        / np.sqrt(2 * j3 + 1)
        * qutip.clebsch(j1, j2, j3, m1, m2, -m3)
    )


def generate_P_op(states: List[State], basis: Basis) -> qutip.Qobj:
    """
    Generates an operator used to determine the summed population in the provided states.
    """
    return qutip.Qobj(
        np.sum([(1 * s).density_matrix(basis) for s in states], 0), type="oper"
    )


def convert_qutip_to_numpy(string: str):
    """
    Converts a qutip str expression to something that can be used by
    numexpr (basically asin -> arscin etc.)
    """
    new_string = string
    for key, value in qutip_to_numpy.items():
        new_string = re.sub(key, value, new_string)

    return new_string


def sub_visualization_labels(string: str):
    """
    Converts Greek letters to form \[letter] so plots will render them nicely
    """
    new_string = string
    for key, value in visualization_subs.items():
        new_string = re.sub(key, value, new_string)

    return new_string
