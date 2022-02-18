from abc import ABC, abstractmethod
from collections import defaultdict
from cProfile import label
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .quantum_system import QuantumSystem
from .states import BasisState
from .utils import sub_visualization_labels


class Visualization:
    """
    Parent class for visualizations of quantum systems.
    """

    @abstractmethod
    def plot(self, ax):
        """
        Applies the visualization onto provided axes
        """


class Visualizer:
    """
    Class for visualizing a quantum system
    """

    def __init__(
        self,
        quantum_system: QuantumSystem,
        vertical: dict,
        horizontal: dict,
    ):
        self.qs = quantum_system

        # For convenience, store the states of the quantum system as an attribute
        self.states = quantum_system.basis.basis_states

        # Generate state visualizations
        self.state_vis = self.generate_state_vis(vertical, horizontal)

        # Generate coupling visualizations
        self.coupling_vis = self.generate_coupling_vis()

        # Generate decay visualizations
        self.decay_vis = self.generate_decay_vis()

        self.plot()

    def plot(self):
        """
        Plot the visualization of the quantum system.
        """
        # Initialize figure and axes
        self.fig, self.ax = plt.subplots(figsize=(16, 9))
        self.fig.set_facecolor("xkcd:pale grey")

        # Turn off everything for the axes
        plt.axis("off")

        for vis in self.coupling_vis:
            vis.plot(self.ax)

        for vis in self.decay_vis:
            vis.plot(self.ax)

        for vis in self.state_vis:
            vis.plot(self.ax)

    def generate_state_vis(
        self,
        vertical: dict,
        horizontal: dict,
    ):
        """
        Generates state visualizations for the provided states.

        The coordinates for each state are provided based on the lists vertical
        and later, which provide pairs of quantum numbers and and the spacings
        related to those.


        """
        # First find the values that the quantum numbers in the lists
        # vertical and lateral can take so can determine positions of states
        qn_values_v = {}
        for qn in vertical.keys():
            qn_values_v[qn] = self._find_QN_values(self.states, qn)

        qn_values_h = {}
        for qn in horizontal.keys():
            qn_values_h[qn] = self._find_QN_values(self.states, qn)

        # Loop over states and generate visualization objects for them
        state_vis = []
        for state in self.states:
            x = 0
            y = 0
            label_offset = (0, -2)

            # Excited states plotted higher than other states
            if state in self.qs.excited_states.basis_states:
                y += 50
                label_offset = (0, 2)

            # Determine the position where the state should be plotted
            for name, value in state.qn:
                if name in vertical.keys():
                    y += vertical[name] * qn_values_v[name].index(value)
                if name in horizontal.keys():
                    x += horizontal[name] * qn_values_h[name].index(value)

            state_vis.append(
                StateVis(pos=(x, y), label=state.__repr__(), label_offset=label_offset)
            )

        return state_vis

    def _find_QN_values(self, states: List[BasisState], qn: str):
        """
        Finds the unique values that a quantum number takes within states
        and returns them in an ordered list
        """
        values = []
        for state in states:
            values.append(getattr(state.qn, qn))

        # Find unique values and sort
        values = list(set(values))
        values.sort()

        return values

    def generate_coupling_vis(self):
        """
        Generates visualizations for couplings.
        """
        coupling_vis = []
        for coupling in self.qs.couplings:
            # Use the matrix of the coupling to determine which states are coupled
            if coupling.matrix_sym is not None:
                M = np.triu(coupling.matrix_sym, k=1)
            elif coupling.matrix is not None:
                M = np.triu(coupling.matrix, k=1)
            else:
                coupling.generate_matrix(self.qs.basis)
                M = np.triu(coupling.matrix, k=1)

            rows, columns = np.nonzero(M)
            for i, j in zip(rows, columns):
                # Find the visualizations of the coupled states
                state1_vis = self.state_vis[i]
                state2_vis = self.state_vis[j]

                # Find positions of the state visualizations
                pos1 = state1_vis.pos
                pos2 = state2_vis.pos

                # Calculate midpoint between the states
                x = (pos1[0] + pos2[0]) / 2 + 1.5
                y = (pos1[1] + pos2[1]) / 2

                # Generate coupling visualization
                coupling_vis.append(CouplingVis((x, y), pos1, pos2, M[i, j]))

        return coupling_vis

    def generate_decay_vis(self):
        """
        Generates visualizations for couplings.
        """
        decay_vis = []
        for decay in self.qs.decays:
            # Use the matrix of the coupling to determine which states are coupled
            if decay.matrix_sym is not None:
                M = decay.matrix_sym
            elif decay.matrix is not None:
                M = decay.matrix
            else:
                decay.generate_matrix(self.qs.basis)
            rows, columns = np.nonzero(M)
            for i, j in zip(rows, columns):
                # Find the visualizations of the coupled states
                state1_vis = self.state_vis[i]
                state2_vis = self.state_vis[j]

                # Find positions of the state visualizations
                pos1 = state1_vis.pos
                pos2 = state2_vis.pos

                # Calculate midpoint between the states
                x = (pos1[0] + pos2[0]) / 2 + 1.5
                y = (pos1[1] + pos2[1]) / 2

                # Generate coupling visualization
                decay_vis.append(DecayVis((x, y), pos1, pos2, M[i, j]))

        return decay_vis


@dataclass
class StateVis:
    """
    Class for visualizing a state.
    """

    pos: Tuple[float]
    label: str = None
    label_offset: Tuple[float] = (0, -2)
    length: float = 50

    def __post_init__(self):
        self.label = sub_visualization_labels(str(self.label))
        self.label = rf"$  {self.label}$"

    def plot(self, ax: plt.Axes):
        """
        Plots the visualization onto given axes
        """
        ax.hlines(
            y=self.pos[1],
            xmin=self.pos[0] - self.length / 2,
            xmax=self.pos[0] + self.length / 2,
            colors="k",
            linewidth=2,
        )

        ax.annotate(
            self.label,
            xy=(self.pos[0] + self.label_offset[0], self.pos[1] + self.label_offset[1]),
            fontsize=16,
        )


@dataclass
class CouplingVis:
    """
    Class for visualizing a coupling
    """

    xy: Tuple[float]
    pos1: Tuple[float]
    pos2: Tuple[float]
    label: str = None

    def __post_init__(self):
        self.label = sub_visualization_labels(str(self.label))
        self.label = rf"$  {self.label}$"

    def plot(self, ax: plt.Axes):
        """
        Plots the visualization onto given axes.

        Makes two arrows with text showing strength of coupling in the middle.
        """

        arrow_args = {
            "arrowstyle": "<|-|>",
            "color": "tab:blue",
            "linewidth": 4,
            "connectionstyle": "arc3",
            "alpha": 0.5,
        }
        ax.annotate("", self.pos1, xytext=self.pos2, arrowprops=arrow_args)
        ax.annotate(self.label, self.xy, fontsize=16)


@dataclass
class DecayVis:
    """
    Class for visualizing a decay
    """

    xy: Tuple[float]
    pos1: Tuple[float]
    pos2: Tuple[float]
    label: str = None

    def __post_init__(self):
        self.label = sub_visualization_labels(str(self.label))
        self.label = rf"$ {self.label}$"

    def plot(self, ax: plt.Axes):
        """
        Plots the visualization onto given axes.

        Makes two arrows with text showing strength of coupling in the middle.
        """

        arrow_args = {
            "arrowstyle": "-",
            "color": "tab:red",
            "linewidth": 4,
            "connectionstyle": "arc3",
            "alpha": 0.5,
        }
        with mpl.rc_context({"path.sketch": (5, 20, 1)}):
            ax.annotate("", self.pos1, xytext=self.pos2, arrowprops=arrow_args)
        ax.annotate(self.label, self.xy, fontsize=16)


@dataclass
class EnergyVis:
    """
    Class for labeling energies
    """
