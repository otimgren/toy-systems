from dataclasses import dataclass
from typing import List, Tuple

from .couplings import Coupling
from .decays import Decay, DecayList
from .hamiltonian import Hamiltonian
from .states import Basis


@dataclass
class QuantumSystem:
    """
    Class for representing quantum systems.

    A QuantumSystem is made up of a Basis of BasisStates that define the Hilbert
    space of the system, Couplings that describe the energies and couplings
    between the states, and Decays that describe the decays of quantum states.

    The Couplings are used to generate the Hamiltonian for the system, which
    together with the Decays can be used for time evolving the system.
    """

    basis: Basis
    couplings: List[Coupling]
    decays: List[Decay]

    def __post_init__(self) -> None:
        # Define a Hamiltonian
        self.H = Hamiltonian(self.couplings, basis=self.basis)

        # Define a Decay list
        self.C_list = DecayList(self.decays, basis=self.basis)

        # Classify states into excited and ground states
        self.excited_states = None
        self.ground_states = None
        self.classify_states()

    def get_qobjs(self) -> Tuple:
        """
        Returns a tuple of parameters required for time-evolution using QuTiP.

        outputs:
        self.H.qobjs    : List of tuples of qutip.Qobjs representing the
                          Hamiltonian and its time dependence

        self.C_list     : List of tuples of qutip.Qobjs representing the
                          Decays and their time dependence
        """
        # Combine the args of self.H.qobj and self.C_list
        self.H.qobj.args.update(self.C_list.args)

        return self.H.qobj, self.C_list.qobjs

    def classify_states(self) -> None:
        """
        Classifies the basis states as either ground or excited based on the
        provided decays.
        """
        excited = []
        for decay in self.decays:
            excited.append(decay.excited)

        ground = [gs for gs in self.basis.basis_states if gs not in excited]

        self.excited_states = Basis(list(excited), "excited_states")
        self.ground_states = Basis(list(ground), "ground_states")
