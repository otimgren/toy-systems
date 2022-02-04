from symtable import Symbol

from sympy import symbols

from toy_systems.basis import Basis
from toy_systems.couplings import ToyCoupling, ToyEnergy
from toy_systems.decays import Decay
from toy_systems.hamiltonian import Hamiltonian
from toy_systems.states import BasisState, JQuantumNumbers, QuantumNumbers, State


def main():

    # test quantum numbers, basis states and states
    quantum_numbers = JQuantumNumbers(J=1, mJ=0)
    basis_state = BasisState(quantum_numbers)
    print(basis_state)

    basis_state2 = BasisState(JQuantumNumbers(J=1, mJ=1))

    state = 1 * basis_state + 0.5 * basis_state2

    print(state.normalize())

    # Test basis
    basis_states = (
        BasisState(JQuantumNumbers(J=1, mJ=-1)),
        BasisState(JQuantumNumbers(J=1, mJ=0)),
        BasisState(JQuantumNumbers(J=1, mJ=+1)),
    )
    basis = Basis(basis_states, name="test")

    print(basis)
    basis.print_basis()

    # Test couplings and hamiltonian
    # Symbolic matrix
    coupling = ToyCoupling(
        state_a=basis_state, state_b=basis_state2, Omega=symbols("Omega")
    )
    print(coupling)

    energy2 = ToyEnergy(state=basis_state2, energy=symbols("Delta"))

    hamiltonian = Hamiltonian(
        basis=Basis((basis_state, basis_state2)), couplings=[coupling, energy2]
    )
    print(hamiltonian)

    # Generate the matrix representation of the hamiltonian
    hamiltonian.generate_matrix()
    print(hamiltonian)

    # Test decays
    decay = Decay(excited=basis_state2, ground=basis_state, gamma=symbols("gamma"))
    print(decay)
    decay.generate_decay_matrix(basis=Basis((basis_state, basis_state2)))
    print(decay.matrix)


if __name__ == "__main__":
    main()
