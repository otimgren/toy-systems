from symtable import Symbol

import qutip
from sympy import symbols

from toy_systems.couplings import ToyCoupling, ToyEnergy
from toy_systems.decays import Decay
from toy_systems.hamiltonian import Hamiltonian
from toy_systems.states import Basis, BasisState, JQuantumNumbers


def main():
    def test_func(x):
        return 2 * x

    result = qutip.parallel_map(test_func, [1, 2, 3])

    print(result)


if __name__ == "__main__":
    main()
