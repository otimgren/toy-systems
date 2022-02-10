from setuptools import find_packages, setup

VERSION = "1.0"
DESCRIPTION = "Package for making toy models for quantum physics."

setup(
    name="toy_systems",
    version=VERSION,
    author="Oskari Timgren",
    description=DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy>=1.21.5", "sympy>=1.9", "qutip>=4.6.2"],
)
