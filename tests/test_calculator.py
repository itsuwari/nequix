import os
import numpy as np
import pytest
import ase.build

from nequix.calculator import NequixCalculator


def test_calculator_nequix_mp_1():
    atoms = ase.build.bulk("C", "diamond", a=3.567, cubic=True)
    calc = NequixCalculator(model_path="models/nequix-mp-1.nqx")
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)
    print(energy, forces, stress)

    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))
