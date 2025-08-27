from pathlib import Path

import equinox as eqx
import jraph
import numpy as np
import urllib.request
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequix.data import (
    atomic_numbers_to_indices,
    dict_to_graphstuple,
    preprocess_graph,
)
from nequix.model import load_model


class NequixCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    URLS = {"nequix-mp-1": "https://figshare.com/files/57245573"}

    def __init__(
        self,
        model_name: str = "nequix-mp-1",
        model_path: str = None,
        capacity_multiplier: float = 1.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_path is None:
            model_path = Path("~/.cache/nequix/models/").expanduser() / f"{model_name}.nqx"
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(self.URLS[model_name], model_path)

        self.model, self.config = load_model(model_path)
        self.atom_indices = atomic_numbers_to_indices(self.config["atomic_numbers"])
        self._capacity = None
        self._capacity_multiplier = capacity_multiplier

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        graph = dict_to_graphstuple(
            preprocess_graph(atoms, self.atom_indices, self.model.cutoff, False)
        )

        # maintain edge capacity with _capacity_multiplier over edges,
        # recalculate if numbers (system) changes
        if self._capacity is None or ("numbers" in system_changes):
            self._capacity = int(np.ceil(graph.n_edge[0] * self._capacity_multiplier))
        elif graph.n_edge[0] > self._capacity:
            self._capacity = int(np.ceil(self._capacity * self._capacity_multiplier))

        padded_graph = jraph.pad_with_graphs(
            graph, n_node=graph.n_node[0] + 1, n_edge=self._capacity, n_graph=2
        )
        energy, forces, stress = eqx.filter_jit(self.model)(padded_graph)

        # take energy and forces without padding
        energy = np.array(energy[0])
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = np.array(forces[: len(atoms)])
        self.results["stress"] = full_3x3_to_voigt_6_stress(np.array(stress[0]))
