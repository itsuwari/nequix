import queue
import threading
import multiprocessing
from pathlib import Path

import ase
import ase.io
import ase.neighborlist
import h5py
import jax
import jraph
import matscipy.neighbours
import numpy as np
import yaml
from tqdm import tqdm
import json as _json
import os


def preprocess_graph(
    atoms: ase.Atoms,
    atom_indices: dict[int, int],
    cutoff: float,
    targets: bool,
) -> dict:
    """Convert ASE Atoms to a dict backing a jraph.GraphsTuple.

    - Uses matscipy neighbour list for periodic cells; falls back to ASE for molecules.
    - Targets (energy/forces/stress) are optional and filled safely downstream.
    """
    # Robust periodic detection; matscipy requires invertible cell
    cell = atoms.cell.array if hasattr(atoms.cell, "array") else np.asarray(atoms.cell)
    det = 0.0
    try:
        det = float(np.linalg.det(cell))
    except Exception:
        det = 0.0

    if atoms.pbc.any() and abs(det) > 1e-12:
        src, dst, shift = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
    else:
        src, dst = ase.neighborlist.neighbor_list("ij", atoms, cutoff)
        shift = np.zeros((len(src), 3), dtype=np.float32)

    graph_dict = {
        "n_node": np.array([len(atoms)]).astype(np.int32),
        "n_edge": np.array([len(src)]).astype(np.int32),
        "senders": dst.astype(np.int32),
        "receivers": src.astype(np.int32),
        "species": np.array([atom_indices[n] for n in atoms.get_atomic_numbers()]).astype(np.int32),
        "positions": atoms.positions.astype(np.float32),
        "shifts": shift.astype(np.float32),
        "cell": atoms.cell.astype(np.float32),
    }

    if targets:
        # Energy may be provided by calculator or atoms.info['energy'] (e.g., MSR-ACC JSON)
        energy = None
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            energy = atoms.info.get("energy")
        if energy is not None:
            graph_dict["energy"] = np.array([float(energy)], dtype=np.float32)

        # Forces/stress might not exist for energy-only datasets
        try:
            graph_dict["forces"] = atoms.get_forces().astype(np.float32)
        except Exception:
            pass
        try:
            graph_dict["stress"] = atoms.get_stress(voigt=False).astype(np.float32)
        except Exception:
            pass

    # Propagate molecular charge if available (defaults to 0 for neutral)
    try:
        ch = atoms.info.get("charge", 0.0)
    except Exception:
        ch = 0.0
    graph_dict["charge"] = np.array([float(ch)], dtype=np.float32)

    return graph_dict


def dict_to_graphstuple(graph_dict: dict) -> jraph.GraphsTuple:
    """Create GraphsTuple with safe defaults for missing targets.

    - forces: zeros [N,3] if absent
    - energy: [NaN] if absent (training typically requires energies present)
    - stress: zeros [3,3] if absent
    """
    n_nodes = graph_dict["positions"].shape[0]
    forces = graph_dict.get("forces")
    if forces is None:
        forces = np.zeros((n_nodes, 3), dtype=np.float32)
    energy = graph_dict.get("energy")
    if energy is None:
        energy = np.array([np.nan], dtype=np.float32)
    stress = graph_dict.get("stress")
    if stress is None:
        stress = np.zeros((3, 3), dtype=np.float32)
    charge = graph_dict.get("charge")
    if charge is None:
        charge = np.array([0.0], dtype=np.float32)

    return jraph.GraphsTuple(
        n_node=graph_dict["n_node"],
        n_edge=graph_dict["n_edge"],
        nodes={
            "species": graph_dict["species"],
            "positions": graph_dict["positions"],
            "forces": forces,
        },
        edges={"shifts": graph_dict["shifts"]},
        senders=graph_dict["senders"],
        receivers=graph_dict["receivers"],
        globals={
            "cell": graph_dict["cell"][None, ...],
            "energy": energy,
            "stress": stress[None, ...],
            "charge": charge,
        },
    )


def atomic_numbers_to_indices(atomic_numbers: list[int]) -> dict[int, int]:
    """Convert list of atomic numbers to dictionary of atomic number to index."""
    return {n: i for i, n in enumerate(sorted(atomic_numbers))}


def preprocess_file(
    file_path: str, atomic_indices: dict[int, int], cutoff: float
) -> list[dict]:  # Now returns list of dicts
    data = ase.io.read(file_path, index=":", format="extxyz")
    return [preprocess_graph(atoms, atomic_indices, cutoff, True) for atoms in data]


def save_graphs_to_hdf5(graphs, output_path, progress_bar=True, attrs: dict | None = None):
    """Save graphs to HDF5 file with optional file-level attributes."""
    with h5py.File(output_path, "w") as f:
        f.attrs["n_graphs"] = len(graphs)
        if attrs:
            for k, v in attrs.items():
                try:
                    f.attrs[k] = v
                except Exception:
                    pass
        for i, graph_dict in enumerate(
            tqdm(graphs, desc="saving graphs", disable=not progress_bar)
        ):
            grp = f.create_group(f"graph_{i}")
            for key, value in graph_dict.items():
                grp.create_dataset(key, data=value)

def _atoms_from_msracc_json(path: Path) -> ase.Atoms:
    """Create ASE Atoms from an MSR-ACC JSON file.

    - Reads atomic numbers and geometry
    - Selects an atomization energy (TAE, Hartree) and converts to eV in atoms.info['energy'].
      Stored convention: energy = +TAE[eV] (positive binding/atomization energy).
    """
    with open(path, "r") as f:
        data = _json.load(f)
    if "atomic_numbers" in data:
        Z = np.array(data["atomic_numbers"], dtype=np.int32)
    else:
        # Some variants store element symbols
        from ase.data import chemical_symbols

        Z = np.array([chemical_symbols.index(s) for s in data["elements"]], dtype=np.int32)
    geom = np.array(data.get("geometry"), dtype=np.float32).reshape(-1, 3)
    atoms = ase.Atoms(numbers=Z, positions=geom, pbc=False)

    # Energy selection for MSR-ACC
    # Priority:
    # 1) If NEQUIX_MSRACC_ENERGY_MODE is set to 'ccsd_all' or 'ccsd_val', compute from W1-F12 components
    # 2) Else, if W1-F12 components exist, default to 'ccsd_all' (HF + delta-CCSD + delta-CV)
    # 3) Else fall back to explicit single-key energies, with optional NEQUIX_MSRACC_ENERGY_KEY override
    extras = data.get("extras", {})
    energy_h = None

    def has(key):
        return (key in extras) and (extras[key] is not None)

    # 0) Explicit single-key override takes precedence if provided
    override_key = os.environ.get("NEQUIX_MSRACC_ENERGY_KEY")
    if override_key and has(override_key):
        energy_h = float(extras[override_key])

    mode = os.environ.get("NEQUIX_MSRACC_ENERGY_MODE", "").strip().lower()
    if energy_h is None and (mode in {"ccsd_all", "ccsd_val"} or (has("tae[HF]@w1-f12") and has("tae[delta-CCSD]@w1-f12"))):
        # Build CCSD valence; add core-valence if required/available
        if has("tae[HF]@w1-f12") and has("tae[delta-CCSD]@w1-f12"):
            energy_h = float(extras["tae[HF]@w1-f12"]) + float(extras["tae[delta-CCSD]@w1-f12"])
            use_cv = (mode == "ccsd_all") or (mode == "" and has("tae[delta-CV]@w1-f12"))
            if use_cv and has("tae[delta-CV]@w1-f12"):
                energy_h += float(extras["tae[delta-CV]@w1-f12"])
        # If components missing, fall through to single-key strategy

    if energy_h is None:
        candidates = [
            "tae@ccsd(t)/6-31g*",
            "tae@w1-f12",
        ]
        for k in candidates:
            if k and k in extras and extras[k] is not None:
                energy_h = float(extras[k])
                break
    if energy_h is not None:
        # MSR-ACC stores atomization energies (TAE) in Hartree. Use positive binding energy in eV.
        # Convention: atoms.info['energy'] = +TAE[eV]
        atoms.info["energy"] = energy_h * 27.211386245988  # Hartree → eV, positive
    # Also carry molecular charge if present
    if "molecular_charge" in data:
        try:
            atoms.info["charge"] = float(data["molecular_charge"])  # usually 0.0
        except Exception:
            atoms.info["charge"] = 0.0
    return atoms


def process_worker_files(args):
    """Process a chunk of files (.extxyz or .json) for one worker."""
    worker_id, file_paths, output_path, atomic_indices, cutoff = args
    all_graphs = []
    for file_path in tqdm(file_paths, desc="reading graphs", disable=worker_id != 0):
        fp = Path(file_path)
        if fp.suffix == ".extxyz":
            data = ase.io.read(fp, index=":", format="extxyz")
            graphs = [preprocess_graph(atoms, atomic_indices, cutoff, True) for atoms in data]
            all_graphs.extend(graphs)
        elif fp.suffix == ".json":
            atoms = _atoms_from_msracc_json(fp)
            graphs = [preprocess_graph(atoms, atomic_indices, cutoff, True)]
            all_graphs.extend(graphs)
    # Annotate energy type when saving JSON (TAE in eV) for downstream checks
    attrs = {"energy_type": "tae_eV"} if (len(file_paths) > 0 and str(file_paths[0]).endswith('.json')) else None
    save_graphs_to_hdf5(all_graphs, output_path, progress_bar=worker_id == 0, attrs=attrs)
    return len(all_graphs)


# pytorch-like dataset that reads xyz files and returns jraph.GraphsTuple
class Dataset:
    def __init__(
        self,
        file_path: str,
        atomic_numbers: list[int],
        cache_dir: str = None,
        split: str = None,
        cutoff: float = 5.0,
        valid_frac: float = 0.1,
        seed: int = 42,
    ):
        self.atomic_indices = atomic_numbers_to_indices(atomic_numbers)
        file_path = Path(file_path)
        cache_dir = Path(cache_dir) if cache_dir is not None else file_path.parent
        cache_dir = cache_dir / f"{file_path.stem}_cutoff_{cutoff}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.hdf5_files = sorted(cache_dir.glob("chunk_*.h5"))

        if not self.hdf5_files:
            self._create_cache(file_path, cache_dir, cutoff)
            self.hdf5_files = sorted(cache_dir.glob("chunk_*.h5"))

        self._file_handles = None

        self.index_map = []
        for file_idx, file_handle in enumerate(self.file_handles):
            n_graphs = file_handle.attrs["n_graphs"]
            for local_idx in range(n_graphs):
                self.index_map.append((file_idx, local_idx))

        if split is not None:
            rng = np.random.RandomState(seed=seed)
            perm = rng.permutation(len(self.index_map))
            train_idx, valid_idx = np.split(perm, [int(len(perm) * (1 - valid_frac))])
            indices = train_idx if split == "train" else valid_idx
            self.index_map = [self.index_map[i] for i in indices]

    @property
    def file_handles(self):
        if self._file_handles is None:
            self._file_handles = [h5py.File(hdf5_file, "r") for hdf5_file in self.hdf5_files]
        return self._file_handles

    def __getstate__(self):
        state = self.__dict__.copy()
        # file handles are not picklable
        state["_file_handles"] = None
        return state

    def _create_cache(self, file_path, cache_dir, cutoff):
        if file_path.is_dir():
            # Prefer EXTXYZ; if none, use JSON (MSR-ACC)
            file_paths = sorted(file_path.glob("*.extxyz"))
            if not file_paths:
                file_paths = sorted(file_path.glob("*.json"))
            # Number of workers: bounded by CPU and number of files
            import os as _os
            cpu = max(1, int((_os.cpu_count() or 8)))
            env_w = int(_os.environ.get("NEQUIX_PREPROCESS_WORKERS", "0") or 0)
            n_workers = env_w if env_w > 0 else min(8, cpu, max(1, len(file_paths)))
            chunk_size = len(file_paths) // n_workers + 1
            tasks = []
            for worker_id in range(n_workers):
                start = worker_id * chunk_size
                end = min(start + chunk_size, len(file_paths))
                if start < len(file_paths):
                    worker_files = file_paths[start:end]
                    output_path = cache_dir / f"chunk_{worker_id:04d}.h5"
                    tasks.append(
                        (
                            worker_id,
                            worker_files,
                            output_path,
                            self.atomic_indices,
                            cutoff,
                        )
                    )

            if tasks:
                # Use spawn context to avoid forking a multithreaded JAX process
                ctx = multiprocessing.get_context("spawn")
                # Use small maxtasksperchild to bound memory; unordered to avoid head-of-line blocking
                with ctx.Pool(n_workers, maxtasksperchild=8) as p:
                    for _ in tqdm(p.imap_unordered(process_worker_files, tasks), total=len(tasks)):
                        pass
        else:
            # Single file; assume EXTXYZ
            data = ase.io.read(file_path, index=":", format="extxyz")
            graphs = [
                preprocess_graph(atoms, self.atomic_indices, cutoff, True) for atoms in tqdm(data)
            ]
            save_graphs_to_hdf5(graphs, cache_dir / "chunk_0000.h5")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> jraph.GraphsTuple:
        file_idx, local_idx = self.index_map[idx]
        grp = self.file_handles[file_idx][f"graph_{local_idx}"]
        graph_dict = {}
        for key in grp:
            graph_dict[key] = grp[key][:]
        return dict_to_graphstuple(graph_dict)

    def __del__(self):
        fhs = getattr(self, "_file_handles", None)
        if fhs:
            for fh in fhs:
                try:
                    fh.close()
                except Exception:
                    pass


def _dataloader_worker(dataset, index_queue, output_queue):
    while True:
        try:
            # Block briefly to avoid busy-spin; exit cleanly on sentinel
            index = index_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if index is None:
            break
        output_queue.put((index, dataset[index]))


# multiprocess data loader with dynamic batching, based on
# https://teddykoker.com/2020/12/dataloader/
# https://github.com/google-deepmind/jraph/blob/51f5990/jraph/ogb_examples/data_utils.py
class DataLoader:
    def __init__(
        self,
        dataset,
        max_n_nodes: int,
        max_n_edges: int,
        avg_n_nodes: int,
        avg_n_edges: int,
        batch_size=1,
        seed=0,
        shuffle=False,
        buffer_factor=1.1,
        graphs_buffer_factor=2.0,
        num_workers=4,
        prefetch_factor=2,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.idxs = np.arange(len(self.dataset))
        self.idx = 0
        self._generator = None  # created in __iter__
        self.n_node = max(batch_size * avg_n_nodes * buffer_factor, max_n_nodes) + 1
        self.n_edge = max(batch_size * avg_n_edges * buffer_factor, max_n_edges)
        # Allow packing more small graphs if budgets permit
        self.n_graph = max(batch_size + 1, int(batch_size * graphs_buffer_factor))
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._started = False
        self.index_queue = None
        self.output_queue = None
        self.workers = []
        self.prefetch_idx = 0

    def _start_workers(self):
        if self._started:
            return

        # Use a spawn context to avoid os.fork from a multithreaded JAX process
        self._started = True
        ctx = multiprocessing.get_context("spawn")
        self.index_queue = ctx.Queue()
        self.output_queue = ctx.Queue()

        for _ in range(self.num_workers):
            worker = ctx.Process(
                target=_dataloader_worker,
                args=(self.dataset, self.index_queue, self.output_queue),
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(seed=hash((self.seed, epoch)) % 2**32)

    def _prefetch(self):
        prefetch_limit = self.idx + self.prefetch_factor * self.num_workers * self.batch_size
        while self.prefetch_idx < len(self.dataset) and self.prefetch_idx < prefetch_limit:
            self.index_queue.put(self.idxs[self.prefetch_idx])
            self.prefetch_idx += 1

    def make_generator(self):
        cache = {}
        self.prefetch_idx = 0

        while True:
            if self.idx >= len(self.dataset):
                return

            self._prefetch()

            real_idx = self.idxs[self.idx]

            if real_idx in cache:
                item = cache[real_idx]
                del cache[real_idx]
            else:
                while True:
                    try:
                        (index, data) = self.output_queue.get(timeout=0)
                    except queue.Empty:
                        continue

                    if index == real_idx:
                        item = data
                        break
                    else:
                        cache[index] = data

            yield item
            self.idx += 1

    def __iter__(self):
        self._start_workers()
        self.idx = 0
        if self.shuffle:
            self.idxs = self.rng.permutation(np.arange(len(self.dataset)))
        self._generator = jraph.dynamically_batch(
            self.make_generator(),
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
        )
        return self

    def __next__(self):
        return next(self._generator)


class ParallelLoader:
    def __init__(self, loader: DataLoader, n: int):
        self.loader = loader
        self.n = n

    def __iter__(self):
        it = iter(self.loader)
        while True:
            try:
                yield jax.tree.map(lambda *x: np.stack(x), *[next(it) for _ in range(self.n)])
            except StopIteration:
                return


# simple threaded prefetching for dataloader (lets us build our dyanamic batches async)
def prefetch(loader, queue_size=4):
    q = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()

    def worker():
        try:
            for item in loader:
                if stop_event.is_set():
                    return
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            elif isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        thread.join(timeout=1.0)


# def default_collate_fn(graphs):
#     # NB: using batch_np is considerably faster than batch with jax
#     return pad_graph_to_nearest_power_of_two(jraph.batch_np(graphs))


# based on https://github.com/ACEsuit/mace/blob/d39cc6b/mace/data/utils.py#L300
def average_atom_energies(dataset: Dataset) -> list[float]:
    """Compute the average energy of each species in the dataset."""
    atomic_indices = dataset.atomic_indices
    A = np.zeros((len(dataset), len(atomic_indices)), dtype=np.float32)
    B = np.zeros((len(dataset),), dtype=np.float32)
    for i, graph in tqdm(enumerate(dataset), total=len(dataset)):
        A[i] = np.bincount(graph.nodes["species"], minlength=len(atomic_indices))
        B[i] = graph.globals["energy"][0]
    E0s = np.linalg.lstsq(A, B, rcond=None)[0].tolist()
    idx_to_atomic_number = {v: k for k, v in atomic_indices.items()}
    atom_energies = {idx_to_atomic_number[i]: e0 for i, e0 in enumerate(E0s)}
    print("computed energies, add to config yml file to avoid recomputing:")
    print(yaml.dump({"atom_energies": atom_energies}))
    return E0s


def dataset_stats(dataset: Dataset, atom_energies: list[float]) -> dict:
    """Compute the statistics of the dataset."""
    energies, forces_list, n_neighbors, n_nodes, n_edges = [], [], [], [], []
    atom_energies = np.array(atom_energies)
    # Track species coverage to detect elements not present in the dataset
    n_species = atom_energies.shape[0]
    species_counts = np.zeros((n_species,), dtype=np.int64)
    for graph in tqdm(dataset, total=len(dataset)):
        graph_e0 = np.sum(atom_energies[graph.nodes["species"]])
        # graph.n_node is [1]; get scalar
        nn = int(np.array(graph.n_node).reshape(-1)[0])
        ne = int(np.array(graph.n_edge).reshape(-1)[0])
        # species coverage
        sc = np.bincount(graph.nodes["species"], minlength=n_species)
        species_counts += sc
        energies.append((graph.globals["energy"][0] - graph_e0) / nn)
        f = graph.nodes.get("forces", None)
        if not isinstance(f, np.ndarray) or f.ndim != 2 or (f.shape[1] if f.ndim>=2 else 0) != 3:
            f = np.zeros((nn, 3), dtype=np.float32)
        forces_list.append(f)
        n_neighbors.append(ne / max(nn, 1))
        n_nodes.append(nn)
        n_edges.append(ne)
    mean = float(np.mean(np.array(energies, dtype=np.float32))) if energies else 0.0
    rms = float(np.sqrt(np.mean(np.concatenate(forces_list, axis=0) ** 2))) if forces_list else 0.0
    # For energy-only datasets, forces are zeros → rms≈0. Clamp to 1.0 to avoid degenerate scale.
    if float(abs(rms)) < 1e-8:
        rms = 1.0
    n_neighbors = float(np.mean(np.array(n_neighbors, dtype=np.float32))) if n_neighbors else 0.0
    stats = {
        "shift": float(mean),
        "scale": float(rms),
        "avg_n_neighbors": float(n_neighbors),
        "avg_n_nodes": float(np.mean(n_nodes)) if n_nodes else 0.0,
        "avg_n_edges": float(np.mean(n_edges)) if n_edges else 0.0,
        "max_n_nodes": int(np.max(n_nodes)) if n_nodes else 0,
        "max_n_edges": int(np.max(n_edges)) if n_edges else 0,
        "species_counts": species_counts.tolist(),
    }
    print("computed dataset statistics, add to config yml file to avoid recomputing:")
    print(yaml.dump(stats))
    return stats
