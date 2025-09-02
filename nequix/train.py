import argparse
import functools
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import yaml
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook

from nequix.data import (
    DataLoader,
    Dataset,
    ParallelLoader,
    average_atom_energies,
    dataset_stats,
    prefetch,
)
from nequix.model import Nequix, NequixMoE, save_model, weight_decay_mask, load_model, node_graph_idx


@eqx.filter_jit
def loss(
    model,
    batch,
    energy_weight,
    force_weight,
    stress_weight,
    loss_type="huber",
    lambda_inf: float = 0.0,
    rng: jax.Array | None = None,
    force_norm_weight: float = 0.0,
    teacher=None,
    distill_force_weight: float = 0.0,
    distill_stress_weight: float = 0.0,
    ckpt_l2_weight: float = 0.0,
):
    """Return huber loss and MAE of energy and force in eV and eV/Å respectively"""
    energy, forces, stress = model(batch)
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    config = {
        "mse": {"energy": "mse", "force": "mse", "stress": "mse"},
        "huber": {"energy": "huber", "force": "huber", "stress": "huber"},
        "mae": {"energy": "mae", "force": "l2", "stress": "mae"},
    }[loss_type]

    loss_fns = {
        "mae": lambda pred, true: jnp.abs(pred - true),
        "mse": lambda pred, true: (pred - true) ** 2,
        "huber": lambda pred, true: optax.losses.huber_loss(pred, true, delta=0.1),
    }

    # Optionally switch to TAE mode: E_tae = sum(E_atom) - E_mol
    # In this mode, dataset energy should be +TAE (positive binding energy, in eV).
    # If the model is constructed with include_atom_priors_in_energy=False, its energy head can
    # directly predict E_tae and be compared to the dataset energy without transformation.
    energy_mode = batch.globals.get("energy_mode", jnp.array(0, dtype=jnp.int32))
    # compute graph-wise sum of atomic energies if needed
    def _sum_atom():
        gidx = node_graph_idx(batch)
        ae_nodes = model.atom_energies[batch.nodes["species"]]
        return jraph.segment_sum(ae_nodes, gidx, num_segments=batch.n_node.shape[0])

    sum_atom = _sum_atom()
    # predicted energy to compare: either E_mol (default) or E_tae
    # If training on TAE and the model already excludes atomic priors (i.e., predicts binding/interaction energy),
    # compare dataset target directly to model energy. Otherwise use legacy (sum_atom - energy).
    def _pred_for_mode(e):
        return jnp.where(
            energy_mode == 1,
            jnp.where(getattr(model, "include_atom_priors_in_energy", True), sum_atom - e, e),
            e,
        )
    pred_energy = _pred_for_mode(energy)
    # true energy: use dataset-provided energy.
    # For TAE runs, dataset energy is set to binding energy (E_bind = +TAE), see data preprocessing.
    true_energy = batch.globals["energy"]

    # energy per atom loss
    energy_loss_per_atom = jnp.sum(
        loss_fns[config["energy"]](pred_energy / batch.n_node, true_energy / batch.n_node) * graph_mask
    ) / jnp.sum(graph_mask)

    if config["force"] == "l2":
        # l2 norm loss for forces
        # NOTE: double where trick is needed to avoid nan's
        force_diff_squared = jnp.sum((forces - batch.nodes["forces"]) ** 2, axis=-1)
        safe_force_diff_squared = jnp.where(force_diff_squared == 0.0, 1.0, force_diff_squared)
        force_loss = jnp.sum(
            jnp.where(force_diff_squared == 0.0, 0.0, jnp.sqrt(safe_force_diff_squared)) * node_mask
        ) / jnp.sum(node_mask)
    else:
        force_loss = jnp.sum(
            loss_fns[config["force"]](forces, batch.nodes["forces"]) * node_mask[:, None]
        ) / (3 * jnp.sum(node_mask))

    # Optional force norm regularizer (helps keep forces bounded on energy-only datasets)
    pred_force_norm = jnp.sum(jnp.abs(forces) * node_mask[:, None]) / (3 * jnp.sum(node_mask))

    if stress_weight > 0:
        stress_loss = jnp.sum(
            loss_fns[config["stress"]](stress, batch.globals["stress"]) * graph_mask[:, None, None]
        ) / (9 * jnp.sum(graph_mask))
    else:
        stress_loss = 0

    total_loss = (
        energy_weight * energy_loss_per_atom
        + force_weight * force_loss
        + stress_weight * stress_loss
    )
    total_loss = total_loss + jnp.asarray(force_norm_weight, dtype=total_loss.dtype) * pred_force_norm

    # Track short- vs long-range energy components for logging
    try:
        e_short_vec = model.short_energy(batch)
        e_total_vec = energy  # model output energy per graph
        e_lr_vec = jnp.where(getattr(model, "use_qeq_head", False), e_total_vec - e_short_vec, jnp.zeros_like(e_short_vec))
        graph_mask = jraph.get_graph_padding_mask(batch).astype(e_short_vec.dtype)
        energy_short = jnp.sum(e_short_vec * graph_mask) / jnp.clip(jnp.sum(graph_mask), a_min=1.0)
        energy_lr = jnp.sum(e_lr_vec * graph_mask) / jnp.clip(jnp.sum(graph_mask), a_min=1.0)
    except Exception:
        energy_short = jnp.array(0.0, dtype=jnp.float32)
        energy_lr = jnp.array(0.0, dtype=jnp.float32)

    # Optional dissociation penalty: encourage short-range head to vanish at large separations
    # Compute unconditionally and scale by lambda_inf to stay JIT-safe.
    try:
        # Per-graph random scaling factor in [1.5, 2.0]
        n_graph = batch.n_node.shape[0]
        if rng is None:
            rng = jax.random.PRNGKey(0)
        lam_rng = 1.5 + 0.5 * jax.random.uniform(rng, (n_graph, 1), dtype=batch.nodes["positions"].dtype)

        gidx = node_graph_idx(batch)
        # Compute center of mass per graph and scale displacements from COM
        ones = jnp.ones((batch.nodes["positions"].shape[0], 1), dtype=batch.nodes["positions"].dtype)
        sum_pos = jraph.segment_sum(batch.nodes["positions"], gidx, num_segments=n_graph)
        counts = jraph.segment_sum(ones, gidx, num_segments=n_graph)
        com = sum_pos / jnp.clip(counts, a_min=1.0, a_max=None)
        pos_scaled = com[gidx] + lam_rng[gidx] * (batch.nodes["positions"] - com[gidx])

        # Replace positions and compute per-graph short energy
        batch_scaled = batch._replace(nodes={**batch.nodes, "positions": pos_scaled})
        e_short = model.short_energy(batch_scaled)
        graph_mask = jraph.get_graph_padding_mask(batch).astype(e_short.dtype)
        l_inf = jnp.sum(jnp.abs(e_short) * graph_mask) / jnp.clip(jnp.sum(graph_mask), a_min=1.0)
        lam_w = jnp.asarray(lambda_inf, dtype=l_inf.dtype)
        total_loss = total_loss + lam_w * l_inf
    except Exception:
        pass

    # metrics:

    # MAE energy (per-atom), computed on selected mode
    energy_mae_per_atom = jnp.sum(
        jnp.abs(pred_energy / batch.n_node - true_energy / batch.n_node) * graph_mask
    ) / jnp.sum(graph_mask)

    # MAE forces
    force_mae = jnp.sum(jnp.abs(forces - batch.nodes["forces"]) * node_mask[:, None]) / (
        3 * jnp.sum(node_mask)
    )

    # MAE stress
    stress_mae_per_atom = jnp.sum(
        jnp.abs(stress - batch.globals["stress"])
        / jnp.where(batch.n_node > 0, batch.n_node, 1.0)[:, None, None]
        * graph_mask[:, None, None]
    ) / (9 * jnp.sum(graph_mask))

    # Optional atom energy anchor regularizer
    atom_ref = batch.globals.get("atom_ref")
    atom_ref_w = batch.globals.get("atom_ref_w")
    atom_ref_base_w = batch.globals.get("atom_ref_base_w")
    if (atom_ref is not None) and (atom_ref_w is not None):
        diff2 = (model.atom_energies - atom_ref) ** 2  # [n_species]
        w = jnp.asarray(atom_ref_w)
        if w.ndim == 0:
            lam = w
            atom_reg = jnp.mean(diff2)
            total_loss = total_loss + jnp.asarray(lam, dtype=diff2.dtype) * atom_reg
        else:
            # Relative per-species multipliers; scale by base scalar if provided
            atom_reg = jnp.sum(w * diff2) / jnp.clip(jnp.sum(w), a_min=1.0)
            lam_base = jnp.asarray(atom_ref_base_w if atom_ref_base_w is not None else 1.0, dtype=diff2.dtype)
            total_loss = total_loss + lam_base * atom_reg

    # Optional distillation from teacher (e.g., pretrained with forces)
    if (teacher is not None) and (distill_force_weight != 0.0 or distill_stress_weight != 0.0):
        # Ensure teacher computes forces/stress irrespective of caller flag
        globs_t = dict(batch.globals)
        globs_t["need_forces"] = jnp.array(1, dtype=jnp.int32)
        batch_t = batch._replace(globals=globs_t)
        t_energy, t_forces, t_stress = teacher(batch_t)
        # Force distillation
        if distill_force_weight != 0.0:
            node_mask = jraph.get_node_padding_mask(batch)
            dist_f = jnp.sum(jnp.abs(forces - t_forces) * node_mask[:, None]) / (
                3 * jnp.sum(node_mask)
            )
            total_loss = total_loss + jnp.asarray(distill_force_weight, dtype=total_loss.dtype) * dist_f
        # Stress distillation
        if distill_stress_weight != 0.0:
            graph_mask = jraph.get_graph_padding_mask(batch)
            dist_s = jnp.sum(jnp.abs(stress - t_stress) * graph_mask[:, None, None]) / (
                9 * jnp.sum(graph_mask)
            )
            total_loss = total_loss + jnp.asarray(distill_stress_weight, dtype=total_loss.dtype) * dist_s

    # Optional parameter L2 anchor to initial checkpoint (continue-training regularizer)
    if (teacher is not None) and (ckpt_l2_weight != 0.0):
        mp = eqx.filter(model, eqx.is_array)
        tp = eqx.filter(teacher, eqx.is_array)
        diffsq = jax.tree.map(lambda a, b: jnp.sum((a - b) ** 2), mp, tp)
        sq = sum(jax.tree.leaves(diffsq)) if isinstance(diffsq, (list, tuple)) else diffsq
        # Normalize by total parameter count
        sizes = jax.tree.map(lambda a: a.size, mp)
        denom = sum(jax.tree.leaves(sizes))
        denom = jnp.maximum(jnp.array(denom, dtype=jnp.float32), 1.0)
        reg = sq / denom
        total_loss = total_loss + jnp.asarray(ckpt_l2_weight, dtype=total_loss.dtype) * reg

    return total_loss, {
        "energy_mae_per_atom": energy_mae_per_atom,
        "force_mae": force_mae,
        "stress_mae_per_atom": stress_mae_per_atom,
        "force_norm": pred_force_norm,
        "energy_short": energy_short,
        "energy_lr": energy_lr,
    }


def evaluate(
    model,
    dataloader,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=1.0,
    loss_type="huber",
    energy_mode_flag: int = 0,
    atom_ref_vec=None,
    atom_ref_weight=0.0,
    atom_ref_base_weight=None,
    prefetch_queue_size: int = 16,
):
    """Return loss and RMSE of energy and force in eV and eV/Å respectively"""
    total_metrics = defaultdict(int)
    total_count = 0
    # Allow larger prefetch queue to keep accelerator fed
    prefetch_q = kwargs.get('prefetch_queue_size', 16) if 'kwargs' in locals() else 16
    for batch in prefetch(dataloader, queue_size=prefetch_queue_size):
        # Inject flags for TAE mode and optional anchors
        globs = dict(batch.globals)
        globs["energy_mode"] = jnp.array(int(energy_mode_flag), dtype=jnp.int32)
        # Energy-only fast path when no force/stress terms are used in evaluation
        need_forces = 1 if (float(force_weight) != 0.0 or float(stress_weight) != 0.0) else 0
        globs["need_forces"] = jnp.array(need_forces, dtype=jnp.int32)
        # Support scalar or (relative) per-species weights + optional base scalar
        if atom_ref_vec is not None and (jnp.any(jnp.asarray(atom_ref_weight) != 0.0)):
            globs["atom_ref"] = atom_ref_vec
            globs["atom_ref_w"] = jnp.asarray(atom_ref_weight, dtype=jnp.float32)
            if atom_ref_base_weight is not None:
                globs["atom_ref_base_w"] = jnp.asarray(atom_ref_base_weight, dtype=jnp.float32)
        batch = batch._replace(globals=globs)
        n_graphs = jnp.sum(jraph.get_graph_padding_mask(batch))
        val_loss, metrics = loss(model, batch, energy_weight, force_weight, stress_weight, loss_type)
        total_metrics["loss"] += val_loss * n_graphs
        for key, value in metrics.items():
            total_metrics[key] += value * n_graphs
        total_count += n_graphs

    for key, value in total_metrics.items():
        total_metrics[key] = value / total_count

    return total_metrics


def save_training_state(path, model, ema_model, optim, opt_state, step, epoch, best_val_loss):
    state = {
        "model": model,
        "ema_model": ema_model,
        "optim": optim,
        "opt_state": opt_state,
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    with open(path, "wb") as f:
        cloudpickle.dump(state, f)


def load_training_state(path):
    with open(path, "rb") as f:
        state = cloudpickle.load(f)
    return (
        state["model"],
        state["ema_model"],
        state["optim"],
        state["opt_state"],
        state["step"],
        state["epoch"],
        state["best_val_loss"],
    )


def train(config_path: str):
    """Train a Nequix model from a config file. See configs/nequix-mp-1.yaml for an example."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # use TMPDIR for slurm jobs if available
    config["cache_dir"] = config.get("cache_dir") or os.environ.get("TMPDIR")

    train_dataset = Dataset(
        file_path=config["train_path"],
        cache_dir=config["cache_dir"],
        atomic_numbers=config["atomic_numbers"],
        split="train",
        cutoff=config["cutoff"],
        valid_frac=config["valid_frac"],
    )
    val_dataset = Dataset(
        file_path=config["train_path"],
        cache_dir=config["cache_dir"],
        atomic_numbers=config["atomic_numbers"],
        split="val",
        cutoff=config["cutoff"],
        valid_frac=config["valid_frac"],
    )

    if "atom_energies" in config:
        atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    else:
        atom_energies = average_atom_energies(train_dataset)
        # Cache atom_energies back into config file for future runs
        try:
            ae_map = {int(Z): float(e) for Z, e in zip(config["atomic_numbers"], atom_energies)}
            config["atom_energies"] = ae_map
            with open(config_path, "w") as wf:
                yaml.safe_dump(config, wf, sort_keys=False)
        except Exception:
            pass

    stats_keys = [
        "shift",
        "scale",
        "avg_n_neighbors",
        "max_n_edges",
        "max_n_nodes",
        "avg_n_nodes",
        "avg_n_edges",
    ]
    if all(key in config for key in stats_keys):
        stats = {key: config[key] for key in stats_keys}
    else:
        stats = dataset_stats(train_dataset, atom_energies)
        # Cache stats back into config file for future runs
        try:
            for k, v in stats.items():
                config[k] = v
            with open(config_path, "w") as wf:
                yaml.safe_dump(config, wf, sort_keys=False)
        except Exception:
            pass

    # Optional: print species coverage table (Z: count) to console
    if bool(config.get("print_species_coverage", False)):
        try:
            counts = stats.get("species_counts")
            if counts is None:
                # compute lightweight coverage from train_dataset if absent
                species_counts = [0] * len(config["atomic_numbers"])
                for graph in train_dataset:
                    for s in np.array(graph.nodes["species"]).reshape(-1):
                        species_counts[int(s)] += 1
                counts = species_counts
            Zs = list(config["atomic_numbers"])  # ordered mapping index -> Z
            print("Species coverage (index->Z: count):")
            for i, (Z, c) in enumerate(zip(Zs, counts)):
                marker = " (absent)" if int(c) == 0 else ""
                print(f"  {i:2d} -> {int(Z):2d}: {int(c)}{marker}")
            missing = sum(1 for c in counts if int(c) == 0)
            print(f"Missing species: {missing}/{len(Zs)} (routed to base in MoE; strong atom-ref anchors applied)")
            print("Energies are in eV (MSR-ACC JSON is converted Hartree→eV; EXTXYZ assumed eV).")
        except Exception as e:
            print(f"Warning: failed to print species coverage: {e}")

    num_devices = len(jax.devices())
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=int(config.get("num_workers", 16)),
        prefetch_factor=int(config.get("prefetch_factor", 4)),
        buffer_factor=float(config.get("buffer_factor", 1.2)),
        graphs_buffer_factor=float(config.get("graphs_buffer_factor", 2.0)),
    )
    train_loader = ParallelLoader(train_loader, num_devices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=int(config.get("num_workers", 16)),
        prefetch_factor=int(config.get("prefetch_factor", 4)),
        buffer_factor=float(config.get("buffer_factor", 1.2)),
        graphs_buffer_factor=float(config.get("graphs_buffer_factor", 2.0)),
    )

    # Optional: enable background 'wandb sync' helper only when explicitly requested.
    # Default is disabled to avoid noisy warnings when syncing is slow/inactive.
    enable_osh = os.environ.get("WANDB_OSH_ENABLE", "0") == "1"
    wandb_sync = TriggerWandbSyncHook() if enable_osh else (lambda: None)
    wandb.init(project="nequix", config=config)

    key = jax.random.key(0)

    # Optional atom-reference anchor: either inline map (atom_ref) or YAML file (atom_ref_path)
    anchor_vec_1d = None
    anchor_weight = 0.0
    weights_rel_vec = None
    if bool(config.get("train_on_tae", False)):
        try:
            if config.get("atom_ref"):
                anchor_vec_1d = jnp.array(
                    [config["atom_ref"][Z] for Z in config["atomic_numbers"]], dtype=jnp.float32
                )
                anchor_weight = float(config.get("atom_ref_weight", 0.0))
            elif config.get("atom_ref_path"):
                with open(config["atom_ref_path"], "r") as arf:
                    arf_yaml = yaml.safe_load(arf)
                ref_map = arf_yaml.get("atom_energies", arf_yaml)
                anchor_vec_1d = jnp.array(
                    [float(ref_map[int(Z)]) for Z in config["atomic_numbers"]], dtype=jnp.float32
                )
                # Provide a small default weight if none specified
                anchor_weight = float(config.get("atom_ref_weight", 1.0e-4))
            else:
                # Default: if known atomref file exists, auto-use it unless disabled
                default_path = Path("data/atomrefs/wb97m-v-def2-qzvp.yml")
                if default_path.exists():
                    with open(default_path, "r") as arf:
                        arf_yaml = yaml.safe_load(arf)
                    ref_map = arf_yaml.get("atom_energies", arf_yaml)
                    anchor_vec_1d = jnp.array(
                        [float(ref_map[int(Z)]) for Z in config["atomic_numbers"]], dtype=jnp.float32
                    )
                    anchor_weight = float(config.get("atom_ref_weight", 1.0e-4))
        except Exception as e:
            print(f"Warning: failed to load atom_ref anchor: {e}")
            anchor_vec_1d = None
            anchor_weight = 0.0
            weights_rel_vec = None

    # If we have an anchor and species coverage, build per-species weights to strongly constrain absent elements
    species_counts = None
    try:
        species_counts = jnp.array(config.get("species_counts", []), dtype=jnp.int64)
    except Exception:
        species_counts = None
    # Pull from freshly computed stats, if present
    try:
        if species_counts is None and isinstance(stats, dict) and ("species_counts" in stats):
            species_counts = jnp.array(stats["species_counts"], dtype=jnp.int64)
    except Exception:
        pass
    if anchor_vec_1d is not None and species_counts is not None and species_counts.size == anchor_vec_1d.shape[0]:
        # Build relative multipliers (absent species get stronger anchor)
        # Use ratio = strong/base to avoid changing global scale; apply base weight separately
        base_w = float(anchor_weight)
        strong_w_abs = float(config.get("atom_ref_strong_weight", max(anchor_weight * 50.0, 1.0e-3)))
        ratio = (strong_w_abs / base_w) if base_w > 0.0 else 0.0
        present = species_counts > 0
        weights_rel_vec = jnp.where(present, 1.0, ratio).astype(jnp.float32)
    teacher_model_single = None
    # Default base model (will be overridden by MoE if enabled)
    model = Nequix(
        key,
        n_species=len(config["atomic_numbers"]),
        hidden_irreps=config["hidden_irreps"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        index_weights=config["index_weights"],
        layer_norm=config["layer_norm"],
        shift=stats["shift"],
        scale=stats["scale"],
        avg_n_neighbors=stats["avg_n_neighbors"],
        atom_energies=atom_energies,
        learn_atom_energies=config.get("learn_atom_energies", False),
        include_atom_priors_in_energy=not config.get("train_on_tae", False),
        use_qeq_head=config.get("use_qeq_head", False),
        qeq_eps=float(config.get("qeq_eps", 1.0e-3)),
        qeq_ridge=float(config.get("qeq_ridge", 1.0e-6)),
        coulomb_mode=str(config.get("coulomb_mode", "auto")),
        ewald_alpha=float(config.get("ewald_alpha", 0.25)),
        ewald_kmax=int(config.get("ewald_kmax", 3)),
        ewald_rcut=float(config.get("ewald_rcut", 0.0)),
    )

    # Optional Mixture-of-Experts: base expert from checkpoint + new TAE expert
    if bool(config.get("use_moe", False)):
        if not config.get("init_from"):
            raise ValueError("use_moe=True requires init_from (pretrained base model)")
        base_model, base_cfg = load_model(config["init_from"])
        # Update non-trainables on base to current stats
        try:
            base_model = eqx.tree_at(lambda m: m.shift, base_model, float(stats["shift"]))
            base_model = eqx.tree_at(lambda m: m.scale, base_model, float(stats["scale"]))
        except Exception:
            pass
        # Build TAE expert configured for binding energy (no atom priors), VACUUM Coulomb unless overridden
        tae_model = Nequix(
            key,
            n_species=len(config["atomic_numbers"]),
            hidden_irreps=config["hidden_irreps"],
            lmax=config["lmax"],
            cutoff=config["cutoff"],
            n_layers=config["n_layers"],
            radial_basis_size=config["radial_basis_size"],
            radial_mlp_size=config["radial_mlp_size"],
            radial_mlp_layers=config["radial_mlp_layers"],
            radial_polynomial_p=config["radial_polynomial_p"],
            mlp_init_scale=config["mlp_init_scale"],
            index_weights=config["index_weights"],
            layer_norm=config["layer_norm"],
            shift=stats["shift"],
            scale=stats["scale"],
            avg_n_neighbors=stats["avg_n_neighbors"],
            atom_energies=atom_energies,
            learn_atom_energies=config.get("learn_atom_energies", False),
            include_atom_priors_in_energy=False,
            use_qeq_head=config.get("use_qeq_head", True),
            qeq_eps=float(config.get("qeq_eps", 1.0e-3)),
            qeq_ridge=float(config.get("qeq_ridge", 1.0e-6)),
            coulomb_mode=str(config.get("coulomb_mode", "vacuum")),
            ewald_alpha=float(config.get("ewald_alpha", 0.25)),
            ewald_kmax=int(config.get("ewald_kmax", 3)),
            ewald_rcut=float(config.get("ewald_rcut", 0.0)),
        )
        gating_mode = str(config.get("moe_gating_mode", "rule"))
        freeze_base = bool(config.get("moe_freeze_base", True))
        # Default to using base forces when doing TAE training (helps preserve force quality)
        default_force_from = "base" if bool(config.get("train_on_tae", False)) else "blend"
        force_from = str(config.get("moe_force_from", default_force_from))
        # Build seen-species mask (per config order)
        seen_mask = None
        try:
            if isinstance(stats, dict) and ("species_counts" in stats):
                counts = jnp.array(stats["species_counts"], dtype=jnp.int64)
                seen_mask = (counts > 0).astype(jnp.float32)
        except Exception:
            pass
        model = NequixMoE(
            base_model,
            tae_model,
            gating_mode=gating_mode,
            freeze_base=freeze_base,
            force_from=force_from,
            avoid_unseen_to_tae=bool(config.get("moe_avoid_unseen_to_tae", bool(config.get("train_on_tae", False)))) ,
            seen_species_mask=seen_mask,
            key=key,
        )
    else:
        # Optional fine-tune from checkpoint (single-expert path)
        if "init_from" in config and config["init_from"]:
            pretrained_model, pretrained_cfg = load_model(config["init_from"])
            # Replace full model params
            model = pretrained_model
            # Update dataset-dependent constants (non-trainables) to current stats
            try:
                model = eqx.tree_at(lambda m: m.atom_energies, model, jnp.array(atom_energies))
            except Exception:
                pass
            try:
                model = eqx.tree_at(lambda m: m.shift, model, float(stats["shift"]))
                model = eqx.tree_at(lambda m: m.scale, model, float(stats["scale"]))
            except Exception:
                pass
            # Use the loaded checkpoint as teacher for force/stress distillation if requested
            teacher_model_single = model

    param_count = sum(p.size for p in jax.tree.flatten(eqx.filter(model, eqx.is_array))[0])
    wandb.run.summary["param_count"] = param_count
    try:
        if isinstance(stats, dict) and ("species_counts" in stats):
            wandb.run.summary["species_counts"] = stats["species_counts"]
            wandb.run.summary["species_missing"] = int(sum(1 for c in stats["species_counts"] if c == 0))
    except Exception:
        pass


    # NB: this is not exact because of dynamic batching but should be close enough.
    # Guard against tiny datasets so schedules remain well-defined.
    global_batch = max(1, config["batch_size"] * max(1, jax.device_count()))
    steps_per_epoch = max(1, len(train_dataset) // global_batch)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config["learning_rate"] * config["warmup_factor"],
        peak_value=config["learning_rate"],
        end_value=config.get("end_lr", 1e-6),
        warmup_steps=config["warmup_epochs"] * steps_per_epoch,
        decay_steps=config["n_epochs"] * steps_per_epoch,
    )

    # Determine logging cadence: aim for N logs per epoch if provided
    logs_per_epoch = int(config.get("logs_per_epoch", 0) or 0)
    if logs_per_epoch > 0:
        log_every_steps = max(1, steps_per_epoch // logs_per_epoch)
    else:
        log_every_steps = int(config.get("log_every", 50))

    if config["optimizer"] == "adamw":
        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=config["weight_decay"],
                mask=weight_decay_mask(model),
            ),
        )
    elif config["optimizer"] == "muon":
        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.contrib.muon(
                learning_rate=schedule,
                weight_decay=config["weight_decay"] if config["weight_decay"] != 0.0 else None,
                weight_decay_mask=weight_decay_mask(model),
            ),
        )
    else:
        raise ValueError(f"optimizer {config['optimizer']} not supported")

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    devices = list(jax.devices())
    model = jax.device_put_replicated(model, devices)
    # Replicate teacher if provided; otherwise reuse model handle (distillation weights should be zero in that case)
    if teacher_model_single is not None:
        teacher_model = jax.device_put_replicated(teacher_model_single, devices)
    else:
        teacher_model = model
    opt_state = jax.device_put_replicated(opt_state, list(jax.devices()))
    ema_model = jax.tree.map(lambda x: x.copy(), model)  # copy model
    step = jnp.array(0)
    start_epoch = 0
    best_val_loss = float("inf")

    if "resume_from" in config:
        model, ema_model, optim, opt_state, step, start_epoch, best_val_loss = load_training_state(
            config["resume_from"]
        )
        if config.get("reset_lr_on_resume", False):
            step = jnp.array(0)

    # Optional dissociation penalty schedule
    lambda_inf_base = float(config.get("lambda_inf", 0.0))
    lambda_inf_stop_epoch = int(config.get("lambda_inf_stop_epoch", 0))
    # Optional force/stress distillation weights (preserve pretrained forces)
    distill_force_weight = float(config.get("distill_force_weight", 0.0))
    distill_stress_weight = float(config.get("distill_stress_weight", 0.0))
    # Optional parameter L2 anchor to checkpoint
    ckpt_l2_weight = float(config.get("ckpt_l2_weight", 0.0))
    if teacher_model_single is None and (distill_force_weight > 0.0 or distill_stress_weight > 0.0):
        print("Warning: distillation requested but no teacher checkpoint loaded via init_from; disabling distillation.")
        distill_force_weight = 0.0
        distill_stress_weight = 0.0

    # @eqx.filter_jit
    @functools.partial(eqx.filter_pmap, in_axes=(0, 0, None, 0, 0, None, None, 0, None, None, None), axis_name="device")
    def train_step(model, ema_model, step, opt_state, batch, lambda_inf, fnorm_w, teacher, d_fw, d_sw, ckpt_w):
        # training step
        device_idx = jax.lax.axis_index("device")
        rng = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(0), step), device_idx)
        (total_loss, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model,
            batch,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
            lambda_inf,
            rng,
            fnorm_w,
            teacher,
            d_fw,
            d_sw,
            ckpt_w,
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)

        # update EMA model
        # don't weight early steps as much (from https://github.com/fadel/pytorch_ema)
        decay = jnp.minimum(config["ema_decay"], (1 + step) / (10 + step))
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        model_params = eqx.filter(model, eqx.is_array)
        new_ema_params = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, model_params
        )
        ema_model = eqx.combine(ema_static, new_ema_params)

        return (
            model,
            ema_model,
            opt_state,
            total_loss,
            metrics,
        )

    for epoch in range(start_epoch, config["n_epochs"]):
        start_time = time.time()
        train_loader.loader.set_epoch(epoch)
        # Accumulate training loss across this epoch for summary
        epoch_train_loss_sum = 0.0
        epoch_train_count = 0
        epoch_train_short_sum = 0.0
        epoch_train_lr_sum = 0.0
        epoch_train_graphs = 0
        for batch in prefetch(train_loader, queue_size=int(config.get("prefetch_queue_size", 16))):
            # Inject TAE training flags and optional atom-energy anchors
            if config.get("train_on_tae", False):
                energy_mode = jnp.ones((num_devices,), dtype=jnp.int32)
                globs = {**batch.globals, "energy_mode": energy_mode}
                # Apply atom-ref anchor if available
                if anchor_vec_1d is not None and float(anchor_weight) > 0.0:
                    atom_ref_vec = jnp.tile(anchor_vec_1d[None, :], (num_devices, 1))
                    globs["atom_ref"] = atom_ref_vec
                    if weights_rel_vec is not None:
                        globs["atom_ref_w"] = jnp.tile(weights_rel_vec[None, :], (num_devices, 1))
                        globs["atom_ref_base_w"] = jnp.full((num_devices,), float(anchor_weight), dtype=jnp.float32)
                    else:
                        globs["atom_ref_w"] = jnp.full((num_devices,), float(anchor_weight), dtype=jnp.float32)
                # Energy-only fast path flag: skip forces/stress if not used
                # Consider current schedules for lambda_inf and force_norm_weight
                lam_cur = lambda_inf_base if (lambda_inf_stop_epoch <= 0 or epoch < lambda_inf_stop_epoch) else 0.0
                f_w_cur = float(config.get("force_norm_weight", 0.0))
                f_w_cur = f_w_cur if (int(config.get("force_norm_stop_epoch", 0)) <= 0 or epoch < int(config.get("force_norm_stop_epoch", 0))) else 0.0
                need_forces = (
                    (float(config["force_weight"]) != 0.0)
                    or (float(config["stress_weight"]) != 0.0)
                    or (float(lam_cur) != 0.0)
                    or (float(f_w_cur) != 0.0)
                    or (float(distill_force_weight) != 0.0)
                    or (float(distill_stress_weight) != 0.0)
                )
                globs["need_forces"] = jnp.full((num_devices,), 1 if need_forces else 0, dtype=jnp.int32)
                batch = batch._replace(globals=globs)
            else:
                # Even outside TAE mode, we can skip force/stress if none are used
                globs = dict(batch.globals)
                lam_cur = lambda_inf_base if (lambda_inf_stop_epoch <= 0 or epoch < lambda_inf_stop_epoch) else 0.0
                f_w_cur = float(config.get("force_norm_weight", 0.0))
                f_w_cur = f_w_cur if (int(config.get("force_norm_stop_epoch", 0)) <= 0 or epoch < int(config.get("force_norm_stop_epoch", 0))) else 0.0
                need_forces = (
                    (float(config["force_weight"]) != 0.0)
                    or (float(config["stress_weight"]) != 0.0)
                    or (float(lam_cur) != 0.0)
                    or (float(f_w_cur) != 0.0)
                    or (float(distill_force_weight) != 0.0)
                    or (float(distill_stress_weight) != 0.0)
                )
                globs["need_forces"] = jnp.full((num_devices,), 1 if need_forces else 0, dtype=jnp.int32)
                batch = batch._replace(globals=globs)

            batch_time = time.time() - start_time
            start_time = time.time()
            # Current lambda_inf: drop to 0 after configured stop epoch
            lam_cur = lambda_inf_base if (lambda_inf_stop_epoch <= 0 or epoch < lambda_inf_stop_epoch) else 0.0
            # Force-norm regularizer weight schedule (optional stop epoch)
            f_w_cur = float(config.get("force_norm_weight", 0.0))
            f_w_cur = f_w_cur if (int(config.get("force_norm_stop_epoch", 0)) <= 0 or epoch < int(config.get("force_norm_stop_epoch", 0))) else 0.0
            (model, ema_model, opt_state, total_loss, metrics) = train_step(
                model, ema_model, step, opt_state, batch,
                jnp.array(lam_cur, dtype=jnp.float32), jnp.array(f_w_cur, dtype=jnp.float32),
                teacher_model,
                jnp.array(distill_force_weight, dtype=jnp.float32), jnp.array(distill_stress_weight, dtype=jnp.float32),
                jnp.array(ckpt_l2_weight, dtype=jnp.float32),
            )
            train_time = time.time() - start_time
            step = step + 1
            # accumulate epoch training loss (host float)
            try:
                epoch_train_loss_sum += float(total_loss.mean().item())
            except Exception:
                try:
                    epoch_train_loss_sum += float(jax.device_get(total_loss).mean())
                except Exception:
                    pass
            epoch_train_count += 1
            # accumulate energy component means weighted by graphs in batch
            try:
                bsz = int(jax.vmap(jraph.get_graph_padding_mask)(batch).sum().item())
            except Exception:
                bsz = 0
            try:
                es = float(metrics["energy_short"].mean().item())
                elr = float(metrics["energy_lr"].mean().item())
            except Exception:
                es = 0.0
                elr = 0.0
            epoch_train_short_sum += es * max(1, bsz)
            epoch_train_lr_sum += elr * max(1, bsz)
            epoch_train_graphs += max(1, bsz)
            # Use host integer step for modulo to avoid JAX truth-value issues
            try:
                step_i = int(step)
            except Exception:
                try:
                    step_i = int(step.item())
                except Exception:
                    step_i = int(jax.device_get(step))
            if step_i % log_every_steps == 0:
                logs = {}
                logs["train/loss"] = total_loss.mean().item()
                logs["learning_rate"] = schedule(step).item()
                logs["train/batch_time"] = batch_time
                logs["train/train_time"] = train_time
                for key, value in metrics.items():
                    logs[f"train/{key}"] = value.mean().item()
                logs["train/batch_size"] = (
                    jax.vmap(jraph.get_graph_padding_mask)(batch).sum().item()
                )
                wandb.log(logs, step=step)
                print(f"step: {step}, logs: {logs}")
                start_time = time.time()

        ema_model_single = jax.tree.map(lambda x: x[0], ema_model)
        # Validation: match training objective for energy if in TAE mode
        use_tae = 1 if config.get("train_on_tae", False) else 0
        # Use the same anchor at eval
        atom_ref_vec = anchor_vec_1d if use_tae else None
        # Evaluate with same base weight and relative per-species multipliers
        if use_tae and (anchor_vec_1d is not None) and float(anchor_weight) > 0.0:
            atom_ref_w = weights_rel_vec if weights_rel_vec is not None else float(anchor_weight)
            atom_ref_base_w = float(anchor_weight) if (weights_rel_vec is not None) else None
        else:
            atom_ref_w = 0.0
            atom_ref_base_w = None

        val_metrics = evaluate(
            ema_model_single,
            val_loader,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
            energy_mode_flag=use_tae,
            atom_ref_vec=atom_ref_vec,
            atom_ref_weight=atom_ref_w,
            atom_ref_base_weight=(atom_ref_base_w if 'atom_ref_base_w' in locals() else None),
            prefetch_queue_size=int(config.get("prefetch_queue_size", 16)),
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_model(Path(wandb.run.dir) / "checkpoint.nqx", ema_model_single, config)

        save_training_state(
            Path(wandb.run.dir) / "state.pkl",
            model,
            ema_model,
            optim,
            opt_state,
            step,
            epoch + 1,
            best_val_loss,
        )

        if "state_path" in config:
            save_training_state(
                config["state_path"],
                model,
                ema_model,
                optim,
                opt_state,
                step,
                epoch + 1,
                best_val_loss,
            )

        logs = {}
        for key, value in val_metrics.items():
            try:
                logs[f"val/{key}"] = value.item()
            except AttributeError:
                # value may be a Python float/int already
                logs[f"val/{key}"] = float(value)
        # Add epoch-aggregated training loss for side-by-side comparison
        if epoch_train_count > 0:
            logs["train/epoch_loss"] = float(epoch_train_loss_sum / max(1, epoch_train_count))
        if epoch_train_graphs > 0:
            logs["train/epoch_energy_short"] = float(epoch_train_short_sum / max(1, epoch_train_graphs))
            logs["train/epoch_energy_lr"] = float(epoch_train_lr_sum / max(1, epoch_train_graphs))
        logs["epoch"] = epoch
        wandb.log(logs, step=step)
        print(f"epoch: {epoch}, logs: {logs}")
        wandb_sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    train(args.config_path)


if __name__ == "__main__":
    main()
