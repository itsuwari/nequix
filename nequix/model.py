import json
import math
from typing import Callable, Optional, Sequence

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph

from nequix.layer_norm import RMSLayerNorm


def bessel_basis(x: jax.Array, num_basis: int, r_max: float) -> jax.Array:
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
    x = x[:, None]
    return prefactor * jnp.where(
        x == 0.0,
        bessel_weights / r_max,  # prevent division by zero
        jnp.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: jax.Array, r_max: float, p: float) -> jax.Array:
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(x, p))
    out = out + (p * (p + 2.0) * jnp.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * jnp.power(x, p + 2.0))
    return out * jnp.where(x < 1.0, 1.0, 0.0)


class Linear(eqx.Module):
    weights: jax.Array
    bias: Optional[jax.Array]
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        scale = math.sqrt(init_scale / in_size)
        self.weights = jax.random.normal(key, (in_size, out_size)) * scale
        self.bias = jnp.zeros(out_size) if use_bias else None
        self.use_bias = use_bias

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


class MLP(eqx.Module):
    layers: list[Linear]
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        sizes,
        activation=jax.nn.silu,
        *,
        init_scale: float = 1.0,
        use_bias: bool = False,
        key: jax.Array,
    ):
        self.activation = activation

        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            Linear(
                sizes[i],
                sizes[i + 1],
                key=keys[i],
                use_bias=use_bias,
                # don't scale last layer since no activation
                init_scale=init_scale if i < len(sizes) - 2 else 1.0,
            )
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class NequixConvolution(eqx.Module):
    output_irreps: e3nn.Irreps = eqx.field(static=True)
    index_weights: bool = eqx.field(static=True)
    avg_n_neighbors: float = eqx.field(static=True)

    radial_mlp: MLP
    linear_1: e3nn.equinox.Linear
    linear_2: e3nn.equinox.Linear
    skip: e3nn.equinox.Linear
    layer_norm: Optional[RMSLayerNorm]

    def __init__(
        self,
        key: jax.Array,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        index_weights: bool = True,
        layer_norm: bool = False,
    ):
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors
        self.index_weights = index_weights

        tp_irreps = e3nn.tensor_product(input_irreps, sh_irreps, filter_ir_out=output_irreps)

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.linear_1 = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=input_irreps,
            key=k1,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            key=k2,
        )

        # add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
        output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = e3nn.equinox.Linear(
            irreps_in=tp_irreps,
            irreps_out=output_irreps,
            key=k3,
        )

        # skip connection has per-species weights
        self.skip = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=output_irreps,
            linear_type="indexed" if index_weights else "vanilla",
            num_indexed_weights=n_species if index_weights else None,
            force_irreps_out=True,
            key=k4,
        )

        if layer_norm:
            self.layer_norm = RMSLayerNorm(
                irreps=output_irreps,
                centering=False,
                std_balance_degrees=True,
            )
        else:
            self.layer_norm = None

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        messages = self.linear_1(features)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.output_irreps)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_message

        messages_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(jax.lax.stop_gradient(self.avg_n_neighbors))

        skip = self.skip(species, features) if self.index_weights else self.skip(features)
        features = self.linear_2(messages_agg) + skip

        if self.layer_norm is not None:
            features = self.layer_norm(features)

        return e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.tanh,
            even_gate_act=jax.nn.silu,
        )


class Nequix(eqx.Module):
    lmax: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)
    radial_basis_size: int = eqx.field(static=True)
    radial_polynomial_p: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    shift: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    atom_energies: jax.Array
    layers: list[NequixConvolution]
    readout: e3nn.equinox.Linear
    # Optional QEq long-range head
    use_qeq_head: bool = eqx.field(static=True)
    qeq_eps: float = eqx.field(static=True)
    qeq_ridge: float = eqx.field(static=True)
    coulomb_mode: str = eqx.field(static=True)
    ewald_alpha: float = eqx.field(static=True)
    ewald_kmax: int = eqx.field(static=True)
    ewald_rcut: float = eqx.field(static=True)
    chi_readout: Optional[e3nn.equinox.Linear] = None
    u_readout: Optional[e3nn.equinox.Linear] = None
    learn_atom_energies: bool = eqx.field(static=True)
    include_atom_priors_in_energy: bool = eqx.field(static=True)

    def __init__(
        self,
        key,
        n_species,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "128x0e + 128x1o + 128x2e + 128x3o",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        index_weights: bool = True,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
        layer_norm: bool = False,
        learn_atom_energies: bool = False,
        include_atom_priors_in_energy: bool = True,
        # QEq head controls
        use_qeq_head: bool = False,
        qeq_eps: float = 1.0e-3,
        qeq_ridge: float = 1.0e-6,
        coulomb_mode: str = "auto",
        ewald_alpha: float = 0.25,
        ewald_kmax: int = 3,
        ewald_rcut: float = 0.0,
    ):
        self.lmax = lmax
        self.cutoff = cutoff
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.shift = shift
        self.scale = scale
        self.atom_energies = (
            jnp.array(atom_energies)
            if atom_energies is not None
            else jnp.zeros(n_species, dtype=jnp.float32)
        )
        self.learn_atom_energies = learn_atom_energies
        self.include_atom_priors_in_energy = include_atom_priors_in_energy
        self.use_qeq_head = bool(use_qeq_head)
        self.qeq_eps = float(qeq_eps)
        self.qeq_ridge = float(qeq_ridge)
        self.coulomb_mode = str(coulomb_mode)
        self.ewald_alpha = float(ewald_alpha)
        self.ewald_kmax = int(ewald_kmax)
        self.ewald_rcut = float(ewald_rcut)
        input_irreps = e3nn.Irreps(f"{n_species}x0e")
        sh_irreps = e3nn.s2_irreps(lmax)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        self.layers = []

        key, *subkeys = jax.random.split(key, n_layers + 1)
        for i in range(n_layers):
            self.layers.append(
                NequixConvolution(
                    key=subkeys[i],
                    input_irreps=input_irreps if i == 0 else hidden_irreps,
                    output_irreps=hidden_irreps if i < n_layers - 1 else hidden_irreps.filter("0e"),
                    sh_irreps=sh_irreps,
                    n_species=n_species,
                    radial_basis_size=radial_basis_size,
                    radial_mlp_size=radial_mlp_size,
                    radial_mlp_layers=radial_mlp_layers,
                    mlp_init_scale=mlp_init_scale,
                    avg_n_neighbors=avg_n_neighbors,
                    index_weights=index_weights,
                    layer_norm=layer_norm,
                )
            )

        self.readout = e3nn.equinox.Linear(
            irreps_in=hidden_irreps.filter("0e"), irreps_out="0e", key=key
        )

        # Heads for QEq: electronegativity chi and diagonal hardness U_ii
        if self.use_qeq_head:
            k_chi, k_u = jax.random.split(key, 2)
            self.chi_readout = e3nn.equinox.Linear(
                irreps_in=hidden_irreps.filter("0e"), irreps_out="0e", key=k_chi
            )
            self.u_readout = e3nn.equinox.Linear(
                irreps_in=hidden_irreps.filter("0e"), irreps_out="0e", key=k_u
            )

    def _node_features(
        self,
        displacements: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        # input features are one-hot encoded species
        features = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"), jax.nn.one_hot(species, self.n_species)
        )

        # safe norm (avoids nan for r = 0)
        square_r_norm = jnp.sum(displacements**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

        radial_basis = (
            bessel_basis(r_norm, self.radial_basis_size, self.cutoff)
            * polynomial_cutoff(
                r_norm,
                self.cutoff,
                self.radial_polynomial_p,
            )[:, None]
        )

        # compute spherical harmonics of edge displacements
        sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(self.lmax),
            displacements,
            normalize=True,
            normalization="component",
        )

        for layer in self.layers:
            features = layer(
                features,
                species,
                sh,
                radial_basis,
                senders,
                receivers,
            )

        return features

    def node_energies(
        self,
        displacements: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ):
        features = self._node_features(displacements, species, senders, receivers)
        node_energies = self.readout(features)

        # scale and shift energies
        node_energies = node_energies * jax.lax.stop_gradient(self.scale) + jax.lax.stop_gradient(
            self.shift
        )

        # optionally add isolated atom energies to each node as prior
        if self.include_atom_priors_in_energy:
            prior = self.atom_energies[species, None]
            if not self.learn_atom_energies:
                prior = jax.lax.stop_gradient(prior)
            node_energies = node_energies + prior

        return node_energies.array

    def short_energy(self, data: jraph.GraphsTuple, positions: Optional[jax.Array] = None) -> jax.Array:
        """Compute per-graph short-range energy (sum of node_energies) without QEq.

        Returns an array of shape [n_graph].
        """
        if positions is None:
            positions = data.nodes["positions"]
        # Build edge displacements with PBC shifts
        cell_per_edge = jnp.repeat(
            data.globals["cell"],
            data.n_edge,
            axis=0,
            total_repeat_length=data.edges["shifts"].shape[0],
        )
        offsets = jnp.einsum("ij,ijk->ik", data.edges["shifts"], cell_per_edge)
        r = positions[data.senders] - positions[data.receivers] + offsets
        node_e = self.node_energies(r, data.nodes["species"], data.senders, data.receivers)
        # Sum per graph
        return jraph.segment_sum(node_e[:, 0], node_graph_idx(data), num_segments=data.n_node.shape[0])

    def __call__(self, data: jraph.GraphsTuple):
        # Optional fast-path: skip force/stress if caller requests energy-only via globals["need_forces"] == 0
        # Accept scalar or per-graph flag; default is to compute forces.
        need_forces_flag = data.globals.get("need_forces")
        if need_forces_flag is None:
            need_forces_flag = jnp.array(1, dtype=jnp.int32)

        def energy_components(positions_eps: tuple[jax.Array, jax.Array]):
            positions, eps = positions_eps
            eps_sym = (eps + eps.swapaxes(1, 2)) / 2
            eps_sym_per_node = jnp.repeat(
                eps_sym,
                data.n_node,
                axis=0,
                total_repeat_length=data.nodes["positions"].shape[0],
            )
            # apply strain to positions and cell
            positions = positions + jnp.einsum("ik,ikj->ij", positions, eps_sym_per_node)
            cell = data.globals["cell"] + jnp.einsum("bij,bjk->bik", data.globals["cell"], eps_sym)
            cell_per_edge = jnp.repeat(
                cell,
                data.n_edge,
                axis=0,
                total_repeat_length=data.edges["shifts"].shape[0],
            )
            offsets = jnp.einsum("ij,ijk->ik", data.edges["shifts"], cell_per_edge)
            r = positions[data.senders] - positions[data.receivers] + offsets
            node_energies = self.node_energies(
                r, data.nodes["species"], data.senders, data.receivers
            )

            # Short-range energy is the sum of node energies
            E_short = jnp.sum(node_energies)

            if not self.use_qeq_head:
                return E_short, node_energies

            # Long-range QEq head with per-graph vmapped KKT and optional Ewald kernel
            features = self._node_features(r, data.nodes["species"], data.senders, data.receivers)
            chi_all = self.chi_readout(features).array[:, 0]
            uii_all = jax.nn.softplus(self.u_readout(features).array[:, 0]) + 1.0e-6

            nnode = data.n_node.astype(jnp.int32)
            gcount = nnode.shape[0]
            starts = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(nnode)[:-1]])
            max_n = jnp.maximum(1, jnp.max(nnode))

            cell_all = data.globals["cell"]
            try:
                Q_all = jnp.reshape(data.globals["charge"], (gcount,)).astype(jnp.float32)
            except Exception:
                Q_all = jnp.zeros((gcount,), dtype=jnp.float32)

            pos_all = positions

            def min_image_diffs(pos_g, cell_g):
                inv_cell = jnp.linalg.inv(cell_g)
                frac = pos_g @ inv_cell
                dfrac = frac[:, None, :] - frac[None, :, :]
                dfrac = dfrac - jnp.round(dfrac)
                return dfrac @ cell_g

            def pairwise_coulomb(cell_g, pos_g, valid_mask_g):
                M = pos_g.shape[0]
                m = valid_mask_g.astype(jnp.float32)
                vol = jnp.abs(jnp.linalg.det(cell_g))
                use_ewald = (self.coulomb_mode == "ewald") | (
                    (self.coulomb_mode == "auto") & (vol > 1e-8)
                )
                rvec = jax.lax.cond(
                    use_ewald,
                    lambda _: min_image_diffs(pos_g, cell_g),
                    lambda _: pos_g[:, None, :] - pos_g[None, :, :],
                    operand=None,
                )
                r2 = jnp.sum(rvec * rvec, axis=-1) + (self.qeq_eps ** 2)
                r = jnp.sqrt(r2)
                J_vac = jnp.where(jnp.eye(M, dtype=bool), 0.0, 1.0 / r)
                J = J_vac

                def recip_part(cell):
                    inv_cell = jnp.linalg.inv(cell)
                    B = 2.0 * jnp.pi * inv_cell.T
                    k = self.ewald_kmax
                    rng = jnp.arange(-k, k + 1)
                    grid = jnp.stack(jnp.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3)
                    nonzero = jnp.any(grid != 0, axis=1)
                    G = grid @ B  # [K,3]
                    G2 = jnp.sum(G * G, axis=1)
                    alpha = self.ewald_alpha
                    denom = jnp.where(nonzero, G2, 1.0)
                    w = (4.0 * jnp.pi / jnp.maximum(vol, 1e-8)) * jnp.exp(-G2 / (4.0 * (alpha ** 2))) / denom
                    w = w * nonzero.astype(w.dtype)
                    R = rvec.reshape(M * M, 3)
                    phase = R @ G.T  # [M*M, K]
                    c = jnp.cos(phase)
                    phi = (c * w[None, :]).sum(axis=1)
                    return phi.reshape(M, M)

                def real_erfc_part(r, M):
                    alpha = self.ewald_alpha
                    erfc = jax.scipy.special.erfc
                    return jnp.where(jnp.eye(M, dtype=bool), 0.0, erfc(alpha * r) / r)

                def build_ewald(_):
                    phi_recip = recip_part(cell_g)
                    phi_real = real_erfc_part(r, M)
                    # Real-space images within distance cutoff ewald_rcut
                    rcut = self.ewald_rcut
                    if rcut > 0.0:
                        k = self.ewald_kmax
                        rng = jnp.arange(-k, k + 1)
                        grid = jnp.stack(jnp.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3)
                        nonzero = jnp.any(grid != 0, axis=1)
                        T = grid @ cell_g  # [K,3]
                        RT = rvec[:, :, None, :] + T[None, None, :, :]
                        RT2 = jnp.sum(RT * RT, axis=-1)
                        RTnorm = jnp.sqrt(RT2 + 1e-12)
                        alpha = self.ewald_alpha
                        addk = jax.scipy.special.erfc(alpha * RTnorm) / RTnorm
                        mask = (RTnorm <= rcut) & nonzero[None, None, :]
                        addk = addk * mask.astype(addk.dtype)
                        add = jnp.sum(addk, axis=2)
                        add = jnp.where(jnp.eye(M, dtype=bool), 0.0, add)
                        phi_real = phi_real + add
                    K = phi_recip + phi_real
                    return jnp.where(jnp.eye(M, dtype=bool), 0.0, K)

                J = jax.lax.cond(use_ewald, build_ewald, lambda _: J, operand=None)
                J = J * (m[:, None] * m[None, :])
                return J

            def solve_one(start_g, count_g, cell_g, Q_g):
                M = max_n
                idx = jnp.arange(M, dtype=jnp.int32)
                valid = idx < count_g
                node_idx = start_g + jnp.where(valid, idx, 0)
                pos_g = jnp.where(valid[:, None], pos_all[node_idx], 0.0)
                chi_g = jnp.where(valid, chi_all[node_idx], 0.0)
                uii_g = jnp.where(valid, uii_all[node_idx], 0.0)
                Jg = pairwise_coulomb(cell_g, pos_g, valid)
                diag = (uii_g + self.qeq_ridge) * valid + (1.0 - valid) * 1e6
                A = Jg + jnp.diag(diag)
                S = valid.astype(jnp.float32)
                # Faster KKT via Schur complement using Cholesky of A
                # Solve A X = [chi, S]
                eye = jnp.eye(M, dtype=jnp.float32)
                L = jnp.linalg.cholesky(A + 1e-8 * eye)
                # Forward/back solves for two RHS
                B = jnp.stack([chi_g, S], axis=1)  # (M,2)
                Y = jax.scipy.linalg.solve_triangular(L, B, lower=True)
                X = jax.scipy.linalg.solve_triangular(L.T, Y, lower=False)
                w = X[:, 0]  # A^{-1} chi
                v = X[:, 1]  # A^{-1} S
                denom = jnp.dot(S, v) + 1e-12
                numer = Q_g + jnp.dot(S, w)
                lam = -numer / denom
                q = (-w - v * lam) * valid
                E = 0.5 * jnp.dot(q, (A @ q)) + jnp.dot(chi_g, q)
                return q, E

            q_all, E_lr_all = jax.vmap(solve_one)(starts, nnode, cell_all, Q_all)
            E_lr = jnp.sum(E_lr_all)
            return E_short + E_lr, node_energies

        eps0 = jnp.zeros_like(data.globals["cell"])

        def with_forces(_):
            (minus_forces, virial), node_energies = eqx.filter_grad(energy_components, has_aux=True)(
                (data.nodes["positions"], eps0)
            )
            # padded nodes may have nan forces, so we mask them
            node_mask = jraph.get_node_padding_mask(data)
            minus_forces_m = jnp.where(node_mask[:, None], minus_forces, 0.0)
            # compute total energies across each subgraph
            graph_energies = jraph.segment_sum(
                node_energies,
                node_graph_idx(data),
                num_segments=data.n_node.shape[0],
                indices_are_sorted=True,
            )
            det = jnp.abs(jnp.linalg.det(data.globals["cell"]))[:, None, None]
            det = jnp.where(det > 0.0, det, 1.0)  # padded graphs have det = 0
            stress = virial / det
            graph_mask = jraph.get_graph_padding_mask(data)
            stress = jnp.where(graph_mask[:, None, None], stress, 0.0)
            return graph_energies[:, 0], -minus_forces_m, stress

        def energy_only(_):
            E, node_energies = energy_components((data.nodes["positions"], eps0))
            graph_energies = jraph.segment_sum(
                node_energies,
                node_graph_idx(data),
                num_segments=data.n_node.shape[0],
                indices_are_sorted=True,
            )
            # zeros for forces/stress with correct shapes
            F = jnp.zeros_like(data.nodes["positions"])  # [N,3]
            S = jnp.zeros_like(data.globals["cell"])     # [B,3,3]
            return graph_energies[:, 0], F, S

        # If any graph requests forces, compute them; else energy-only fast path
        # Support scalar or vector flags
        do_forces = jnp.any(need_forces_flag != 0)
        return jax.lax.cond(do_forces, with_forces, energy_only, operand=None)


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    # based on https://github.com/google-deepmind/jraph/blob/51f5990/jraph/_src/models.py#L209-L216
    n_graph = data.n_node.shape[0]
    # equivalent to jnp.sum(n_node), but jittable
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node)
    return node_gr_idx


def weight_decay_mask(model):
    """weight decay mask (only apply decay to linear weights)"""

    def is_layer(x):
        return isinstance(x, Linear) or isinstance(x, e3nn.equinox.Linear)

    def set_mask(x):
        if isinstance(x, Linear):
            mask = jax.tree.map(lambda _: True, x)
            mask = eqx.tree_at(lambda m: m.bias, mask, False)
            return mask
        elif isinstance(x, e3nn.equinox.Linear):
            return jax.tree.map(lambda _: True, x)
        else:
            return jax.tree.map(lambda _: False, x)

        return mask

    mask = jax.tree.map(set_mask, model, is_leaf=is_layer)
    return mask


def save_model(path: str, model: eqx.Module, config: dict):
    """Save a model and its config to a file."""
    with open(path, "wb") as f:
        config_str = json.dumps(config)
        f.write((config_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(path: str) -> tuple[Nequix, dict]:
    """Load a model and its config from a file."""
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        model = Nequix(
            key=jax.random.key(0),
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
            shift=config["shift"],
            scale=config["scale"],
            avg_n_neighbors=config["avg_n_neighbors"],
            # NOTE: atom_energies will be in model weights
        )
        model = eqx.tree_deserialise_leaves(f, model)
        return model, config


class NequixMoE(eqx.Module):
    """Mixture-of-Experts wrapper combining a pretrained 'base' Nequix and a TAE expert.

    - Gating per-graph chooses between experts; supports 'rule' (by cell volume) or 'learned' (logit on log-volume).
    - Returns blended energy/forces/stress: w * TAE + (1 - w) * Base.
    - Base can be frozen via stop-gradient to preserve pretrained knowledge.
    """
    base: Nequix
    tae: Nequix
    gating_mode: str = eqx.field(static=True)
    freeze_base: bool = eqx.field(static=True)
    vol_eps: float = eqx.field(static=True)
    gate_bias: jax.Array
    gate_scale: jax.Array
    # Control where forces/stress come from: 'blend' (default), 'base', or 'tae'
    force_from: str = eqx.field(static=True)
    # Optional: avoid routing graphs with unseen species to TAE expert
    avoid_unseen_to_tae: bool = eqx.field(static=True)
    seen_species_mask: Optional[jax.Array] = None  # [n_species] bool or 0/1

    def __init__(
        self,
        base: Nequix,
        tae: Nequix,
        gating_mode: str = "rule",
        freeze_base: bool = True,
        vol_eps: float = 1e-8,
        force_from: str = "blend",
        avoid_unseen_to_tae: bool = False,
        seen_species_mask: Optional[jax.Array] = None,
        *,
        key: jax.Array | None = None,
    ):
        self.base = base
        self.tae = tae
        self.gating_mode = gating_mode
        self.freeze_base = freeze_base
        self.vol_eps = vol_eps
        self.force_from = force_from
        self.avoid_unseen_to_tae = bool(avoid_unseen_to_tae)
        self.seen_species_mask = seen_species_mask
        if key is None:
            key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        self.gate_bias = jax.random.normal(k1, ()) * 0.01
        self.gate_scale = jax.random.normal(k2, ()) * 0.01

    def _gate_weights(self, data: jraph.GraphsTuple) -> jax.Array:
        # Use cell volume as a proxy: small/zero → molecule (TAE); large → periodic
        vol = jnp.abs(jnp.linalg.det(data.globals["cell"]))  # [B]
        if self.gating_mode == "rule":
            w = (vol <= self.vol_eps).astype(jnp.float32)
        else:
            # Learned smooth gate: sigmoid(bias - scale * log1p(vol)) ⇒ small vol → w≈1
            w = jax.nn.sigmoid(self.gate_bias - self.gate_scale * jnp.log1p(vol))
        return w  # [B]

    def __call__(self, data: jraph.GraphsTuple):
        if self.freeze_base:
            base_model = eqx.tree_map(lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x, self.base)
        else:
            base_model = self.base
        Eb, Fb, Sb = base_model(data)
        Et, Ft, St = self.tae(data)
        w = self._gate_weights(data)  # [B]
        # If configured, force base (w=0) for any graph containing unseen species
        if self.avoid_unseen_to_tae and (self.seen_species_mask is not None):
            gidx = node_graph_idx(data)
            species = data.nodes["species"]  # [N]
            seen_mask = self.seen_species_mask
            # unseen indicator per node
            unseen = 1.0 - seen_mask[species].astype(jnp.float32)
            unseen_per_graph = jraph.segment_sum(unseen, gidx, num_segments=data.n_node.shape[0])
            has_unseen = unseen_per_graph > 0
            w = jnp.where(has_unseen, 0.0, w)
        # Blend energies per-graph
        E = (1.0 - w) * Eb + w * Et
        # Blend forces per-node according to their graph
        if self.force_from == "base":
            F = Fb
            S = Sb
        elif self.force_from == "tae":
            F = Ft
            S = St
        else:
            gidx = node_graph_idx(data)
            w_nodes = w[gidx][:, None]
            F = (1.0 - w_nodes) * Fb + w_nodes * Ft
            # Blend stress per-graph
            w_graph = w[:, None, None]
            S = (1.0 - w_graph) * Sb + w_graph * St
        return E, F, S
