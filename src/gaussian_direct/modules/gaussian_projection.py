import math

import torch as pt

from .runtime import replay_dropout_rng
def build_gaussian_projector(
    *,
    projection_mode,
    dim,
    surface_residue_topk,
    sigma_tangent,
    sigma_normal,
    multiscale_sigmas,
    residue_amplitude_mode,
    residue_amplitude_min,
    residue_amplitude_max,
    phys_feature_dim,
    true_feature_dim,
    atom_feature_dim,
    surface_patch_count,
    dropout,
):
    if projection_mode != "per_residue_sigma":
        raise ValueError("projection_mode must be per_residue_sigma")
    return PerResidueAdaptiveGaussianResidueToSurfaceProjector(
        dim=dim,
        geom_dim=6,
        residue_topk=surface_residue_topk,
        sigma_pairs=multiscale_sigmas,
        learn_amplitude=residue_amplitude_mode == "bounded",
        amplitude_min=residue_amplitude_min,
        amplitude_max=residue_amplitude_max,
        dropout=dropout,
    )


def parse_sigma_pairs(text_or_pairs):
    if isinstance(text_or_pairs, str):
        pairs = []
        for part in text_or_pairs.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                a, b = part.split(":", 1)
            elif "/" in part:
                a, b = part.split("/", 1)
            else:
                value = float(part)
                a, b = value, max(value * 0.5, 1e-3)
            pairs.append((float(a), float(b)))
        return tuple(pairs)
    return tuple((float(a), float(b)) for a, b in text_or_pairs)


def inverse_softplus(value):
    value = pt.as_tensor(value, dtype=pt.float32).clamp_min(1e-6)
    return pt.log(pt.expm1(value).clamp_min(1e-6))


class PerResidueAdaptiveGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Project residue features with residue-conditioned Gaussian bandwidths."""

    accepts_geometry_cache = True

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        residue_topk=16,
        sigma_pairs=((4.0, 2.0), (8.0, 4.0), (14.0, 7.0)),
        sigma_min=0.5,
        learn_amplitude=False,
        amplitude_min=0.25,
        amplitude_max=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_pairs = parse_sigma_pairs(sigma_pairs)
        if not self.sigma_pairs:
            raise ValueError("sigma_pairs must contain at least one scale.")
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        base_sigma = pt.as_tensor(self.sigma_pairs[mid], dtype=pt.float32)
        self.sigma_min = float(sigma_min)
        self.learn_amplitude = bool(learn_amplitude)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        if self.amplitude_max <= self.amplitude_min:
            raise ValueError("amplitude_max must be greater than amplitude_min")
        self.context_proj = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim))
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.sigma_head = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 2),
        )
        last = self.sigma_head[-1]
        pt.nn.init.zeros_(last.weight)
        with pt.no_grad():
            last.bias.copy_(inverse_softplus(base_sigma - self.sigma_min))
        self.amplitude_head = None
        if self.learn_amplitude:
            self.amplitude_head = pt.nn.Sequential(
                pt.nn.LayerNorm(dim),
                pt.nn.Linear(dim, dim),
                pt.nn.GELU(),
                pt.nn.Dropout(dropout),
                pt.nn.Linear(dim, 1),
            )
            amp_last = self.amplitude_head[-1]
            pt.nn.init.zeros_(amp_last.weight)
            target = (1.0 - self.amplitude_min) / max(self.amplitude_max - self.amplitude_min, 1e-6)
            target = min(max(target, 1e-4), 1.0 - 1e-4)
            with pt.no_grad():
                amp_last.bias.fill_(math.log(target / (1.0 - target)))
        self.norm = pt.nn.LayerNorm(dim)

    def _geometry(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        return idx, d_tangent, d_normal

    def forward(self, h_res, residue_centers, cache, geometry_cache=None):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        if geometry_cache is None:
            idx, d_tangent, d_normal = self._geometry(
                surface_pos, surface_normal, residue_centers
            )
        else:
            idx = geometry_cache["idx"].to(device=h_res.device, dtype=pt.long)
            d_tangent = geometry_cache["d_tangent"].to(
                device=h_res.device, dtype=h_res.dtype
            )
            d_normal = geometry_cache["d_normal"].to(
                device=h_res.device, dtype=h_res.dtype
            )

        sigma_res = self.sigma_min + pt.nn.functional.softplus(self.sigma_head(h_res))
        sigma_pair = sigma_res[idx].clamp_min(self.sigma_min)
        score = -0.5 * ((d_tangent / sigma_pair[..., 0]) ** 2 + (d_normal / sigma_pair[..., 1]) ** 2)
        amplitude = None
        if self.amplitude_head is not None:
            amp_raw = self.amplitude_head(h_res).squeeze(-1)
            amplitude = self.amplitude_min + (self.amplitude_max - self.amplitude_min) * pt.sigmoid(amp_raw)
            score = score + amplitude[idx].clamp_min(1e-6).log()
        weight = pt.softmax(score, dim=-1)
        ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
        if self.context_proj is None or self.geom_proj is None or self.norm is None:
            replay_dropout_rng(
                h_res,
                (int(surface_pos.shape[0]), int(h_res.shape[-1])),
                probability=float(self.canonical_replay_dropout_probability),
                training=self.training,
            )
            h_surf0 = h_res.new_zeros((surface_pos.shape[0], h_res.shape[-1]))
        else:
            h_surf0 = self.norm(self.context_proj(ctx) + self.geom_proj(surface_geom))
        self.last_diagnostics = {
            "sigma_tangent_mean": sigma_res[:, 0].mean().detach(),
            "sigma_tangent_min": sigma_res[:, 0].min().detach(),
            "sigma_tangent_max": sigma_res[:, 0].max().detach(),
            "sigma_tangent_std": sigma_res[:, 0].std(unbiased=False).detach(),
            "sigma_normal_mean": sigma_res[:, 1].mean().detach(),
            "sigma_normal_min": sigma_res[:, 1].min().detach(),
            "sigma_normal_max": sigma_res[:, 1].max().detach(),
            "sigma_normal_std": sigma_res[:, 1].std(unbiased=False).detach(),
        }
        if amplitude is not None:
            self.last_diagnostics.update(
                {
                    "residue_amplitude_mean": amplitude.mean().detach(),
                    "residue_amplitude_min": amplitude.min().detach(),
                    "residue_amplitude_max": amplitude.max().detach(),
                    "residue_amplitude_std": amplitude.std(unbiased=False).detach(),
                }
            )
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight

    def forward_batched(self, requests):
        """Project fixed-size surfaces while padding only the residue axis."""

        surface_counts = {
            int(request["cache"]["surface_pos"].shape[0]) for request in requests
        }
        residue_counts = [int(request["h_res"].shape[0]) for request in requests]
        if len(surface_counts) != 1 or min(residue_counts) < self.residue_topk:
            results = []
            for request in requests:
                values = self.forward(**request)
                results.append(
                    {
                        "values": values,
                        "diagnostics": dict(self.last_diagnostics),
                        "surface_residue_dist": None,
                    }
                )
            return results

        device = requests[0]["h_res"].device
        dtype = requests[0]["h_res"].dtype
        batch_size = len(requests)
        max_residues = max(residue_counts)
        h_res = pt.nn.utils.rnn.pad_sequence(
            [request["h_res"] for request in requests], batch_first=True
        )
        centers = pt.nn.utils.rnn.pad_sequence(
            [request["residue_centers"] for request in requests], batch_first=True
        )
        surface_pos_values = []
        surface_normal_values = []
        surface_geom_values = []
        for request in requests:
            cache = request["cache"]
            surface_pos = cache["surface_pos"].to(device=device, dtype=dtype)
            surface_normal = cache.get("surface_normal")
            if surface_normal is None:
                center = request["residue_centers"].mean(dim=0, keepdim=True)
                surface_normal = surface_pos - center
                surface_normal = surface_normal / surface_normal.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
            else:
                surface_normal = surface_normal.to(device=device, dtype=dtype)
            surface_pos_values.append(surface_pos)
            surface_normal_values.append(surface_normal)
            surface_geom_values.append(
                cache["surface_geom"].to(device=device, dtype=dtype)
            )
        surface_pos = pt.stack(surface_pos_values, dim=0)
        surface_normal = pt.stack(surface_normal_values, dim=0)
        surface_geom = pt.stack(surface_geom_values, dim=0)

        geometry_caches = [request.get("geometry_cache") for request in requests]
        surface_residue_dist = None
        if all(cache is not None for cache in geometry_caches):
            idx = pt.stack(
                [cache["idx"].to(device=device, dtype=pt.long) for cache in geometry_caches]
            )
            d_tangent = pt.stack(
                [cache["d_tangent"].to(device=device, dtype=dtype) for cache in geometry_caches]
            )
            d_normal = pt.stack(
                [cache["d_normal"].to(device=device, dtype=dtype) for cache in geometry_caches]
            )
        else:
            surface_residue_dist = pt.cdist(surface_pos, centers).clamp_min(0.0)
            residue_mask = (
                pt.arange(max_residues, device=device).view(1, 1, -1)
                < pt.as_tensor(residue_counts, device=device).view(-1, 1, 1)
            )
            masked_dist = surface_residue_dist.masked_fill(
                ~residue_mask, float("inf")
            )
            idx = pt.topk(
                masked_dist, k=self.residue_topk, dim=-1, largest=False
            ).indices
            batch_index = pt.arange(batch_size, device=device).view(
                batch_size, 1, 1
            )
            rel = centers[batch_index, idx] - surface_pos.unsqueeze(2)
            d_normal = (rel * surface_normal.unsqueeze(2)).sum(dim=-1).abs()
            d_tangent = (
                rel.square().sum(dim=-1) - d_normal.square()
            ).clamp_min(0.0).sqrt()

        sigma_res = self.sigma_min + pt.nn.functional.softplus(self.sigma_head(h_res))
        batch_index = pt.arange(batch_size, device=device).view(batch_size, 1, 1)
        sigma_pair = sigma_res[batch_index, idx].clamp_min(self.sigma_min)
        score = -0.5 * (
            (d_tangent / sigma_pair[..., 0]).square()
            + (d_normal / sigma_pair[..., 1]).square()
        )
        amplitude = None
        if self.amplitude_head is not None:
            amp_raw = self.amplitude_head(h_res).squeeze(-1)
            amplitude = self.amplitude_min + (
                self.amplitude_max - self.amplitude_min
            ) * pt.sigmoid(amp_raw)
            score = score + amplitude[batch_index, idx].clamp_min(1e-6).log()
        weight = pt.softmax(score, dim=-1)
        context = (h_res[batch_index, idx] * weight.unsqueeze(-1)).sum(dim=2)
        if self.context_proj is None or self.geom_proj is None or self.norm is None:
            replay_dropout_rng(
                h_res,
                (batch_size, int(surface_pos.shape[1]), int(h_res.shape[-1])),
                probability=float(self.canonical_replay_dropout_probability),
                training=self.training,
            )
            h_surf0 = h_res.new_zeros(
                (batch_size, surface_pos.shape[1], h_res.shape[-1])
            )
        else:
            h_surf0 = self.norm(
                self.context_proj(context) + self.geom_proj(surface_geom)
            )

        residue_mask = (
            pt.arange(max_residues, device=device).unsqueeze(0)
            < pt.as_tensor(residue_counts, device=device).unsqueeze(1)
        )

        def masked_stats(values):
            count = pt.as_tensor(
                residue_counts, device=device, dtype=values.dtype
            ).clamp_min(1.0)
            mean = (values * residue_mask).sum(dim=-1) / count
            variance = (
                (values - mean.unsqueeze(-1)).square() * residue_mask
            ).sum(dim=-1) / count
            minimum = values.masked_fill(~residue_mask, float("inf")).min(dim=-1).values
            maximum = values.masked_fill(~residue_mask, -float("inf")).max(dim=-1).values
            return mean, minimum, maximum, variance.clamp_min(0.0).sqrt()

        sigma_t_stats = masked_stats(sigma_res[..., 0])
        sigma_n_stats = masked_stats(sigma_res[..., 1])
        diagnostics = {
            "sigma_tangent_mean": sigma_t_stats[0].detach(),
            "sigma_tangent_min": sigma_t_stats[1].detach(),
            "sigma_tangent_max": sigma_t_stats[2].detach(),
            "sigma_tangent_std": sigma_t_stats[3].detach(),
            "sigma_normal_mean": sigma_n_stats[0].detach(),
            "sigma_normal_min": sigma_n_stats[1].detach(),
            "sigma_normal_max": sigma_n_stats[2].detach(),
            "sigma_normal_std": sigma_n_stats[3].detach(),
        }
        if amplitude is not None:
            amp_stats = masked_stats(amplitude)
            diagnostics.update(
                {
                    "residue_amplitude_mean": amp_stats[0].detach(),
                    "residue_amplitude_min": amp_stats[1].detach(),
                    "residue_amplitude_max": amp_stats[2].detach(),
                    "residue_amplitude_std": amp_stats[3].detach(),
                }
            )

        results = []
        for sample_index, residue_count in enumerate(residue_counts):
            sample_dist = (
                surface_residue_dist[sample_index, :, :residue_count]
                if surface_residue_dist is not None
                else None
            )
            results.append(
                {
                    "values": (
                        h_surf0[sample_index],
                        surface_pos[sample_index],
                        surface_normal[sample_index],
                        surface_geom[sample_index],
                        idx[sample_index],
                        weight[sample_index],
                    ),
                    "diagnostics": {
                        key: value[sample_index] for key, value in diagnostics.items()
                    },
                    "surface_residue_dist": sample_dist,
                }
            )
        return results
