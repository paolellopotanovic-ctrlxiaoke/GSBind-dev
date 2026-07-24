"""Single-residue Gaussian residue-to-surface projection families."""

import math

import torch as pt


class GaussianResidueToSurfaceProjector(pt.nn.Module):
    def __init__(self, dim=128, geom_dim=6, residue_topk=8, sigma_t=6.0, sigma_n=3.0, dropout=0.1):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_t = float(sigma_t)
        self.sigma_n = float(sigma_n)
        self.res_proj = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim))
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def _weights(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        score = -0.5 * ((d_tangent / self.sigma_t) ** 2 + (d_normal / self.sigma_n) ** 2)
        weight = pt.softmax(score, dim=-1)
        return idx, weight, d_tangent, d_normal

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, weight, _d_tangent, _d_normal = self._weights(surface_pos, surface_normal, residue_centers)
        h_context = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(self.res_proj(h_context) + self.geom_proj(surface_geom))
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight

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
class MultiScaleGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Project residue features onto surface points with several 2D Gaussian kernels.

    The old projector uses one tangent/normal bandwidth. This version keeps the
    same input/output contract but lets each surface point mix short-, mid-, and
    long-range residue context. It is still a representation change only; no
    additional loss is introduced.
    """

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        residue_topk=16,
        sigma_pairs=((4.0, 2.0), (8.0, 4.0), (14.0, 7.0)),
        dropout=0.1,
    ):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_pairs = parse_sigma_pairs(sigma_pairs)
        if not self.sigma_pairs:
            raise ValueError("sigma_pairs must contain at least one scale.")
        self.context_proj = pt.nn.ModuleList(
            [pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim)) for _ in self.sigma_pairs]
        )
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.scale_router = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim + len(self.sigma_pairs) * 2),
            pt.nn.Linear(geom_dim + len(self.sigma_pairs) * 2, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, len(self.sigma_pairs)),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def _geometry(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        self.last_surface_residue_dist = dist
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal = (rel * normal).sum(dim=-1).abs()
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal.square()).clamp_min(0.0).sqrt()
        return idx, d_tangent, d_normal

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        contexts = []
        stats = []
        for proj, (sigma_t, sigma_n) in zip(self.context_proj, self.sigma_pairs):
            score = -0.5 * ((d_tangent / float(sigma_t)) ** 2 + (d_normal / float(sigma_n)) ** 2)
            weight = pt.softmax(score, dim=-1)
            ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
            contexts.append(proj(ctx))
            stats.append(pt.stack([weight.max(dim=-1).values, weight.square().sum(dim=-1)], dim=-1))
        router_in = pt.cat([surface_geom, *stats], dim=-1)
        scale_weight = pt.softmax(self.scale_router(router_in), dim=-1)
        stacked = pt.stack(contexts, dim=1)
        h_context = (stacked * scale_weight.unsqueeze(-1)).sum(dim=1)
        # Return the mid-scale weights for diagnostics/backward-compatible callers.
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        sigma_t, sigma_n = self.sigma_pairs[mid]
        score = -0.5 * ((d_tangent / float(sigma_t)) ** 2 + (d_normal / float(sigma_n)) ** 2)
        weight = pt.softmax(score, dim=-1)
        h_surf0 = self.norm(h_context + self.geom_proj(surface_geom))
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class GlobalWeightedMultiScaleGaussianResidueToSurfaceProjector(MultiScaleGaussianResidueToSurfaceProjector):
    """Multiscale Gaussian projection with one learnable global scale mixture.

    This keeps the same sigma bank as the fixed multiscale projector, but removes
    per-surface-point routing so the only new degree of freedom is a global
    softmax over Gaussian scales.
    """

    def __init__(self, *args, init_logits=None, **kwargs):
        super().__init__(*args, **kwargs)
        if init_logits is None:
            init_logits = pt.zeros(len(self.sigma_pairs), dtype=pt.float32)
        self.global_scale_logits = pt.nn.Parameter(pt.as_tensor(init_logits, dtype=pt.float32).clone())

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        contexts = []
        mid_weight = None
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        for scale_idx, (proj, (sigma_t, sigma_n)) in enumerate(zip(self.context_proj, self.sigma_pairs)):
            score = -0.5 * ((d_tangent / float(sigma_t)) ** 2 + (d_normal / float(sigma_n)) ** 2)
            weight = pt.softmax(score, dim=-1)
            if scale_idx == mid:
                mid_weight = weight
            ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
            contexts.append(proj(ctx))
        scale_weight = pt.softmax(self.global_scale_logits.to(device=h_res.device, dtype=h_res.dtype), dim=-1)
        stacked = pt.stack(contexts, dim=0)
        h_context = (stacked * scale_weight.view(-1, 1, 1)).sum(dim=0)
        h_surf0 = self.norm(h_context + self.geom_proj(surface_geom))
        entropy = -(scale_weight * scale_weight.clamp_min(1e-8).log()).sum()
        self.last_diagnostics = {
            "sigma_global_weight_entropy": entropy.detach(),
            "sigma_global_weight_max": scale_weight.max().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, mid_weight


class LearnableScaleGaussianResidueToSurfaceProjector(GlobalWeightedMultiScaleGaussianResidueToSurfaceProjector):
    """Multiscale projection where the Gaussian bandwidths themselves are learned.

    Unlike ``multiscale_global``, this does not only learn a softmax over a fixed
    sigma bank. It initializes from the same bank, then optimizes the tangent and
    normal bandwidth values directly with a positive softplus parameterization.
    """

    def __init__(self, *args, sigma_min=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_min = float(sigma_min)
        init = pt.as_tensor(self.sigma_pairs, dtype=pt.float32)
        self.learned_sigma_raw = pt.nn.Parameter(inverse_softplus(init - self.sigma_min))

    def _learned_sigmas(self, device, dtype):
        return self.sigma_min + pt.nn.functional.softplus(self.learned_sigma_raw.to(device=device, dtype=dtype))

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        sigmas = self._learned_sigmas(device=h_res.device, dtype=h_res.dtype)
        contexts = []
        mid_weight = None
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        for scale_idx, proj in enumerate(self.context_proj):
            sigma_t = sigmas[scale_idx, 0].clamp_min(self.sigma_min)
            sigma_n = sigmas[scale_idx, 1].clamp_min(self.sigma_min)
            score = -0.5 * ((d_tangent / sigma_t) ** 2 + (d_normal / sigma_n) ** 2)
            weight = pt.softmax(score, dim=-1)
            if scale_idx == mid:
                mid_weight = weight
            ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
            contexts.append(proj(ctx))
        scale_weight = pt.softmax(self.global_scale_logits.to(device=h_res.device, dtype=h_res.dtype), dim=-1)
        stacked = pt.stack(contexts, dim=0)
        h_context = (stacked * scale_weight.view(-1, 1, 1)).sum(dim=0)
        h_surf0 = self.norm(h_context + self.geom_proj(surface_geom))
        entropy = -(scale_weight * scale_weight.clamp_min(1e-8).log()).sum()
        self.last_diagnostics = {
            "sigma_global_weight_entropy": entropy.detach(),
            "sigma_global_weight_max": scale_weight.max().detach(),
            "sigma_tangent_mean": sigmas[:, 0].mean().detach(),
            "sigma_tangent_min": sigmas[:, 0].min().detach(),
            "sigma_tangent_max": sigmas[:, 0].max().detach(),
            "sigma_normal_mean": sigmas[:, 1].mean().detach(),
            "sigma_normal_min": sigmas[:, 1].min().detach(),
            "sigma_normal_max": sigmas[:, 1].max().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, mid_weight


class PerResidueAdaptiveGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Project residue features with residue-conditioned Gaussian bandwidths."""

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

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

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


class FullLearnableCovarianceGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Project residue features with residue-conditioned amplitude and local 3D covariance."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        residue_topk=16,
        sigma_pairs=((4.0, 2.0), (8.0, 4.0), (14.0, 7.0)),
        sigma_min=0.5,
        amplitude_min=0.25,
        amplitude_max=2.0,
        offdiag_scale=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_pairs = parse_sigma_pairs(sigma_pairs)
        if not self.sigma_pairs:
            raise ValueError("sigma_pairs must contain at least one scale.")
        mid = min(len(self.sigma_pairs) // 2, len(self.sigma_pairs) - 1)
        base_t, base_n = self.sigma_pairs[mid]
        self.sigma_min = float(sigma_min)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        self.offdiag_scale = float(offdiag_scale)
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
        self.gaussian_head = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 7),
        )
        last = self.gaussian_head[-1]
        pt.nn.init.zeros_(last.weight)
        with pt.no_grad():
            base_diag = pt.as_tensor([base_t, base_t, base_n], dtype=pt.float32)
            last.bias[:3].copy_(inverse_softplus(base_diag - self.sigma_min))
            last.bias[3:6].zero_()
            target = (1.0 - self.amplitude_min) / max(self.amplitude_max - self.amplitude_min, 1e-6)
            target = min(max(target, 1e-4), 1.0 - 1e-4)
            last.bias[6].fill_(math.log(target / (1.0 - target)))
        self.norm = pt.nn.LayerNorm(dim)

    def _geometry(self, surface_pos, surface_normal, residue_centers):
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        normal = surface_normal.unsqueeze(1)
        d_normal_signed = (rel * normal).sum(dim=-1)
        d2 = (rel * rel).sum(dim=-1).clamp_min(0.0)
        d_tangent = (d2 - d_normal_signed.square()).clamp_min(0.0).sqrt()
        return idx, d_tangent, d_normal_signed

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal_signed = self._geometry(surface_pos, surface_normal, residue_centers)

        params = self.gaussian_head(h_res)
        diag = self.sigma_min + pt.nn.functional.softplus(params[:, :3])
        offdiag = self.offdiag_scale * pt.tanh(params[:, 3:6])
        amp_raw = params[:, 6]
        amplitude = self.amplitude_min + (self.amplitude_max - self.amplitude_min) * pt.sigmoid(amp_raw)

        diag_pair = diag[idx].clamp_min(self.sigma_min)
        off_pair = offdiag[idx]
        x0 = d_tangent
        x1 = pt.zeros_like(x0)
        x2 = d_normal_signed

        l00 = diag_pair[..., 0]
        l11 = diag_pair[..., 1]
        l22 = diag_pair[..., 2]
        l10 = off_pair[..., 0]
        l20 = off_pair[..., 1]
        l21 = off_pair[..., 2]
        y0 = x0 / l00.clamp_min(1e-6)
        y1 = (x1 - l10 * y0) / l11.clamp_min(1e-6)
        y2 = (x2 - l20 * y0 - l21 * y1) / l22.clamp_min(1e-6)
        mahal = y0.square() + y1.square() + y2.square()
        score = -0.5 * mahal + amplitude[idx].clamp_min(1e-6).log()

        weight = pt.softmax(score, dim=-1)
        ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(self.context_proj(ctx) + self.geom_proj(surface_geom))
        self.last_diagnostics = {
            "fullcov_sigma_t1_mean": diag[:, 0].mean().detach(),
            "fullcov_sigma_t1_min": diag[:, 0].min().detach(),
            "fullcov_sigma_t1_max": diag[:, 0].max().detach(),
            "fullcov_sigma_t2_mean": diag[:, 1].mean().detach(),
            "fullcov_sigma_t2_min": diag[:, 1].min().detach(),
            "fullcov_sigma_t2_max": diag[:, 1].max().detach(),
            "fullcov_sigma_normal_mean": diag[:, 2].mean().detach(),
            "fullcov_sigma_normal_min": diag[:, 2].min().detach(),
            "fullcov_sigma_normal_max": diag[:, 2].max().detach(),
            "fullcov_offdiag_10_mean": offdiag[:, 0].mean().detach(),
            "fullcov_offdiag_20_mean": offdiag[:, 1].mean().detach(),
            "fullcov_offdiag_21_mean": offdiag[:, 2].mean().detach(),
            "residue_amplitude_mean": amplitude.mean().detach(),
            "residue_amplitude_min": amplitude.min().detach(),
            "residue_amplitude_max": amplitude.max().detach(),
            "residue_amplitude_std": amplitude.std(unbiased=False).detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class PhysChemGaussianFieldResidueToSurfaceProjector(MultiScaleGaussianResidueToSurfaceProjector):
    """Project learned residue context plus continuous physchem scalar fields."""

    accepts_phys_features = True

    def __init__(self, *args, phys_feature_dim=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.phys_feature_dim = max(1, int(phys_feature_dim))
        self.field_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(self.phys_feature_dim),
            pt.nn.Linear(self.phys_feature_dim, self.context_proj[0][1].out_features),
            pt.nn.GELU(),
            pt.nn.Linear(self.context_proj[0][1].out_features, self.context_proj[0][1].out_features),
        )

    def forward(self, h_res, residue_centers, cache, residue_phys_features=None):
        h_surf0, surface_pos, surface_normal, surface_geom, idx, weight = super().forward(h_res, residue_centers, cache)
        if residue_phys_features is None:
            phys = h_res.new_zeros((h_res.shape[0], self.phys_feature_dim))
        else:
            phys = residue_phys_features.to(device=h_res.device, dtype=h_res.dtype)
            if phys.shape[1] < self.phys_feature_dim:
                phys = pt.cat([phys, phys.new_zeros(phys.shape[0], self.phys_feature_dim - phys.shape[1])], dim=-1)
            elif phys.shape[1] > self.phys_feature_dim:
                phys = phys[:, : self.phys_feature_dim]
        field = (phys[idx.to(device=h_res.device, dtype=pt.long)] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(h_surf0 + self.field_proj(field))
        self.last_diagnostics = {
            "phys_field_abs_mean": field.abs().mean().detach(),
            "phys_field_std": field.std(unbiased=False).detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class MixtureBallGaussianResidueToSurfaceProjector(pt.nn.Module):
    """Each residue emits a small mixture over Gaussian balls before surface pooling."""

    def __init__(
        self,
        dim=128,
        geom_dim=6,
        residue_topk=16,
        sigma_pairs=((3.0, 1.5), (5.0, 2.5), (8.0, 4.0), (14.0, 7.0)),
        dropout=0.1,
    ):
        super().__init__()
        self.residue_topk = int(residue_topk)
        self.sigma_pairs = parse_sigma_pairs(sigma_pairs)
        if not self.sigma_pairs:
            raise ValueError("sigma_pairs must contain at least one mixture component.")
        self.num_balls = len(self.sigma_pairs)
        self.context_proj = pt.nn.Sequential(pt.nn.LayerNorm(dim), pt.nn.Linear(dim, dim))
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.ball_weight = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, self.num_balls),
        )
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

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        idx, d_tangent, d_normal = self._geometry(surface_pos, surface_normal, residue_centers)

        residue_mix = pt.softmax(self.ball_weight(h_res), dim=-1)
        local_mix = residue_mix[idx.to(device=h_res.device, dtype=pt.long)]
        scores = []
        for sigma_t, sigma_n in self.sigma_pairs:
            scores.append(-0.5 * ((d_tangent / float(sigma_t)) ** 2 + (d_normal / float(sigma_n)) ** 2))
        score = pt.stack(scores, dim=-1)
        score = score + local_mix.clamp_min(1e-8).log()
        weight = pt.softmax(pt.logsumexp(score, dim=-1), dim=-1)
        ctx = (h_res[idx] * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(self.context_proj(ctx) + self.geom_proj(surface_geom))
        entropy = -(residue_mix * residue_mix.clamp_min(1e-8).log()).sum(dim=-1)
        self.last_diagnostics = {
            "mixture_ball_entropy": entropy.mean().detach(),
            "mixture_ball_max_weight": residue_mix.max(dim=-1).values.mean().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class Local3DFrameResidueToSurfaceProjector(pt.nn.Module):
    """ScanNet-style local residue neighborhood reader for each surface point."""

    def __init__(self, dim=128, geom_dim=6, residue_topk=16, dropout=0.1):
        super().__init__()
        self.residue_topk = int(residue_topk)
        local_dim = dim + geom_dim + 7
        self.score = pt.nn.Sequential(
            pt.nn.LayerNorm(local_dim),
            pt.nn.Linear(local_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.message = pt.nn.Sequential(
            pt.nn.LayerNorm(local_dim),
            pt.nn.Linear(local_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.geom_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(geom_dim),
            pt.nn.Linear(geom_dim, dim),
            pt.nn.GELU(),
            pt.nn.Linear(dim, dim),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, h_res, residue_centers, cache):
        surface_pos = cache["surface_pos"].to(device=h_res.device, dtype=h_res.dtype)
        surface_normal = cache.get("surface_normal")
        if surface_normal is None:
            center = residue_centers.mean(dim=0, keepdim=True)
            surface_normal = surface_pos - center
            surface_normal = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            surface_normal = surface_normal.to(device=h_res.device, dtype=h_res.dtype)
        surface_geom = cache["surface_geom"].to(device=h_res.device, dtype=h_res.dtype)
        dist = pt.cdist(surface_pos, residue_centers).clamp_min(0.0)
        k = min(self.residue_topk, residue_centers.shape[0])
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = residue_centers[idx] - surface_pos.unsqueeze(1)
        dist_k = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / dist_k
        normal = surface_normal.unsqueeze(1)
        signed_normal = (rel * normal).sum(dim=-1, keepdim=True)
        d_normal = signed_normal.abs()
        d_tangent = (dist_k.square() - d_normal.square()).clamp_min(0.0).sqrt()
        geom = surface_geom.unsqueeze(1).expand(-1, k, -1)
        local = pt.cat([h_res[idx], geom, dist_k, d_tangent, signed_normal, direction], dim=-1)
        score = self.score(local).squeeze(-1)
        weight = pt.softmax(score, dim=-1)
        msg = self.message(local)
        ctx = (msg * weight.unsqueeze(-1)).sum(dim=1)
        h_surf0 = self.norm(ctx + self.geom_proj(surface_geom))
        entropy = -(weight * weight.clamp_min(1e-8).log()).sum(dim=-1) / max(math.log(max(k, 2)), 1e-8)
        self.last_diagnostics = {
            "local3d_attention_entropy": entropy.mean().detach(),
            "local3d_attention_max": weight.max(dim=-1).values.mean().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class Local3DAtomPointCloudResidueToSurfaceProjector(MultiScaleGaussianResidueToSurfaceProjector):
    """Fuse Gaussian residue projection with a local atom point-cloud reader."""

    accepts_atom_features = True

    def __init__(self, *args, atom_feature_dim=123, atom_topk=32, dropout=0.1, **kwargs):
        super().__init__(*args, dropout=dropout, **kwargs)
        dim = self.norm.normalized_shape[0]
        geom_dim = self.geom_proj[0].normalized_shape[0]
        self.atom_topk = int(atom_topk)
        self.atom_feature_proj = pt.nn.Sequential(
            pt.nn.LayerNorm(int(atom_feature_dim)),
            pt.nn.Linear(int(atom_feature_dim), dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        local_dim = dim + geom_dim + 6
        self.atom_score = pt.nn.Sequential(
            pt.nn.LayerNorm(local_dim),
            pt.nn.Linear(local_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, 1),
        )
        self.atom_message = pt.nn.Sequential(
            pt.nn.LayerNorm(local_dim),
            pt.nn.Linear(local_dim, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.atom_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 2),
            pt.nn.Linear(dim * 2, dim),
            pt.nn.Sigmoid(),
        )
        self.atom_norm = pt.nn.LayerNorm(dim)

    def _atom_context(self, h_surf0, surface_pos, surface_normal, surface_geom, atom_pos, atom_features):
        if atom_pos is None or atom_features is None or atom_pos.numel() == 0:
            zero_ctx = h_surf0.new_zeros(h_surf0.shape)
            weight = h_surf0.new_zeros((h_surf0.shape[0], 1))
            return zero_ctx, weight
        atom_pos = atom_pos.to(device=h_surf0.device, dtype=h_surf0.dtype)
        atom_features = atom_features.to(device=h_surf0.device, dtype=h_surf0.dtype)
        k = min(self.atom_topk, int(atom_pos.shape[0]))
        dist = pt.cdist(surface_pos, atom_pos).clamp_min(0.0)
        idx = pt.topk(dist, k=k, dim=-1, largest=False).indices
        rel = atom_pos[idx] - surface_pos.unsqueeze(1)
        d = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / d
        normal = surface_normal.unsqueeze(1)
        signed_normal = (rel * normal).sum(dim=-1, keepdim=True)
        d_tangent = (d.square() - signed_normal.square()).clamp_min(0.0).sqrt()
        atom_h = self.atom_feature_proj(atom_features)[idx]
        geom = surface_geom.unsqueeze(1).expand(-1, k, -1)
        local = pt.cat([atom_h, geom, d, d_tangent, signed_normal, direction], dim=-1)
        score = self.atom_score(local).squeeze(-1)
        weight = pt.softmax(score, dim=-1)
        message = self.atom_message(local)
        ctx = (message * weight.unsqueeze(-1)).sum(dim=1)
        return ctx, weight

    def forward(self, h_res, residue_centers, cache, atom_pos=None, atom_features=None):
        h_surf0, surface_pos, surface_normal, surface_geom, idx, weight = super().forward(h_res, residue_centers, cache)
        atom_ctx, atom_weight = self._atom_context(h_surf0, surface_pos, surface_normal, surface_geom, atom_pos, atom_features)
        gate = self.atom_gate(pt.cat([h_surf0, atom_ctx], dim=-1))
        h_surf0 = self.atom_norm(h_surf0 + gate * atom_ctx)
        entropy = -(atom_weight * atom_weight.clamp_min(1e-8).log()).sum(dim=-1)
        if atom_weight.shape[1] > 1:
            entropy = entropy / max(math.log(atom_weight.shape[1]), 1e-8)
        self.last_diagnostics = {
            "atom_pointcloud_attention_entropy": entropy.mean().detach(),
            "atom_pointcloud_attention_max": atom_weight.max(dim=-1).values.mean().detach(),
            "atom_pointcloud_gate_mean": gate.mean().detach(),
        }
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight


class Local3DPatchPointCloudResidueToSurfaceProjector(Local3DAtomPointCloudResidueToSurfaceProjector):
    """Atom point-cloud reader followed by surface-to-patch compression and gating."""

    def __init__(self, *args, patch_count=32, dropout=0.1, **kwargs):
        super().__init__(*args, dropout=dropout, **kwargs)
        dim = self.norm.normalized_shape[0]
        geom_dim = self.geom_proj[0].normalized_shape[0]
        self.patch_count = int(patch_count)
        self.patch_assignment = pt.nn.Sequential(
            pt.nn.LayerNorm(dim + geom_dim + 3),
            pt.nn.Linear(dim + geom_dim + 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, self.patch_count),
        )
        self.patch_gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim),
            pt.nn.Linear(dim, dim // 2),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim // 2, 1),
            pt.nn.Sigmoid(),
        )
        self.patch_norm = pt.nn.LayerNorm(dim)

    def forward(self, h_res, residue_centers, cache, atom_pos=None, atom_features=None):
        h_surf0, surface_pos, surface_normal, surface_geom, idx, weight = super().forward(
            h_res,
            residue_centers,
            cache,
            atom_pos=atom_pos,
            atom_features=atom_features,
        )
        assign_logits = self.patch_assignment(pt.cat([h_surf0, surface_geom, surface_normal], dim=-1))
        assign = pt.softmax(assign_logits, dim=-1)
        usage = assign.sum(dim=0).clamp_min(1e-6)
        patch = (assign.transpose(0, 1) @ h_surf0) / usage.unsqueeze(-1)
        patch_gate = self.patch_gate(patch)
        surface_gate = assign @ patch_gate
        h_surf0 = self.patch_norm(h_surf0 * (1.0 + 0.1 * surface_gate))
        entropy = -(assign.clamp_min(1e-8) * assign.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(math.log(max(self.patch_count, 2)), 1e-8)
        usage_norm = usage / usage.sum().clamp_min(1e-8)
        self.last_diagnostics.update(
            {
                "pointcloud_patch_assignment_entropy": entropy.mean().detach(),
                "pointcloud_patch_usage_active": (usage_norm > (0.25 / max(self.patch_count, 1))).float().sum().detach(),
                "pointcloud_patch_gate_mean": patch_gate.mean().detach(),
                "pointcloud_patch_gate_std": patch_gate.std(unbiased=False).detach(),
            }
        )
        return h_surf0, surface_pos, surface_normal, surface_geom, idx, weight
