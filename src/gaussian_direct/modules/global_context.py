import torch as pt

from .gvp import masked_mean


class GlobalContextFeedback(pt.nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.gate = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
            pt.nn.Sigmoid(),
        )
        self.update = pt.nn.Sequential(
            pt.nn.LayerNorm(dim * 3),
            pt.nn.Linear(dim * 3, dim),
            pt.nn.GELU(),
            pt.nn.Dropout(dropout),
            pt.nn.Linear(dim, dim),
        )
        self.norm = pt.nn.LayerNorm(dim)

    def forward(self, h, mask):
        mask = mask.to(dtype=h.dtype, device=h.device)
        mean = masked_mean(h, mask, dim=1)
        masked_h = h.masked_fill(mask.unsqueeze(-1) < 0.5, pt.finfo(h.dtype).min)
        maxv = masked_h.max(dim=1).values
        maxv = pt.where(pt.isfinite(maxv), maxv, pt.zeros_like(maxv))
        global_context = 0.5 * (mean + maxv)
        expanded_context = global_context.unsqueeze(1).expand_as(h)
        update_input = pt.cat([h, expanded_context, h * expanded_context], dim=-1)
        return self.norm(h + self.gate(update_input) * self.update(update_input))
