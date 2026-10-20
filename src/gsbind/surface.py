"""Intervene only on Gaussian surface-to-residue gate/update/fusion inputs."""

import torch


SURFACE_CONDITIONS = ['baseline', 'identity', 'S_retain50', 'S_retain00', 'S_noise101', 'S_noise211', 'S_noise307']


class SurfaceContextCutoff:
    def __init__(self, gaussian, condition='baseline'):
        if condition not in SURFACE_CONDITIONS:
            raise ValueError('Only final Gaussian-surface interventions are supported')
        self.gaussian = gaussian
        self.condition = condition
        self.stage = -1
        self.audit = []
        self.handles = []

    def __enter__(self):
        self.handles = [self.gaussian.core_residue_gate.register_forward_pre_hook(self.gate),
                        self.gaussian.core_residue_update.register_forward_pre_hook(self.update),
                        self.gaussian.same_level_fusion.register_forward_pre_hook(self.fusion)]
        return self

    def replace(self, arguments, stage):
        original = arguments[0]
        if self.condition in ['baseline', 'identity']:
            return arguments
        residue, surface = original.chunk(2, dim=-1)
        if self.condition.startswith('S_noise'):
            seed = int(self.condition.replace('S_noise', ''))
            generator = torch.Generator(device=surface.device)
            generator.manual_seed(seed + 1009 * stage)
            direction = torch.randn(surface.shape, generator=generator, dtype=surface.dtype, device=surface.device)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            replacement = surface + direction * surface.norm(dim=-1, keepdim=True)
        else:
            fraction = .5 if self.condition == 'S_retain50' else 0.
            replacement = surface - (1-fraction) * surface
        self.audit.append(dict(stage=stage, rows=len(surface), shift_rms=float((replacement-surface).square().mean().sqrt().detach().cpu())))
        return (torch.cat([residue, replacement], dim=-1), *arguments[1:])

    def gate(self, module, arguments):
        self.stage += 1
        return self.replace(arguments, self.stage)

    def update(self, module, arguments):
        return self.replace(arguments, self.stage)

    def fusion(self, module, arguments):
        return self.replace(arguments, 2)

    def __exit__(self, exception_type, exception, traceback):
        for handle in self.handles:
            handle.remove()
