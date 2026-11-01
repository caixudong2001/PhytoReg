"""Four independent per-track readouts for PhytoReg."""
import torch
from torch import nn


class IndependentReadouts(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_tracks, registry_size, num_species, eps=1e-5):
        super().__init__()
        self.projections = nn.ModuleDict({
            n: nn.Sequential(nn.LayerNorm(hidden_dim, eps=eps), nn.Linear(hidden_dim, output_dim, bias=False))
            for n in ('signal', 'peak', 'profile', 'total')
        })
        self.weights = nn.ParameterDict({n: nn.Parameter(torch.zeros(num_tracks, output_dim)) for n in self.projections})
        self.biases = nn.ParameterDict({n: nn.Parameter(torch.zeros(num_tracks)) for n in self.projections})
        self.register_buffer('lookup', torch.full((registry_size,), -1, dtype=torch.long))
        self.signal_delta_weight = nn.Parameter(torch.zeros(num_tracks, output_dim))
        self.signal_delta_bias = nn.Parameter(torch.zeros(num_tracks))
        self.register_buffer('signal_scale', torch.ones(num_tracks))
        self.register_buffer('feature_mean', torch.zeros(num_species, output_dim))
        self.register_buffer('feature_std', torch.ones(num_species, output_dim))
        self.register_buffer('normalization_group', torch.zeros(num_tracks, dtype=torch.long))
        self.output_dim = output_dim

    def forward(self, h, track_indices, valid=None, positions=None):
        valid = torch.ones_like(track_indices, dtype=torch.bool) if valid is None else valid.bool()
        if torch.any(valid & ((track_indices < 0) | (track_indices >= len(self.lookup)))):
            raise ValueError('Track is outside the registry')
        rows = self.lookup[track_indices.clamp(0, len(self.lookup)-1)]
        if torch.any(valid & (rows < 0)):
            raise ValueError('No independent output for this track')
        rows = rows.clamp_min(0)
        outputs = {}
        for name in ('peak', 'profile'):
            z = self.projections[name](h)
            outputs[name] = (torch.einsum('bld,bkd->blk', z, self.weights[name][rows])
                             + self.biases[name][rows][:, None]) * valid[:, None]
        # The signal head combines two learned affine terms, one using
        # species-specific feature normalization.
        group = self.normalization_group[rows]
        first = valid.long().argmax(1, keepdim=True)
        chosen_group = group.gather(1, first)[:, 0]
        if torch.any(valid & (group != chosen_group[:, None])):
            raise ValueError('A sequence cannot combine tracks from different species')
        z = self.projections['signal'](h)
        base = (torch.einsum('bld,bkd->blk', z, self.weights['signal'][rows])
                / self.output_dim**0.5 + self.biases['signal'][rows][:, None]) * valid[:, None]
        # Evaluate each affine branch with the shared signal projection.
        normalized = ((self.projections['signal'](h) - self.feature_mean[chosen_group][:, None])
                      / self.feature_std[chosen_group][:, None])
        delta = (torch.einsum('bld,bkd->blk', normalized, self.signal_delta_weight[rows])
                 + self.signal_delta_bias[rows][:, None]) * self.signal_scale[rows][:, None]
        outputs['signal'] = base + delta * valid[:, None]
        if positions is None:
            pooled = h.mean(1)
        else:
            accumulation = h.float() if h.dtype in (torch.float16, torch.bfloat16) else h
            pooled = ((accumulation * positions[:, :, None]).sum(1)
                      / positions.sum(1, keepdim=True).clamp_min(1)).to(h.dtype)
        z = self.projections['total'](pooled)
        total = (torch.einsum('bd,bkd->bk', z, self.weights['total'][rows]) + self.biases['total'][rows]) * valid
        mask = valid[:, None].expand_as(outputs['profile'])
        if positions is not None:
            mask = mask & positions[:, :, None]
        stable = outputs['profile'].float().masked_fill(~mask, float('-inf'))
        maximum = stable.amax(1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.)
        return dict(signal_z=outputs['signal'], signal_logits=outputs['signal'],
                    peak_logits=outputs['peak'], profile_logits=outputs['profile'],
                    window_total_z=total, window_total_logits=total,
                    signal_profile_score=torch.exp(stable-maximum))
