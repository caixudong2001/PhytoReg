"""Shared spatial pathway and task-adaptive peak readout for PhytoReg."""
import torch
from torch import nn
from torch.nn import functional as F

TASKS = ('Histone', 'OCR', 'TF')


class SymmetricConv(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(width, width, 3))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.kaiming_uniform_(self.weight, a=5**.5)

    def forward(self, x):
        weight = torch.cat((self.weight[:, :, 1:].flip(-1), self.weight), -1)
        return F.conv1d(x, weight, self.bias, padding=2*self.dilation, dilation=self.dilation)


def local_mean(x, support, kernel):
    numerator = F.avg_pool1d(x, kernel, 1, kernel//2, count_include_pad=True)
    denominator = F.avg_pool1d(support.float(), kernel, 1, kernel//2, count_include_pad=True)
    return numerator / denominator.clamp_min(1./kernel)


class SpatialPathway(nn.Module):
    """One set of convolution weights shared by all three assay tasks."""
    def __init__(self, width, num_conditions):
        super().__init__()
        self.stem = nn.Conv1d(2, width, 1)
        self.branches = nn.ModuleList(SymmetricConv(width, d) for d in (1, 4, 16, 64))
        self.mix = nn.Conv1d(2*width, width, 1)
        self.condition = nn.Embedding(num_conditions, 4)
        self.gate = nn.Linear(4, 4)
        self.readout_weight = nn.Parameter(torch.zeros(3, width))
        self.readout_bias = nn.Parameter(torch.zeros(3))
        self.center_weight = nn.Parameter(torch.zeros(3, 2))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, peak, signal, tasks, conditions, region, input_mean, input_std):
        raw = torch.stack((peak.detach().float(), signal.detach().float()), 1)
        x = (raw-input_mean[tasks, :, None])/input_std[tasks, :, None]
        support = region[None, None].expand(len(peak), 1, -1)
        x = torch.where(support, x, 0.)
        center = F.silu(self.stem(x))*support
        gates = self.gate(self.condition(conditions)).softmax(-1)
        combined = torch.zeros_like(center)
        for i, (branch, dilation) in enumerate(zip(self.branches, (1, 4, 16, 64))):
            pooled = local_mean(center, support, 2*dilation+1)
            combined = combined + F.silu(branch(pooled))*gates[:, i, None, None]
        hidden = F.silu(self.mix(torch.cat((center, combined), 1)))
        residual = ((hidden*self.readout_weight[tasks, :, None]).sum(1)
                    + self.readout_bias[tasks, None]
                    + (x*self.center_weight[tasks, :, None]).sum(1))
        return torch.where(region[None], peak.detach().float()+residual, peak.detach().float())


class TaskFeatureReadout(nn.Module):
    def __init__(self, hidden_dim, output_dim, tracks, eps):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.projection = nn.Linear(hidden_dim, output_dim, bias=False)
        self.weight = nn.Parameter(torch.zeros(tracks, output_dim))
        self.bias = nn.Parameter(torch.zeros(tracks))

    def forward(self, h, rows):
        z = self.projection(self.norm(h.detach().float()))[None]
        return (torch.einsum('bld,bkd->blk', z, self.weight[rows][None])
                + self.bias[rows][None, None])[0].T


class PeakDecoder(nn.Module):
    """Frozen context priors, a shared spatial pathway, and task readouts.

    All learned parameters and calibration coefficients belong to the single
    PhytoReg state dictionary. No observed signal or labels enter this module.
    """
    def __init__(self, config, catalog):
        super().__init__()
        size = config.head.registry_size
        self.register_buffer('tasks', torch.full((size,), -1, dtype=torch.long))
        self.register_buffer('conditions', torch.zeros(size, dtype=torch.long))
        self.register_buffer('enabled', torch.zeros(size, dtype=torch.bool))
        self.register_buffer('tf_lookup', torch.full((size,), -1, dtype=torch.long))
        tf_count = 0
        for row in catalog:
            i = row['registry_index']
            self.tasks[i] = TASKS.index(row['task_group'])
            self.conditions[i] = row.get('spatial_condition_index', 0)
            self.enabled[i] = row.get('peak_enabled', True)
            if row['task_group'] == 'TF':
                self.tf_lookup[i] = tf_count
                tf_count += 1
        self.register_buffer('input_mean', torch.zeros(3, 2))
        self.register_buffer('input_std', torch.ones(3, 2))
        self.register_buffer('local_weight', torch.zeros(3, 8))
        self.register_buffer('local_bias', torch.zeros(3))
        self.register_buffer('context_gain', torch.ones(3))
        self.register_buffer('readout_gain', torch.ones(3))
        self.spatial = SpatialPathway(config.spatial.width, config.spatial.num_conditions)
        self.context = SpatialPathway(config.spatial.context_width, config.spatial.num_conditions)
        self.calibration = SpatialPathway(config.spatial.context_width, config.spatial.num_conditions)
        args = config.head.hidden_dim, config.head.output_dim, tf_count, config.head.eps
        self.task_readout = TaskFeatureReadout(*args)
        self.context_readout = TaskFeatureReadout(*args)
        self.feature_reference = TaskFeatureReadout(*args)
        self.context.requires_grad_(False)
        self.calibration.requires_grad_(False)
        self.context_readout.requires_grad_(False)
        self.feature_reference.requires_grad_(False)

    def _local_prior(self, peak, signal, tasks, region):
        p = torch.where(region[None], peak.detach().float(), 0.)
        s = torch.where(region[None], signal.detach().float(), 0.)
        features = [p, s]
        x = torch.stack((p, s), 1)
        support = region[None, None].expand(len(p), 1, -1)
        for kernel in (3, 9, 33):
            means = local_mean(x, support, kernel)
            features.extend((means[:, 1], means[:, 0]-p))
        residual = ((torch.stack(features, -1)*self.local_weight[tasks, None]).sum(-1)
                    + self.local_bias[tasks, None])
        return torch.where(region[None], peak.detach().float()+residual, peak.detach().float())

    def _score(self, pathway, readout, h, peak, signal, ids, region):
        tasks = self.tasks[ids]
        prior = self._local_prior(peak, signal, tasks, region)
        value = prior + (pathway(peak, signal, tasks, self.conditions[ids], region,
                                self.input_mean, self.input_std)-peak.detach())
        tf = tasks == 2
        if torch.any(tf):
            rows = self.tf_lookup[ids[tf]]
            ref = self.feature_reference(h, rows)
            delta = readout(h, rows)-ref
            value = value.clone()
            value[tf] = value[tf]+torch.where(region[None], delta, 0.)
        gain = self.context_gain[tasks, None]
        chosen = gain[:, 0] != 1
        if torch.any(chosen):
            with torch.no_grad():
                baseline = self._local_prior(peak[chosen], signal[chosen], tasks[chosen], region)
                baseline = baseline + (self.context(peak[chosen], signal[chosen], tasks[chosen],
                    self.conditions[ids[chosen]], region, self.input_mean, self.input_std)-peak[chosen].detach())
            value = value.clone()
            value[chosen] = torch.where(gain[chosen] == 0, baseline,
                                       baseline+gain[chosen]*(value[chosen]-baseline))
        return torch.where(region[None], value, peak.detach())

    def forward(self, h, peak, signal, indices, valid, positions):
        refined = peak.detach().clone()
        for b in range(len(peak)):
            columns = torch.where(valid[b] & self.enabled[indices[b]])[0]
            if not len(columns):
                continue
            ids = indices[b, columns]
            p, s = peak[b, :, columns].T, signal[b, :, columns].T
            if not torch.isfinite(p[:, positions[b]]).all() or not torch.isfinite(s[:, positions[b]]).all():
                raise ValueError('Active spatial inputs must be finite')
            with torch.autocast(device_type=peak.device.type, enabled=False):
                value = self._score(self.spatial, self.task_readout, h[b], p, s, ids, positions[b])
                gain = self.readout_gain[self.tasks[ids], None]
                chosen = gain[:, 0] != 1
                if torch.any(chosen):
                    with torch.no_grad():
                        base = self._score(self.calibration, self.context_readout, h[b],
                                          p[chosen], s[chosen], ids[chosen], positions[b])
                    value = value.clone()
                    value[chosen] = torch.where(gain[chosen] == 0, base,
                                               base+gain[chosen]*(value[chosen]-base))
                refined[b, :, columns] = value.T
        return refined
