"""PhytoReg: frozen sequence representation and a task-adaptive spatial readout."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import NTv3Backbone
from .config import BackboneConfig, HeadConfig, PhytoRegConfig
from .heads import IndependentReadouts
from .spatial import PeakDecoder, TASKS

FORMAT = 'PhytoReg'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class PhytoReg(nn.Module):
    """Two-stage DNA regulatory model for a fixed experimental track registry.

    Tracks use registry indices, not contiguous head rows. A batch may mix
    species, but each sequence's selected tracks must belong to one species.
    """
    model_name = 'PhytoReg'

    def __init__(self, config: PhytoRegConfig, registry_indices, normalization_groups, *, track_catalog):
        super().__init__()
        self.config = config
        self.backbone = NTv3Backbone(config.backbone)
        self.head = IndependentReadouts(**asdict(config.head))
        indices = torch.as_tensor(registry_indices, dtype=torch.long)
        groups = torch.as_tensor(normalization_groups, dtype=torch.long)
        if indices.shape != (config.head.num_tracks,) or groups.shape != indices.shape:
            raise ValueError('Registry indices and species groups must have num_tracks entries')
        if (indices.unique().numel() != indices.numel()
                or torch.any((indices < 0) | (indices >= config.head.registry_size))):
            raise ValueError('Registry indices must be unique and within the registry')
        if torch.any((groups < 0) | (groups >= config.head.num_species)):
            raise ValueError('Invalid species normalization group')
        self.head.lookup[indices] = torch.arange(indices.numel())
        self._padding_index = int(indices[0])
        self.head.normalization_group.copy_(groups)
        self.register_buffer('label_center', torch.zeros(config.head.registry_size))
        self.register_buffer('label_scale', torch.ones(config.head.registry_size))
        self.region_total_length = nn.Linear(1, 1, bias=False) if config.region_total_length else None
        if self.region_total_length is not None:
            nn.init.ones_(self.region_total_length.weight)
        self.metadata = {}
        self.track_catalog = [dict(row) for row in track_catalog]
        self.peak_decoder = PeakDecoder(config, self.track_catalog)
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(False)
        if self.region_total_length is not None:
            self.region_total_length.requires_grad_(False)
        self._validate_catalog()
        self.train(self.training)

    @property
    def device(self):
        return self.label_center.device

    def train(self, mode=True):
        """Train the spatial/task readout while keeping first-stage statistics fixed."""
        super().train(mode)
        self.backbone.eval()
        self.head.eval()
        self.peak_decoder.context.eval()
        self.peak_decoder.calibration.eval()
        self.peak_decoder.context_readout.eval()
        self.peak_decoder.feature_reference.eval()
        if self.region_total_length is not None:
            self.region_total_length.eval()
        return self

    def tracks_for(self, species: str, task: str | None = None) -> Tensor:
        """Return registry indices [K] for a species and optional Histone/OCR/TF."""
        indices = [r['registry_index'] for r in self.track_catalog
                   if r['species_id'] == species and (task is None or r['task_group'] == task)]
        if not indices:
            raise ValueError(f'No registered tracks for species={species!r}, task={task!r}')
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _track_mask(self, indices, mask, batch):
        if indices.dtype != torch.long or indices.ndim != 2 or indices.shape[0] != batch or indices.shape[1] == 0:
            raise ValueError('track_indices must be int64 [B,K], with K > 0')
        if indices.device != self.device:
            raise ValueError('track_indices and model must be on the same device')
        valid = indices >= 0 if mask is None else mask
        if valid.dtype != torch.bool or valid.shape != indices.shape or valid.device != indices.device:
            raise ValueError('track_valid_mask must be boolean [B,K] on the model device')
        if torch.any(valid & ((indices < 0) | (indices >= self.config.head.registry_size))):
            raise ValueError('Active track index is outside the registry')
        # Masked padding may be any sentinel; never use it to index label arrays.
        safe = indices.masked_fill(~valid, self._padding_index)
        if torch.any(valid & (self.head.lookup[safe] < 0)):
            raise ValueError('No independent output exists for this track')
        return safe, valid

    def from_features(self, h: Tensor, track_indices: Tensor, *,
                      track_valid_mask: Tensor | None = None,
                      position_valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Run both readout stages on central 32-bp features with known region masks."""
        if h.ndim != 3 or h.shape[-1] != self.config.head.hidden_dim or h.shape[1] == 0:
            raise ValueError('features must have shape [B,L,hidden_dim] with L > 0')
        if h.device != self.device or not h.is_floating_point():
            raise ValueError('features must be floating point on the model device')
        indices, valid = self._track_mask(track_indices, track_valid_mask, h.shape[0])
        positions = (torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
                     if position_valid_mask is None else position_valid_mask)
        if positions.dtype != torch.bool or positions.shape != h.shape[:2] or positions.device != h.device:
            raise ValueError('position_valid_mask must be boolean [B,L] on the model device')
        with torch.autocast(device_type=h.device.type, enabled=False):
            with torch.no_grad():
                p = self.head(h.detach().float(), indices, valid, positions)
                signal = ((p['signal_z'] - self.label_center[indices][:, None])
                          / self.label_scale[indices][:, None]) * valid[:, None]
                total = p['window_total_z']
                if self.region_total_length is not None:
                    total = (total + self.region_total_length(
                        torch.log1p(positions.sum(1, keepdim=True).float()))) * valid
            peak = self.peak_decoder(h.detach().float(), p['peak_logits'], signal, indices, valid, positions)
            return dict(signal_z=signal, peak_logits=peak,
                        profile_logits=p['profile_logits'], total_z=total,
                        profile_score=p['signal_profile_score'])

    def forward(self, dna: Tensor, track_indices: Tensor, *,
                track_valid_mask: Tensor | None = None,
                position_valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Accept A/C/G/T/N tokens [B,bp] or the original one-hot [B,bp,5]."""
        if dna.device != self.device:
            raise ValueError('DNA and model must be on the same device')
        if dna.ndim == 2:
            if dna.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
                raise ValueError('DNA tokens must be integers A/C/G/T/N = 0/1/2/3/4')
            if torch.any((dna < 0) | (dna > 4)):
                raise ValueError('DNA tokens must be in [0,4]')
            dna = F.one_hot(dna.long(), num_classes=5).float()
        with torch.no_grad():
            features = self.backbone.forward_features(dna)['regulatory_32bp']
        return self.from_features(features, track_indices, track_valid_mask=track_valid_mask,
                                  position_valid_mask=position_valid_mask)

    def signal_in_registry_space(self, signal_z, track_indices, track_valid_mask=None):
        """Undo TRAIN standardization, leaving the registry's original signal transform."""
        indices, valid = self._track_mask(track_indices, track_valid_mask, signal_z.shape[0])
        return (signal_z * self.label_scale[indices][:, None]
                + self.label_center[indices][:, None]) * valid[:, None]

    @classmethod
    def from_pretrained(cls, directory, *, device='cpu'):
        """Load one PhytoReg directory after verifying every package checksum."""
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError('Expected a PhytoReg model directory')
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('format') != FORMAT or manifest.get('model_name') != 'PhytoReg':
            raise ValueError('Expected PhytoReg model format')
        if manifest.get('weights_file') != 'model.pt':
            raise ValueError('Expected model.pt weights')
        files = manifest.get('files', [])
        names = [row['path'] for row in files]
        if len(set(names)) != len(names) or not {'config.json', 'tracks.json', 'metadata.json'} <= set(names):
            raise ValueError('Incomplete PhytoReg package manifest')
        for row in files:
            name = row['path']
            if Path(name).name != name or not (directory / name).resolve().is_relative_to(directory.resolve()):
                raise ValueError('Package files must be within the model directory')
            if file_sha256(directory / name) != row['sha256']:
                raise ValueError(f'Package checksum mismatch: {name}')
        weights = directory / 'model.pt'
        if not weights.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Weights must be within the model directory')
        actual = file_sha256(weights)
        if actual != manifest['weights_sha256']:
            raise ValueError('Model checksum mismatch')
        config = PhytoRegConfig.from_json(directory / 'config.json')
        state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
        lookup = state['head.lookup']
        indices = torch.nonzero(lookup >= 0).flatten()
        if indices.numel() != config.head.num_tracks or not torch.equal(
                lookup[indices].sort().values, torch.arange(config.head.num_tracks)):
            raise ValueError('Invalid registry-to-head mapping in checkpoint')
        indices = indices[lookup[indices].argsort()]
        catalog = json.loads((directory / 'tracks.json').read_text(encoding='utf-8'))
        model = cls(config, indices, state['head.normalization_group'], track_catalog=catalog)
        model.load_state_dict(state, strict=True)
        active = model.head.lookup >= 0
        if not torch.isfinite(model.label_center[active]).all() or not torch.all(
                torch.isfinite(model.label_scale[active]) & (model.label_scale[active] > 0)):
            raise ValueError('Invalid TRAIN label normalization')
        if not torch.all(torch.isfinite(model.head.feature_std) & (model.head.feature_std > 0)):
            raise ValueError('Invalid feature normalization')
        model.metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        model.metadata.update(model_name='PhytoReg', loaded_weights_sha256=actual)
        model._validate_catalog()
        return model.to(device).eval()

    def _validate_catalog(self):
        indices = [r['registry_index'] for r in self.track_catalog]
        if len(indices) != self.config.head.num_tracks or len(set(indices)) != len(indices):
            raise ValueError('Track catalog does not cover the independent heads exactly')
        groups_by_species = {}
        for row in self.track_catalog:
            index, head_row = row['registry_index'], row['head_row']
            if not 0 <= index < len(self.head.lookup) or int(self.head.lookup[index]) != head_row:
                raise ValueError('Track catalog disagrees with checkpoint registry')
            group = int(self.head.normalization_group[head_row])
            groups_by_species.setdefault(row['species_id'], set()).add(group)
        if (any(len(groups) != 1 for groups in groups_by_species.values())
                or len({next(iter(g)) for g in groups_by_species.values()}) != len(groups_by_species)):
            raise ValueError('Track catalog species disagree with normalization groups')
        for row in self.track_catalog:
            index = row['registry_index']
            condition = row.get('spatial_condition_index', 0)
            if (row['task_group'] not in TASKS or not 0 <= condition < self.config.spatial.num_conditions
                    or int(self.peak_decoder.tasks[index]) != TASKS.index(row['task_group'])
                    or int(self.peak_decoder.conditions[index]) != condition
                    or bool(self.peak_decoder.enabled[index]) != row.get('peak_enabled', True)):
                raise ValueError('Track catalog disagrees with spatial readout metadata')
        tf_ids = [r['registry_index'] for r in self.track_catalog if r['task_group'] == 'TF']
        if sorted(self.peak_decoder.tf_lookup[tf_ids].tolist()) != list(range(len(tf_ids))):
            raise ValueError('Invalid TF feature readout mapping')
        for name in ('context_gain', 'readout_gain'):
            gain = getattr(self.peak_decoder, name)
            if not torch.all(torch.isfinite(gain) & (gain >= 0) & (gain <= 1)):
                raise ValueError('Invalid spatial calibration coefficients')
        if not torch.all(torch.isfinite(self.peak_decoder.input_std) & (self.peak_decoder.input_std > 0)):
            raise ValueError('Invalid spatial input normalization')

    def save_pretrained(self, directory):
        """Save the PhytoReg format without optimizer state; never overwrite."""
        self._validate_catalog()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {k: v for k, v in self.metadata.items() if k != 'loaded_weights_sha256'}
        metadata['model_name'] = 'PhytoReg'
        contents = {'config.json': self.config.to_dict(), 'tracks.json': self.track_catalog,
                    'metadata.json': metadata}
        files = []
        for name, content in contents.items():
            path = directory / name
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            files.append(dict(path=name, sha256=file_sha256(path)))
        weights = directory / 'model.pt'
        torch.save({k: v.detach().cpu() for k, v in self.state_dict().items()}, weights)
        manifest = dict(model_name='PhytoReg', format=FORMAT, weights_file=weights.name,
                        weights_sha256=file_sha256(weights), files=files)
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
